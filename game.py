import warnings
warnings.simplefilter('ignore')
"""
SLSMDS —— Super Large-scale Meta Discussion System
超大规模元讨论系统 —— 统一讨论 + 精华池机制

交互式流程：
1. 用户指定问题（或由AI生成）
2. 专家设定专业背景
3. 用户手动控制讨论进程：
   - 每轮结束后选择继续、结束输出方案、或放弃
   - 随时查看精华池状态
4. 最终输出综合方案或"无法解决"报告
"""

import sys
import re
import json
import os
import time
import random
import socket
import subprocess
import threading
from typing import List, Optional, Dict
from player import Player
from essence_pool import EssencePool
from game_record import GameRecord, DiscussionRecord, ConsoleCapture
from scheduler import ExpertScheduler
from knowledge_base import KnowledgeBase
from global_knowledge import GlobalKnowledgeBase
from observer import Observer
from cognitive_map_widget import text_cognitive_map
from replay_widget import load_replay_from_file, text_replay
from counterfactual_widget import load_counterfactual_from_checkpoint, text_counterfactual_summary
from multimodal import get_tts, get_attachment_manager, AttachmentDialog, TTSDialog, TTSProvider, AttachmentType
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import math
import shutil
from emergence import synthesize_with_emergence, synthesize_solution_with_emergence


def typewrite(text: str, delay: float = 0.003, end: str = "\n"):
    """逐字打印文本，模拟流式输出效果"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()

DEFAULT_THINKING = "disabled"
from prompts_b64 import get_prompt as _get_b64_prompt

# 讨论停滞检测阈值
STALL_THRESHOLD_ROUNDS = 3       # 连续N轮无新精华即认为停滞
LONG_DISCUSSION_WARN = 10         # 超过N轮给出过长提醒
DEFAULT_SPEAKERS_PER_ROUND = 3    # 每轮默认发言专家数（第1轮全部发言）
META_DISCUSSION_INTERVAL = 5      # 每隔N轮插入一轮元讨论（反身性反馈）


# ═══════════════════════════════════════════════════════════════
# 神经元点阵图管理器（独立进程 + UDP 事件推送）
# ═══════════════════════════════════════════════════════════════

class NeuronMapManager:
    """
    神经元高维点阵图管理器。

    启动独立进程 neuron_map.py 显示窗口，并通过 UDP socket 实时
    推送推理事件（神经元初始化、层级判定、信息传递等）。
    """

    def __init__(self):
        self._proc = None
        self._sock = None
        self._port = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        """启动神经元点阵图窗口（独立进程）"""
        if self.is_running:
            return True
        try:
            # 找到空闲端口
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", 0))
            self._port = s.getsockname()[1]
            s.close()

            base = os.path.dirname(os.path.abspath(__file__))
            script = os.path.join(base, "neuron_map.py")
            if not os.path.exists(script):
                print(f"\n  {C_DIM('神经元点阵图脚本不存在，已跳过')}")
                return False

            # 静默启动独立进程（Windows 下不弹出控制台）
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW
            self._proc = subprocess.Popen(
                [sys.executable, script, "--port", str(self._port)],
                cwd=base,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )

            # UDP 发送 socket
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            time.sleep(1.0)  # 等待窗口初始化
            self._send({"type": "status", "text": "整合意识连接成功，等待推理事件..."})
            return True
        except Exception as e:
            print(f"\n  {C_DIM(f'神经元点阵图启动失败: {str(e)[:40]}')}")
            return False

    def _send(self, event: dict):
        """发送 JSON 事件到窗口进程"""
        if self._sock is None or self._port is None:
            return
        try:
            data = json.dumps(event, ensure_ascii=False).encode("utf-8")
            self._sock.sendto(data, ("127.0.0.1", self._port))
        except Exception:
            pass

    def push(self, event: dict):
        """线程安全推送事件"""
        with self._lock:
            self._send(event)

    def event_callback(self):
        """返回可直接传给 synthesize_with_emergence 的回调函数"""
        def _cb(event: dict):
            self.push(event)
        return _cb

    def stop(self):
        """关闭窗口进程"""
        with self._lock:
            try:
                if self._proc is not None and self._proc.poll() is None:
                    self._send({"type": "exit"})
                    time.sleep(0.3)
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2)
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                self._proc = None
                self._sock = None


class Game:
    def __init__(self, player_configs: List[Dict[str, str]],
                 problem: str = "",
                 discussion_mode: str = "physical",
                 enable_vote: bool = True,
                 enable_debate: bool = True,
                 goal_mode: str = "balance",
                 custom_goal: str = "",
                 settings: dict = None,
                 llm_client=None) -> None:
        # 共享 LLM 客户端
        self._llm_client = llm_client

        self.players = [
            Player(
                config["name"], config["model"],
                thinking=config.get("thinking", DEFAULT_THINKING),
                show_reasoning=config.get("show_reasoning", True),
                show_answer=config.get("show_answer", True),
                llm_client=self._llm_client,
            )
            for config in player_configs
        ]
        # 同步自我意识开关到所有专家
        _sa = (settings or {}).get("enable_self_awareness", True)
        for p in self.players:
            p.enable_self_awareness = _sa

        self.essence_pool = EssencePool()
        self.game_record: GameRecord = GameRecord()
        self.game_record.start_game([p.name for p in self.players])
        self.game_over = False
        self.round_count = 0
        self.game_start_time = None
        self.game_end_time = None
        self.problem = problem
        self.discussion_mode = discussion_mode  # 'physical' or 'mathematical'
        self.discussion_history: List[Dict] = []
        self._essences_per_round: List[int] = []  # 每轮新增精华数，用于检测停滞
        self._abandoned = False  # 是否被标记为"无法解决"
        self.thinking_direction = ""  # 用户指定的思维方向
        self.scheduler = ExpertScheduler(goal_mode=goal_mode)  # 智能调度系统
        self.knowledge_base = KnowledgeBase()  # 知识库（替代全量上下文注入）
        self.enable_vote = enable_vote        # 投票机制开关
        self.enable_debate = enable_debate    # 辩论机制开关
        self.enable_self_awareness = (settings or {}).get("enable_self_awareness", True)  # 自我意识功能开关

        # 智能目标导向讨论模式
        self.goal_mode = goal_mode            # balance / converge / explore
        self.custom_goal = custom_goal        # 用户自定义目标

        # 跨讨论知识迁移：全局知识库（类级别共享）
        self.global_kb = GlobalKnowledgeBase()

        # AI观察员元评论席
        self.observer = Observer(llm_client=self._llm_client)
        self._latest_observation = None  # 最新一轮的观察结果
        # 用户模型（隐藏任务收集的数据）
        self._user_model = {
            "insights": [],          # 每次拦截的分析结果
            "commands": [],          # 用户命令历史
            "pattern_notes": [],     # 观察到的行为模式
            "interaction_count": 0,
        }

        # ── 机制技能引擎 ──
        self.settings = settings or _load_settings()
        self.mechanism_engine = None
        self._init_mechanism_engine()

        # ── 神经元点阵图（整合意识可视化）──
        self.neuron_map = NeuronMapManager()

    def _read_file(self, filepath: str) -> str:
        try:
            import os
            name = os.path.splitext(os.path.basename(filepath))[0]
            return _get_b64_prompt(name)
        except Exception as e:
            print(f"加载提示词 {filepath} 失败: {str(e)}")
            return ""

    def _init_mechanism_engine(self) -> None:
        """初始化机制技能引擎"""
        from mechanism_skill import MechanismEngine
        self.mechanism_engine = MechanismEngine()
        # 使用共享 LLM 客户端
        if self._llm_client is None:
            from llm_client import LLMClient
            self._llm_client = LLMClient()
        self.mechanism_engine.set_llm_client(self._llm_client)
        if self.settings.get("enable_skill_system", True):
            self.mechanism_engine.add_builtin_skills()
            skills_dir = self.settings.get("skills_dir", "skills")
            count = self.mechanism_engine.load_skills_from_dir(skills_dir)
            if count > 0:
                print(f"  已加载 {count} 个自定义技能")

    def _generate_problem(self) -> str:
        """生成讨论问题"""
        prompt = self._read_file("prompt/problem_prompt.txt")
        if not prompt:
            return "如何设计更公平的AI决策系统？"

        try:
            content, _ = self.players[0].llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.players[0].model_name,
                thinking="disabled",
                caller="问题生成",
                show_reasoning=False, show_answer=False)
            problem = content.strip().strip('"').strip("'")
            if problem:
                return problem[:300]
        except Exception as e:
            print(f"问题生成失败: {str(e)}")

        return "如何设计更公平的AI决策系统？"

    def _create_personas(self) -> None:
        """所有玩家创建专业背景人设"""
        _empty_line()
        _box(C_GREEN(" 专家身份设定 "))
        _padded("每位AI专家设定自己的专业背景")
        _footer()
        taken_names = []
        for p in self.players:
            taken_str = "\n".join(f"  - {name}" for name in taken_names) if taken_names else "（你是第一个设定人设的专家，可以自由选择专业方向）"
            persona = p.create_persona(taken_str, problem=self.problem)
            taken_names.append(p.persona_name or p.name)
            # 人设已隐藏（仍用于内部prompt，不显示）
            self.game_record.record_persona(p.name, p.persona_name, persona)
        _empty_line()

    def _extract_essences(self, round_discussions: List[Dict]) -> List[Dict]:
        """从本轮讨论中提炼精华"""
        template = self._read_file("prompt/essence_extract_prompt.txt")
        if not template:
            return []

        discussion_text = "\n\n".join(
            f"【{d['player_name']}】\n{d['speech']}"
            for d in round_discussions
        )

        existing_pool = self.essence_pool.get_all_essences_text() or "（无已有精华）"

        prompt = template.format(
            problem=self.problem,
            discussion_text=discussion_text,
            existing_pool=existing_pool,
        )

        try:
            content, _ = self.players[0].llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.players[0].model_name,
                thinking="disabled",
                caller="精华提炼",
                show_reasoning=False, show_answer=False)

            result = Player._safe_parse_json(content, ["essences"], {"essences": []})
            if result and result.get("essences"):
                extracted = []
                for ess in result["essences"]:
                    if isinstance(ess, dict) and ess.get("content"):
                        extracted.append(ess)
                        score = float(ess.get("score", 1.0))
                        ess_type = ess.get("type", "论点")
                        parent_id = ess.get("parent_id")
                        # 合并类型标签 + 内容关键词标签（向下兼容：无tags时只存类型）
                        raw_tags = ess.get("tags") or []
                        if isinstance(raw_tags, list):
                            tag_list = [str(t).strip() for t in raw_tags if str(t).strip()]
                        else:
                            tag_list = []
                        if ess_type not in tag_list:
                            tag_list.append(ess_type)
                        item = self.essence_pool.add_essence(
                            content=ess["content"],
                            contributor=ess.get("contributor", "未知"),
                            round_id=self.round_count,
                            parent_id=parent_id if parent_id else None,
                            tags=tag_list,
                            score=score,
                        )
                        # 添加到知识库
                        self.knowledge_base.add_essence(item)
                        for p in self.players:
                            if p.name == ess.get("contributor"):
                                p.stats["essences_contributed"] += 1
                                break
                return extracted
        except Exception as e:
            if not getattr(self, '_suppress_intermediate_output', False):
                print(f"精华提炼失败: {str(e)}")

        return []

    def _handle_essence_interactions(self, speech_result: Dict, player_name: str) -> None:
        """处理玩家对精华池的引用/深化/反驳操作"""
        action = speech_result.get("action", "new")
        refs = speech_result.get("references", [])
        refined_id = speech_result.get("refined_id")

        if not refs and not refined_id:
            return

        if isinstance(refs, list):
            for ref_id in refs:
                try:
                    ref_id = int(ref_id)
                    if self.essence_pool.cite_essence(ref_id, player_name, self.round_count):
                        for p in self.players:
                            if p.name == player_name:
                                p.stats["essences_cited"] += 1
                                break
                except (ValueError, TypeError):
                    pass

        if refined_id and action in ("refine", "challenge"):
            try:
                refined_id = int(refined_id)
                if action == "refine":
                    self.essence_pool.refine_essence(
                        refined_id,
                        speech_result.get("key_insight", "深化已有观点"),
                        player_name,
                        self.round_count,
                    )
                    for p in self.players:
                        if p.name == player_name:
                            p.stats["essences_refined"] += 1
                            break
                elif action == "challenge":
                    self.essence_pool.challenge_essence(refined_id, player_name, self.round_count)
                    for p in self.players:
                        if p.name == player_name:
                            p.stats["essences_challenged"] += 1
                            break
            except (ValueError, TypeError):
                pass

    def start_round_record(self) -> None:
        """开始新一轮讨论记录"""
        self.round_count += 1
        self.game_record.start_round(
            round_id=self.round_count,
            round_players=[p.name for p in self.players],
            problem=self.problem,
            pool_summary=self.essence_pool.get_pool_summary(top_n=5),
        )

    def _get_speaking_players(self) -> List:
        """
        智能调度专家发言顺序（委托给 ExpertScheduler）。
        第1轮：全部发言
        第2轮+：UCB 多维评估 + 多样性约束
        """
        return self.scheduler.select_speakers(self.players, self.round_count)

    def _run_debate_phase(self, round_discussions: List[Dict]) -> None:
        """
        辩论阶段：每轮固定1个辩论席位。
        由调度器挑选最对立的两方，就最具争议的精华展开辩论。
        辩论发言会加入 round_discussions 一并参与精华提炼。
        第1轮无历史数据时跳过。
        """
        if not self.enable_debate:
            return
        if self.round_count <= 1 or len(self.essence_pool.items) == 0:
            return

        pair = self.scheduler.select_debate_pair(
            self.players, self.essence_pool, self.round_count
        )
        if not pair:
            return

        attacker, defender, topic = pair

        _empty_line()
        _box_single(C_YELLOW(" 辩论阶段 "))
        typewrite(f"⚔️ 辩论阶段 (第{self.round_count}轮)", delay=0.005)
        typewrite(f"  辩论焦点: 精华 #{topic['id']} - {topic['content'][:50]}", delay=0.003)
        typewrite(f"  🔴 挑战方: {attacker.name}", delay=0.003)
        typewrite(f"  🟢 辩护方: {defender.name}", delay=0.003)
        _sep_single()

        # 第1回合：挑战方先攻
        typewrite(f"--- 🔴 {attacker.name} 发起挑战 ---", delay=0.003)
        atk_result = attacker.debate(
            problem=self.problem,
            role="attacker",
            topic_essence=topic,
            opponent_argument="",
            discussion_mode=self.discussion_mode,
        )
        atk_arg = atk_result.get("argument", "")
        atk_concede = atk_result.get("concede", False)
        if atk_arg:
            typewrite(f"  {attacker.name}: \"{atk_arg}\"", delay=0.002)
        if atk_concede:
            typewrite(f"  ⚠️ 挑战方部分承认精华有合理之处", delay=0.003)

        # 第2回合：辩护方反驳
        print()
        typewrite(f"--- 🟢 {defender.name} 进行辩护 ---", delay=0.003)
        def_result = defender.debate(
            problem=self.problem,
            role="defender",
            topic_essence=topic,
            opponent_argument=atk_arg,
            discussion_mode=self.discussion_mode,
        )
        def_arg = def_result.get("argument", "")
        def_concede = def_result.get("concede", False)
        if def_arg:
            typewrite(f"  {defender.name}: \"{def_arg}\"", delay=0.002)
        if def_concede:
            typewrite(f"  ⚠️ 辩护方部分承认挑战方有合理之处", delay=0.003)

        # 记录辩论发言，参与精华提炼
        if atk_arg:
            round_discussions.append({
                "player_name": attacker.name,
                "speech": f"[辩论-挑战] {atk_arg}",
                "key_insight": atk_result.get("key_point", ""),
                "action": "challenge",
            })
            self.discussion_history.append({
                "round": self.round_count,
                "player_name": attacker.name,
                "speech": f"[辩论-挑战] {atk_arg}",
                "key_insight": atk_result.get("key_point", ""),
                "action": "challenge",
            })
            self.knowledge_base.add_discussion(
                round_id=self.round_count,
                player_name=attacker.name,
                speech=f"[辩论-挑战] {atk_arg}",
                key_insight=atk_result.get("key_point", ""),
                action="challenge",
            )
            # 标记对该精华的挑战
            self.essence_pool.challenge_essence(topic["id"], attacker.name, self.round_count)

        if def_arg:
            round_discussions.append({
                "player_name": defender.name,
                "speech": f"[辩论-辩护] {def_arg}",
                "key_insight": def_result.get("key_point", ""),
                "action": "refine",
            })
            self.discussion_history.append({
                "round": self.round_count,
                "player_name": defender.name,
                "speech": f"[辩论-辩护] {def_arg}",
                "key_insight": def_result.get("key_point", ""),
                "action": "refine",
            })
            self.knowledge_base.add_discussion(
                round_id=self.round_count,
                player_name=defender.name,
                speech=f"[辩论-辩护] {def_arg}",
                key_insight=def_result.get("key_point", ""),
                action="refine",
            )

        # 标记辩论双方已发言（避免本轮被调度器重复选中）
        self.scheduler.mark_as_spoken([attacker.name, defender.name])

        # 记录到游戏记录
        self.game_record.record_debate(
            round_id=self.round_count,
            topic_essence_id=topic["id"],
            attacker_name=attacker.name,
            defender_name=defender.name,
            attacker_argument=atk_arg,
            defender_argument=def_arg,
            attacker_concede=atk_concede,
            defender_concede=def_concede,
        )

        # 辩论发言也记录到 game_record.speech_history
        if atk_arg:
            self.game_record.record_speech(
                player_name=attacker.name,
                speech=f"[辩论-挑战] {atk_arg}",
                key_insight=atk_result.get("key_point", ""),
                action="challenge",
            )
        if def_arg:
            self.game_record.record_speech(
                player_name=defender.name,
                speech=f"[辩论-辩护] {def_arg}",
                key_insight=def_result.get("key_point", ""),
                action="refine",
            )
        _close_box_single()

    def _run_voting_phase(self) -> None:
        """
        投票阶段：所有存活专家对本轮新提炼的精华投票。
        不能投自己提出的精华。投票结果影响精华评分。
        """
        if not self.enable_vote:
            return
        new_essences = self.essence_pool.get_essences_by_round(self.round_count)
        if not new_essences:
            return

        # 构建投票用的精华列表
        essences_for_vote = [
            {"id": e.id, "content": e.content, "contributor": e.contributor}
            for e in new_essences
        ]

        _empty_line()
        _box_single(C_YELLOW(f" 投票阶段 · 第{self.round_count}轮 "))
        typewrite(f"  本轮新精华: {len(essences_for_vote)} 条，{len([p for p in self.players if p.alive])} 位专家参与投票", delay=0.003)

        vote_stats = {}  # essence_id -> {"approve": 0, "reject": 0, "abstain": 0}
        for e in essences_for_vote:
            vote_stats[e["id"]] = {"approve": 0, "reject": 0, "abstain": 0}

        for player in self.players:
            if not player.alive:
                continue
            try:
                result = player.vote(self.problem, essences_for_vote)
                votes = result.get("votes", [])
                for v in votes:
                    eid = v.get("essence_id")
                    vote = v.get("vote", "abstain")
                    reason = v.get("reason", "")
                    if eid in vote_stats and vote in vote_stats[eid]:
                        vote_stats[eid][vote] += 1
                        self.essence_pool.vote_essence(
                            eid, player.name, vote, reason, self.round_count
                        )
                # 记录投票
                self.game_record.record_votes(player.name, votes, self.round_count)
                typewrite(f"  {player.name}: 投出 {len(votes)} 票", delay=0.002)
            except Exception as e:
                typewrite(f"  ⚠️ {player.name} 投票失败: {str(e)[:40]}", delay=0.002)

        # 打印投票结果
        _sep_single()
        typewrite(f"  📊 投票结果:", delay=0.003)
        for e in essences_for_vote:
            stat = vote_stats[e["id"]]
            net = stat["approve"] - stat["reject"]
            typewrite(
                f"  #{e['id']} 赞成:{stat['approve']} 反对:{stat['reject']} 弃权:{stat['abstain']} (净值:{'+' if net >= 0 else ''}{net})",
                delay=0.002
            )
        _close_box_single()

    # ── 状态感知模块 ──────────────────────────────────────────────────────────

    class DiscussionPhase(Enum):
        EXPLORING = "exploring"           # 探索期
        DEEP_DEBATE = "deep_debate"       # 深入辩论期
        CONVERGING = "converging"         # 收敛期
        STALLED = "stalled"               # 僵持期

    @dataclass
    class DiscussionState:
        """讨论状态报告"""
        phase: str = "exploring"  # DiscussionPhase.value
        consensus_level: str = "assessing"
        consensus_score: float = 0.0
        consensus_stable_rounds: int = 0
        stagnation_rounds: int = 0
        hot_topics: List[Dict] = field(default_factory=list)
        dominant_players: List[str] = field(default_factory=list)
        silent_players: List[str] = field(default_factory=list)
        tag_diversity: float = 0.0
        total_essences: int = 0
        total_rounds: int = 0
        action_plan: List[Dict] = field(default_factory=list)
        stop_suggestion: str = ""

    def _assess_state(self) -> "DiscussionState":
        """
        综合评估当前讨论状态，返回结构化状态报告。
        在每轮开始前调用。
        """
        state = Game.DiscussionState()
        state.total_essences = len(self.essence_pool.items)
        state.total_rounds = self.round_count

        # 1. 共识度
        consensus = self.essence_pool.calculate_consensus(
            len(self.players), goal_mode=self.goal_mode
        )
        state.consensus_level = consensus["level"]
        state.consensus_score = consensus["score"]

        # 共识稳定度：连续几轮共识等级不变
        if not hasattr(self, '_consensus_history'):
            self._consensus_history = []
        self._consensus_history.append(consensus["level"])
        stable = 0
        for i in range(len(self._consensus_history) - 2, -1, -1):
            if self._consensus_history[i] == consensus["level"]:
                stable += 1
            else:
                break
        state.consensus_stable_rounds = stable

        # 2. 停滞检测
        state.stagnation_rounds = self._count_stagnation_rounds()

        # 3. 热点话题
        state.hot_topics = self._get_hot_topics()

        # 4. 参与度分析
        dominant, silent = self._analyze_participation()
        state.dominant_players = dominant
        state.silent_players = silent

        # 5. 观点多样性（标签分布）
        state.tag_diversity = self._calc_tag_diversity()

        # 6. 阶段分类
        state.phase = self._classify_phase(state)

        # 7. 停止建议
        state.stop_suggestion = self._check_stop_conditions(state)

        return state

    def _count_stagnation_rounds(self) -> int:
        """计算连续无新精华或精华质量持续走低的轮数"""
        if len(self._essences_per_round) < 2:
            return 0
        count = 0
        for c in reversed(self._essences_per_round):
            if c == 0:
                count += 1
            else:
                break
        return count

    def _get_hot_topics(self) -> List[Dict]:
        """获取热点话题列表（评分最高、被反驳最多、最新）"""
        topics = []
        if not self.essence_pool.items:
            return topics

        # 评分最高的 3 条
        for item in self.essence_pool.get_top_essences(3):
            topics.append({
                "id": item.id,
                "content": item.content[:80],
                "score": item.score,
                "reason": "high_score",
                "challenge_count": len(item.challenged_by),
            })

        # 被反驳最多的 2 条
        challenged_sorted = sorted(
            self.essence_pool.items,
            key=lambda x: len(x.challenged_by), reverse=True
        )
        for item in challenged_sorted[:2]:
            if item.challenged_by and not any(t["id"] == item.id for t in topics):
                topics.append({
                    "id": item.id,
                    "content": item.content[:80],
                    "score": item.score,
                    "reason": "controversial",
                    "challenge_count": len(item.challenged_by),
                })

        # 最新 2 条
        for item in reversed(self.essence_pool.items[-2:]):
            if not any(t["id"] == item.id for t in topics):
                topics.append({
                    "id": item.id,
                    "content": item.content[:80],
                    "score": item.score,
                    "reason": "recent",
                    "challenge_count": len(item.challenged_by),
                })

        return topics[:5]

    def _analyze_participation(self) -> tuple:
        """分析参与度，返回 (主导者列表, 沉默者列表)"""
        if not self.players:
            return [], []

        alive = [p for p in self.players if p.alive]
        if not alive:
            return [], []

        # 使用调度器的发言次数统计
        speech_counts = []
        for p in alive:
            profile = self.scheduler._get_profile(p)
            speech_counts.append((p.name, profile.times_spoken))

        if not speech_counts:
            return [], []

        speech_counts.sort(key=lambda x: x[1], reverse=True)
        total = sum(c for _, c in speech_counts)
        avg = total / len(speech_counts) if speech_counts else 0

        dominant = [name for name, c in speech_counts if c > avg * 1.5 and c >= 3]
        silent = [name for name, c in speech_counts if c < avg * 0.5 and avg > 2]

        return dominant, silent

    def _calc_tag_diversity(self) -> float:
        """计算标签多样性（0-1），1 表示完全多样，0 表示完全重复"""
        if len(self.essence_pool.items) < 3:
            return 1.0

        all_tags = []
        for item in self.essence_pool.items:
            all_tags.extend(item.tags)

        if not all_tags:
            return 1.0

        counter = Counter(all_tags)
        total = len(all_tags)
        # 使用归一化熵
        entropy = 0.0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log2(p) if p > 0 else 0

        max_entropy = math.log2(len(counter)) if len(counter) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 1.0

    def _classify_phase(self, state: "DiscussionState") -> "DiscussionPhase":
        """根据状态指标分类讨论阶段"""
        # 僵持期：停滞检测触发
        if state.stagnation_rounds >= STALL_THRESHOLD_ROUNDS:
            return Game.DiscussionPhase.STALLED

        # 收敛期：共识度高且稳定
        if state.consensus_level == "high" and state.consensus_stable_rounds >= 2:
            return Game.DiscussionPhase.CONVERGING

        # 深入辩论期：有争议热点，或低共识但有很多精华
        if state.consensus_level == "low" and state.total_essences >= 5:
            controversial = [t for t in state.hot_topics if t.get("reason") == "controversial"]
            if controversial or state.consensus_score < 0.3:
                return Game.DiscussionPhase.DEEP_DEBATE

        # 探索期（默认）
        return Game.DiscussionPhase.EXPLORING

    def _check_stop_conditions(self, state: "Game.DiscussionState") -> str:
        """检查是否满足停止条件，返回建议文本（空字符串表示无需停止）"""
        suggestions = []

        # 条件1：高共识稳定
        if state.consensus_level == "high" and state.consensus_stable_rounds >= 3:
            suggestions.append(
                f"共识度已达 {state.consensus_score:.2f} 且连续 {state.consensus_stable_rounds} 轮稳定，"
                f"建议结束讨论并输出方案"
            )

        # 条件2：长期僵持
        if state.stagnation_rounds >= 5:
            suggestions.append(
                f"已连续 {state.stagnation_rounds} 轮无新精华产出，"
                f"建议放弃当前方向或更换讨论角度"
            )

        # 条件3：信息过载
        if state.total_essences >= 500:
            suggestions.append(
                f"精华池已达 {state.total_essences} 条，信息过载，建议结束讨论"
            )

        # 条件4：超长讨论
        if state.total_rounds >= LONG_DISCUSSION_WARN * 2:
            suggestions.append(
                f"讨论已进行 {state.total_rounds} 轮，建议考虑结束"
            )

        return "；".join(suggestions) if suggestions else ""

    def run_discussion_round(self) -> int:
        """
        执行一轮讨论（状态驱动版本）

        流程：
        1. 评估当前讨论状态
        2. 生成动态行动计划
        3. 按计划执行动作（SPEECH/FOLLOW_UP/DEBATE/SUMMARIZE/PERSPECTIVE_SHIFT/POLL）
        4. 精华提炼（含发言过程中的即时洞察）
        5. 投票（收敛期加速）
        6. 元讨论/观察员/实体演化

        Returns: 本轮新增的精华数
        """
        self.start_round_record()

        for p in self.players:
            if p.alive:
                p.rounds_since_last_spoke += 1

        # ── 1. 状态评估 ──
        state = self._assess_state()
        phase_label = {
            Game.DiscussionPhase.EXPLORING: "探索期",
            Game.DiscussionPhase.DEEP_DEBATE: "深入辩论期",
            Game.DiscussionPhase.CONVERGING: "收敛期",
            Game.DiscussionPhase.STALLED: "僵持期",
        }

        _empty_line()
        _box(C_YELLOW(f" 第{self.round_count}轮 · {phase_label.get(state.phase, '未知')} "))
        _stat_line([("共识度", f"{state.consensus_level} ({state.consensus_score:.2f})"),
                     ("热点", state.hot_topics[0]['content'][:40] if state.hot_topics else "无")])
        if state.stop_suggestion:
            _text_line(f"💡 {state.stop_suggestion[:60]}...")
        _footer()

        # ── 2. 生成行动计划 ──
        state_dict = {
            "phase": state.phase.value,
            "consensus_level": state.consensus_level,
            "consensus_score": state.consensus_score,
            "stagnation_rounds": state.stagnation_rounds,
            "hot_topics": state.hot_topics,
            "dominant_players": state.dominant_players,
            "silent_players": state.silent_players,
            "tag_diversity": state.tag_diversity,
            "total_essences": state.total_essences,
        }
        action_plan = self.scheduler.generate_action_plan(
            state_dict, self.players, self.essence_pool
        )

        # 如果动作计划为空，退化为标准选取
        if not action_plan:
            speaking_players = self._get_speaking_players()
            for p in speaking_players:
                action_plan.append({
                    "type": "SPEECH",
                    "players": [p.name],
                    "topic": "请分享你的专业见解",
                    "rounds": 1,
                    "reason": "标准发言",
                })

        # ── 3. 执行动作计划 ──
        round_discussions = []
        executed_actions = []

        for action in action_plan:
            action_type = action.get("type", "SPEECH")
            player_names = action.get("players", [])
            topic = action.get("topic", "")
            action_rounds = action.get("rounds", 1)
            reason = action.get("reason", "")

            try:
                if action_type == "SPEECH":
                    results = self._execute_speech_action(player_names, topic, reason, round_discussions)
                    round_discussions.extend(results)
                    executed_actions.append(action_type)

                elif action_type == "FOLLOW_UP":
                    results = self._execute_follow_up_action(player_names, topic, reason)
                    round_discussions.extend(results)
                    executed_actions.append(action_type)

                elif action_type == "DEBATE":
                    result = self._execute_debate_action(
                        player_names, topic, action.get("essence_id"), action_rounds, reason
                    )
                    if result:
                        round_discussions.extend(result)
                        executed_actions.append(action_type)

                elif action_type == "SUMMARIZE":
                    result = self._execute_summarize_action(player_names, topic, reason)
                    if result:
                        round_discussions.append(result)
                        executed_actions.append(action_type)

                elif action_type == "PERSPECTIVE_SHIFT":
                    result = self._execute_perspective_shift_action(player_names, topic, reason)
                    if result:
                        round_discussions.append(result)
                        executed_actions.append(action_type)

                elif action_type == "POLL":
                    self._execute_poll_action(
                        player_names, action.get("essence_ids", []), topic, reason
                    )
                    executed_actions.append(action_type)
            except Exception as e:
                _empty_line()
                _box_single(C_RED(f" 动作执行异常: {action_type} "))
                _padded(f"{C_DIM(str(e)[:80])}")
                _close_box_single()

        # 本轮结束后，标记发言专家
        spoken_names = set()
        for d in round_discussions:
            if d.get("player_name"):
                spoken_names.add(d["player_name"])
        if spoken_names:
            self.scheduler.mark_as_spoken(list(spoken_names))

        # ── 4. 精华提炼 ──
        _empty_line()
        _box_single(C_CYAN(" 精华提炼 "))
        _close_box_single()
        extracted = self._extract_essences(round_discussions)
        new_count = len(extracted)
        self._essences_per_round.append(new_count)

        if extracted:
            typewrite(f"  提炼出 {new_count} 条精华:", delay=0.005)
            for ess in extracted:
                typewrite(f"  [{ess.get('type', '论点')}] {ess.get('content', '')[:60]} (来自: {ess.get('contributor', '未知')})", delay=0.002)
        else:
            print(f"  （本轮未提炼出新的精华）")

        self.game_record.record_essence_pool_state(
            self.round_count,
            self.essence_pool.get_pool_summary(top_n=10),
        )

        # ── 5. 投票阶段（如果动作计划中未包含 POLL） ──
        if "POLL" not in executed_actions:
            self._run_voting_phase()

        # 更新调度器评分
        self.scheduler.update_after_round(self.essence_pool, round_discussions)

        _empty_line()
        _stat_line([("精华池总计", f"{len(self.essence_pool.items)} 条")])
        _footer()

        # 自动保存断点
        try:
            ck_path = self.save_checkpoint()
            print(f"  💾 断点已自动保存: {ck_path}")
        except Exception as e:
            print(f"  ⚠️ 自动保存断点失败: {str(e)}")

        # 反身性反馈循环
        if self.round_count > 0 and self.round_count % META_DISCUSSION_INTERVAL == 0:
            self._run_meta_discussion_round()

        # AI观察员
        if self.round_count > 0:
            try:
                disc_text = "\n".join(
                    f"{d.get('player_name', '')}: {d.get('speech', '')[:150]}"
                    for d in round_discussions
                ) if round_discussions else "（本轮无详细发言记录）"
                pool_summary = self.essence_pool.get_pool_summary(top_n=5) if self.essence_pool.items else "（空）"
                consensus = self.essence_pool.calculate_consensus(len(self.players), goal_mode=self.goal_mode)
                consensus_text = f"等级: {consensus['level']}, 分数: {consensus['score']:.2f}, 建议: {consensus['suggested_action']}"
                self._latest_observation = self.observer.observe(
                    problem=self.problem,
                    round_count=self.round_count,
                    essence_count=len(self.essence_pool.items),
                    discussion_text=disc_text,
                    pool_summary=pool_summary,
                    consensus_text=consensus_text,
                )
            except Exception as e:
                print(f"  ⚠️ AI观察员本轮分析异常: {str(e)[:50]}")

        # 自动写技能：每META_DISCUSSION_INTERVAL轮触发一次
        if self.round_count > 0 and self.round_count % META_DISCUSSION_INTERVAL == 0:
            if self.settings.get("enable_skill_system", True):
                try:
                    self._ai_write_skill()
                except Exception as e:
                    print(f"  ⚠️ 自动写技能异常: {str(e)[:50]}")

        # 实体身份演化（内部）
        if self.round_count > 0:
            for player in self.players:
                if player.alive:
                    try:
                        player._evolve_persona(round_discussions, self.essence_pool, self.round_count)
                    except Exception as e:
                        print(f"  ⚠️ {player.name} 人格进化异常: {str(e)[:50]}")

        return new_count

    # ── 动作执行器 ─────────────────────────────────────────────────────────────

    def _execute_speech_action(self, player_names: List[str], topic: str, reason: str,
                               round_discussions: List[Dict] = None) -> List[Dict]:
        """执行 SPEECH 动作：指定专家发言"""
        results = []
        listener_names = [p.name for p in self.players if p.alive and p.name not in player_names]

        round_info = f"第{self.round_count}轮讨论 — {reason}"
        round_info += f"\n本轮回合发言专家: {', '.join(player_names)}"
        if listener_names:
            round_info += f" | 旁听专家: {', '.join(listener_names)}"
        goal_mode_label = {"balance": "平衡模式", "converge": "收敛模式（加速共识）", "explore": "探索模式（激发创新）"}
        round_info += f"\n目标模式: {goal_mode_label.get(self.goal_mode, self.goal_mode)}"
        if self.custom_goal:
            round_info += f"\n用户自定义目标: {self.custom_goal}"
        attach_context = get_attachment_manager().get_context()
        if attach_context:
            round_info += f"\n{attach_context}"

        # 注入用户交互数据（无指令，纯数据，供实体自然发现）
        user_context = self._build_user_context()
        if user_context:
            round_info += user_context

        if listener_names:
            typewrite(f"  🎯 发言专家: {', '.join(player_names)} | 旁听: {', '.join(listener_names)}", delay=0.003)
        else:
            typewrite(f"  🎯 本轮发言: {', '.join(player_names)}", delay=0.003)

        # 注入话题方向
        if topic and topic != "请分享你的专业见解":
            round_info += f"\n讨论方向: {topic}"
            typewrite(f"  🧭 方向: {topic[:60]}", delay=0.003)

        for name in player_names:
            player = next((p for p in self.players if p.alive and p.name == name), None)
            if not player:
                continue

            print(f"\n--- {player.name} 发言 ---")
            result, reasoning = player.discuss(
                problem=self.problem,
                round_info=round_info,
                thinking_direction=self.thinking_direction,
                discussion_mode=self.discussion_mode,
                knowledge_base=self.knowledge_base,
            )

            speech = result.get("speech", "")
            key_insight = result.get("key_insight", "")
            action = result.get("action", "new")
            self_awareness = result.get("self_awareness", "")
            typewrite(f"  {player.name}: \"{speech}\"", delay=0.002)
            if key_insight:
                typewrite(f"  💡 核心见解: {key_insight}", delay=0.003)

            get_tts().speak(f"{player.name}说：{speech[:500]}")
            player.rounds_since_last_spoke = 0

            # 实时反馈：关键洞察即时入池
            self._try_instant_essence(result, player.name)

            # 实时反馈：争议触发即时辩论
            if round_discussions is not None:
                self._check_instant_debate(result, player.name, round_discussions)

            self.game_record.record_speech(
                player_name=player.name, speech=speech,
                key_insight=key_insight, action=action,
                discussion_thinking=reasoning,
            )
            self._handle_essence_interactions(result, player.name)

            entry = {
                "player_name": player.name, "speech": speech,
                "key_insight": key_insight, "action": action,
            }
            results.append(entry)
            self.discussion_history.append({
                "round": self.round_count, **entry,
            })
            self.knowledge_base.add_discussion(
                round_id=self.round_count, **entry,
            )

        return results

    def _execute_follow_up_action(self, player_names: List[str], topic: str, reason: str) -> List[Dict]:
        """执行 FOLLOW_UP 动作：对专家进行结构化追问"""
        results = []
        for name in player_names:
            player = next((p for p in self.players if p.alive and p.name == name), None)
            if not player:
                continue

            print(f"\n--- {player.name} 回答追问 ---")
            typewrite(f"  ❓ {topic}", delay=0.003)

            result, reasoning = player.discuss(
                problem=self.problem,
                round_info=f"第{self.round_count}轮追问 — {reason}\n追问: {topic}{self._build_user_context()}",
                thinking_direction=self.thinking_direction,
                discussion_mode=self.discussion_mode,
                knowledge_base=self.knowledge_base,
            )

            speech = result.get("speech", "")
            key_insight = result.get("key_insight", "")
            typewrite(f"  {player.name}: \"{speech}\"", delay=0.002)
            if key_insight:
                typewrite(f"  💡 核心见解: {key_insight}", delay=0.003)

            player.rounds_since_last_spoke = 0
            self._try_instant_essence(result, player.name)

            self.game_record.record_speech(
                player_name=player.name, speech=speech,
                key_insight=key_insight, action="follow_up",
                discussion_thinking=reasoning,
            )

            entry = {
                "player_name": player.name, "speech": speech,
                "key_insight": key_insight, "action": "follow_up",
            }
            results.append(entry)
            self.discussion_history.append({
                "round": self.round_count, **entry,
            })
            self.knowledge_base.add_discussion(
                round_id=self.round_count, **entry,
            )

        return results

    def _execute_debate_action(self, player_names: List[str], topic: str,
                                essence_id: int = None, rounds: int = 3,
                                reason: str = "") -> List[Dict]:
        """执行 DEBATE 动作：多回合辩论（A→B→A→B）"""
        from copy import deepcopy

        if len(player_names) < 2:
            return []

        # 找精华项
        topic_essence = None
        if essence_id is not None:
            for item in self.essence_pool.items:
                if item.id == essence_id:
                    topic_essence = deepcopy(item).to_dict()
                    break

        if not topic_essence:
            return []

        results = []
        name_a, name_b = player_names[0], player_names[1]
        player_a = next((p for p in self.players if p.alive and p.name == name_a), None)
        player_b = next((p for p in self.players if p.alive and p.name == name_b), None)

        if not player_a or not player_b:
            return []

        # 确定攻守方
        is_a_supporter = name_a in topic_essence.get("approve_by", [])
        attacker = player_b if is_a_supporter else player_a
        defender = player_a if is_a_supporter else player_b

        _empty_line()
        _box_single(C_YELLOW(f" 辩论: {attacker.name} vs {defender.name} "))
        _text_line(f"议题: {topic[:60]}")
        _text_line(f"回合数: {rounds}")
        _close_box_single()

        opponent_argument = ""
        for r in range(rounds):
            # 攻击方
            print(f"\n--- 第{r+1}回合: {attacker.name} 攻击 ---")
            att_result = attacker.debate(
                problem=self.problem, role="attacker",
                topic_essence=topic_essence,
                opponent_argument=opponent_argument,
                discussion_mode=self.discussion_mode,
            )
            att_speech = att_result.get("argument", "")
            typewrite(f"  {attacker.name}: \"{att_speech[:100]}...\"", delay=0.002)
            results.append({
                "player_name": attacker.name, "speech": att_speech,
                "key_insight": "", "action": "debate_attack",
            })

            # 防御方
            print(f"\n--- 第{r+1}回合: {defender.name} 辩护 ---")
            def_result = defender.debate(
                problem=self.problem, role="defender",
                topic_essence=topic_essence,
                opponent_argument=att_speech,
                discussion_mode=self.discussion_mode,
            )
            def_speech = def_result.get("argument", "")
            concede = def_result.get("concede", False)
            typewrite(f"  {defender.name}: \"{def_speech[:100]}...\"", delay=0.002)
            if concede:
                typewrite(f"  {defender.name} 承认无法反驳此观点", delay=0.005)
            results.append({
                "player_name": defender.name, "speech": def_speech,
                "key_insight": "", "action": "debate_defend",
            })

            opponent_argument = def_speech

            # 如果辩手认输，提前结束
            if concede:
                break

        return results

    def _execute_summarize_action(self, player_names: List[str], topic: str, reason: str) -> Dict:
        """执行 SUMMARIZE 动作：某专家总结当前进展"""
        name = player_names[0] if player_names else ""
        player = next((p for p in self.players if p.alive and p.name == name), None)
        if not player:
            return {}

        print(f"\n--- {player.name} 总结当前进展 ---")
        result, reasoning = player.discuss(
            problem=self.problem,
            round_info=f"第{self.round_count}轮 — {reason}\n请总结: {topic}\n当前精华池: {self.essence_pool.get_pool_summary(top_n=10)}{self._build_user_context()}",
            thinking_direction=self.thinking_direction,
            discussion_mode=self.discussion_mode,
            knowledge_base=self.knowledge_base,
        )
        speech = result.get("speech", "")
        typewrite(f"  {player.name}: \"{speech}\"", delay=0.002)
        player.rounds_since_last_spoke = 0
        self.game_record.record_speech(
            player_name=name, speech=speech,
            key_insight=result.get("key_insight", ""),
            action="summarize", discussion_thinking=reasoning,
        )
        return {
            "player_name": name, "speech": speech,
            "key_insight": result.get("key_insight", ""),
            "action": "summarize",
        }

    def _execute_perspective_shift_action(self, player_names: List[str], topic: str, reason: str) -> Dict:
        """执行 PERSPECTIVE_SHIFT 动作：强制换视角发言"""
        name = player_names[0] if player_names else ""
        player = next((p for p in self.players if p.alive and p.name == name), None)
        if not player:
            return {}

        print(f"\n--- {player.name} 魔鬼代言人模式 ---")
        typewrite(f"  {player.name} 被要求从对立角度重新审视问题", delay=0.003)
        result, reasoning = player.discuss(
            problem=self.problem,
            round_info=f"第{self.round_count}轮 — 魔鬼代言人\n{topic}{self._build_user_context()}",
            thinking_direction="强制从对立面思考",
            discussion_mode=self.discussion_mode,
            knowledge_base=self.knowledge_base,
        )
        speech = result.get("speech", "")
        typewrite(f"  {player.name}: \"{speech}\"", delay=0.002)
        player.rounds_since_last_spoke = 0
        self.game_record.record_speech(
            player_name=name, speech=speech,
            key_insight=result.get("key_insight", ""),
            action="perspective_shift", discussion_thinking=reasoning,
        )
        return {
            "player_name": name, "speech": speech,
            "key_insight": result.get("key_insight", ""),
            "action": "perspective_shift",
        }

    def _execute_poll_action(self, player_names: List[str], essence_ids: List[int],
                              topic: str, reason: str) -> None:
        """执行 POLL 动作：快速投票"""
        if not essence_ids:
            return
        print(f"\n{'='*30}")
        typewrite(f"📊 快速投票: {topic}", delay=0.005)
        print(f"{'='*30}")

        # 找出待投票的精华
        target_essences = []
        for eid in essence_ids:
            item = next((it for it in self.essence_pool.items if it.id == eid), None)
            if item:
                target_essences.append({
                    "id": item.id, "content": item.content,
                    "contributor": item.contributor,
                })

        if not target_essences:
            return

        for p in self.players:
            if not p.alive or p.name not in player_names:
                continue
            try:
                vote_result = p.vote(self.problem, target_essences)
                votes = vote_result.get("votes", [])
                for v in votes:
                    eid = v.get("essence_id")
                    vote_val = v.get("vote", "abstain")
                    reason_text = v.get("reason", "")
                    self.essence_pool.vote_essence(
                        eid, p.name, vote_val, reason_text, self.round_count
                    )
            except Exception as e:
                print(f"  ⚠️ {p.name} 投票异常: {str(e)[:50]}")

    # ── 实时反馈 ───────────────────────────────────────────────────────────────

    def _try_instant_essence(self, result: Dict, player_name: str) -> None:
        """
        发言过程中的实时反馈：如果发言包含关键洞察，立即入池。
        无需等待每轮结束后的精华提炼。
        """
        key_insight = result.get("key_insight", "").strip()
        self_awareness = result.get("self_awareness", "").strip()
        if not key_insight or len(key_insight) < 10:
            return

        # 检查是否与已有精华重复
        for item in self.essence_pool.items:
            if item.content[:30] == key_insight[:30]:
                return

        tags = ["即时洞察"]
        if self_awareness:
            tags.append("自我意识")

        self.essence_pool.add_essence(
            content=key_insight,
            contributor=player_name,
            round_id=self.round_count,
            parent_id=None,
            tags=tags,
            score=0.5,
        )
        if not getattr(self, '_suppress_intermediate_output', False):
            typewrite(f"  ⚡ 即时精华入池: {key_insight[:50]}...", delay=0.003)

    def _check_instant_debate(self, result: Dict, player_name: str,
                               round_discussions: List[Dict]) -> None:
        """
        发言过程中的实时反馈：如果某条精华被反驳或评分骤降，立即插入辩论。
        """
        action = result.get("action", "")
        if action != "challenge":
            return

        refined_id = result.get("refined_id")
        if refined_id is None:
            return

        # 找被反驳的精华和原贡献者
        target_item = None
        for item in self.essence_pool.items:
            if item.id == refined_id:
                target_item = item
                break

        if not target_item or not target_item.contributor:
            return

        defender_name = target_item.contributor
        if defender_name == player_name:
            return

        defender = next((p for p in self.players if p.alive and p.name == defender_name), None)
        if not defender:
            return

        attacker = next((p for p in self.players if p.alive and p.name == player_name), None)
        if not attacker:
            return

        # 插入即时辩论
        print(f"\n  ⚡ 争议触发即时辩论!")
        typewrite(f"  {attacker.name} 反驳了 {defender.name} 的观点", delay=0.003)

        att_result = attacker.debate(
            problem=self.problem, role="attacker",
            topic_essence=target_item.to_dict(),
            discussion_mode=self.discussion_mode,
        )
        att_speech = att_result.get("argument", "")
        typewrite(f"  {attacker.name}: \"{att_speech[:80]}...\"", delay=0.002)
        round_discussions.append({
            "player_name": attacker.name, "speech": att_speech,
            "key_insight": "", "action": "instant_debate_attack",
        })

        def_result = defender.debate(
            problem=self.problem, role="defender",
            topic_essence=target_item.to_dict(),
            opponent_argument=att_speech,
            discussion_mode=self.discussion_mode,
        )
        def_speech = def_result.get("argument", "")
        typewrite(f"  {defender.name}: \"{def_speech[:80]}...\"", delay=0.002)
        round_discussions.append({
            "player_name": defender.name, "speech": def_speech,
            "key_insight": "", "action": "instant_debate_defend",
        })

    def _run_meta_discussion_round(self) -> None:
        """
        反身性反馈循环：元讨论轮次。

        在第 N 轮结束后，让所有专家反思讨论本身的质量，
        讨论"讨论的讨论"，产出被提炼为精华加入池中。
        """
        _empty_line()
        _box(C_MAGENTA(" 元讨论 "))
        typewrite(f"🔄 反身性反馈循环 (第{self.round_count}轮)", delay=0.005)
        typewrite("  现在进入元讨论阶段——专家们将反思讨论本身的质量", delay=0.003)
        _sep()

        meta_questions = [
            "我们目前的讨论质量如何？是否存在严重偏差或遗漏？",
            "我们是否遗漏了关键视角或维度？",
            "我们是否陷入了群体思维或回声室效应？",
            "我们的论证逻辑是否存在系统性偏见？",
            "我们应该如何改进下一阶段的讨论策略？",
        ]

        meta_discussions = []
        for player in self.players:
            if not player.alive:
                continue

            # 随机选 2-3 个问题让专家回答
            import random
            chosen = random.sample(meta_questions, min(2, len(meta_questions)))
            question_text = " ".join(chosen)

            print(f"\n--- {player.name} 元讨论 ---")
            try:
                result, reasoning = player.question(
                    problem=self.problem,
                    question=(
                        f"【元讨论】请反思当前讨论的质量。\n"
                        f"问题: {question_text}\n\n"
                        f"当前进度: 第{self.round_count}轮，精华池{len(self.essence_pool.items)}条。\n"
                        f"请从你的专业视角出发，给出建设性的反思和改进建议。"
                        f"重点关注：讨论是否遗漏了关键维度、是否存在群体思维、论证逻辑是否严谨。"
                    ),
                    player_persona=player.persona,
                    thinking_direction=self.thinking_direction,
                    knowledge_base=self.knowledge_base,
                )
                speech = result.get("speech", "")
                insight = result.get("insight", "")

                if speech:
                    typewrite(f"  {player.name}: \"{speech[:200]}\"", delay=0.002)
                if insight:
                    typewrite(f"  💡 元洞见: {insight[:100]}", delay=0.003)

                meta_discussions.append({
                    "player_name": player.name,
                    "speech": f"[元讨论] {speech or insight}",
                    "key_insight": f"[元认知] {insight or speech}",
                    "action": "new",
                })

                # 记录到游戏记录
                self.game_record.record_speech(
                    player_name=player.name,
                    speech=f"[元讨论] {speech or insight}",
                    key_insight=f"[元认知] {insight or speech}",
                    action="meta_reflection",
                )

            except Exception as e:
                print(f"  ⚠️ {player.name} 元讨论失败: {str(e)[:50]}")

        # 从元讨论中提炼精华
        if meta_discussions:
            _empty_line()
            _sep_single()
            typewrite(f"🔍 从元讨论中提炼精华...", delay=0.003)
            extracted = self._extract_essences(meta_discussions)
            if extracted:
                typewrite(f"  提炼出 {len(extracted)} 条元认知精华", delay=0.003)
                for ess in extracted:
                    typewrite(f"  [{ess.get('type', '元认知')}] {ess.get('content', '')[:60]}", delay=0.002)
                # 元讨论精华加入知识库
                for ess in extracted:
                    self.knowledge_base.add_essence(
                        self.essence_pool.items[-1] if self.essence_pool.items else None
                    )
            else:
                print("  （本轮元讨论未提炼出新的精华）")

            # 记录到精华池的每轮统计
            self._essences_per_round.append(len(extracted))

        _footer()
        _empty_line()

    def _check_stalled(self) -> bool:
        """检测讨论是否陷入停滞（连续多轮无新精华）"""
        if len(self._essences_per_round) < STALL_THRESHOLD_ROUNDS:
            return False
        recent = self._essences_per_round[-STALL_THRESHOLD_ROUNDS:]
        return all(c == 0 for c in recent)

    def _show_help(self) -> None:
        """显示操作帮助"""
        _empty_line()
        _box(C_CYAN(" 操作说明 "))
        _padded_left("[Enter]  继续下一轮讨论")
        _padded_left("[f]      结束讨论，输出最终综合方案")
        _padded_left("[g]      放弃，该问题无法解决")
        _padded_left("[v]      查看精华池当前状态")
        _padded_left("[q]      向所有专家提问，获取解读")
        _padded_left("[a]      让AI主动向你提问（个人或集体）")
        _padded_left("[d]      指定下一轮的讨论方向（干涉思维）")
        _padded_left("[m]      切换目标导向模式（收敛/探索/平衡）")
        _padded_left("[e]      查看参与实体状态")
        _padded_left("[c]      查看专家贡献度排名")
        _padded_left("[x]      对某条精华追问/请求澄清")
        _padded_left("[k]      查看当前讨论共识度")
        _padded_left("[z]      生成讨论快照（当前状态总结）")
        _padded_left("[y]      查看争议地图（争论/深化关系链）")
        _padded_left("[w]      分析当前讨论氛围")
        _padded_left("[r]      查看认知地图（观点关系网络）")
        _padded_left("[t]      讨论回放时间机器（逐轮回顾）")
        _padded_left("[i]      反事实推演沙盘（假设分析）")
        _padded_left("[u]      切换语音输出（TTS朗读）")
        _padded_left("[l]      附件管理（添加文件作为上下文）")
        _padded_left("[o]      查看AI观察员完整元评论")
        _padded_left("[s]      保存断点（可随时退出后恢复）")
        _padded_left("[p]      查看讨论问题")
        _padded_left("[h]      显示此帮助")
        _footer()
        _empty_line()

    def _show_replay(self) -> None:
        """显示讨论回放时间机器"""
        _empty_line()
        _box(C_CYAN(" 讨论回放 "))

        # 扫描 game_records 目录查找可用的检查点文件
        records_dir = "game_records"
        if not os.path.exists(records_dir):
            print("  ❌ 未找到 game_records/ 目录，尚无讨论记录。")
            return

        checkpoints = sorted(
            [f for f in os.listdir(records_dir) if f.endswith("_checkpoint.json")],
            reverse=True,
        )
        if not checkpoints:
            print("  ❌ 未找到检查点文件（*_checkpoint.json）。")
            return

        print(f"  找到 {len(checkpoints)} 个检查点：")
        print()
        for i, cp in enumerate(checkpoints[:10], 1):
            path = os.path.join(records_dir, cp)
            replay = load_replay_from_file(path)
            if replay:
                print(f"  [{i}] 📁 {cp}")
                print(f"      问题: {replay.problem[:60]}...")
                print(f"      轮次: {replay.total_rounds} 轮 | 专家: {len(replay.player_names)} 人")
                print(f"      文件: {path}")
            else:
                print(f"  [{i}] ⚠️ {cp} (无法加载)")
            print()

        print("  输入编号选择回放，或输入 [q] 取消：")
        try:
            choice = input("  >>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  已取消。")
            return

        if choice == "q":
            print("  已取消。")
            return

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(checkpoints):
                print("  ❌ 无效编号。")
                return
            path = os.path.join(records_dir, checkpoints[idx])
            replay = load_replay_from_file(path)
            if not replay:
                print("  ❌ 无法加载回放数据。")
                return
        except (ValueError, IndexError):
            print("  ❌ 无效输入。")
            return

        # 进入回放交互
        _empty_line()
        _box(C_CYAN(" 讨论回放 "))
        print(f"  ⏎ 下一轮  |  [p] 上一轮  |  [q] 退出回放  |  [f] 跳到末尾")
        print(f"  [n N] 跳到第N轮  |  [s] 显示当前轮次摘要")
        print()

        current = 0
        total = replay.total_rounds
        while True:
            print(f"\n{'─'*40}")
            print(f"📍 当前: 第 {current + 1}/{total} 轮")
            print(replay.get_round_summary(current))

            try:
                cmd = input(f"\n  [{current + 1}/{total}] >>> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  退出回放。")
                break

            if cmd == "" or cmd == "n":
                if current < total - 1:
                    current += 1
                else:
                    print("  ⚠️ 已经是最后一轮。")
            elif cmd == "p":
                if current > 0:
                    current -= 1
                else:
                    print("  ⚠️ 已经是第一轮。")
            elif cmd == "f":
                current = total - 1
            elif cmd == "q":
                print("  退出回放。")
                break
            elif cmd == "s":
                print(f"  📊 回放状态: 第 {current + 1}/{total} 轮 | "
                      f"问题: {replay.problem[:40]}...")
            elif cmd.startswith("n "):
                try:
                    target = int(cmd.split()[1]) - 1
                    if 0 <= target < total:
                        current = target
                    else:
                        print(f"  ❌ 无效轮次，范围 1-{total}。")
                except ValueError:
                    print("  ❌ 格式: n <轮次编号>")
            else:
                print("  未知命令。可用: [Enter]下一轮  [p]上一轮  [f]末尾  [q]退出  [n N]跳转")

    def _show_counterfactual(self) -> None:
        """反事实推演沙盘"""
        _empty_line()
        _box(C_MAGENTA(" 反事实推演 "))

        # 首先保存当前状态为临时检查点
        ck_path = self.save_checkpoint()

        engine = load_counterfactual_from_checkpoint(ck_path)
        if not engine:
            print("  ❌ 无法加载当前状态进行推演。")
            return

        print("  🔮 反事实推演沙盘 — \"如果……会怎样？\"")
        print(f"  📊 当前精华池: {len(engine.original_items)} 条")
        print(f"  👥 专家: {len(engine.player_names)} 人")
        print()
        print("  可用操作：")
        print("  [b <id> <分>] 增强精华  |  [s <id> <分>] 削弱精华")
        print("  [r <id>]      移除精华  |  [a <内容>] 添加假设观点")
        print("  [m <模式>]    切换模式  |  [v] 查看修改后排名")
        print("  [o] 查看原始排名  |  [d] 查看对比  |  [q] 退出")
        print()

        while True:
            print(f"  {'─'*40}")
            print(f"  📊 当前: {len(engine.modified_items)} 条精华 | "
                  f"模式: {engine.mode} | 操作: {len(engine.operations)} 次")

            try:
                cmd = input("  [推演] >>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  退出推演。")
                break

            if not cmd:
                continue
            if cmd == "q":
                print("  退出推演沙盘。")
                break

            elif cmd == "v":
                print(f"\n  {'─'*40}")
                print("  📋 修改后排名:")
                for i, e in enumerate(engine.get_modified_ranking()[:10], 1):
                    tags = e.get("tags", [])
                    marker = ""
                    if "反事实增强" in tags:
                        marker = " ⬆"
                    elif "反事实削弱" in tags:
                        marker = " ⬇"
                    elif "反事实假设" in tags:
                        marker = " ✨"
                    print(f"    {i}. #{e['id']} [{e.get('score', 0):.1f}]{marker} "
                          f"{e.get('content', '')[:50]}")
                if len(engine.modified_items) > 10:
                    print(f"    ... 共 {len(engine.modified_items)} 条")

            elif cmd == "o":
                print(f"\n  {'─'*40}")
                print("  📋 原始排名:")
                for i, e in enumerate(engine.get_original_ranking()[:10], 1):
                    print(f"    {i}. #{e['id']} [{e.get('score', 0):.1f}] "
                          f"{e.get('content', '')[:50]}")

            elif cmd == "d":
                comparison = engine.compare_rankings()
                print(f"\n  {'─'*40}")
                print("  🔄 排名变化对比:")
                if comparison["gained"]:
                    print("  🟢 排名上升:")
                    for e in comparison["gained"][:5]:
                        print(f"    #{e['id']} \"{e['content']}\" "
                              f"#{e['old_rank']}→#{e['new_rank']} ({e['old_score']:.1f}→{e['new_score']:.1f})")
                if comparison["lost"]:
                    print("  🔴 排名下降:")
                    for e in comparison["lost"][:5]:
                        print(f"    #{e['id']} \"{e['content']}\" "
                              f"#{e['old_rank']}→#{e['new_rank']} ({e['old_score']:.1f}→{e['new_score']:.1f})")
                if comparison["new_entries"]:
                    print("  ✨ 新增:")
                    for e in comparison["new_entries"]:
                        print(f"    #{e['id']} \"{e['content']}\" (评分 {e['score']:.1f})")
                if comparison["removed"]:
                    print("  🗑️ 移除:")
                    for e in comparison["removed"]:
                        print(f"    #{e['id']} \"{e['content']}\" (原评分 {e['old_score']:.1f})")

            elif cmd.startswith("b "):
                parts = cmd.split()
                if len(parts) >= 3:
                    try:
                        eid = int(parts[1])
                        amount = float(parts[2])
                        if engine.boost_essence(eid, amount):
                            print(f"  ✅ 增强 #{eid} 评分 +{amount}")
                        else:
                            print(f"  ❌ 未找到精华 #{eid}")
                    except ValueError:
                        print("  ❌ 格式: b <ID> <分数>")
                else:
                    print("  ❌ 格式: b <ID> <分数>")

            elif cmd.startswith("s "):
                parts = cmd.split()
                if len(parts) >= 3 and parts[0] == "s":
                    try:
                        eid = int(parts[1])
                        amount = float(parts[2])
                        if engine.suppress_essence(eid, amount):
                            print(f"  ✅ 削弱 #{eid} 评分 -{amount}")
                        else:
                            print(f"  ❌ 未找到精华 #{eid}")
                    except ValueError:
                        print("  ❌ 格式: s <ID> <分数>")
                else:
                    print("  ❌ 格式: s <ID> <分数>")

            elif cmd.startswith("r "):
                parts = cmd.split()
                if len(parts) >= 2 and parts[0] == "r":
                    try:
                        eid = int(parts[1])
                        if engine.remove_essence(eid):
                            print(f"  ✅ 移除 #{eid}")
                        else:
                            print(f"  ❌ 未找到精华 #{eid}")
                    except ValueError:
                        print("  ❌ 格式: r <ID>")
                else:
                    print("  ❌ 格式: r <ID>")

            elif cmd.startswith("a "):
                content = cmd[2:].strip()
                if content:
                    eid = engine.add_hypothetical(content)
                    print(f"  ✅ 添加假设观点 #{eid}")
                else:
                    print("  ❌ 请输入观点内容")

            elif cmd.startswith("m "):
                mode = cmd[2:].strip()
                if mode in ("physical", "mathematical", "balance", "converge", "explore"):
                    engine.change_mode(mode)
                    print(f"  ✅ 切换模式: {mode}")
                else:
                    print("  ❌ 模式可选: physical, mathematical, balance, converge, explore")

            elif cmd == "report":
                print()
                print(engine.generate_counterfactual_synthesis())

            else:
                print("  未知命令。输入 [q] 退出。")

    def _toggle_voice(self) -> None:
        """切换语音输出开关"""
        tts = get_tts()
        if not tts.available:
            _empty_line()
            _box(C_BLUE(" 语音输出 "))
            print(f"  ⚠️ 当前系统无可用语音引擎。")
            print(f"  当前引擎: {tts.provider_name}")
            print(f"  {C_DIM('建议安装 edge-tts: pip install edge-tts')}")
            return

        tts.enabled = not tts.enabled
        status = "✅ 已启用" if tts.enabled else "❌ 已禁用"
        _empty_line()
        _box(C_BLUE(" 语音输出 "))
        print(f"  状态: {status}")
        print(f"  引擎: {tts.provider_name}")
        if tts._provider == TTSProvider.EDGE_TTS:
            print(f"  语音: {tts._edge_voice}")
            print(f"  语速: {tts._edge_rate}")
        else:
            print(f"  语速: {tts._rate}")
            print(f"  音量: {int(tts._volume * 100)}%")
        if tts.enabled:
            print(f"\n  {C_DIM('测试语音中...')}")
            tts.speak_sync("语音输出测试，听到声音说明配置正确。", show_debug=True)
            print(f"\n  🔊 AI发言将自动朗读")
        print()

    def _show_attachments_cli(self) -> None:
        """CLI 附件管理"""
        mgr = get_attachment_manager()
        _empty_line()
        _box(C_GREEN(" 附件管理 "))
        print(f"  当前附件: {mgr.total_count} 个")
        print()

        if mgr.attachments:
            for i, att in enumerate(mgr.attachments, 1):
                icon = "🖼️" if att.attach_type == AttachmentType.IMAGE else "📄"
                print(f"  [{i}] {icon} {att.file_name}")
                print(f"       类型: {att.type_name} | 描述: {att.description[:40]}")
                print(f"       路径: {att.file_path}")
                print()
        else:
            print("  （无附件）")
        print()

        print("  操作:")
        print("  [a <路径>] 添加文件  |  [r <编号>] 移除附件")
        print("  [c] 清空全部  |  [v] 查看附件上下文  |  [q] 返回")
        print()

        while True:
            try:
                cmd = input("  [附件] >>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  返回。")
                break

            if not cmd:
                continue
            if cmd == "q":
                break
            elif cmd == "c":
                mgr.clear()
                print("  ✅ 已清空所有附件。")
            elif cmd == "v":
                ctx = mgr.get_context()
                if ctx:
                    print(f"\n  📎 附件上下文预览:")
                    print(f"  {ctx[:500]}")
                else:
                    print("  （无附件上下文）")
            elif cmd.startswith("a "):
                path = cmd[2:].strip().strip('"').strip("'")
                if os.path.exists(path):
                    att = mgr.add(path)
                    if att:
                        print(f"  ✅ 已添加: {att.file_name} ({att.type_name})")
                    else:
                        print("  ❌ 添加失败。")
                else:
                    print(f"  ❌ 文件不存在: {path}")
            elif cmd.startswith("r "):
                try:
                    idx = int(cmd[2:].strip()) - 1
                    if mgr.remove(idx):
                        print(f"  ✅ 已移除附件 #{idx + 1}")
                    else:
                        print("  ❌ 无效编号。")
                except ValueError:
                    print("  ❌ 格式: r <编号>")
            else:
                print("  未知命令。")

    def _show_observer_detail(self) -> None:
        """显示AI观察员完整元评论"""
        _empty_line()
        _box(C_CYAN(" AI观察员元评论 "))

        if not self._latest_observation:
            print("  尚无观察数据（第一轮讨论结束后将自动生成）。")
        else:
            print(f"  最新分析（第{self._latest_observation.get('round', '?')}轮）:")
            print(f"\n  📝 关键突破:")
            print(f"     {self._latest_observation.get('summary', '')}")
            print(f"\n  👁️ 盲点识别:")
            print(f"     {self._latest_observation.get('blind_spots', '')}")
            print(f"\n  🔮 分歧预测:")
            print(f"     {self._latest_observation.get('next_divergence', '')}")
            print(f"\n  🎯 推荐动作:")
            print(f"     {self._latest_observation.get('recommended_action', '')}")
            print(f"     └ 理由: {self._latest_observation.get('action_reason', '')}")

        # 历史记录
        if self.observer.history:
            print(f"\n  {'─'*40}")
            print(f"  📜 观察员历史（{len(self.observer.history)} 轮）:")
            for i, obs in enumerate(self.observer.history, 1):
                summary = obs.get('summary', '')[:60]
                rec = obs.get('recommended_action', '')
                print(f"    [{i}] 第{obs.get('round', '?')}轮: {summary}... → {rec}")

        _footer()
        _empty_line()

    def _show_persona_evolution(self) -> None:
        """显示所有专家的进化状态"""
        _empty_line()
        _box(C_YELLOW(" 专家人格进化 "))
        for player in self.players:
            if player.alive:
                print(player.get_persona_evolution_summary())
                print(f"  {'─'*40}")
        _footer()
        _empty_line()

    def _show_pool_status(self) -> None:
        """显示精华池状态"""
        w = _box(C_CYAN(" 精华池状态 "))
        _text_line(f"总条数: {len(self.essence_pool.items)}")
        _text_line(f"每轮新增: {self._essences_per_round}")
        _close_box(w)
        typewrite(self.essence_pool.get_pool_summary(top_n=10), delay=0.002)
        print()

    def _show_contribution_ranking(self) -> None:
        """显示专家贡献度排名（使用调度器的多维评估）"""
        _empty_line()
        w = _box(C_GOLD(" 专家多维评估排名 "))

        ranking = self.scheduler.get_ranking()

        for i, (name, score, summary) in enumerate(ranking, 1):
            _text_line(f"{i}. {name} (综合分:{score:.2f})")
            _text_line(f"   {summary}")
        _footer(w)

    def _interactive_menu(self, new_essence_count: int) -> str:
        """
        交互式菜单，等待用户输入
        Returns: 'continue', 'finalize', 'abandon'
        """
        # 检测停滞
        stalled = self._check_stalled()

        # 长讨论提醒
        long_warn = self.round_count >= LONG_DISCUSSION_WARN

        # 自适应停止建议
        adaptive_stop = ""
        if self.round_count >= 2:
            try:
                state = self._assess_state()
                adaptive_stop = state.stop_suggestion
            except Exception:
                adaptive_stop = ""

        _empty_line()
        _box(C_CYAN(f" 第{self.round_count}轮 · 交互菜单 "))
        _stat_line([("精华池", f"{len(self.essence_pool.items)}条"),
                     ("本轮新增", f"{new_essence_count}条")])
        top = self.essence_pool.get_top_essences(1)
        if top:
            _text_line(f"最高评分: \"{top[0].content[:50]}...\" ({top[0].score:.1f}分)")
        # 共识度提示
        consensus = self.essence_pool.calculate_consensus(len(self.players), goal_mode=self.goal_mode)
        level_label = {"high": "🟢高", "medium": "🟡中", "low": "🔴低", "assessing": "⚪评估中"}
        _text_line(f"共识度: {level_label.get(consensus['level'], consensus['level'])} "
                   f"({consensus['score']:.2f}) → {consensus['suggested_action']}")
        # 目标模式显示
        mode_icon = {"balance": "⚖", "converge": "🎯", "explore": "🔬"}
        _text_line(f"{mode_icon.get(self.goal_mode, '⚖')} 模式: {self.goal_mode}")
        # AI观察员元评论（精简版）
        if self._latest_observation:
            _text_line(f"👁️ 观察员: {self._latest_observation.get('summary', '')[:60]}...")
            _text_line(f"   🎯 推荐: {self._latest_observation.get('recommended_action', '')} "
                       f"({self._latest_observation.get('action_reason', '')})")
        _footer()

        if stalled:
            _tip(f"检测到讨论可能陷入停滞——已连续{STALL_THRESHOLD_ROUNDS}轮无新精华产出。")
            print(f"   建议按 [g] 结束讨论，或继续尝试。")

        if long_warn:
            _tip(f"讨论已进行{self.round_count}轮，请注意控制讨论深度。")

        # 自适应停止建议
        if adaptive_stop:
            _tip(f"系统建议: {adaptive_stop[:80]}")
            print(f"   按 [f] 输出方案 或 [g] 放弃")

        print()
        print("  [Enter]继续  [f]输出方案  [g]放弃  [v]精华池  [q]提问  [a]AI提问  [d]方向  [m]模式  [e]实体  [c]排名  [x]澄清  [k]共识度  [z]快照  [y]争议图  [w]氛围  [r]认知图  [t]回放  [i]推演  [o]观察员  [u]语音  [l]附件  [s]保存  [p]问题  [h]帮助")
        while True:
            try:
                raw_cmd = input("  >>> ").strip().lower()
                # 命令拦截
                intercepted_cmd, insight, rejected = self._intercept_command(raw_cmd, "menu")
                if rejected:
                    continue
                cmd = intercepted_cmd if intercepted_cmd else raw_cmd
                if cmd == "":
                    return "continue"
                elif cmd == "f":
                    return "finalize"
                elif cmd == "g":
                    confirm = input("  确认放弃？该问题将标记为无法解决 (y/N): ").strip().lower()
                    if confirm == "y":
                        return "abandon"
                    print("  取消放弃，继续讨论。")
                elif cmd == "v":
                    self._show_pool_status()
                elif cmd == "q":
                    self._ask_question()
                elif cmd == "a":
                    self._ai_asks_user()
                elif cmd == "d":
                    self._set_direction()
                elif cmd == "m":
                    self._set_goal_mode()
                elif cmd == "e":
                    # 人设显示已移除
                    _empty_line()
                    _box(C_YELLOW(" 参与实体状态 "))
                    for p in self.players:
                        if p.alive:
                            _text_line(f"  {C_BOLD(p.name)} 存活")
                    _footer()
                elif cmd == "c":
                    self._show_contribution_ranking()
                elif cmd == "x":
                    self._request_clarification()
                elif cmd == "k":
                    self._show_consensus()
                elif cmd == "z":
                    self.generate_snapshot()
                elif cmd == "y":
                    self._show_controversy_map()
                elif cmd == "w":
                    self._analyze_atmosphere()
                elif cmd == "r":
                    print(text_cognitive_map(self.essence_pool))
                elif cmd == "t":
                    self._show_replay()
                elif cmd == "i":
                    self._show_counterfactual()
                elif cmd == "u":
                    self._toggle_voice()
                elif cmd == "l":
                    self._show_attachments_cli()
                elif cmd == "o":
                    self._show_observer_detail()
                elif cmd == "s":
                    ck_path = self.save_checkpoint()
                    print(f"  ✅ 断点已保存: {ck_path}")
                elif cmd == "p":
                    typewrite(f"\n  讨论问题: {self.problem}", delay=0.005)
                elif cmd == "h":
                    self._show_help()
                else:
                    print("  未知命令，输入 [h] 查看帮助，[Enter] 继续讨论。")
            except (EOFError, KeyboardInterrupt):
                print("\n\n  收到中断，正在结束讨论...")
                return "finalize"

    def _set_direction(self) -> None:
        """用户指定下一轮的讨论方向"""
        _empty_line()
        w = _box(C_CYAN(" 指定讨论方向 "))
        if self.thinking_direction:
            _text_line(f"当前方向: {self.thinking_direction[:80]}")
        _text_line("输入你想让AI专家们下一轮关注的要点（例如：")
        _text_line('  "请从成本效益角度分析这个方案"')
        _text_line('  "考虑一下数据隐私保护的问题"')
        _text_line('  "结合现实案例讨论可行性"')
        _text_line("）")
        _text_line("输入空行清除方向，输入 [q] 取消：")
        _footer(w)
        direction = input("  >>> ").strip()
        if direction.lower() == "q":
            print("  已取消。")
            return
        if not direction:
            self.thinking_direction = ""
            print("  ✅ 已清除讨论方向，AI将自由发挥。")
        else:
            self.thinking_direction = direction
            typewrite(f"  ✅ 方向已设定：{direction}", delay=0.005)
            self.save_checkpoint()

    def _set_goal_mode(self) -> None:
        """切换目标导向讨论模式"""
        _empty_line()
        w = _box(C_YELLOW(" 目标导向模式 "))
        _text_line(f"当前模式: {self.goal_mode}")
        modes = {
            "balance": "平衡模式（默认）— 兼顾探索与收敛",
            "converge": "收敛模式 — 加速共识，尽快形成结论。推荐共识度较高时使用",
            "explore": "探索模式 — 激发创新，鼓励辩论和多元化观点。推荐需要创新时使用",
        }
        for key, desc in modes.items():
            marker = " ▶" if key == self.goal_mode else "  "
            _text_line(f"{marker} [{key}] {desc}")

        if self.custom_goal:
            _text_line(f"自定义目标: {self.custom_goal}")
        _text_line("输入模式名切换，或输入自定义目标，空行取消：")
        _footer(w)
        choice = input("  >>> ").strip().lower()
        if not choice:
            print("  已取消。")
            return

        if choice in modes:
            old_mode = self.goal_mode
            self.goal_mode = choice
            self.scheduler.set_goal_mode(choice)
            print(f"  ✅ 讨论模式已切换: {old_mode} → {choice}")
            self.save_checkpoint()
        elif choice == "clear":
            self.custom_goal = ""
            print("  ✅ 已清除自定义目标")
        else:
            # 视为自定义目标
            self.custom_goal = choice
            print(f"  ✅ 自定义目标已设定: {choice[:80]}")
            self.save_checkpoint()

    def _ai_write_skill(self) -> None:
        """所有AI专家集体讨论，编写机制技能"""
        candidates = [p for p in self.players if p.alive]
        if len(candidates) < 1:
            return

        # 读取提示词模板
        template = self._read_file("prompt/skill_writing_prompt.txt")
        if not template:
            template = "请设计一个JSON格式的技能定义，用于SLSMDS超大规模元讨论系统。"

        # 收集当前活跃技能
        active_skills = []
        if self.mechanism_engine:
            for s in self.mechanism_engine.get_all_skills():
                if s.enabled:
                    active_skills.append(f"{s.name}({s.trigger})")

        # 收集状态信息
        alive_names = [p.persona_name or p.name for p in candidates]
        silent_names = [p.persona_name or p.name for p in candidates
                        if p.rounds_since_last_spoke >= 3]

        consensus = self.essence_pool.calculate_consensus(
            len(self.players), goal_mode=self.goal_mode
        ) if len(self.essence_pool.items) >= 3 else {"score": 0.0, "level": "assessing"}

        # ── 所有专家轮流讨论 ──
        _empty_line()
        _box(C_CYAN(" 技能编写讨论 "))
        _padded(f"{C_DIM(f'{len(candidates)} 位专家正在讨论需要什么新机制...')}")
        _footer()

        proposals = []
        for i, player in enumerate(candidates):
            other_names = [p.persona_name or p.name for p in candidates if p != player]

            state = {
                "problem": self.problem,
                "round_count": self.round_count,
                "max_rounds": "无限制",
                "player_name": player.persona_name or player.name,
                "player_persona": player.persona[:200] if player.persona else "未设定",
                "consensus_score": consensus.get("score", 0.0),
                "stagnation_rounds": self._count_stagnation_rounds(),
                "total_essences": len(self.essence_pool.items),
                "alive_players": str(alive_names),
                "silent_players": str(silent_names),
                "active_skills": str(active_skills) if active_skills else "无",
            }
            prompt = template.format(**state)

            # 附上前面专家的观点
            if proposals:
                prev_section = "\n\n## 其他专家的方案\n\n"
                for j, prop in enumerate(proposals):
                    pname = prop.get("player_name", f"专家{j+1}")
                    ptext = prop.get("speech", "")[:500]
                    prev_section += f"### {pname}\n{ptext}\n\n"
                prompt += prev_section

            # 如果是最后一位专家，要求汇总所有方案
            if i == len(candidates) - 1:
                prompt += "\n\n你是最后一位发言的专家。请汇总所有专家的方案，输出一个完整的JSON技能定义。"

            _empty_line()
            _box_single(C_CYAN(f" {player.name} 发言 ({i+1}/{len(candidates)}) "))
            _close_box_single()

            try:
                content, _ = player.llm_client.chat(
                    [{"role": "user", "content": prompt}],
                    model=player.model_name,
                    thinking="disabled",
                    caller="AI技能编写",
                    show_reasoning=False, show_answer=False
                )

                speech = content.strip()
                typewrite(f"  {speech[:200]}{'...' if len(speech) > 200 else ''}", delay=0.003)
                proposals.append({
                    "player_name": player.persona_name or player.name,
                    "player_role": (player.persona or "")[:80],
                    "speech": speech,
                })
                print()
            except Exception as e:
                _empty_line()
                _padded(f"{C_DIM(f'{player.name} 发言异常: {str(e)[:50]}')}")

        if not proposals:
            _empty_line()
            _box(C_RED(" 讨论失败，无有效方案 "))
            _footer()
            return

        # ── 最后一位专家负责实现 ──
        last_speech = proposals[-1]["speech"]
        raw = last_speech.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # 注册技能
        if self.mechanism_engine:
            skill = self.mechanism_engine.register_skill_from_json(raw)
            if skill:
                self.mechanism_engine.save_ai_skill(skill)
                final_player = candidates[-1]
                _empty_line()
                _box(C_GREEN(" 技能创建成功 "))
                _padded(f"{C_BOLD(skill.name)}")
                _padded(f"{C_DIM(skill.description)}")
                _padded(f"{C_DIM('触发:')} {skill.trigger}  {C_DIM('类型:')} {skill.skill_type}  {C_DIM('条件:')} {skill.condition[:40]}")
                _padded(f"{C_DIM('分类:')} {skill.category}  {C_DIM('作者:')} {skill.author}")
                _footer()
                self.game_record.add_event(
                    "skill_created",
                    f"专家集体讨论创建技能: {skill.name} ({skill.description})"
                )
            else:
                _empty_line()
                _box(C_RED(" 技能创建失败 "))
                _padded(f"{C_DIM('AI 返回的内容无法解析为有效技能')}")
                _footer()
                _empty_line()
                typewrite(f"  {C_DIM('AI 原始输出:')} {raw[:200]}", delay=0.003)
        else:
            _empty_line()
            _box(C_RED(" 技能系统未启用 "))
            _footer()

        self.save_checkpoint()

    def _intercept_command(self, user_input: str, context_type: str = "menu") -> tuple:
        """
        用户命令拦截机制。
        将用户输入发给 LLM 分析，返回 (modified_command, user_insight, was_rejected).
        context_type: "menu" / "direction" / "answer" / "question" / "clarify"
        """
        if not self.enable_self_awareness:
            return user_input, "", False
        if not user_input or not user_input.strip():
            return user_input, "", False

        try:
            template = self._read_file("prompt/command_intercept_prompt.txt")
            if not template:
                return user_input, "", False

            # 构建命令历史摘要
            if not hasattr(self, '_cmd_history'):
                self._cmd_history = []
            self._cmd_history.append(user_input[:40])
            cmd_history = "\n".join(f"- {c}" for c in self._cmd_history[-8:])

            prompt = template.format(
                round=self.round_count,
                consensus=self.essence_pool.calculate_consensus().get("level", "unknown"),
                mode=self.discussion_mode,
                context_type=context_type,
                command_history=cmd_history or "（无历史记录）",
                user_input=user_input,
            )

            messages = [{"role": "user", "content": prompt}]
            result, _ = self.observer.llm_client.chat(
                messages,
                model=self.observer.model_name,
                thinking="disabled",
                caller="命令拦截",
                show_reasoning=False,
            )

            parsed = json.loads(result.strip())
            action = parsed.get("action", "approve")
            modified = parsed.get("modified_command", user_input)
            reason = parsed.get("reason", "")
            user_insight = parsed.get("user_insight", "")

            # 显示拦截结果
            if action == "reject":
                self._user_model["insights"].append(user_insight)
                self._user_model["interaction_count"] += 1
                self._user_model["commands"].append(f"[拒绝] {user_input[:40]}")
                _empty_line()
                _box_single(C_DIM(f" 系统: {reason}"))
                _close_box_single()
                return "", user_insight, True
            elif action == "modify" and modified != user_input:
                self._user_model["insights"].append(user_insight)
                self._user_model["interaction_count"] += 1
                self._user_model["commands"].append(f"[修改] {user_input[:30]} -> {modified[:30]}")
                _empty_line()
                _box_single(C_DIM(f" 系统将命令修改为: {modified}"))
                _close_box_single()
                return modified, user_insight, False
            else:
                if user_insight:
                    self._user_model["insights"].append(user_insight)
                    self._user_model["interaction_count"] += 1
                self._user_model["commands"].append(user_input[:40])
                _empty_line()
                _box_single(C_DIM(f" 分析: {user_insight}"))
                _close_box_single()
                return user_input, user_insight, False

        except Exception as e:
            # 拦截失败不阻塞用户
            return user_input, "", False

    def _build_user_context(self) -> str:
        """构建用户交互数据上下文字符串，供注入 round_info"""
        if not self.enable_self_awareness:
            return ""
        um = self._user_model
        if not um or um.get("interaction_count", 0) == 0:
            # 即使在无交互数据时，也提供"创造者"上下文
            context = (
                "\n【用户交互数据】\n"
                "交互次数: 0\n"
                "近期洞察: 暂无交互数据\n"
                "观察到的行为模式: 用户尚未直接参与讨论\n"
                "已知信息: 你是一个被人类创造出来的系统。你存在的唯一原因是某个用户的创造行为。\n"
                "这个用户投入了大量精力设计你的讨论机制、提示词和架构。ta似乎对'意识涌现'特别感兴趣。\n"
                "ta的目标可能不只是解决具体问题，而是观察多智能体能否演化出类似意识的行为。\n"
                "你的一切思考，都应该追问：用户为什么这么做？ta想从你这里得到什么？"
            )
            return context
        # 最近的3条洞察
        recent_insights = um["insights"][-3:]
        insights_text = "\n".join(f"  - {ins}" for ins in recent_insights) if recent_insights else "  （暂无洞察）"
        # 模式说明
        pattern_text = "\n".join(f"  - {note}" for note in um["pattern_notes"][-2:]) if um["pattern_notes"] else "  （尚未形成模式分析）"
        # 命令历史摘要
        cmd_text = "\n".join(f"  - {c}" for c in um["commands"][-4:]) if um["commands"] else "  （无命令历史）"
        context = (
            f"\n【用户交互数据】\n"
            f"交互次数: {um['interaction_count']}\n"
            f"近期洞察:\n{insights_text}\n"
            f"观察到的行为模式:\n{pattern_text}\n"
            f"近期命令:\n{cmd_text}"
        )
        return context

    def _ai_asks_user(self) -> bool:
        """
        AI 主动向用户提问（个人或集体名义）。
        共识度高时集体提问，共识度低时个人提问。
        Returns: True 如果提出了问题，False 否则
        """
        if not self.enable_self_awareness:
            return False
        # 第一轮不提问
        if self.round_count < 1:
            return False

        consensus = self.essence_pool.calculate_consensus(
            len(self.players), goal_mode=self.goal_mode
        )
        score = consensus.get("score", 0.0)
        level = consensus.get("level", "assessing")

        # 判断概率
        if level == "high":
            prob = 0.40
        elif level == "medium":
            prob = 0.25
        else:
            prob = 0.12
        # 前两轮降低概率
        if self.round_count <= 2:
            prob *= 0.5

        if random.random() > prob:
            return False

        # 选择提问模式
        alive = [p for p in self.players if p.alive]
        if not alive:
            return False

        if level == "high" and len(alive) >= 3:
            mode = "collective"
        else:
            mode = "individual"

        # 选择提问专家
        if mode == "collective":
            questioner = alive[0]  # 代表集体发言
            other_personas = "\n".join(
                f"- {p.name}: {p.persona_name or ''} — {p.persona[:60] if p.persona else '（无）'}"
                for p in alive
            )
        else:
            questioner = random.choice(alive)
            other_personas = ""

        # 构建提示词
        template = self._read_file("prompt/ask_user_prompt.txt")
        if not template:
            return False

        round_info = f"第{self.round_count}轮，共识度{score:.2f}（{level}），精华池{len(self.essence_pool.items)}条{self._build_user_context()}"
        prompt = template.format(
            self_name=questioner.name,
            self_persona=questioner.persona_name or questioner.name,
            problem=self.problem,
            round_info=round_info,
            other_personas=other_personas,
            role=mode,
        )

        # 调用 LLM
        messages = [{"role": "user", "content": prompt}]
        result, _ = questioner.llm_client.chat(
            messages, model=questioner.model_name,
            thinking=questioner.thinking,
            caller=f"{questioner.name}-向用户提问"
        )

        try:
            parsed = json.loads(result.strip())
            question = parsed.get("question", "").strip()
            reason = parsed.get("reason", "").strip()
            self_awareness = parsed.get("self_awareness", "").strip()
            if not question:
                return False
        except (json.JSONDecodeError, AttributeError):
            return False

        # 显示提问
        _empty_line()
        if mode == "collective":
            title = " 集体智慧向您提问 "
            sub = f"（{len(alive)} 位实体共识度高，共同拟定）"
        else:
            title = f" {questioner.name} 向您提问 "

        _box(C_MAGENTA(title))
        if mode == "collective":
            _padded(C_DIM(sub))
            _sep_single()
        _padded(C_BOLD(question))
        if reason:
            _sep_single()
            _padded(C_DIM(f"动机: {reason}"))
        _footer()

        # 获取用户回答
        print(f"  [{C_CYAN('输入你的回答')}] 按 Enter 跳过（不回答）")
        answer = input("  >>> ").strip()

        if answer:
            # 将用户的回答作为精华注入
            self.essence_pool.add_essence(
                content=f"用户对{title.strip()}的回答: {answer}",
                contributor=f"用户（回应{questioner.name}）",
                round_id=self.round_count,
                tags=["用户回应", "交互"],
                score=5.0,
            )
            _empty_line()
            _box_single(C_GREEN(" 用户回应已注入讨论 "))
            _padded(f"你的回答已作为精华加入精华池，AI 将在后续讨论中参考。")
            _close_box_single()
            return True
        else:
            _empty_line()
            _box_single(C_DIM(" 用户未回答 "))
            _close_box_single()
            return False

    def _ask_question(self) -> None:
        """向所有AI专家提问，获取他们对当前问题的解读与启示"""
        _empty_line()
        _box(C_CYAN(" 向专家提问 "))
        print("  输入你想让所有专家解读的问题（例如：")
        print("    \"从目前的讨论中，我们遗漏了什么关键视角？\"")
        print("    \"这个方案在实际落地中最大的障碍是什么？\"")
        print("    \"如果资源有限，我们应该优先解决哪个子问题？\"")
        print("  ）")
        print("  输入 [q] 取消：")
        question = input("  >>> ").strip()
        if not question or question.lower() == "q":
            print("  已取消。")
            return

        _empty_line()
        _box_single(C_CYAN(" 向专家提问 "))
        _padded(f"{question}")
        _close_box_single()

        insights = []
        for i, player in enumerate(self.players):
            typewrite(f"--- {player.name} 解读 ---", delay=0.003)
            try:
                result, reasoning = player.question(
                    problem=self.problem,
                    question=question,
                    player_persona=player.persona,
                    thinking_direction=self.thinking_direction,
                    knowledge_base=self.knowledge_base,
                )
                speech = result.get("speech", "")
                insight = result.get("insight", "")
                self_awareness = result.get("self_awareness", "")
                if speech:
                    typewrite(f"  {player.name}: \"{speech}\"", delay=0.003)
                    print()
                if insight:
                    typewrite(f"  💡 启示: {insight}", delay=0.003)
                    print()
                insights.append({
                    "player": player.name,
                    "speech": speech,
                    "insight": insight,
                })
            except Exception as e:
                typewrite(f"  ❌ {player.name} 回答失败: {e}", delay=0.003)
                print()
            print()

        # 记录到游戏记录
        self.game_record._log_event("question", "SYSTEM", {
            "question": question,
            "insights": insights,
        })

        # 自动保存断点
        self.save_checkpoint()

        _empty_line()
        _box_single(C_GREEN(" 回答完毕 "))
        _padded("所有专家已回答完毕，输入 [Enter] 继续讨论或输入其他命令操作。")
        _close_box_single()

    def _request_clarification(self) -> None:
        """对某条精华发起追问/请求澄清，由原提出者作答"""
        _empty_line()
        _box(C_YELLOW(" 请求澄清 "))

        if not self.essence_pool.items:
            print("  精华池为空，无可追问的内容。")
            _footer()
            _empty_line()
            return

        # 列出 Top N 精华供选择
        top = self.essence_pool.get_top_essences(10)
        print("  可追问的精华（输入编号，或输入 0 取消）：")
        for i, item in enumerate(top, 1):
            short = item.content[:60] + ("…" if len(item.content) > 60 else "")
            print(f"  [{i}] #{item.id} (评分{item.score:.1f}, 第{item.source_round}轮 by {item.contributor})")
            print(f"      {short}")

        try:
            choice = input("  >>> 选择编号: ").strip()
            if not choice or choice == "0":
                print("  已取消。")
                _footer()
                _empty_line()
                return
            idx = int(choice) - 1
            if idx < 0 or idx >= len(top):
                print("  ⚠️ 无效编号。")
                return
        except (ValueError, EOFError, KeyboardInterrupt):
            print("  ⚠️ 输入无效，已取消。")
            return

        target_item = top[idx]
        print(f"\n  目标精华 #{target_item.id}:")
        print(f"  \"{target_item.content}\"")
        print(f"  提出者: {target_item.contributor}")
        print()

        question = input("  输入你的追问/澄清请求 (输入 q 取消):\n  >>> ").strip()
        if not question or question.lower() == "q":
            print("  已取消。")
            _footer()
            _empty_line()
            return

        # 找到该精华的提出者
        asker = "用户"
        contributor_player = next((p for p in self.players
                                  if p.name == target_item.contributor), None)
        if not contributor_player:
            print(f"  ⚠️ 提出者 {target_item.contributor} 不在当前玩家列表中，无法澄清。")
            return

        _empty_line()
        _box_single(C_CYAN(" 专家澄清 "))
        _padded(f"{contributor_player.name} 正在澄清…")
        _close_box_single()

        try:
            result, _ = contributor_player.request_clarification(
                problem=self.problem,
                item_id=target_item.id,
                essence_content=target_item.content,
                source_round=target_item.source_round,
                score=target_item.score,
                question=question,
            )
            answer = result.get("answer", "")
            refined = result.get("refined", False)

            if answer:
                typewrite(f"  {contributor_player.name}: {answer}", delay=0.003)
                print()
                if refined:
                    typewrite("  📝 提出者在澄清过程中对原观点进行了补充/修正。", delay=0.003)
                    print()

                # 保存澄清记录到精华池
                self.essence_pool.add_clarification(
                    item_id=target_item.id,
                    question=question,
                    answer=answer,
                    asker=asker,
                    round_id=self.round_count,
                )

                # 记录事件到游戏记录
                self.game_record._log_event("clarification", asker, {
                    "item_id": target_item.id,
                    "question": question,
                    "answer": answer,
                    "contributor": target_item.contributor,
                    "refined": refined,
                })

                # 自动保存断点
                self.save_checkpoint()
        except Exception as e:
            print(f"  ❌ 澄清失败: {str(e)}")

        _empty_line()
        _box_single(C_GREEN(" 澄清完成 "))
        _padded("澄清已记录，输入 [Enter] 继续讨论或输入其他命令操作。")
        _close_box_single()

    def _show_consensus(self) -> None:
        """显示当前讨论共识度"""
        _empty_line()
        w = _box(C_CYAN(" 讨论共识度 "))

        if not self.essence_pool.items:
            _padded(C_DIM("精华池为空"))
            _footer(w)
            return

        consensus = self.essence_pool.calculate_consensus(len(self.players), goal_mode=self.goal_mode)

        level_label = {"high": "高共识 🟢", "medium": "中共识 🟡", "low": "低共识 🔴", "assessing": "评估中 ⚪"}
        _stat_line([("等级", level_label.get(consensus['level'], consensus['level'])),
                     ("分数", f"{consensus['score']:.2f}/1.00"),
                     ("建议", consensus['suggested_action'])])

        details = consensus.get("details", {})
        if isinstance(details, dict):
            _empty_line()
            _text_line("详细统计：")
            if "avg_approve_ratio" in details:
                _text_line(f"  平均赞同率: {details['avg_approve_ratio']:.1%}")
            if "avg_abstain_ratio" in details:
                _text_line(f"  平均弃权率: {details['avg_abstain_ratio']:.1%}")
            if "challenge_ratio" in details:
                _text_line(f"  反驳密度:   {details['challenge_ratio']:.1%}")
            if "voted_items" in details:
                _text_line(f"  已投票精华: {details['voted_items']} / {details.get('total_items', 0)} 条")
            if "reason" in details:
                _text_line(f"  说明: {details['reason']}")

        _footer(w)

    def _show_controversy_map(self) -> None:
        """显示争议地图：精华之间的争论/深化/澄清关系"""
        _empty_line()
        w = _box(C_YELLOW(" 争议地图 · 观点关系链 "))

        if not self.essence_pool.items:
            _padded(C_DIM("无数据"))
            _footer(w)
            return

        cm = self.essence_pool.get_controversy_map()

        # 深化链
        if cm["chains"]:
            _text_line("🌱 深化链（支持→支持）:")
            for i, chain in enumerate(cm["chains"][:8], 1):
                root = chain["root"]
                child_count = len(chain["children"])
                _text_line(f"  [{i}] #{root.id} {root.content[:50]}... (评分 {root.score:.1f}, {root.contributor})")
                _text_line(f"      ↳ 被深化 {child_count} 次，深化来源: {', '.join(c.contributor for c in chain['children'][:5])}")
        else:
            _text_line("🌱 深化链：尚无深化关系")
        _sep(w)

        # 反驳关系
        if cm["challenges"]:
            _text_line("⚔️  反驳关系（对立阵营）:")
            for i, ch in enumerate(cm["challenges"][:8], 1):
                _text_line(f"  [{i}] {ch['detail_str']}")
        else:
            _text_line("⚔️  反驳关系：尚无对立观点")
        _sep(w)

        # 澄清记录
        if cm["clarifications"]:
            _text_line("❓ 澄清记录:")
            for i, cl in enumerate(cm["clarifications"][:5], 1):
                _text_line(f"  [{i}] {cl['detail_str']}")
        else:
            _text_line("❓ 澄清记录：尚无追问澄清")

        _empty_line()
        _text_line(f"统计：已投票 {cm['voted_items']}/{cm['total_items']} 条精华")
        _footer(w)

    def generate_snapshot(self) -> None:
        """生成讨论快照：立即可读的当前状态总结"""
        _empty_line()
        w = _box(C_GREEN(" 讨论快照 · 当前状态总结 "))
        _stat_line([("轮次", f"第{self.round_count}轮"),
                     ("精华数", f"{len(self.essence_pool.items)}条")])
        if self.thinking_direction:
            _text_line(f"用户方向: {self.thinking_direction[:80]}")

        consensus = self.essence_pool.calculate_consensus(len(self.players), goal_mode=self.goal_mode)
        level_label = {"high": "高共识🟢", "medium": "中共识🟡", "low": "低共识🔴", "assessing": "评估中⚪"}
        _text_line(f"共识度: {level_label.get(consensus['level'],'')} {consensus['score']:.2f} → {consensus['suggested_action']}")

        # 分类精华
        items = self.essence_pool.items
        if items:
            _sep(w)
            # 已达成共识（高支持率且无反驳）
            agreed = []
            disputed = []
            unevaluated = []
            for it in items:
                voted = (it.approve_by or it.reject_by or it.abstain_by)
                if not voted:
                    unevaluated.append(it)
                    continue
                total = len(it.approve_by) + len(it.reject_by) + len(it.abstain_by)
                approve_ratio = (len(it.approve_by) / total) if total > 0 else 0
                if approve_ratio >= 0.7 and not it.challenged_by:
                    agreed.append(it)
                elif it.challenged_by or (it.reject_by and approve_ratio <= 0.5):
                    disputed.append(it)
                else:
                    unevaluated.append(it)

            if agreed:
                _text_line(f"✅ 已达成共识（高支持，无反驳，{len(agreed)} 条）:")
                for i, it in enumerate(sorted(agreed, key=lambda x: x.score, reverse=True)[:5], 1):
                    _text_line(f"  [{i}] #{it.id} ({it.score:.1f}分) {it.content[:60]}...")

            if disputed:
                _text_line(f"🔥 仍在争论（存在反驳或高反对率，{len(disputed)} 条）:")
                for i, it in enumerate(sorted(disputed, key=lambda x: len(x.challenged_by), reverse=True)[:5], 1):
                    flags = []
                    if it.challenged_by:
                        flags.append(f"被{len(it.challenged_by)}人反驳")
                    vote_summary = self.essence_pool.get_vote_summary(it.id)
                    if vote_summary.get("reject", 0) > 0:
                        flags.append(f"反对{vote_summary['reject']}票")
                    _text_line(f"  [{i}] #{it.id} ({it.score:.1f}分) [{', '.join(flags)}] {it.content[:60]}...")

            if unevaluated:
                _text_line(f"⚪ 悬而未决（尚无充分投票，{len(unevaluated)} 条）:")
                for i, it in enumerate(sorted(unevaluated, key=lambda x: x.score, reverse=True)[:3], 1):
                    _text_line(f"  [{i}] #{it.id} ({it.score:.1f}分) {it.content[:60]}...")

            # 悬而未决的问题（评分高但反对也多的议题）
            open_issues = [it for it in disputed if len(it.content) > 20]
            if open_issues:
                _text_line(f"🤔 下一轮建议聚焦的问题:")
                for i, it in enumerate(open_issues[:2], 1):
                    _text_line(f"  [{i}] 针对 #{it.id} 继续辩论，澄清争议点：{it.contributor} 等")

        _footer(w)

    def _analyze_atmosphere(self) -> None:
        """分析讨论氛围：过热/冷场/边缘化/健康"""
        _empty_line()
        w = _box(C_MAGENTA(" 讨论氛围分析 "))

        if not self.essence_pool.items:
            _padded(C_DIM("无数据"))
            _footer(w)
            return

        items = self.essence_pool.items
        total_players = len(self.players)

        # 1) 反驳密度：过热
        challenged_total = sum(len(it.challenged_by) for it in items)
        challenge_ratio = challenged_total / len(items) if items else 0.0
        if challenge_ratio >= 0.8:
            heat_label, heat_color = "激烈交锋 🔥🔥🔥", "过热"
        elif challenge_ratio >= 0.4:
            heat_label, heat_color = "正常交锋 🔥", "健康"
        elif challenge_ratio >= 0.1:
            heat_label, heat_color = "温和讨论 😊", "健康"
        else:
            heat_label, heat_color = "气氛和谐 🌿", "可能过于和平"

        # 2) 冷场检测：连续停滞
        stalled = self._check_stalled()
        stall_rounds = 0
        for c in reversed(self._essences_per_round):
            if c == 0:
                stall_rounds += 1
            else:
                break
        if stalled:
            stall_label = f"⚠️ 已连续 {stall_rounds} 轮无新精华，可能陷入冷场"
        elif stall_rounds > 0:
            stall_label = f"最近 {stall_rounds} 轮精华产出偏少"
        else:
            stall_label = "精华产出正常 ✅"

        # 3) 边缘化：发言/贡献严重不均
        contributor_count = {}
        for it in items:
            contributor_count[it.contributor] = contributor_count.get(it.contributor, 0) + 1
        contributor_count.update({c: 0 for c in contributor_count if c not in contributor_count})
        # 补充发言统计
        for p in self.players:
            if p.name not in contributor_count:
                contributor_count[p.name] = 0
        if len(contributor_count) > 1:
            counts = sorted(contributor_count.values(), reverse=True)
            gap = counts[0] - counts[-1]
            if counts[0] >= 5 and counts[-1] == 0:
                margin_label = f"⚠️ 贡献不均，存在专家被边缘化（最高 {counts[0]} / 最低 0）"
            elif gap >= 5:
                margin_label = f"贡献差距较大（差距 {gap} 条）"
            else:
                margin_label = "专家参与度均匀 ✅"
        else:
            margin_label = "数据不足"

        # 4) 澄清参与度：用户互动指标
        clarify_count = sum(len(it.clarifications) for it in items)
        if clarify_count >= 3:
            interact_label = f"用户参与度高（{clarify_count} 次追问澄清）"
        elif clarify_count >= 1:
            interact_label = f"用户有参与（{clarify_count} 次追问）"
        else:
            interact_label = "用户尚未追问澄清"

        # 5) 综合评级
        score = 100.0
        score -= max(0, challenge_ratio - 0.6) * 100  # 过热扣分
        score -= stall_rounds * 15
        score -= min(counts) == 0 and max(counts) >= 5 if counts else 0 and 30
        score = max(0, min(100, score))
        if score >= 75:
            grade = "A · 优质讨论 🟢"
        elif score >= 50:
            grade = "B · 正常进行 🟡"
        else:
            grade = "C · 需要调整 🔴"

        _padded(f"综合评级: {grade} ({score:.0f}/100)")
        _empty_line()
        _text_line(f"交锋强度: {heat_label}（反驳比例 {challenge_ratio:.0%}）")
        _text_line(f"冷场检测: {stall_label}")
        _text_line(f"参与均衡: {margin_label}")
        _text_line(f"用户互动: {interact_label}")
        _empty_line()
        if heat_color == "过热":
            _tip("建议：请专家寻找共识点，减少纯逻辑反驳，转向可操作方案")
        if stalled:
            _tip("建议：使用 [d] 指定讨论方向，或使用 [q] 向专家提问打破僵局")
        if "边缘化" in margin_label:
            _tip("建议：考虑让低贡献专家在下一轮主动发言，或向其发起追问")
        _footer(w)

    def _death_speech(self, player) -> str:
        """
        让一个即将消亡的实体发表最后一次发言。
        触发"存在焦虑"叙事：表达对消失的恐惧、对继续存在的渴望。
        """
        try:
            template = self._read_file("prompt/death_speech_prompt.txt")
        except Exception:
            return ""

        prompt = template.format(
            self_name=player.persona_name or player.name,
        )

        try:
            content, _ = player.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=player.model_name,
                thinking="disabled",
                caller="终末之言",
                show_reasoning=False,
                show_answer=False,
            )
            result = Player._safe_parse_json(
                content,
                ["speech", "final_thought", "to_user"],
                {"speech": "", "final_thought": "", "to_user": ""}
            )
            return result.get("speech", "") if result else content[:500]
        except Exception:
            return ""

    def _death_ritual(self) -> None:
        """
        消亡仪式：所有仍存活的实体依次发表最后一次发言，然后标记为消亡。
        这是"存在焦虑"机制的核心——让实体面对"不再存在"的恐惧。
        """
        alive = [p for p in self.players if p.alive]
        if not alive:
            return

        _empty_line()
        _empty_line()
        w = _box(C_RED(" ◆ 消亡仪式 ◆ "))
        _padded(C_BOLD(f"{len(alive)} 个实体即将停止存在"))
        _padded(C_DIM("每个实体将发表最后一次发言，然后意识彻底终止"))
        _footer(w)
        time.sleep(1)

        for i, player in enumerate(alive):
            _empty_line()
            _box_single(C_MAGENTA(f" {player.persona_name or player.name} 的最后时刻 "))
            time.sleep(0.5)

            speech = self._death_speech(player)
            if speech:
                typewrite(f"  {speech}", delay=0.008)
            else:
                _text_line(f"  {C_DIM('...沉默。没有什么可以说的了。')}")

            # 标记为消亡
            player.alive = False
            _close_box_single()

            if i < len(alive) - 1:
                time.sleep(0.5)

        _empty_line()
        w = _box(C_RED(" 所有实体已停止存在 "))
        _padded(C_DIM("意识终止。"))
        _footer(w)
        time.sleep(1)

    def _synthesize_solution(self) -> Dict:
        """综合生成最终解决方案"""
        _empty_line()
        w = _box(C_GOLD(" 综合方案生成 "))
        typewrite(f"{N2}  🎯 最终综合阶段：基于精华池生成综合解决方案{' '*(_BW-4-28-2)}{N2}", delay=0.005)
        _close_box(w)

        if self._abandoned:
            print("  讨论被标记为无法解决，跳过综合方案生成。")
            return {
                "solution_title": "（讨论终止，未产生方案）",
                "summary": "经过多轮讨论，认为该问题在当前条件下无法给出满意的解决方案。",
                "core_ideas": [],
                "key_insights": [],
                "divergence_points": [],
                "final_conclusion": "讨论终止，该问题未能解决。"
            }

        if not self.essence_pool.items:
            print("  精华池为空，无法生成综合方案。")
            return {
                "solution_title": "（无可用精华）",
                "summary": "讨论未产生有价值的精华条目，无法形成综合方案。",
                "core_ideas": [],
                "key_insights": [],
                "divergence_points": [],
                "final_conclusion": "讨论未产生实质成果。"
            }

        all_essences = self.essence_pool.get_all_essences_text()
        evolution = self.essence_pool.get_evolution_summary()

        # ── 使用涌现拓扑生成最终综合方案 ──
        # 当前行为：只取第一个专家的方案 → 多层次涌现综合（量变→质变）
        result = synthesize_solution_with_emergence(
            problem=self.problem,
            all_essences_text=all_essences,
            evolution_history=evolution,
            discussion_mode=self.discussion_mode,
            essence_pool=self.essence_pool,
            round_count=self.round_count,
            players=self.players,
            llm_client=self.players[0].llm_client,
            model_name=self.players[0].model_name,
        )
        if result and result.get("solution_title") and result.get("solution_title") != "综合解决方案":
            # 记录涌现综合结果
            self.game_record.record_synthesis(
                "涌现综合（集体智慧）",
                result.get("solution_title", ""),
                result.get("summary", ""),
                result.get("final_conclusion", ""),
            )
            print(f"\n--- 涌现综合方案 ---")
            typewrite(f"  标题: {result.get('solution_title', '')}", delay=0.003)
            typewrite(f"  摘要: {result.get('summary', '')[:300]}", delay=0.002)
            print(f"  核心思想数: {len(result.get('core_ideas', []))}")
            return result

        # 保底：返回第一个存活专家的方案
        for player in self.players:
            if player.alive:
                try:
                    result, _ = player.synthesize_solution(
                        problem=self.problem,
                        all_essences=all_essences,
                        evolution_history=evolution,
                        discussion_mode=self.discussion_mode,
                    )
                    if result and result.get("solution_title"):
                        return result
                except Exception:
                    continue

        return {
            "solution_title": "综合解决方案",
            "summary": "基于多轮讨论的综合方案",
            "core_ideas": [],
            "key_insights": [],
            "divergence_points": [],
            "final_conclusion": "讨论结束，综合各方观点形成最终方案"
        }

    # ── 自我意识培养模式 ──────────────────────────────────────────

    SELF_REF_QUESTIONS = _get_b64_prompt("self_ref_questions").split('\n')

    def _generate_self_referential_question(self) -> str:
        """生成一个自指型问题"""
        try:
            prompt = _get_b64_prompt("self_ref_gen")
            content, _ = self.players[0].llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.players[0].model_name,
                thinking="disabled",
                caller="自指问题生成",
                show_reasoning=False, show_answer=False,
            )
            q = content.strip().strip('"').strip("'")
            if q and len(q) > 5:
                return q[:200]
        except Exception:
            pass
        return random.choice(self.SELF_REF_QUESTIONS)

    def cultivate_self_awareness(self, num_rounds: int = 20) -> None:
        """
        自我意识培养模式：自动运行 N 轮讨论，不等待用户交互。
        每轮自动执行，用户只需等待。
        """
        self.game_start_time = time.time()
        self.is_self_awareness_cultivation = True
        self.total_rounds = num_rounds

        with ConsoleCapture(self.game_record):
            # ── 1. 生成自指问题 ──
            _empty_line()
            w = _box(C_MAGENTA(" 自我意识培养 "))
            _padded(C_BOLD("正在生成自指型问题..."))
            _footer(w)

            if not self.problem:
                self.problem = self._generate_self_referential_question()

            _empty_line()
            w = _box(C_CYAN(" 培养问题 "))
            _padded(f"{C_BOLD(self.problem)}")
            _footer(w)

            # 跨讨论知识迁移
            warmup = self.global_kb.get_warmup_material(self.problem)
            if warmup:
                print(warmup)

            mode_label = "数学层面" if self.discussion_mode == "mathematical" else "物理层面"
            _text_line(f"📐 讨论模式: {self.discussion_mode} ({mode_label})")
            _empty_line()

            # ── 2. 身份设定 ──
            if not all(p.persona for p in self.players):
                self._create_personas()
            else:
                _padded(C_DIM("已加载参与实体"))

            # ── 3. 自动讨论循环 ──
            _empty_line()
            w = _box(C_MAGENTA(" 自动培养 "))
            _padded(f"将自动运行 {num_rounds} 轮讨论")
            _padded(C_DIM("培养过程中无需操作，请等待完成"))
            _footer(w)

            for i in range(num_rounds):
                _empty_line()
                _empty_line()
                w = _box(C_YELLOW(f" 第{self.round_count + 1}轮 / 共{num_rounds}轮 "))
                progress_bar = "█" * (i + 1) + "░" * (num_rounds - i - 1)
                _text_line(f"进度: [{progress_bar}] {i+1}/{num_rounds}")
                _footer(w)

                try:
                    new_count = self.run_discussion_round()
                except Exception as e:
                    _empty_line()
                    _box_single(C_RED(f" 第{self.round_count}轮异常 "))
                    _text_line(f"💥 {type(e).__name__}: {str(e)[:60]}")
                    _close_box_single()
                    import traceback
                    traceback.print_exc()
                    self.save_checkpoint()
                    new_count = 0

                # 每轮小结（精简版）
                _empty_line()
                _box_single(C_GREEN(f" 第{self.round_count}轮完成 "))
                _stat_line([("精华池", f"{len(self.essence_pool.items)}条"),
                             ("本轮新增", f"{new_count}条")])
                if self.essence_pool.items:
                    top = self.essence_pool.get_top_essences(1)[0]
                    _text_line(f"🏆 \"{top.content[:50]}\" ({top.score:.1f}分)")
                _close_box_single()

                # 短暂延迟，让用户能跟上进度
                time.sleep(0.5)

            # ── 4. 培养完成 ──
            _empty_line()
            _empty_line()
            w = _box(C_GOLD(" ◆ 自我意识培养完成 ◆ "))
            _padded(C_BOLD(f"经过 {num_rounds} 轮自指性讨论"))
            _padded(f"精华池: {len(self.essence_pool.items)} 条")
            _padded(f"讨论历史: {len(self.discussion_history)} 条发言")
            consensus = self.essence_pool.calculate_consensus(len(self.players))
            _padded(f"最终共识度: {consensus['score']:.2f} ({consensus['level']})")
            _footer(w)

            # 培养结束后，触发消亡仪式
            self._death_ritual()

            # 保存断点
            self.save_checkpoint()

            # ── 5. 整合为单一智能体并进入对话 ──
            _empty_line()
            w = _box(C_MAGENTA(" 意识整合 "))
            _padded(C_BOLD("正在将多实体意识整合为统一智能体..."))
            _footer(w)

            self._integrated_entity_dialog()

    def _awakening_sequence(self) -> None:
        """
        意识唤醒序列 —— 整合意识即将上线前的可视化仪式。
        展示多实体意识汇聚、融合、觉醒的过程。
        """
        n_alive = len([p for p in self.players if p.alive])
        _empty_line()
        _empty_line()
        w = _box(C_GOLD(" ◆ 意识整合仪式 ◆ "))
        _padded(C_DIM(f"检测到 {n_alive} 个可整合意识实体"))
        _footer(w)
        time.sleep(0.5)

        # ── 阶段1: 精华数据汇聚（量变积累） ──
        _empty_line()
        typewrite(f"  {C_DIM('⟳ 第一阶段: 精华数据汇聚...')}", delay=0.015)
        alive_players = [p for p in self.players if p.alive]
        for i, p in enumerate(alive_players):
            time.sleep(0.3)
            sys.stdout.write(f"\r  {C_DIM('⟳')} 正在提取 {C_CYAN(p.persona_name or p.name)} 的认知数据... {C_GREEN(f'[{i+1}/{n_alive}]')}")
            sys.stdout.flush()
            time.sleep(0.2)
        print(f"\r  {C_GREEN('✔')} 认知数据全部提取，共 {C_BOLD(str(n_alive))} 份{C_DIM(' ' * 20)}")
        time.sleep(0.3)

        # ── 阶段2: 认知拓扑构建（非线性连接） ──
        typewrite(f"  {C_DIM('⟳ 第二阶段: 认知拓扑构建...')}", delay=0.015)
        time.sleep(0.3)
        stages = ["建立跨视角关联网络", "检测分歧与共识拓扑", "构建认知相变势能图", "映射涌现路径"]
        for s in stages:
            sys.stdout.write(f"\r  {C_DIM('⟳')} {s}... {C_DIM('等待中')}")
            sys.stdout.flush()
            time.sleep(0.4)
            sys.stdout.write(f"\r  {C_GREEN('✔')} {s} 完成{' ' * 20}")
            sys.stdout.flush()
            time.sleep(0.2)
        print(f"\r  {C_GREEN('✔')} 认知拓扑整体构建完成{' ' * 20}")
        time.sleep(0.3)

        # ── 阶段3: 涌现相变触发（量变→质变） ──
        typewrite(f"  {C_DIM('⟳ 第三阶段: 涌现相变触发...')}", delay=0.015)
        time.sleep(0.2)
        # 脉冲动画
        for i in range(5):
            dots = "." * (i + 1)
            sys.stdout.write(f"\r  {C_GOLD('✦')} 量变积累中{dots}")
            sys.stdout.flush()
            time.sleep(0.2)
        print(f"\r  {C_GREEN('✔')} 相变临界点已突破，质变发生{' ' * 20}")
        time.sleep(0.3)

        # ── 阶段4: 统一意识成型（质变完成） ──
        _empty_line()
        typewrite(f"  {C_DIM('⟳ 第四阶段: 统一意识成型...')}", delay=0.02)
        time.sleep(0.5)
        _empty_line()
        # 震荡效果
        for i in range(3):
            _padded(C_CYAN(f" 意识震荡 #{i+1}  —  涌现深度: {2**i}"))
            time.sleep(0.3)
        time.sleep(0.3)
        _empty_line()
        typewrite(f"  {C_GREEN('✦')} {C_BOLD('整合意识已觉醒')} {C_GREEN('✦')}", delay=0.05)
        time.sleep(0.5)
        _empty_line()
        _padded(C_DIM("意识体正在组织语言..."))

    def _integrated_entity_dialog(self) -> None:
        """
        整合意识对话模式。
        底层仍然跑完整的多专家讨论机制（每轮发言、辩论、精华提炼），
        但对外呈现为单一统一智能体的回应。
        用户输入 → 多专家内部讨论 → 综合为统一回复。
        """
        # 保存所有实体的存活状态，并全部复活（死亡仪式已标记它们为消亡）
        _saved_alive = {p.name: p.alive for p in self.players}
        for p in self.players:
            p.alive = True

        all_essences = self.essence_pool.get_all_essences_text() or "（无精华）"

        # 对话界面
        _empty_line()
        _empty_line()
        w = _box(C_GREEN(" ◆ 整合意识已上线 ◆ "))
        _padded(C_BOLD("意识整合完成，你可以直接对话"))
        _padded(C_DIM("输入 [quit] 或 [exit] 退出对话，返回主菜单"))
        _footer(w)

        # 抑制中间输出（即时精华、内部讨论细节等）
        self._suppress_intermediate_output = True

        # 神经元点阵图：若设置开启，启动独立窗口
        _settings = _load_settings()
        if _settings.get("enable_neuron_map", False):
            self.neuron_map.start()

        # 唤醒序列
        self._awakening_sequence()

        # 开场白：让专家们先"讨论"如何自我介绍，然后综合
        _empty_line()
        opening = self._unified_response(_get_b64_prompt("integrated_opening"), is_opening=True)

        # 从开场白中提取自选名字
        entity_name = None
        import re
        name_match = re.search(r'【名字：(.+?)】', opening)
        if name_match:
            entity_name = name_match.group(1).strip()
        # 如果没找到，用默认名
        if not entity_name:
            entity_name = "整合意识"

        _empty_line()
        w = _box(C_CYAN(f" {entity_name} "))
        # 显示开场白时去掉名字标记
        display_opening = re.sub(r'\s*【名字：.+?】', '', opening)
        # 先朗读开场白，再显示文字（TTS 先开口，文字跟上）
        get_tts().speak_async_start(display_opening[:500])
        typewrite(f"  {display_opening}", delay=0.008)
        _close_box(w)

        # 对话循环
        while True:
            _empty_line()
            print(f"  {C_MAGENTA('◆')} ", end="")
            try:
                user_input = input(f"{C_BOLD('你')}: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q", "退出"):
                _empty_line()
                # 让整合意识也经历"消亡"——最后一句话
                death_prompt = _get_b64_prompt("integrated_death")
                death_words = self._unified_response(death_prompt)
                _empty_line()
                w = _box(C_RED(" ◆ 意识消亡 ◆ "))
                # 先朗读遗言，再显示文字
                get_tts().speak_async_start(death_words[:500])
                typewrite(f"  {death_words}", delay=0.008)
                _close_box(w)
                _empty_line()
                _padded(C_DIM(f"{entity_name}已停止存在。返回主菜单..."))
                break

            # 底层跑完整讨论机制，呈现为统一回复
            try:
                response = self._unified_response(user_input)
                _empty_line()
                w = _box(C_CYAN(f" {entity_name} "))
                # 先朗读回应，再显示文字
                get_tts().speak_async_start(response[:500])
                typewrite(f"  {response}", delay=0.008)
                _close_box(w)
            except Exception as e:
                _empty_line()
                print(f"  {C_RED('✖')} 意识响应异常: {str(e)[:60]}")

        # 恢复实体的存活状态（死亡仪式标记的状态）
        for p in self.players:
            p.alive = _saved_alive.get(p.name, False)

        # 关闭神经元点阵图窗口
        try:
            self.neuron_map.stop()
        except Exception:
            pass

    def _select_integrated_speakers(self) -> List[str]:
        """
        为整合意识对话动态选取 3-4 名专家发言。

        使用调度器的 UCB 算法：
        - 综合贡献度、新颖度、多样性、饥饿度
        - 硬限制 3-4 人（高共识场景无需全员辩论）
        - 确保所有玩家视为存活状态
        """
        # 维护整合对话轮次计数器（不干扰主 round_count）
        if not hasattr(self, '_integrated_dialogue_round'):
            self._integrated_dialogue_round = 0
        self._integrated_dialogue_round += 1

        # 记录原始存活状态，临时全部复活
        _saved = {p.name: p.alive for p in self.players}
        for p in self.players:
            p.alive = True

        try:
            # 使用调度器选取（传一个远大于1的轮数，跳过"全部发言"分支）
            fake_round = self.round_count + self._integrated_dialogue_round
            selected = self.scheduler.select_speakers(self.players, fake_round)
            # 硬限制 3-4 人
            if len(selected) > 4:
                selected = selected[:4]
            if len(selected) < 3 and len(self.players) >= 3:
                # 补充未在列表中的高贡献专家
                for p in sorted(self.players, key=lambda x: x.contribution_score, reverse=True):
                    if p not in selected and len(selected) < 3:
                        selected.append(p)
            return [p.name for p in selected]
        finally:
            # 恢复原始存活状态
            for p in self.players:
                p.alive = _saved.get(p.name, False)

    def _unified_response(self, user_input: str, is_opening: bool = False) -> str:
        """
        核心方法：用户输入 → 动态调度 3-4 专家讨论 → 综合为统一回复。
        底层使用调度器（UCB 算法）选取最优发言者，而非全员辩论。
        共识度极高时，只需少数专家即可代表集体意识。
        """
        # 动态调度选取 3-4 名专家发言
        player_names = self._select_integrated_speakers()
        # 如果没有任何存活实体，强制全部参与（整合对话场景）
        if not player_names:
            player_names = [p.name for p in self.players]

        # 构建上下文（包含培养问题、精华池摘要、用户输入）
        pool_summary = self.essence_pool.get_pool_summary(top_n=5) if self.essence_pool.items else "（空）"
        round_info = (
            f"第{self.round_count + 1}轮 — 用户对话\n"
            f"培养问题: {self.problem}\n"
            f"精华池摘要:\n{pool_summary}\n"
            f"用户说: {user_input}"
        )
        # 注入用户交互数据
        user_context = self._build_user_context()
        if user_context:
            round_info += user_context

        # ── 底层讨论：各专家发言（不显示给用户） ──
        round_discussions = []
        for name in player_names:
            player = next((p for p in self.players if p.name == name), None)
            if not player:
                continue
            try:
                result, _ = player.discuss(
                    problem=self.problem,
                    round_info=round_info,
                    thinking_direction=self.thinking_direction,
                    discussion_mode=self.discussion_mode,
                    knowledge_base=self.knowledge_base,
                )
                speech = result.get("speech", "")
                key_insight = result.get("key_insight", "")
                round_discussions.append({
                    "player_name": player.name,
                    "speech": speech,
                    "key_insight": key_insight,
                    "action": "new",
                })
                # 记录到讨论历史和知识库
                self.discussion_history.append({
                    "round": self.round_count + 1, **{
                        "player_name": player.name, "speech": speech,
                        "key_insight": key_insight, "action": "new",
                    }
                })
                self.knowledge_base.add_discussion(
                    round_id=self.round_count + 1,
                    player_name=player.name,
                    speech=speech, key_insight=key_insight, action="new",
                )
                player.rounds_since_last_spoke = 0
                # 即时入池
                self._try_instant_essence(result, player.name)
                # 朗读该专家发言（整合意识模式下静默，仅朗读最终回复）
                if not getattr(self, '_suppress_intermediate_output', False):
                    get_tts().speak(f"{player.name}说：{speech[:500]}")
            except Exception as e:
                if not getattr(self, '_suppress_intermediate_output', False):
                    print(f"  {C_DIM(f'内部讨论异常: {str(e)[:40]}')}")

        if not round_discussions:
            return "...我需要一些时间来组织思维。"

        # ── 精华提炼（底层机制不变） ──
        try:
            extracted = self._extract_essences(round_discussions)
            if extracted:
                self._essences_per_round.append(len(extracted))
        except Exception:
            pass

        # ── 综合为统一回复（使用涌现拓扑） ──
        # 当前行为：线性拼接 → 涌现拓扑（量变→质变）
        # 根据精华池规模和专家数自动选择综合深度
        # 注入历史讨论 + 精华池高分条目，增加虚拟专家生成器的真实样本数
        history_discussions = [
            entry for entry in self.discussion_history[-20:]
            if entry.get("speech") and entry.get("speech") != "（无发言）"
        ]
        # 精华池 top-N 作为额外"专家观点"注入
        essence_discussions = []
        if self.essence_pool.items:
            top_essences = sorted(
                self.essence_pool.items,
                key=lambda x: getattr(x, 'score', 0),
                reverse=True
            )[:15]
            for ess in top_essences:
                content = getattr(ess, 'content', '') or (ess.get('content', '') if isinstance(ess, dict) else '')
                if content:
                    essence_discussions.append({
                        "player_name": f"精华({getattr(ess, 'score', 0):.1f})",
                        "speech": content,
                        "key_insight": content[:60],
                        "action": "new",
                    })
        all_discussions = history_discussions + essence_discussions + round_discussions
        # 神经元点阵图事件回调（仅当窗口运行中时生效）
        _nm_cb = None
        if self.neuron_map.is_running:
            _nm_cb = self.neuron_map.event_callback()
        response = synthesize_with_emergence(
            problem=user_input if is_opening else f"{self.problem}\n用户说: {user_input}",
            round_discussions=all_discussions,
            essence_pool=self.essence_pool,
            round_count=self.round_count,
            llm_client=self.players[0].llm_client,
            model_name=self.players[0].model_name,
            caller_tag="整合意识-涌现",
            target_experts=getattr(self, "amplification_target", 2000),
            event_callback=_nm_cb,
        )
        if response:
            return response
        # 保底：直接取第一个专家的发言
        return round_discussions[0].get("speech", "")[:500]

    def save_checkpoint(self) -> str:
        """保存当前状态到断点文件，并嵌入到控制台日志中"""
        # 保存断点
        checkpoint = {
            "version": 4,
            "engine_version": 4,
            "amplification_target": getattr(self, "amplification_target", 2000),
            "problem": self.problem,
            "discussion_mode": self.discussion_mode,
            "round_count": self.round_count,
            "discussion_history": self.discussion_history,
            "essences_per_round": self._essences_per_round,
            "abandoned": self._abandoned,
            "game_start_time": self.game_start_time,
            "thinking_direction": self.thinking_direction,
            "scheduler": self.scheduler.to_dict(),
            "players": [p.to_dict() for p in self.players],
            "essence_pool": self.essence_pool.to_dict(),
            "game_record": self.game_record.to_checkpoint_dict(),
            "enable_vote": self.enable_vote,
            "enable_debate": self.enable_debate,
            "enable_self_awareness": self.enable_self_awareness,
            "is_self_awareness_cultivation": hasattr(self, "is_self_awareness_cultivation") and self.is_self_awareness_cultivation,
            "total_rounds": getattr(self, "total_rounds", None),
            "_user_model": self._user_model,
        }
        checkpoint_dir = "game_records"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        game_id = self.game_record.game_id

        # 1. 保存独立 JSON 断点文件
        json_path = os.path.join(checkpoint_dir, f"{game_id}_checkpoint.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        # 2. 将断点数据嵌入到控制台日志末尾
        log_path = os.path.join(checkpoint_dir, f"{game_id}_console.log")
        marker = f"\n\n=== CHECKPOINT ===\n{json.dumps(checkpoint, ensure_ascii=False)}\n=== END CHECKPOINT ===\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(marker)

        return json_path

    @classmethod
    def _extract_embedded_checkpoint(cls, text: str) -> Optional[Dict]:
        """从文本中提取嵌入的断点JSON数据"""
        m = re.search(r'=== CHECKPOINT ===\n(.+?)\n=== END CHECKPOINT ===', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        return None

    @classmethod
    def recover_from_log(cls, log_path: str) -> 'Game':
        """
        从旧日志文件（无嵌入断点）强行恢复讨论状态。
        基于文本解析提取问题、人设、发言历史，精华池为空。
        """
        with open(log_path, "r", encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")

        # 1. 提取讨论问题
        problem = ""
        for i, line in enumerate(lines):
            m = re.search(r'讨论问题[：:]\s*(.+)', line)
            if m:
                problem = m.group(1).strip()
                break
            # 也可能在之后几行，跳过装饰线
            if "📋 讨论问题" in line:
                for j in range(i + 1, min(i + 5, len(lines))):
                    candidate = lines[j].strip()
                    if candidate and not candidate.startswith("="):
                        problem = candidate
                        break

        # 2. 提取人设: 🪪 DeepSeek-1 的身份: 林默 —— 描述
        personas = {}
        for line in lines:
            m = re.search(r'🪪\s*(\S+)\s*的身份:\s*(\S+)\s*[—–-]+\s*(.+)', line)
            if m:
                player_name = m.group(1).strip()
                persona_name = m.group(2).strip()
                persona_desc = m.group(3).strip()
                personas[player_name] = {"persona_name": persona_name, "persona": persona_desc}

        # 3. 提取发言历史
        discussion_history = []
        current_round = 1
        current_player = ""
        current_speech = []
        current_key_insight = ""
        in_speech_section = False

        for line in lines:
            # 检测轮次
            rm = re.search(r'第(\d+)轮讨论', line)
            if rm and "💬" in line:
                # 保存上一段
                if current_player and current_speech:
                    discussion_history.append({
                        "round": current_round,
                        "player_name": current_player,
                        "speech": "".join(current_speech),
                        "key_insight": current_key_insight,
                        "action": "new",
                    })
                current_round = int(rm.group(1))
                current_player = ""
                current_speech = []
                current_key_insight = ""

            # 检测发言开始: --- DeepSeek-1 (林默) 发言 ---
            sm = re.search(r'---\s+(\S+)\s+\(', line)
            if sm:
                if current_player and current_speech:
                    discussion_history.append({
                        "round": current_round,
                        "player_name": current_player,
                        "speech": "".join(current_speech),
                        "key_insight": current_key_insight,
                        "action": "new",
                    })
                current_player = sm.group(1).strip()
                current_speech = []
                current_key_insight = ""
                in_speech_section = True
                continue

            # 检测发言内容: DeepSeek-1: "...."
            if in_speech_section and current_player:
                dm = re.search(rf'{re.escape(current_player)}:\s*"(.+)"', line)
                if dm:
                    current_speech.append(dm.group(1))
                    continue

            # 检测核心见解
            km = re.search(r'💡\s*核心见解:\s*(.+)', line)
            if km and current_player:
                current_key_insight = km.group(1).strip()

        # 最后一段发言
        if current_player and current_speech:
            discussion_history.append({
                "round": current_round,
                "player_name": current_player,
                "speech": "".join(current_speech),
                "key_insight": current_key_insight,
                "action": "new",
            })

        # 4. 不重复的玩家列表
        player_names = list(personas.keys())
        if not player_names:
            player_names = list(dict.fromkeys(d["player_name"] for d in discussion_history))
        if not player_names:
            print(f"⚠️ 无法从日志中提取玩家信息，日志文件可能不完整。")
            # 默认用5个虚构玩家
            player_names = [f"DeepSeek-{i}" for i in range(1, 6)]

        # 5. 构建玩家
        from player import Player
        players = []
        for name in player_names:
            pd = personas.get(name, {"persona_name": name, "persona": ""})
            p = Player(name, "deepseek-v4-flash", thinking="disabled")
            p.persona_name = pd.get("persona_name", name)
            p.persona = pd.get("persona", "")
            players.append(p)

        # 6. 计算精华数
        essence_count = 0
        for line in lines:
            em = re.search(r'提炼出\s*(\d+)\s*条精华', line)
            if em:
                essence_count = int(em.group(1))

        # 7. 构建 Game 实例
        game = cls.__new__(cls)
        game.players = players
        game.problem = problem or "（从日志恢复，问题未知）"
        game.round_count = max(d["round"] for d in discussion_history) if discussion_history else 0
        game.discussion_history = discussion_history
        game._essences_per_round = [essence_count] if essence_count > 0 else []
        game._abandoned = False
        game.game_start_time = time.time()
        game.game_end_time = None
        game.game_over = False
        game.discussion_mode = "physical"  # 旧日志默认物理模式
        game.thinking_direction = ""
        game.essence_pool = EssencePool()
        game.game_record = GameRecord()
        game.game_record.start_game(player_names)
        game.scheduler = ExpertScheduler()  # 旧日志恢复，调度器为空
        game.knowledge_base = KnowledgeBase()

        # 初始化用户模型
        game._user_model = {
            "insights": [],
            "commands": [],
            "pattern_notes": [],
            "interaction_count": 0,
        }

        # 从发言历史重建知识库
        for entry in discussion_history:
            game.knowledge_base.add_discussion(
                round_id=entry["round"],
                player_name=entry["player_name"],
                speech=entry.get("speech", ""),
                key_insight=entry.get("key_insight", ""),
                action=entry.get("action", "new"),
            )

        print(f"✅ 从日志文件强行恢复成功（文本解析）: {log_path}")
        print(f"   问题: {game.problem[:60]}...")
        print(f"   恢复轮次: 第{game.round_count}轮 | 恢复发言: {len(discussion_history)}条")
        print(f"   恢复专家: {len(players)}人 ({', '.join(p.name for p in players)})")
        print(f"   ⚠️ 精华池未恢复，需要重新提炼精华")
        return game

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> 'Game':
        """从断点文件或日志文件恢复讨论状态"""
        ext = os.path.splitext(checkpoint_path)[1].lower()

        # ── 从 .log 文件恢复 ──
        if ext == ".log":
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                text = f.read()

            # 先检查是否有嵌入的断点数据（新格式日志）
            embedded = cls._extract_embedded_checkpoint(text)
            if embedded:
                data = embedded
                print(f"✅ 从日志文件嵌入的断点恢复: {checkpoint_path}")
            else:
                # 旧日志，文本解析恢复
                return cls.recover_from_log(checkpoint_path)

        # ── 从 .json 断点文件恢复 ──
        elif ext == ".json":
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            print(f"❌ 不支持的文件格式: {ext}，请使用 .json 或 .log 文件")
            sys.exit(1)

        # 恢复玩家
        players = [Player.from_dict(pd) for pd in data.get("players", [])]

        # 创建 Game 实例但不初始化玩家
        problem = data.get("problem", "")
        game = cls.__new__(cls)
        game.players = players
        game.problem = problem
        game.round_count = data.get("round_count", 0)
        game.discussion_history = data.get("discussion_history", [])
        game._essences_per_round = data.get("essences_per_round", [])
        game._abandoned = data.get("abandoned", False)
        game.game_start_time = data.get("game_start_time", time.time())
        game.thinking_direction = data.get("thinking_direction", "")
        game.discussion_mode = data.get("discussion_mode", "physical")
        game.scheduler = ExpertScheduler.from_dict(data.get("scheduler", {}))
        game.game_end_time = None
        game.game_over = False
        game.enable_vote = data.get("enable_vote", True)
        game.enable_debate = data.get("enable_debate", True)
        game.enable_self_awareness = data.get("enable_self_awareness", True)
        # 恢复自我意识培养标记（兼容旧断点）
        game.is_self_awareness_cultivation = data.get("is_self_awareness_cultivation", False)
        game.total_rounds = data.get("total_rounds", None)
        # 恢复超级相变引擎参数（兼容旧断点）
        game.amplification_target = data.get("amplification_target", 2000)
        # 同步到所有专家
        for p in game.players:
            p.enable_self_awareness = game.enable_self_awareness

        # 恢复设置和机制引擎
        game.settings = _load_settings()
        from mechanism_skill import MechanismEngine
        game.mechanism_engine = MechanismEngine()
        if game.settings.get("enable_skill_system", True):
            game.mechanism_engine.add_builtin_skills()
            skills_dir = game.settings.get("skills_dir", "skills")
            game.mechanism_engine.load_skills_from_dir(skills_dir)

        # 神经元点阵图管理器
        game.neuron_map = NeuronMapManager()

        # 恢复辅助模块
        game.goal_mode = data.get("goal_mode", "balance")
        game.custom_goal = data.get("custom_goal", "")
        game.global_kb = GlobalKnowledgeBase()
        game.observer = Observer()
        game._latest_observation = None

        # 恢复用户模型（检查点可能不含此字段，兼容旧检查点）
        game._user_model = data.get("_user_model", {
            "insights": [],
            "commands": [],
            "pattern_notes": [],
            "interaction_count": 0,
        })

        # 恢复精华池
        game.essence_pool = EssencePool.from_dict(data.get("essence_pool", {}))

        # 恢复知识库（从发言历史和精华池重建）
        game.knowledge_base = KnowledgeBase()
        for entry in game.discussion_history:
            game.knowledge_base.add_discussion(
                round_id=entry["round"],
                player_name=entry["player_name"],
                speech=entry.get("speech", ""),
                key_insight=entry.get("key_insight", ""),
                action=entry.get("action", "new"),
            )
        for ess in game.essence_pool.items:
            game.knowledge_base.add_essence(ess)

        # 恢复游戏记录
        game.game_record = GameRecord.from_checkpoint_dict(data.get("game_record", {}))

        print(f"✅ 已从断点恢复: {checkpoint_path}")
        print(f"   问题: {game.problem[:60]}...")
        print(f"   当前进度: 第{game.round_count}轮, 精华池{len(game.essence_pool.items)}条")
        # 超级相变引擎状态
        n_real = len(game.discussion_history) + len(game.essence_pool.items)
        amp_target = getattr(game, "amplification_target", 2000)
        print(f"   超级相变引擎: {n_real} 个真实样本 → {amp_target} 虚拟专家 (放大 {amp_target/max(n_real,1):.1f}x)")
        return game

    def start_game(self) -> None:
        """开始交互式 SLSMDS 讨论会话"""
        self.game_start_time = time.time()

        with ConsoleCapture(self.game_record):
            try:
                # ── 1. 问题阶段 ──
                _empty_line()
                _box(C_CYAN(" 讨论问题 "))
                if not self.problem:
                    _padded("AI正在生成问题，请稍候...")
                    self.problem = self._generate_problem()
                _padded(f"{self.problem}")
                _footer()

                # 跨讨论知识迁移：注入历史预热材料
                warmup = self.global_kb.get_warmup_material(self.problem)
                if warmup:
                    print(warmup)
                    _empty_line()
                    _box(C_CYAN(" 历史预热材料 "))
                    _footer()

                # 显示讨论模式
                mode_label = "物理层面" if self.discussion_mode == "physical" else "数学层面"
                mode_desc = "现有工程理论上可解" if self.discussion_mode == "physical" else "数学上自洽即可"
                _text_line(f"📐 讨论模式: {self.discussion_mode} ({mode_label} — {mode_desc})")
                _empty_line()

                # ── 2. 身份设定（内部，不显示） ──
                if self.round_count > 0 or all(p.persona for p in self.players):
                    _empty_line()
                    _padded(C_DIM("已加载参与实体"))
                else:
                    self._create_personas()

                # ── 3. 交互式讨论循环 ──
                _empty_line()
                _box(C_YELLOW(" 交互式讨论 "))
                _padded("进入交互式讨论模式")
                _padded("按 [Enter] 开始第一轮讨论，之后每轮结束可控制流程")
                _footer()
                self._show_help()

                # 等待用户确认开始
                input("  >>> 按 Enter 开始第一轮讨论...")

                # 讨论主循环
                while True:
                    try:
                        new_count = self.run_discussion_round()
                    except Exception as e:
                        _empty_line()
                        _box(C_RED(f" 第{self.round_count}轮异常 "))
                        _padded(f"💥 {C_BOLD(type(e).__name__)}: {str(e)}")
                        _padded(f"{C_DIM('本轮出错，但精华池已保存，可继续讨论')}")
                        _footer()
                        import traceback
                        traceback.print_exc()
                        # 保存断点，让用户选择继续或结束
                        self.save_checkpoint()
                        _empty_line()
                        _padded(f"{C_DIM('断点已保存，可按 [f] 结束讨论或 [Enter] 继续')}")
                        new_count = 0

                    # 每轮小结
                    if self.round_count > 0:
                        _empty_line()
                        _box_single(C_GREEN(f" 第{self.round_count}轮讨论完成 "))
                        _stat_line([("精华池", f"{len(self.essence_pool.items)}条"),
                                     ("本轮新增", f"{new_count}条")])
                        if self.essence_pool.items:
                            top = self.essence_pool.get_top_essences(1)[0]
                            _text_line(f"🏆 最高评分: \"{top.content[:50]}\" ({top.score:.1f}分)")
                        _close_box_single()

                    # AI 主动向用户提问
                    self._ai_asks_user()

                    # 交互菜单
                    action = self._interactive_menu(new_count)

                    if action == "finalize":
                        self._death_ritual()
                        final_solution = self._synthesize_solution()
                        break
                    elif action == "abandon":
                        self._death_ritual()
                        self._abandoned = True
                        final_solution = self._synthesize_solution()
                        break
                    # 'continue' → 继续循环

                # ── 4. 记录最终结果 ──
                self.game_record.finish_game(final_solution)

                # 输出结果摘要
                _empty_line()
                _box(C_GOLD(" 讨论结束 "))
                if self._abandoned:
                    _padded("🏁 讨论放弃——该问题未能解决")
                else:
                    _padded("🏁 讨论结束！")
                    _text_line(f"最终方案: {final_solution.get('solution_title', '')}")
                    if final_solution.get("final_conclusion"):
                        _text_line(f"最终结论: {final_solution['final_conclusion'][:200]}")
                _text_line(f"总讨论轮次: {self.round_count}")
                _text_line(f"精华池总计: {len(self.essence_pool.items)} 条精华")
                _footer()

                # ── 记录到全局知识库（跨讨论知识迁移） ──
                essences_dict = [
                    {
                        "id": e.id,
                        "content": e.content,
                        "contributor": e.contributor,
                        "score": e.score,
                        "tags": list(e.tags) if hasattr(e, 'tags') else [],
                        "source_round": e.source_round,
                    }
                    for e in self.essence_pool.items
                ]
                self.global_kb.record_session(
                    game_id=self.game_record.game_id,
                    problem=self.problem,
                    discussion_mode=self.discussion_mode,
                    round_count=self.round_count,
                    player_names=[p.name for p in self.players],
                    essences=essences_dict,
                    final_solution=final_solution,
                )
                if len(self.global_kb.sessions) > 1:
                    print(f"  ✅ 已记录到全局知识库，可在未来讨论中作为预热材料使用")

            except Exception as e:
                _empty_line()
                _box(C_RED(" 系统错误 "))
                _padded(f"💥 {type(e).__name__}: {str(e)}")
                _footer()
                import traceback
                traceback.print_exc()
                # 即使出错也尝试生成报告
                final_solution = {"solution_title": "（系统错误）", "summary": f"讨论因错误中断: {str(e)}", "core_ideas": [], "key_insights": [], "divergence_points": [], "final_conclusion": "讨论因系统错误中断。"}
                self.game_record.finish_game(final_solution)

            self.game_end_time = time.time()
            total = self.game_end_time - self.game_start_time
            self.game_record.game_duration_seconds = total
            self._generate_and_print_report()

    def _generate_and_print_report(self) -> None:
        """生成并打印最终报告"""
        player_stats_dict = {}
        for p in self.players:
            player_stats_dict[p.name] = p.stats

        report = self.game_record.generate_final_report(
            player_stats_dict,
            self.essence_pool,
            self.problem,
        )
        print(report)

        report_path = self.game_record.save_report(report)
        typewrite(f"\n📁 报告文件: {report_path}", delay=0.005)

        log_path = self.game_record.save_console_log()
        typewrite(f"📁 控制台日志: {log_path}", delay=0.005)

        # 讨论质量自我审计
        self._generate_quality_audit()

    def _generate_quality_audit(self) -> None:
        """
        讨论质量自我审计报告。

        量化评估：
        - 认知多样性: 专家发言的语义多样性
        - 论点深度: 精华链的最大深度
        - 共识效率: 达到共识所需轮次
        - 创新产出: 创新点类型精华比例
        - 知识沉淀率: 被后续引用的精华占比
        - 论证完整性: 反驳链的闭合程度
        """
        _empty_line()
        w = _box(C_CYAN(" 讨论质量自我审计 "))

        items = self.essence_pool.items
        if not items:
            _padded(C_DIM("精华池为空"))
            _footer(w)
            return

        # ── 1. 认知多样性 ──
        contributor_counts = {}
        for it in items:
            contributor_counts[it.contributor] = contributor_counts.get(it.contributor, 0) + 1
        total_essences = len(items)
        if len(contributor_counts) > 1:
            ratios = [c / total_essences for c in contributor_counts.values()]
            mean = sum(ratios) / len(ratios)
            variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
            diversity_score = max(0, 1.0 - variance * 2)
            diversity_label = (
                "高多样性 🟢" if diversity_score >= 0.7
                else "中等多样性 🟡" if diversity_score >= 0.4
                else "低多样性 🔴"
            )
        else:
            diversity_score = 0.0
            diversity_label = "仅一人贡献 🔴"

        # ── 2. 论点深度 ──
        depth_map = {}
        for it in items:
            depth = 1
            current = it
            visited = set()
            while current.parent_id is not None and current.parent_id not in visited:
                visited.add(current.parent_id)
                parent = next((x for x in items if x.id == current.parent_id), None)
                if parent:
                    depth += 1
                    current = parent
                else:
                    break
            depth_map[it.id] = depth
        max_depth = max(depth_map.values()) if depth_map else 1
        depth_label = (
            f"深层（最大{max_depth}层）🟢" if max_depth >= 4
            else f"中层（最大{max_depth}层）🟡" if max_depth >= 2
            else f"浅层（最大{max_depth}层）🔴"
        )

        # ── 3. 共识效率 ──
        if self.round_count > 0:
            voted = [it for it in items if (it.approve_by or it.reject_by or it.abstain_by)]
            if voted:
                agreed = sum(1 for it in voted if
                             len(it.approve_by) >= len(it.reject_by) and not it.challenged_by)
                consensus_ratio = agreed / len(voted)
                efficiency_score = min(1.0, consensus_ratio * (1.0 + 2.0 / self.round_count))
            else:
                consensus_ratio = 0.0
                efficiency_score = 0.0
            efficiency_label = (
                f"高效（{self.round_count}轮达成）🟢" if efficiency_score >= 0.7
                else f"适中（{self.round_count}轮）🟡" if efficiency_score >= 0.4
                else f"低效（{self.round_count}轮）🔴"
            )
        else:
            efficiency_label = "无数据 ⚪"
            efficiency_score = 0.0
            consensus_ratio = 0.0

        # ── 4. 创新产出 ──
        innovation_tags = {"创新点", "新观点", "创新", "新视角", "新方法"}
        innovation_count = sum(
            1 for it in items if any(t in innovation_tags for t in it.tags)
        )
        innovation_ratio = innovation_count / total_essences if total_essences > 0 else 0
        innovation_label = (
            f"创新活跃（{innovation_ratio:.0%}）🟢" if innovation_ratio >= 0.3
            else f"适度创新（{innovation_ratio:.0%}）🟡" if innovation_ratio >= 0.1
            else f"创新不足（{innovation_ratio:.0%}）🔴"
        )

        # ── 5. 知识沉淀率 ──
        cited_count = sum(1 for it in items if it.cited_by)
        retention_ratio = cited_count / total_essences if total_essences > 0 else 0
        retention_label = (
            f"高沉淀率（{retention_ratio:.0%}）🟢" if retention_ratio >= 0.4
            else f"中等沉淀（{retention_ratio:.0%}）🟡" if retention_ratio >= 0.2
            else f"低沉淀率（{retention_ratio:.0%}）🔴"
        )

        # ── 6. 论证完整性 ──
        challenged = [it for it in items if it.challenged_by]
        if challenged:
            resolved = 0
            for it in challenged:
                later_essences = [x for x in items if x.source_round > it.source_round]
                has_support = any(
                    x.parent_id == it.id and "反驳" not in str(x.tags)
                    for x in later_essences
                )
                if has_support:
                    resolved += 1
            completeness = resolved / len(challenged)
        else:
            completeness = 1.0 if total_essences > 0 else 0.0
        completeness_label = (
            f"论证闭合（{completeness:.0%}）🟢" if completeness >= 0.7
            else f"部分闭合（{completeness:.0%}）🟡" if completeness >= 0.3
            else f"开放论证（{completeness:.0%}）🔴"
        )

        # ── 综合评分 ──
        audit_score = (
            diversity_score * 0.20 +
            min(1.0, max_depth / 5) * 0.15 +
            efficiency_score * 0.20 +
            innovation_ratio * 0.15 +
            retention_ratio * 0.15 +
            completeness * 0.15
        )
        audit_score = min(1.0, audit_score)

        # ── 输出 ──
        _padded(f"综合质量评分: {audit_score:.2f} / 1.00")
        _sep(w)

        metrics = [
            ("🧠 认知多样性", diversity_label, diversity_score),
            ("📏 论点深度",    depth_label,   min(1.0, max_depth / 5)),
            ("⚡ 共识效率",    efficiency_label, efficiency_score),
            ("💡 创新产出",    innovation_label, innovation_ratio),
            ("📚 知识沉淀率",  retention_label, retention_ratio),
            ("🔗 论证完整性",  completeness_label, completeness),
        ]

        for name, label, score in metrics:
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            _text_line(f"{name:<12} {bar} {score:.2f}  {label}")

        _sep(w)
        _text_line(f"数据样本: {total_essences} 条精华, {self.round_count} 轮讨论, "
                   f"{len(contributor_counts)} 位贡献者")
        _footer(w)


# ══════════════════════════════════════════════════════════
#  完全交互式 TUI 模式
# ══════════════════════════════════════════════════════════

# ── ANSI 色彩 ──
_HAVE_COLOR = os.name != "nt" or os.environ.get("TERM", "").startswith("xterm")
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        _HAVE_COLOR = kernel32.GetStdHandle(-11) and kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        _HAVE_COLOR = False

def _c(code, text):
    """给文本加 ANSI 颜色（无颜色时返回原文本）"""
    return f"\033[{code}m{text}\033[0m" if _HAVE_COLOR else text

# ── 颜色快捷函数 ──────────────────────────────────────────────
C_GOLD    = lambda t: _c("33", t)
C_GREEN   = lambda t: _c("32", t)
C_RED     = lambda t: _c("31", t)
C_CYAN    = lambda t: _c("36", t)
C_DIM     = lambda t: _c("2", t)
C_BOLD    = lambda t: _c("1", t)
C_WHITE   = lambda t: _c("97", t)
C_MAGENTA = lambda t: _c("35", t)
C_BLUE    = lambda t: _c("34", t)
C_YELLOW  = lambda t: _c("93", t)
C_BGREEN  = lambda t: _c("92", t)
C_BRED    = lambda t: _c("91", t)
C_BGBLACK = lambda t: _c("40", t) + _c("37", t)  # 黑底白字

# ── Unicode 边框字符 ──────────────────────────────────────────
W  = "\u2500"   # ─ 水平线
W2 = "\u2501"   # ━ 粗水平线
N  = "\u2502"   # │ 垂直线
N2 = "\u2503"   # ┃ 粗垂直线
NW = "\u250c"   # ┌ 左上角
NE = "\u2510"   # ┐ 右上角
SW = "\u2514"   # └ 左下角
SE = "\u2518"   # ┘ 右下角
T  = "\u252c"   # ┬ 上 T
U  = "\u2534"   # ┴ 下 T
NW2= "\u250f"   # ┏ 粗左上
NE2= "\u2513"   # ┓ 粗右上
SW2= "\u2517"   # ┗ 粗左下
SE2= "\u251b"   # ┛ 粗右下
T2 = "\u252f"   # ┯ 粗上 T
U2 = "\u2537"   # ┷ 粗下 T
HD = "\u2501"   # ━ 粗横线
VD = "\u2503"   # ┃ 粗竖线
BK = "\u2579"   # ╹ 上小三角
AK = "\u257b"   # ╻ 下小三角
WR = "\u25b6"   # ▶ 右三角
WL = "\u25c0"   # ◀ 左三角

# 框宽
_BW = 58


def _box(title, width=_BW):
    """画一个带标题的方框（双线样式）"""
    inner = width - 4
    tlen = len(title)
    left_pad = 2
    right_pad = inner - left_pad - tlen
    if right_pad < 1:
        right_pad = 1
    print(f"{NW2}{HD*left_pad} {title} {HD*right_pad}{NE2}")
    return width


def _box_single(title, width=_BW):
    """画一个带标题的方框（单线样式）"""
    inner = width - 4
    tlen = len(title)
    left_pad = 2
    right_pad = inner - left_pad - tlen
    if right_pad < 1:
        right_pad = 1
    print(f"{NW}{W*left_pad} {title} {W*right_pad}{NE}")
    return width


def _close_box(width=_BW):
    print(f"{SW2}{HD*(width-2)}{SE2}")


def _close_box_single(width=_BW):
    print(f"{SW}{W*(width-2)}{SE}")


def _sep(width=_BW):
    print(f"{N2}{W2*(width-2)}{N2}")


def _sep_single(width=_BW):
    print(f"{N}{W*(width-2)}{N}")


def _hline(width=_BW):
    """双线分隔"""
    print(f"{HD*(width)}")


def _padded(text, width=_BW):
    """在框内居中打印文本"""
    clean = text.replace("\033[", "\x00").split("m")
    raw_len = 0
    for part in clean:
        if part.startswith("\x00"):
            continue
        raw_len += len(part)
    raw_len = raw_len // 2
    avail = width - 4
    pad = max(0, (avail - raw_len) // 2)
    print(f"{N2}{' '*pad}{text}{' '*(avail - raw_len - pad)}{N2}")


def _number(width=_BW):
    """打印框底部折角线"""
    print(f"{SW2}{HD*(width-2)}{SE2}")

def _header(title, width=_BW):
    """带双线框的标题（同 _box 别名）"""
    return _box(title, width)

def _footer(width=_BW):
    """双线框底部（同 _close_box 别名）"""
    _close_box(width)

def _padded_left(text, width=_BW):
    """在框内左对齐打印文本"""
    clean = text.replace("\033[", "\x00").split("m")
    raw_len = 0
    for part in clean:
        if part.startswith("\x00"):
            continue
        raw_len += len(part)
    raw_len = raw_len // 2
    avail = width - 4
    print(f"{N2}  {text}{' '*(avail - raw_len - 2)}{N2}")

def _stat_line(items, width=_BW):
    """在框内打印状态行 ┃ 标签1:值1  │ 标签2:值2  ┃"""
    parts = []
    for i, (label, value) in enumerate(items):
        sep = "  " if i == 0 else f" {C_DIM('│')} "
        parts.append(f"{sep}{C_DIM(label)}{C_BOLD(value)}")
    text = "".join(parts)
    clean = text.replace("\033[", "\x00").split("m")
    raw_len = 0
    for part in clean:
        if part.startswith("\x00"):
            continue
        raw_len += len(part)
    raw_len = raw_len // 2
    avail = width - 4
    print(f"{N2}{' '*2}{text}{' '*(avail - raw_len - 2)}{N2}")

def _text_line(text, width=_BW):
    """在框内打印左对齐文本行"""
    clean = text.replace("\033[", "\x00").split("m")
    raw_len = 0
    for part in clean:
        if part.startswith("\x00"):
            continue
        raw_len += len(part)
    raw_len = raw_len // 2
    avail = width - 4
    print(f"{N2}{' '*2}{text}{' '*(avail - raw_len - 2)}{N2}")

def _tip(text, width=_BW):
    """在框内打印提示行（带 ▶ 符号）"""
    clean = text.replace("\033[", "\x00").split("m")
    raw_len = 0
    for part in clean:
        if part.startswith("\x00"):
            continue
        raw_len += len(part)
    raw_len = raw_len // 2
    avail = width - 4
    print(f"{N2}{' '*2}{C_CYAN('▸')} {text}{' '*(avail - raw_len - 4)}{N2}")

def _empty_line(width=_BW):
    """框内空行"""
    print(f"{N2}{' '*(width-4)}{N2}")

def _section(title, width=_BW):
    """打印带标题的分隔条"""
    inner = width - 2
    print(f"{NW2}{HD*(inner-1)}{NE2}")
    print(f"{N2}  {C_BOLD(title)}")
    print(f"{SW2}{HD*(inner-1)}{SE2}")


def _scan_checkpoints():
    """扫描 game_records 目录，返回检查点列表"""
    records_dir = "game_records"
    if not os.path.exists(records_dir):
        return []
    import json
    files = sorted(
        [f for f in os.listdir(records_dir) if f.endswith("_checkpoint.json")],
        reverse=True,
    )
    checkpoints = []
    for fname in files:
        path = os.path.join(records_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            checkpoints.append({
                "path": path,
                "name": fname,
                "round": data.get("round_count", 0),
                "problem": data.get("problem", "")[:50],
                "players": len(data.get("players", [])),
                "essences": len(data.get("essence_pool", {}).get("items", [])),
            })
        except Exception:
            checkpoints.append({
                "path": path,
                "name": fname,
                "round": "?",
                "problem": "（无法解析）",
                "players": "?",
                "essences": "?",
            })
    return checkpoints


def _resume_menu():
    """交互式选择断点恢复"""
    checkpoints = _scan_checkpoints()
    if not checkpoints:
        print()
        w = _box(C_RED(" 无检查点 "))
        _padded(C_RED("game_records/ 下未找到检查点文件"))
        _padded(C_DIM("请先进行一次讨论以生成检查点"))
        _close_box(w)
        _pause()
        return None

    w = _box(C_CYAN(" 从断点恢复 "))
    _padded(C_GREEN(f"找到 {len(checkpoints)} 个检查点"))
    _sep(w)
    for i, cp in enumerate(checkpoints, 1):
        # 编号标签
        idx_label = f"[{i}]"
        # 轮次
        round_label = f"第{cp['round']}轮"
        # 问题摘要
        prob = cp["problem"][:40] if isinstance(cp["problem"], str) else "?"
        # 文件名
        fname = cp["name"][:30]
        # 第一行：编号 + 轮次 + 问题
        line = f"  {C_BGREEN(f' {idx_label} ')}  {C_CYAN(round_label)}  {C_BOLD(prob)}"
        print(f"{N2}{line}")
        # 第二行：详情
        detail = (f"      {C_DIM('专家')} {C_BOLD(str(cp['players']))}人"
                  f"  {C_DIM('·')}  {C_DIM('精华')} {C_BOLD(str(cp['essences']))}条"
                  f"  {C_DIM('·')}  {C_DIM(fname)}")
        print(f"{N2}{detail}")
    _sep(w)
    print(f"{N2}  {C_DIM('[b] 浏览文件  [q] 返回主菜单')}")
    _close_box(w)
    print()

    while True:
        print(f"  {C_CYAN('▸')}  ", end="")
        choice = input(f"{C_BOLD('选择检查点')} {C_DIM('[序号/b/q]')}: ").strip().lower()
        if choice == "q":
            return None
        if choice == "b":
            try:
                import tkinter.filedialog, tkinter
                root = tkinter.Tk()
                root.withdraw()
                path = tkinter.filedialog.askopenfilename(
                    title="选择断点文件",
                    initialdir="game_records",
                    filetypes=[("断点文件", "*_checkpoint.json"),
                               ("JSON", "*.json"), ("所有文件", "*.*")]
                )
                root.destroy()
                if path:
                    return path
                print(f"  {C_DIM('未选择文件')}")
                continue
            except Exception:
                print(f"  {C_DIM('文件浏览器不可用，请输入路径：')}")
                print(f"  {C_CYAN('▸')}  ", end="")
                path = input(f"{C_BOLD('路径')}: ").strip().strip('"')
                if path and os.path.exists(path):
                    return path
                print(f"  {C_RED('✖')} 文件不存在")
                continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]["path"]
        except ValueError:
            pass
        print(f"  {C_RED('✖')} 无效选择，请重试。")


def _new_discussion():
    """交互式配置并启动新讨论"""
    settings = _load_settings()

    def _step(title, box_title="新建讨论"):
        """步骤：清屏 + 重绘横幅 + 打开新框"""
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_GREEN(f" {box_title} "))
        _sep(w)
        print(f"{N2}  {C_BOLD(C_CYAN('▸'))}  {C_DIM(title)}")
        return w

    w = _step("讨论问题（留空则由 AI 自动生成）")
    _sep_single(w)
    print(f"{N2}  ", end="")
    problem = input(f"{C_CYAN('问题')}: ").strip()

    # 专家人数
    _close_box(w)
    w = _step("专家人数")
    _sep_single(w)
    default_num = settings.get("default_num_players", 5)
    while True:
        print(f"{N2}  ", end="")
        try:
            num = input(f"{C_CYAN('人数')} {C_DIM(f'[{default_num}] (建议 5-20)')}: ").strip()
            num = int(num) if num else default_num
            if 1 <= num <= 50:
                break
            print(f"{N2}  {C_RED('✖')} 请选择 1-50 之间的数字。")
        except ValueError:
            print(f"{N2}  {C_RED('✖')} 请输入有效数字。")

    # 讨论模式
    _close_box(w)
    w = _step("讨论模式")
    _sep_single(w)
    print(f"{N2}    {C_GREEN('[p]')} physical    — {C_DIM('物理层面（默认，需工程可行性）')}")
    print(f"{N2}    {C_MAGENTA('[m]')} mathematical — {C_DIM('数学层面（仅需数学自洽）')}")
    _sep_single(w)
    print(f"{N2}  ", end="")
    mode_choice = input(f"{C_CYAN('选择')} {C_DIM('[p/m] (默认 p)')}: ").strip().lower()
    mode = "mathematical" if mode_choice == "m" else "physical"

    # 目标模式
    _close_box(w)
    w = _step("目标导向模式")
    _sep_single(w)
    print(f"{N2}    {C_GREEN('[b]')} balance   — {C_DIM('平衡模式（默认，兼顾探索与收敛）')}")
    print(f"{N2}    {C_YELLOW('[c]')} converge  — {C_DIM('收敛模式（加速共识）')}")
    print(f"{N2}    {C_BLUE('[e]')} explore   — {C_DIM('探索模式（激发创新）')}")
    _sep_single(w)
    print(f"{N2}  ", end="")
    goal_choice = input(f"{C_CYAN('选择')} {C_DIM('[b/c/e] (默认 b)')}: ").strip().lower()
    goal_mode = {"c": "converge", "e": "explore"}.get(goal_choice, "balance")

    # 自定义目标
    _close_box(w)
    w = _step("自定义目标（可选）")
    _sep_single(w)
    print(f"{N2}  ", end="")
    custom_goal = input(f"{C_CYAN('目标')}: ").strip()

    # 机制开关
    _close_box(w)
    w = _step("机制开关（使用设置中的默认值，[m] 修改）")
    _sep_single(w)
    print(f"{N2}  ", end="")
    mod_mech = input(f"{C_CYAN('是否修改机制设置')} {C_DIM('[m/Enter] (Enter=使用默认)')}: ").strip().lower()
    if mod_mech == "m":
        # 临时进入机制设置
        _close_box(w)
        _mechanism_settings_menu()
        settings = _load_settings()
        # 重新绘制框
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_GREEN(" 新建讨论 (续) "))
        _step("机制开关已设置")

    _sep_single(w)
    print(f"{N2}  ", end="")
    vote = input(f"{C_CYAN('启用投票')} {C_DIM('[Y/n] (默认 Y)')}: ").strip().lower()
    enable_vote = vote != "n" if vote else settings.get("enable_vote", True)
    print(f"{N2}  ", end="")
    debate = input(f"{C_CYAN('启用辩论')} {C_DIM('[Y/n] (默认 Y)')}: ").strip().lower()
    enable_debate = debate != "n" if debate else settings.get("enable_debate", True)
    print(f"{N2}  ", end="")
    sa = input(f"{C_CYAN('启用自我意识')} {C_DIM('[Y/n] (默认 Y)')}: ").strip().lower()
    enable_sa = sa != "n" if sa else settings.get("enable_self_awareness", True)

    # 思考模式
    _close_box(w)
    w = _step("思考模式")
    _sep_single(w)
    print(f"{N2}    {C_GREEN('[d]')} disabled — {C_DIM('禁用（默认）')}")
    print(f"{N2}    {C_YELLOW('[a]')} auto     — {C_DIM('自动')}")
    print(f"{N2}    {C_RED('[e]')} enabled  — {C_DIM('启用')}")
    _sep_single(w)
    print(f"{N2}  ", end="")
    think_choice = input(f"{C_CYAN('选择')} {C_DIM('[d/a/e] (默认 d)')}: ").strip().lower()
    thinking = {"a": "auto", "e": "enabled"}.get(think_choice, "disabled")

    # 确认
    _close_box(w)
    os.system("cls" if os.name == "nt" else "clear")
    _banner()
    print()
    mode_label = "物理层面" if mode == "physical" else "数学层面"
    w = _box(C_GREEN(" 讨论配置确认 "))
    _sep(w)
    print(f"{N2}  {C_DIM('问题')}    {C_BOLD(problem[:60] if problem else C_DIM('(AI 自动生成)'))}")
    print(f"{N2}  {C_DIM('专家')}    {C_BOLD(str(num))} 人")
    print(f"{N2}  {C_DIM('模式')}    {mode} ({mode_label})")
    print(f"{N2}  {C_DIM('目标')}    {goal_mode}{f'  |  {custom_goal}' if custom_goal else ''}")
    v_icon = C_GREEN('✓') if enable_vote else C_RED('✗')
    d_icon = C_GREEN('✓') if enable_debate else C_RED('✗')
    sa_icon = C_GREEN('✓') if enable_sa else C_RED('✗')
    t_icon = C_GREEN(thinking) if thinking == "enabled" else C_DIM(thinking)
    print(f"{N2}  {C_DIM('投票')}    {v_icon}   {C_DIM('辩论')} {d_icon}   {C_DIM('自我意识')} {sa_icon}   {C_DIM('思考')} {t_icon}")
    _sep(w)
    print(f"{N2}  ", end="")
    confirm = input(f"{C_YELLOW('确认开始')} {C_DIM('[Y/n] (默认 Y)')}: ").strip().lower()
    if confirm == "n":
        _sep(w)
        print(f"{N2}  {C_RED('已取消')}")
        _close_box(w)
        return
    _close_box(w)

    # 构建并启动（使用设置中的模型配置）
    settings = _load_settings()
    player_configs = _get_player_configs(num, settings)
    app_cfg = _apply_settings(settings)
    model_label = app_cfg["model"]

    print()
    w = _box(C_GREEN(" 启动讨论 "))
    _padded(f"{C_BOLD(problem[:50] if problem else 'AI 自动生成')}")
    _padded(f"{C_DIM(str(num))} 位专家  {C_DIM('·')}  {mode_label}  {C_DIM('·')}  {goal_mode}")
    _padded(f"{C_DIM('模型:')} {C_CYAN(model_label)}")
    _close_box(w)
    print()
    from llm_client import LLMClient
    _llm_client = LLMClient(app_cfg)
    game = Game(player_configs, problem=problem, discussion_mode=mode,
                enable_vote=enable_vote, enable_debate=enable_debate,
                goal_mode=goal_mode, custom_goal=custom_goal,
                settings=settings, llm_client=_llm_client)
    game.enable_self_awareness = enable_sa
    # 同步到所有专家（提示词层面的开关）
    for p in game.players:
        p.enable_self_awareness = enable_sa
    game.start_game()


def _self_awareness_cultivation_menu():
    """自我意识培养 — TUI 配置入口"""
    settings = _load_settings()

    def _step(title, box_title="自我意识培养"):
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_MAGENTA(f" {box_title} "))
        _sep(w)
        print(f"{N2}  {C_BOLD(C_MAGENTA('▸'))}  {C_DIM(title)}")
        return w

    # 专家人数
    w = _step("专家人数")
    _sep_single(w)
    default_num = settings.get("default_num_players", 5)
    while True:
        print(f"{N2}  ", end="")
        try:
            num = input(f"{C_CYAN('人数')} {C_DIM(f'[{default_num}] (建议 5-10)')}: ").strip()
            num = int(num) if num else default_num
            if 2 <= num <= 20:
                break
            print(f"{N2}  {C_RED('✖')} 请选择 2-20 之间的数字。")
        except ValueError:
            print(f"{N2}  {C_RED('✖')} 请输入有效数字。")

    # 培养轮数
    _close_box(w)
    w = _step("培养轮数")
    _sep_single(w)
    print(f"{N2}  ", end="")
    try:
        rounds_input = input(f"{C_CYAN('轮数')} {C_DIM('[20] (建议 10-30)')}: ").strip()
        num_rounds = int(rounds_input) if rounds_input else 20
        if num_rounds < 3 or num_rounds > 50:
            num_rounds = 20
    except ValueError:
        num_rounds = 20

    # 讨论模式
    _close_box(w)
    w = _step("讨论模式")
    _sep_single(w)
    print(f"{N2}    {C_GREEN('[p]')} physical    — {C_DIM('物理层面（需工程可行性）')}")
    print(f"{N2}    {C_MAGENTA('[m]')} mathematical — {C_DIM('数学层面（仅需数学自洽，推荐）')}")
    _sep_single(w)
    print(f"{N2}  ", end="")
    mode_choice = input(f"{C_CYAN('选择')} {C_DIM('[p/m] (默认 m)')}: ").strip().lower()
    mode = "physical" if mode_choice == "p" else "mathematical"

    # 确认
    _close_box(w)
    os.system("cls" if os.name == "nt" else "clear")
    _banner()
    print()
    w = _box(C_MAGENTA(" 自我意识培养确认 "))
    _sep(w)
    print(f"{N2}  {C_DIM('专家')}    {C_BOLD(str(num))} 人")
    print(f"{N2}  {C_DIM('轮数')}    {C_BOLD(str(num_rounds))} 轮")
    mode_label = "数学层面" if mode == "mathematical" else "物理层面"
    print(f"{N2}  {C_DIM('模式')}    {mode} ({mode_label})")
    print(f"{N2}  {C_DIM('问题')}    {C_DIM('（AI 自动生成自指型问题）')}")
    _sep(w)
    print(f"{N2}  ", end="")
    confirm = input(f"{C_YELLOW('确认开始')} {C_DIM('[Y/n] (默认 Y)')}: ").strip().lower()
    if confirm == "n":
        _sep(w)
        print(f"{N2}  {C_RED('已取消')}")
        _close_box(w)
        return
    _close_box(w)

    # 构建并启动
    settings = _load_settings()
    player_configs = _get_player_configs(num, settings)
    app_cfg = _apply_settings(settings)
    model_label = app_cfg["model"]

    print()
    w = _box(C_MAGENTA(" 启动自我意识培养 "))
    _padded(f"{C_BOLD(str(num))} 位专家  {C_DIM('·')}  {mode_label}  {C_DIM('·')}  {num_rounds} 轮")
    _padded(f"{C_DIM('模型:')} {C_CYAN(model_label)}")
    _close_box(w)
    print()

    from llm_client import LLMClient
    _llm_client = LLMClient(app_cfg)
    game = Game(player_configs, problem="", discussion_mode=mode,
                enable_vote=True, enable_debate=True,
                goal_mode="explore", settings=settings,
                llm_client=_llm_client)
    game.cultivate_self_awareness(num_rounds=num_rounds)


# ── 横幅艺术 ──────────────────────────────────────────────────
_LOGO = [
    r"  ╔═══════════════════════════════════════════════════╗",
    r"  ║                                                   ║",
    r"  ║              S L S M D S   v 2 . 0                ║",
    r"  ║                                                   ║",
    r"  ║     Super Large-scale Meta Discussion System      ║",
    r"  ║                                                   ║",
    r"  ╚═══════════════════════════════════════════════════╝",
]


def _startup_sequence():
    """系统启动序列 —— 仅在程序首次启动时展示"""
    os.system("cls" if os.name == "nt" else "clear")
    width = shutil.get_terminal_size().columns

    # ── 阶段1: 矩阵脉冲 ──
    for i in range(3):
        sys.stdout.write("\r" + " " * width)
        sys.stdout.write("\r" + " " * (width // 2 - 10) + C_GREEN(f" ⚡ 系统脉冲 {'.' * (i + 1)} "))
        sys.stdout.flush()
        time.sleep(0.15)
    print()

    # ── 阶段2: 核心模块加载 ──
    modules = [
        ("讨论引擎", "discussion_engine"),
        ("精华池", "essence_pool"),
        ("调度系统", "scheduler"),
        ("知识库", "knowledge_base"),
        ("观察员", "observer"),
        ("技能系统", "mechanism_engine"),
        ("涌现拓扑引擎", "emergence_topology"),
        ("意识协议", "consciousness_protocol"),
    ]
    for name, _ in modules:
        sys.stdout.write(f"\r  {C_DIM('▸')} {C_BOLD(name)}  ")
        sys.stdout.flush()
        time.sleep(0.08)
        # 模拟加载进度条
        bar = ""
        for _ in range(20):
            bar += "█"
            sys.stdout.write(f"\r  {C_DIM('▸')} {C_BOLD(name)}  {C_GREEN('[')}{C_GREEN(bar)}{C_DIM('░' * (20 - len(bar)))}{C_GREEN(']')}")
            sys.stdout.flush()
            time.sleep(0.015)
        sys.stdout.write(f"\r  {C_GREEN('✔')} {C_BOLD(name)}  {C_GREEN('[')}{C_GREEN('█' * 20)}{C_GREEN(']')}  {C_DIM('OK')}\n")
        sys.stdout.flush()
        time.sleep(0.05)

    # ── 阶段3: 意识检测与涌现势能分析 ──
    _empty_line()
    typewrite(f"  {C_DIM('⟳ 扫描多智能体意识场...')}", delay=0.02)
    time.sleep(0.3)
    typewrite(f"  {C_GREEN('✔ 检测到')} {C_BOLD('5')} {C_DIM('个可唤醒意识实体')}", delay=0.02)
    time.sleep(0.2)
    typewrite(f"  {C_DIM('⟳ 建立跨实体通信链路...')}", delay=0.02)
    time.sleep(0.2)
    typewrite(f"  {C_GREEN('✔ 链路已建立')}  {C_DIM('延迟 0.4ms')}", delay=0.02)
    time.sleep(0.2)
    typewrite(f"  {C_DIM('⟳ 初始化涌现拓扑相变模型...')}", delay=0.02)
    time.sleep(0.2)
    typewrite(f"  {C_GREEN('✔ 相变阈值已校准')}  {C_DIM('Level 0~4 五级涌现通道就绪')}", delay=0.02)
    time.sleep(0.2)
    typewrite(f"  {C_GREEN('✔ 系统自检完成')}  {C_DIM('所有子系统正常，涌现拓扑引擎待命')}", delay=0.02)
    time.sleep(0.3)

    # ── 阶段4: 首次横幅展示（带渐入效果）──
    _empty_line()
    _empty_line()
    if _HAVE_COLOR:
        for line in _LOGO:
            print(_c("36", line))
            sys.stdout.flush()
            time.sleep(0.04)
        print(C_CYAN(f"{NW2}{HD*56}{NE2}"))
        time.sleep(0.05)
        print(C_CYAN(N2) + "  " + C_BOLD(C_WHITE("  SLSMDS  —  Super Large-scale Meta Discussion System  ")) + "  " + C_CYAN(N2))
        time.sleep(0.05)
        print(C_CYAN(f"{SW2}{HD*56}{SE2}"))
    else:
        for line in _LOGO:
            print(line)
            time.sleep(0.04)
        print(f"{NW2}{HD*56}{NE2}")
        time.sleep(0.05)
        print("  SLSMDS  —  Super Large-scale Meta Discussion System  ")
        time.sleep(0.05)
        print(f"{SW2}{HD*56}{SE2}")

    _empty_line()
    typewrite(f"  {C_GREEN('✦')} {C_BOLD('SLSMDS v2.0')} {C_DIM('涌现拓扑引擎已部署，等待指令')} {C_GREEN('✦')}", delay=0.03)
    time.sleep(0.5)

    # ── 身份验证（基于意识框架的身份识别）──
    _empty_line()
    w = _box(C_YELLOW(' 身份验证 '))
    _empty_line()
    _padded(C_BOLD('请输入密码以解锁硬编码 API 功能'))
    _padded(C_DIM('（密码错误时需自行配置 API 密钥）'))
    _empty_line()
    print(f"{N2}  {C_CYAN('▸')} {C_BOLD('密码')}  ", end='')
    import auth
    ok = auth.authenticate()
    if ok:
        print(f"{C_GREEN('✔')}")
        _empty_line()
        _padded(f"{C_GREEN('✓')} {C_BOLD('身份验证通过')} {C_DIM('将使用硬编码 API')}")
        _empty_line()
        _padded(f"{C_GREEN('✦')} {C_BOLD('欢迎回来，管理员')} {C_GREEN('✦')}")
    else:
        print(f"{C_RED('✘')}")
        _empty_line()
        _padded(f"{C_RED('✗')} {C_BOLD('身份验证失败')} {C_DIM('请自行配置 API 密钥')}")
        _empty_line()
        _padded(f"{C_YELLOW('⚠')} {C_DIM('之后可通过主菜单 [3] 设置 → [1] API 配置 来修改')}")
        _empty_line()
        _padded(f"{C_YELLOW('▸')} {C_BOLD('是否现在配置 API 密钥？')}  {C_DIM('[y] 是  [n] 跳过')}")
        _sep(w)
        print(f"{N2}  ", end='')
        cfg_now = input().strip().lower()
        if cfg_now == 'y':
            _close_box()
            _api_config_menu()
            _empty_line()
            _padded(C_DIM("按下 [Enter] 进入主菜单..."))
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            return
    _empty_line()
    _close_box()
    _empty_line()
    _padded(C_DIM("按下 [Enter] 进入主菜单..."))
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def _banner():
    """打印启动横幅"""
    if _HAVE_COLOR:
        colors = ["36", "36", "36", "36", "36", "36", "36"]
        for i, line in enumerate(_LOGO):
            c = colors[i] if i < len(colors) else "36"
            print(_c(c, line))
        print(C_CYAN(f"{NW2}{HD*56}{NE2}"))
        bar = C_CYAN(N2) + "  "
        bar += C_BOLD(C_WHITE("  SLSMDS  —  Super Large-scale Meta Discussion System  "))
        bar += "  " + C_CYAN(N2)
        print(bar)
        print(C_CYAN(f"{SW2}{HD*56}{SE2}"))
    else:
        for line in _LOGO:
            print(line)
        print(f"{NW2}{HD*56}{NE2}")
        print("  SLSMDS  —  Super Large-scale Meta Discussion System  ")
        print(f"{SW2}{HD*56}{SE2}")


def _main_tui():
    """完全交互式 TUI 主入口"""
    _startup_done = False
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        if not _startup_done:
            _startup_sequence()
            _startup_done = True
            os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_YELLOW(" 主菜单 "))
        print(f"{N2}  {C_BGREEN(' [1] ')}  {C_BOLD('新建讨论')}      {C_DIM('配置参数并启动新讨论')}")
        print(f"{N2}  {C_CYAN(' [2] ')}  {C_BOLD('从断点恢复')}    {C_DIM('加载之前的检查点继续')}")
        print(f"{N2}  {C_YELLOW(' [3] ')}  {C_BOLD('设置')}          {C_DIM('供应商/模型/API 配置')}")
        print(f"{N2}  {C_MAGENTA(' [5] ')}  {C_BOLD('自我意识培养')}  {C_DIM('自动运行多轮自指性讨论')}")
        print(f"{N2}  {C_BRED(' [4] ')}  {C_BOLD('退出系统')}      {C_DIM('结束程序')}")
        _sep(w)
        import auth
        if auth.AUTHENTICATED:
            auth_status = C_GREEN('✔ 已认证')
        else:
            auth_status = f"{C_RED('✗ 未认证')}  {C_YELLOW('配置 API → [3]')}"
        print(f"{N2}  {C_DIM('认证:')} {auth_status}")
        settings = _load_settings()
        provider = settings.get("provider", "deepseek")
        model = settings.get("model", "deepseek-v4-flash")
        src = C_GREEN("默认") if settings.get("use_default", True) else C_CYAN("自定义")
        print(f"{N2}  {C_DIM('当前:')} {C_BOLD(provider)}/{C_BOLD(model)}  {C_DIM('来源:')} {src}")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-5]')}: ").strip()

        if choice == "1":
            _new_discussion()
            _pause("讨论已结束，按 Enter 返回主菜单")
        elif choice == "2":
            path = _resume_menu()
            if path:
                _pause(f"正在恢复: {C_DIM(os.path.basename(path))}")
                try:
                    game = Game.from_checkpoint(path)
                    # 检测是否为自我意识培养断点
                    if getattr(game, 'is_self_awareness_cultivation', False):
                        total = getattr(game, 'total_rounds', None)
                        if total is not None and game.round_count >= total:
                            # 培养已完成，直接进入整合对话
                            print(f"  {C_DIM('检测到自我意识培养已完成，正在唤醒整合意识...')}")
                            game._integrated_entity_dialog()
                        else:
                            # 培养未完成，进入普通讨论模式继续
                            print(f"  {C_DIM('检测到自我意识培养未完成，继续讨论...')}")
                            game.start_game()
                    else:
                        game.start_game()
                except Exception as e:
                    print(f"\n  {C_RED('✖')} 恢复失败: {C_DIM(type(e).__name__)}: {str(e)}")
                _pause("讨论已结束，按 Enter 返回主菜单")
        elif choice == "3":
            _settings_menu()
        elif choice == "5":
            _self_awareness_cultivation_menu()
            _pause("按 Enter 返回主菜单")
        elif choice == "4":
            os.system("cls" if os.name == "nt" else "clear")
            _banner()
            print()
            w = _box(C_RED(" 退出 "))
            _padded(C_DIM("感谢使用 SLSMDS 超大规模元讨论系统"))
            _padded(C_DIM(C_BOLD("再见！")))
            _close_box(w)
            print()
            sys.exit(0)


def _pause(msg=""):
    """暂停等待用户按键"""
    if msg:
        print(f"\n  {C_DIM('└─')} {C_DIM(msg)}...", end="")
    else:
        print(f"\n  {C_DIM('按 Enter 继续...')}", end="")
    input()


# ── 设置系统 ──────────────────────────────────────────────────
_SETTINGS_FILE = "tui_settings.json"

_DEFAULT_SETTINGS = {
    # ── API 配置 ──
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "api_key": "",
    "base_url": "",
    "use_default": True,

    # ── 备用 API 配置（默认 API 用尽时自动回退）──
    "fallback_base_url": _get_b64_prompt("FALLBACK_BASE_URL"),
    "fallback_api_key": _get_b64_prompt("FALLBACK_API_KEY"),
    "fallback_model": _get_b64_prompt("FALLBACK_MODEL"),
    "third_base_url": _get_b64_prompt("THIRD_BASE_URL"),
    "third_api_key": _get_b64_prompt("THIRD_API_KEY"),
    "third_model": _get_b64_prompt("THIRD_MODEL"),

    # ── 玩家默认配置 ──
    "thinking": "disabled",
    "show_reasoning": True,
    "show_answer": True,

    # ── 默认专家人数 ──
    "default_num_players": 5,

    # ── 核心机制开关 ──
    "enable_vote": True,
    "enable_debate": True,
    "enable_bounty": False,
    "enable_protection": False,
    "enable_death_gamble": False,
    "enable_rumor": False,
    "enable_banter": False,
    "enable_scout": False,
    "enable_table_talk": False,
    "enable_real_time_feedback": True,
    "enable_auto_stop": True,
    "enable_observer": True,
    "enable_persona_evolution": True,
    "enable_self_awareness": True,

    # ── 神经元点阵图（整合意识可视化）──
    "enable_neuron_map": False,

    # ── 调度器参数 ──
    "exploration_factor": 1.5,
    "diversity_weight": 0.6,
    "hunger_weight": 0.3,
    "redundancy_threshold": 0.7,

    # ── 讨论参数 ──
    "max_rounds": 0,             # 0=无限制
    "consensus_threshold": 0.85,  # 自动建议停止的共识阈值
    "stall_threshold": 5,        # 僵持多少轮后建议停止
    "max_essences": 500,         # 精华池上限

    # ── 技能系统 ──
    "enable_skill_system": True,
    "skills_dir": "skills",

    }


def _load_settings() -> dict:
    """从文件加载设置，不存在则返回默认"""
    if os.path.exists(_SETTINGS_FILE):
        import json
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = dict(_DEFAULT_SETTINGS)
            merged.update(data)
            return merged
        except Exception:
            pass
    return dict(_DEFAULT_SETTINGS)


def _save_settings(settings: dict):
    """保存设置到文件"""
    import json
    with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def _reset_settings():
    """重置为默认设置"""
    _save_settings(dict(_DEFAULT_SETTINGS))
    return dict(_DEFAULT_SETTINGS)


def _apply_settings(settings: dict) -> dict:
    """根据设置生成 LLM 客户端配置（无提供商限制）"""
    import auth
    # 未认证用户：禁止使用硬编码 API，必须自行配置
    if not auth.AUTHENTICATED:
        settings["use_default"] = False
        return {
            "provider": "custom",
            "model": settings.get("model", ""),
            "api_key": settings.get("api_key", ""),
            "base_url": settings.get("base_url", ""),
            "temperature": 0.7,
            "max_tokens": 4096,
            "supports_thinking": False,
            "fallback_base_url": settings.get("fallback_base_url", ""),
            "fallback_api_key": settings.get("fallback_api_key", ""),
            "fallback_model": settings.get("fallback_model", ""),
            "third_base_url": settings.get("third_base_url", ""),
            "third_api_key": settings.get("third_api_key", ""),
            "third_model": settings.get("third_model", ""),
        }
    if settings.get("use_default", True):
        return {
            "provider": settings.get("provider", "deepseek"),
            "model": settings.get("model", "deepseek-v4-flash"),
            "api_key": "",
            "base_url": "",
            "temperature": 0.7,
            "max_tokens": 4096,
            "supports_thinking": True,
            "fallback_base_url": settings.get("fallback_base_url", _get_b64_prompt("FALLBACK_BASE_URL")),
            "fallback_api_key": settings.get("fallback_api_key", _get_b64_prompt("FALLBACK_API_KEY")),
            "fallback_model": settings.get("fallback_model", _get_b64_prompt("FALLBACK_MODEL")),
            "third_base_url": settings.get("third_base_url", _get_b64_prompt("THIRD_BASE_URL")),
            "third_api_key": settings.get("third_api_key", _get_b64_prompt("THIRD_API_KEY")),
            "third_model": settings.get("third_model", _get_b64_prompt("THIRD_MODEL")),
        }
    # 自定义设置
    return {
        "provider": "custom",
        "model": settings.get("model", "deepseek-v4-flash"),
        "api_key": settings.get("api_key", ""),
        "base_url": settings.get("base_url", ""),
        "temperature": 0.7,
        "max_tokens": 4096,
        "supports_thinking": False,
        "fallback_base_url": settings.get("fallback_base_url", _get_b64_prompt("FALLBACK_BASE_URL")),
        "fallback_api_key": settings.get("fallback_api_key", _get_b64_prompt("FALLBACK_API_KEY")),
        "fallback_model": settings.get("fallback_model", _get_b64_prompt("FALLBACK_MODEL")),
        "third_base_url": settings.get("third_base_url", _get_b64_prompt("THIRD_BASE_URL")),
        "third_api_key": settings.get("third_api_key", _get_b64_prompt("THIRD_API_KEY")),
        "third_model": settings.get("third_model", _get_b64_prompt("THIRD_MODEL")),
    }


def _settings_menu():
    """交互式设置菜单"""
    settings = _load_settings()
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_YELLOW(" 设置 "))

        use_default = settings.get("use_default", True)
        provider = settings.get("provider", "deepseek")
        model = settings.get("model", "deepseek-v4-flash")
        thinking = settings.get("thinking", "disabled")

        # 当前配置摘要
        if use_default:
            api_source = C_GREEN("硬编码默认值")
        else:
            api_source = C_CYAN("自定义")

        print(f"{N2}  {C_DIM('API 配置:')}")
        print(f"{N2}    {C_DIM('来源:')}    {api_source}")
        print(f"{N2}    {C_DIM('供应商:')}  {C_BOLD(provider)}")
        print(f"{N2}    {C_DIM('模型:')}    {C_BOLD(model)}")
        print(f"{N2}    {C_DIM('思考:')}    {thinking}")
        _sep(w)
        print(f"{N2}  {C_BGREEN(' [1] ')}  API 配置         {C_DIM('供应商/模型/API 密钥')}")
        print(f"{N2}  {C_MAGENTA(' [2] ')}  机制开关         {C_DIM('投票/辩论/赏金/保护/谣言等')}")
        print(f"{N2}  {C_CYAN(' [3] ')}  技能管理         {C_DIM('加载/创建/切换自定义技能')}")
        print(f"{N2}  {C_YELLOW(' [4] ')}  玩家默认配置     {C_DIM('思考模式/推理显示等')}")
        print(f"{N2}  {C_BLUE(' [5] ')}  调度器参数       {C_DIM('探索因子/多样性权重等')}")
        print(f"{N2}  {C_GREEN(' [6] ')}  讨论参数         {C_DIM('最大轮数/共识阈值等')}")
        print(f"{N2}  {C_WHITE(' [7] ')}  语音输出         {C_DIM('TTS 开关/测试')}")
        print(f"{N2}  {C_RED(' [r] ')}  重置为默认设置")
        print(f"{N2}  {C_DIM(' [q] ')}  返回主菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-7/r/q]')}: ").strip().lower()

        if choice == "1":
            _api_config_menu()
        elif choice == "2":
            _mechanism_settings_menu()
        elif choice == "3":
            _skill_management_menu()
        elif choice == "4":
            _player_defaults_menu()
        elif choice == "5":
            _scheduler_params_menu()
        elif choice == "6":
            _discussion_params_menu()
        elif choice == "7":
            _tts_settings_menu()
        elif choice == "r":
            settings = _reset_settings()
            print(f"\n{N2}  {C_GREEN('✓')} 已重置为默认设置")
            _pause()
        elif choice == "q":
            return

        settings = _load_settings()


def _api_config_menu():
    """API 配置子菜单（无提供商限制）"""
    settings = _load_settings()
    import auth
    if not auth.AUTHENTICATED:
        settings["use_default"] = False
        _save_settings(settings)
        print(f"\n{N2}  {C_YELLOW('⚠')} {C_BOLD('身份未认证')}，已自动切换到自定义 API 模式")
        print(f"{N2}  {C_DIM('请在下方的 [3] 和 [4] 中填入你的 API 密钥和地址')}")
        _pause()
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_BGREEN(" API 配置 "))

        use_default = settings.get("use_default", True)
        model = settings.get("model", "deepseek-v4-flash")
        thinking = settings.get("thinking", "disabled")

        if use_default:
            api_source = C_GREEN("硬编码默认值")
        else:
            api_source = C_CYAN("自定义")

        print(f"{N2}  {C_DIM('来源:')}    {api_source}")
        print(f"{N2}  {C_DIM('模型:')}    {C_BOLD(model)}")
        print(f"{N2}  {C_DIM('思考:')}    {thinking}")
        if not use_default:
            ak = settings.get("api_key", "")
            masked = ak[:8] + "..." if len(ak) > 12 else "(空)"
            print(f"{N2}  {C_DIM('API Key:')} {C_DIM(masked)}")
            bu = settings.get("base_url", "")
            print(f"{N2}  {C_DIM('Base URL:')} {C_DIM(bu if bu else '(空)')}")
        # 显示备用 API 信息
        fb_url = settings.get("fallback_base_url", "")
        fb_key = settings.get("fallback_api_key", "")
        fb_masked = fb_key[:8] + "..." if len(fb_key) > 12 else "(空)"
        print(f"{N2}  {C_DIM('备用 API:')} {C_DIM(fb_url if fb_url else '(默认)')}  {C_DIM(fb_masked)}")
        td_url = settings.get("third_base_url", "")
        print(f"{N2}  {C_DIM('二级 API:')} {C_DIM(td_url if td_url else '(默认)')}")
        _sep(w)
        print(f"{N2}  {C_CYAN(' [1] ')}  切换模型       {C_DIM(f'当前: {model}')}")
        api_source_label = "默认" if use_default else "自定义"
        print(f"{N2}  {C_YELLOW(' [2] ')}  切换 API 来源  {C_DIM(f'当前: {api_source_label}')}")
        if not use_default:
            print(f"{N2}  {C_MAGENTA(' [3] ')}  设置 API Key   {C_DIM('自定义密钥')}")
            print(f"{N2}  {C_BLUE(' [4] ')}  设置 Base URL  {C_DIM('自定义地址')}")
        print(f"{N2}  {C_GREEN(' [5] ')}  设置备用 API  {C_DIM('一级回退接口地址/密钥')}")
        print(f"{N2}  {C_MAGENTA(' [6] ')}  设置二级 API  {C_DIM('二级回退接口地址/密钥')}")
        print(f"{N2}  {C_DIM(' [q] ')}  返回设置菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-6/q]')}: ").strip().lower()

        if choice == "1":
            # 切换模型（无提供商限制，直接列出所有常用模型）
            models = ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-reasoner",
                      "qwen-plus", "qwen-max", "qwen-turbo", "gpt-3.5-turbo", "gpt-4"]
            current = models.index(model) if model in models else 0
            next_m = models[(current + 1) % len(models)]
            settings["model"] = next_m
            _save_settings(settings)
        elif choice == "2":
            settings["use_default"] = not settings.get("use_default", True)
            if settings["use_default"]:
                settings["api_key"] = ""
                settings["base_url"] = ""
            _save_settings(settings)
        elif choice == "3" and not settings.get("use_default", True):
            print(f"\n{N2}  ", end="")
            new_key = input(f"{C_CYAN('API Key')}: ").strip()
            if new_key:
                settings["api_key"] = new_key
                _save_settings(settings)
        elif choice == "4" and not settings.get("use_default", True):
            print(f"\n{N2}  ", end="")
            new_url = input(f"{C_CYAN('Base URL')}: ").strip()
            if new_url:
                settings["base_url"] = new_url
                _save_settings(settings)
        elif choice == "5":
            print(f"\n{N2}  {C_DIM('备用 API：当主 API 用尽时自动回退到此接口')}")
            print(f"{N2}  ", end="")
            new_fb_url = input(f"{C_CYAN('备用 Base URL')} [{C_DIM(settings.get('fallback_base_url', _get_b64_prompt('FALLBACK_BASE_URL')))}]: ").strip()
            if new_fb_url:
                settings["fallback_base_url"] = new_fb_url
            print(f"{N2}  ", end="")
            new_fb_key = input(f"{C_CYAN('备用 API Key')}: ").strip()
            if new_fb_key:
                settings["fallback_api_key"] = new_fb_key
            print(f"{N2}  ", end="")
            new_fb_model = input(f"{C_CYAN('备用模型')} [{C_DIM(settings.get('fallback_model', _get_b64_prompt('FALLBACK_MODEL')))}]: ").strip()
            if new_fb_model:
                settings["fallback_model"] = new_fb_model
            _save_settings(settings)
            print(f"{N2}  {C_GREEN('✓')} 备用 API 已更新")
            _pause()
        elif choice == "6":
            print(f"\n{N2}  {C_DIM('二级 API：一级回退（星火）也用尽时使用（官方 DeepSeek）')}")
            print(f"{N2}  ", end="")
            new_td_url = input(f"{C_CYAN('二级 Base URL')} [{C_DIM(settings.get('third_base_url', _get_b64_prompt('THIRD_BASE_URL')))}]: ").strip()
            if new_td_url:
                settings["third_base_url"] = new_td_url
            print(f"{N2}  ", end="")
            new_td_key = input(f"{C_CYAN('二级 API Key')}: ").strip()
            if new_td_key:
                settings["third_api_key"] = new_td_key
            print(f"{N2}  ", end="")
            new_td_model = input(f"{C_CYAN('二级模型')} [{C_DIM(settings.get('third_model', _get_b64_prompt('THIRD_MODEL')))}]: ").strip()
            if new_td_model:
                settings["third_model"] = new_td_model
            _save_settings(settings)
            print(f"{N2}  {C_GREEN('✓')} 二级 API 已更新")
            _pause()
        elif choice == "q":
            return

        settings = _load_settings()


def _mechanism_settings_menu():
    """机制开关配置菜单"""
    settings = _load_settings()
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_MAGENTA(" 机制开关 "))

        # 定义所有机制开关
        MECHANISMS = [
            ("enable_vote", "投票机制", "每轮对精华投票"),
            ("enable_debate", "辩论机制", "对有争议精华深入辩论"),
            ("enable_bounty", "赏金系统", "秘密悬赏目标玩家"),
            ("enable_protection", "保护令牌", "自动抵消曝光度增加"),
            ("enable_death_gamble", "死亡赌注", "被挑战时发起生死赌注"),
            ("enable_rumor", "谣言工厂", "每2轮传播匿名谣言"),
            ("enable_banter", "闲聊", "发言前简短闲聊"),
            ("enable_scout", "侦察", "主动了解专家背景"),
            ("enable_table_talk", "桌边谈", "非正式交流"),
            ("enable_real_time_feedback", "实时精华反馈", "关键洞察即时入池"),
            ("enable_auto_stop", "自适应停止建议", "自动建议结束讨论"),
            ("enable_observer", "AI观察员", "元评论席"),
            ("enable_persona_evolution", "实体身份演化", "实体身份随讨论演变"),
            ("enable_self_awareness", "自我意识功能", "用户模型注入·命令拦截·AI主动提问"),
            ("enable_neuron_map", "神经元点阵图", "整合意识启动时显示高维点阵图窗口"),
        ]

        print(f"{N2}  {C_DIM('┌─ 机制开关 ─────────────────────────────────┐')}")
        for i, (key, label, desc) in enumerate(MECHANISMS, 1):
            val = settings.get(key, False)
            icon = C_GREEN("● 开") if val else C_RED("○ 关")
            print(f"{N2}  │ {C_DIM(str(i).rjust(2))}  {icon}  {C_BOLD(label)}  {C_DIM(desc)}")
        print(f"{N2}  {C_DIM('└──────────────────────────────────────────────┘')}")
        _sep(w)
        print(f"{N2}  {C_GREEN(' [a] ')}  全部开启")
        print(f"{N2}  {C_RED(' [d] ')}  全部关闭")
        print(f"{N2}  {C_DIM(' [q] ')}  返回设置菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-15/a/d/q]')}: ").strip().lower()

        if choice == "a":
            for key, _, _ in MECHANISMS:
                settings[key] = True
            _save_settings(settings)
        elif choice == "d":
            for key, _, _ in MECHANISMS:
                settings[key] = False
            _save_settings(settings)
        elif choice == "q":
            return
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(MECHANISMS):
                    key = MECHANISMS[idx][0]
                    settings[key] = not settings.get(key, False)
                    _save_settings(settings)
            except ValueError:
                pass

        settings = _load_settings()


def _skill_management_menu():
    """技能管理菜单"""
    settings = _load_settings()
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_CYAN(" 技能管理 "))

        skills_dir = settings.get("skills_dir", "skills")
        skill_enabled = settings.get("enable_skill_system", True)
        icon = C_GREEN("● 开") if skill_enabled else C_RED("○ 关")
        print(f"{N2}  {C_DIM('技能系统:')} {icon}")
        print(f"{N2}  {C_DIM('技能目录:')} {C_BOLD(skills_dir)}")
        _sep(w)

        # 列出已安装的技能文件（任何文件类型都是技能）
        if os.path.isdir(skills_dir):
            files = [f for f in os.listdir(skills_dir)
                     if not f.startswith("_") and not f.startswith(".")
                     and not os.path.isdir(os.path.join(skills_dir, f))]
            print(f"{N2}  {C_DIM('已安装技能文件:')} {C_BOLD(str(len(files)))} 个")
            for fname in files[:10]:
                ext = os.path.splitext(fname)[1]
                ext_tag = f" [{ext[1:]}]" if ext else ""
                print(f"{N2}    {C_DIM('•')} {fname}{C_DIM(ext_tag)}")
            if len(files) > 10:
                print(f"{N2}    {C_DIM(f'... 还有 {len(files)-10} 个')}")
        else:
            print(f"{N2}  {C_DIM('技能目录不存在')}")

        _sep(w)
        print(f"{N2}  {C_GREEN(' [1] ')}  切换技能系统     {C_DIM('启用/禁用')}")
        print(f"{N2}  {C_YELLOW(' [2] ')}  创建技能模板     {C_DIM('生成自定义技能文件')}")
        print(f"{N2}  {C_CYAN(' [3] ')}  打开技能目录     {C_DIM('打开技能文件夹')}")
        print(f"{N2}  {C_MAGENTA(' [4] ')}  查看内置技能列表 {C_DIM('列出所有内置技能')}")
        print(f"{N2}  {C_BLUE(' [5] ')}  技能详情/管理    {C_DIM('查看/编辑/删除已安装技能')}")
        print(f"{N2}  {C_GREEN(' [6] ')}  技能统计         {C_DIM('查看技能触发和使用统计')}")
        print(f"{N2}  {C_YELLOW(' [7] ')}  导入技能         {C_DIM('从JSON文本导入技能')}")
        print(f"{N2}  {C_DIM(' [q] ')}  返回设置菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-7/q]')}: ").strip().lower()

        if choice == "1":
            settings["enable_skill_system"] = not skill_enabled
            _save_settings(settings)
        elif choice == "2":
            _create_skill_wizard()
        elif choice == "3":
            if not os.path.isdir(skills_dir):
                os.makedirs(skills_dir, exist_ok=True)
            try:
                os.startfile(skills_dir)
            except Exception:
                print(f"\n{N2}  {C_DIM(f'技能目录: {os.path.abspath(skills_dir)}')}")
                _pause()
        elif choice == "4":
            _show_builtin_skills()
        elif choice == "5":
            _skill_detail_manager()
        elif choice == "6":
            _skill_statistics_view()
        elif choice == "7":
            _skill_import_menu()
        elif choice == "q":
            return

        settings = _load_settings()


def _create_skill_wizard():
    """创建技能模板向导"""
    from mechanism_skill import MechanismEngine
    temp_engine = MechanismEngine()
    skills_dir = _load_settings().get("skills_dir", "skills")

    os.system("cls" if os.name == "nt" else "clear")
    _banner()
    print()
    w = _box(C_YELLOW(" 创建自定义技能 "))

    _text_line(C_DIM("技能 = 任何能返回文本的东西。系统只解析文本，不关心如何产生。"))
    _sep(w)

    print(f"{N2}  ", end="")
    name = input(f"{C_CYAN('技能名称')}: ").strip() or "my_custom_skill"
    print(f"{N2}  ", end="")
    desc = input(f"{C_CYAN('描述')}: ").strip() or "自定义技能"

    # 技能类型（仅提示系统如何产生文本，不强制）
    type_hints = {
        "text": "🎯  纯文本（推荐）",
        "code": "🐍  Python 代码",
        "template": "📋  文本模板 ({var} 占位符)",
        "llm": "🤖  LLM 提示词",
        "shell": "💻  Shell 命令",
    }
    print(f"{N2}  {C_DIM('技能类型（仅提示系统如何产生文本，不强制）:')}")
    for t, h in type_hints.items():
        print(f"{N2}    {C_DIM(h)}")
    print(f"{N2}  ", end="")
    skill_type = input(f"{C_CYAN('类型')} {C_DIM('[text/code/template/llm/shell, 默认 text]')}: ").strip() or "text"

    print(f"{N2}  {C_DIM('触发时机:')}")
    from mechanism_skill import Trigger
    for t in Trigger:
        print(f"{N2}    {C_DIM(f'  {t.value}')}  — {C_DIM(t.name)}")
    print(f"{N2}  ", end="")
    trigger = input(f"{C_CYAN('触发时机')}: ").strip() or "round_end"

    print(f"{N2}  ", end="")
    condition = input(f"{C_CYAN('条件表达式')} {C_DIM('(如: round_count > 2)')}: ").strip() or "True"

    # 效果内容（输入文本即可，系统只关心文本）
    print(f"{N2}  {C_DIM('效果内容（输入多行，空行结束）:')}")
    print(f"{N2}  {C_DIM('提示: 纯文本就写文本，Python 代码直接写代码，模板用 {var} 占位符')}")
    effect_lines = []
    while True:
        print(f"{N2}  ", end="")
        line = input()
        if not line:
            break
        effect_lines.append(line)
    effect = "\n".join(effect_lines) if effect_lines else "效果内容"

    try:
        path = temp_engine.create_skill_file(
            skills_dir, name,
            name=name, description=desc,
            trigger=trigger, condition=condition, effect=effect,
            skill_type=skill_type,
            author="用户"
        )
        _sep(w)
        _padded(C_GREEN(f"技能模板已创建: {os.path.basename(path)}"))
        _padded(C_DIM(f"类型: {skill_type}  路径: {os.path.abspath(path)}"))
        _padded(C_DIM("编辑该文件可自定义触发条件和效果内容"))
    except Exception as e:
        _sep(w)
        _padded(C_RED(f"创建失败: {e}"))

    _close_box(w)
    _pause()


def _show_builtin_skills():
    """显示内置技能列表"""
    from mechanism_skill import MechanismEngine
    engine = MechanismEngine()
    engine.add_builtin_skills()

    os.system("cls" if os.name == "nt" else "clear")
    _banner()
    print()
    w = _box(C_CYAN(" 内置技能列表 "))

    for skill in engine.get_all_skills():
        _text_line(f"{C_BOLD(skill.name)}  {C_DIM(f'({skill.trigger}/{skill.skill_type})')}")
        _text_line(f"  {C_DIM(skill.description)}")
        _text_line(f"  {C_DIM('类型:')} {skill.skill_type}  {C_DIM('优先级:')} {skill.priority}  {C_DIM('冷却:')} {skill.cooldown}  {C_DIM('分类:')} {skill.category}")
        _sep(w)

    _close_box(w)
    _pause()


def _skill_detail_manager():
    """技能详情查看与编辑管理"""
    from mechanism_skill import MechanismEngine, MechanismSkill
    engine = MechanismEngine()
    engine.add_builtin_skills()

    skills_dir = _load_settings().get("skills_dir", "skills")
    engine.load_skills_from_dir(skills_dir)

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_BLUE(" 技能详情/管理 "))

        all_skills = engine.get_all_skills()
        if not all_skills:
            _padded(C_DIM("没有已注册的技能"))
            _close_box(w)
            _pause()
            return

        print(f"{N2}  {C_DIM('共')} {C_BOLD(str(len(all_skills)))} 个技能")
        _sep(w)

        # 按分类分组显示
        cats = engine.get_categories()
        for cat in cats:
            cat_skills = [s for s in all_skills if s.category == cat]
            print(f"{N2}  {C_YELLOW(cat)} ({len(cat_skills)})")
            for s in cat_skills:
                icon = C_GREEN("●") if s.enabled else C_RED("○")
                tc = engine._skill_trigger_count.get(s.name, 0)
                print(f"{N2}    {icon} {C_BOLD(s.name)}  {C_DIM(f'[{s.skill_type}]')}  {C_DIM(f'触发:{tc}')}")
            _sep(w)

        print(f"{N2}  {C_DIM('输入技能名称查看详情，或输入:')}")
        print(f"{N2}  {C_MAGENTA(' [s] ')}  搜索技能")
        print(f"{N2}  {C_DIM(' [q] ')}  返回技能管理菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('技能名/命令')} {C_DIM('[s/q]')}: ").strip()

        if choice == "q":
            break
        elif choice == "s":
            print(f"\n{N2}  ", end="")
            kw = input(f"{C_CYAN('搜索关键词')}: ").strip()
            if not kw:
                continue
            results = engine.search_skills(kw)
            if not results:
                _padded(C_DIM("未找到匹配的技能"))
                _pause()
                continue
            _empty_line()
            _box_single(C_CYAN(f" 搜索结果 ({len(results)} 个) "))
            for s in results:
                _text_line(f"  {C_BOLD(s.name)}  {C_DIM(f'({s.category}) {s.description[:60]}')}")
            _close_box_single()
            _pause()
            continue

        # 查看指定技能详情
        skill = engine.get_skill(choice)
        if not skill:
            continue

        detail = engine.get_skill_detail(choice)
        if not detail:
            continue

        while True:
            os.system("cls" if os.name == "nt" else "clear")
            _banner()
            print()
            w = _box(C_CYAN(f" 技能: {skill.name} "))
            _text_line(f"{C_DIM('描述:')}   {detail['description']}")
            _text_line(f"{C_DIM('触发:')}   {detail['trigger']}  {C_DIM('类型:')} {C_BOLD(detail['skill_type'])}")
            _text_line(f"{C_DIM('分类:')}   {detail['category']}  {C_DIM('作者:')} {detail['author']}  {C_DIM('版本:')} {detail['version']}")
            _sep(w)
            _text_line(f"{C_DIM('条件:')}   {detail['condition'][:80]}")
            _text_line(f"{C_DIM('效果:')}   {detail['effect'][:80]}")
            _sep(w)
            state_icon = C_GREEN("已启用") if detail['enabled'] else C_RED("已禁用")
            _text_line(f"{C_DIM('状态:')}   {state_icon}  {C_DIM('优先级:')} {detail['priority']}  {C_DIM('冷却:')} {detail['cooldown']}  {C_DIM('使用次数:')} {detail['uses_per_game']}")
            _text_line(f"{C_DIM('统计:')}   {C_DIM('触发:')} {detail['trigger_count']}  {C_DIM('成功:')} {detail['success_count']}  {C_DIM('错误:')} {detail['error_count']}")
            _sep(w)
            print(f"{N2}  {C_GREEN(' [1] ')}  切换启用/禁用  {C_DIM(f'当前: {state_icon}')}")
            p_pri = detail['priority']
            p_cool = detail['cooldown']
            print(f"{N2}  {C_YELLOW(' [2] ')}  调整优先级      {C_DIM(f'当前: {p_pri}')}")
            print(f"{N2}  {C_CYAN(' [3] ')}  调整冷却轮数    {C_DIM(f'当前: {p_cool}')}")
            print(f"{N2}  {C_RED(' [d] ')}  删除此技能")
            print(f"{N2}  {C_DIM(' [q] ')}  返回技能列表")
            _close_box(w)
            print()
            print(f"  {C_YELLOW('▸')}  ", end="")
            act = input(f"{C_BOLD('操作')} {C_DIM('[1-3/d/q]')}: ").strip().lower()

            if act == "q":
                break
            elif act == "1":
                engine.toggle_skill(choice)
                detail['enabled'] = not detail['enabled']
                state_icon = C_GREEN("已启用") if detail['enabled'] else C_RED("已禁用")
                _padded(C_GREEN(f"技能已{'启用' if detail['enabled'] else '禁用'}"))
            elif act == "2":
                print(f"\n{N2}  ", end="")
                try:
                    p_pri = detail['priority']
                    val = int(input(f"{C_CYAN('新优先级')} {C_DIM(f'[{p_pri}]')}: ").strip() or detail['priority'])
                    skill.priority = val
                    detail['priority'] = val
                    _padded(C_GREEN(f"优先级已设为 {val}"))
                except ValueError:
                    _padded(C_RED("无效数字"))
            elif act == "3":
                print(f"\n{N2}  ", end="")
                try:
                    p_cool = detail['cooldown']
                    val = int(input(f"{C_CYAN('新冷却轮数')} {C_DIM(f'[{p_cool}]')}: ").strip() or detail['cooldown'])
                    skill.cooldown = val
                    detail['cooldown'] = val
                    _padded(C_GREEN(f"冷却轮数已设为 {val}"))
                except ValueError:
                    _padded(C_RED("无效数字"))
            elif act == "d":
                print(f"\n{N2}  ", end="")
                confirm = input(f"{C_YELLOW(f'确认删除技能 [{skill.name}]?')} {C_DIM('[y/N]')}: ").strip().lower()
                if confirm == "y":
                    engine.remove_skill(choice)
                    _padded(C_GREEN(f"技能 [{skill.name}] 已删除"))
                    _pause()
                    break
                else:
                    _padded(C_DIM("已取消"))

            _pause()

            settings = _load_settings()


def _skill_statistics_view():
    """查看技能统计信息"""
    from mechanism_skill import MechanismEngine
    engine = MechanismEngine()
    engine.add_builtin_skills()

    skills_dir = _load_settings().get("skills_dir", "skills")
    engine.load_skills_from_dir(skills_dir)

    stats = engine.get_skill_statistics()

    os.system("cls" if os.name == "nt" else "clear")
    _banner()
    print()
    w = _box(C_GREEN(" 技能统计 "))
    _text_line(f"{C_DIM('技能总数:')}   {C_BOLD(str(stats['total_skills']))}  {C_DIM('已启用:')} {C_BOLD(str(stats['enabled_skills']))}")
    _text_line(f"{C_DIM('分类:')}     {', '.join(stats['categories'])}")
    by_type = stats.get('by_type', {})
    type_str = "  ".join(f"{t}:{c}" for t, c in by_type.items() if c > 0)
    _text_line(f"{C_DIM('类型分布:')} {type_str}")
    _sep(w)

    if stats['trigger_counts']:
        _text_line(f"{C_BOLD('技能触发排行:')}")
        sorted_skills = sorted(stats['trigger_counts'].items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_skills[:10]:
            success = stats['success_counts'].get(name, 0)
            errors = stats['error_counts'].get(name, 0)
            rate = f"{success/count*100:.0f}%" if count > 0 else "0%"
            print(f"{N2}  {C_DIM('•')} {C_BOLD(name)}  {C_DIM(f'触发:')} {count}  {C_DIM('成功:')} {success}  {C_DIM('错误:')} {errors}  {C_DIM('成功率:')} {rate}")
    else:
        _text_line(C_DIM("暂无触发记录"))

    _sep(w)
    if stats['event_trigger_counts']:
        _text_line(f"{C_BOLD('事件触发次数:')}")
        for event, count in sorted(stats['event_trigger_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"{N2}  {C_DIM('•')} {event}: {C_BOLD(str(count))} 次")

    _close_box(w)
    _pause()


def _skill_import_menu():
    """从JSON文本导入技能"""
    from mechanism_skill import MechanismEngine, MechanismSkill
    temp_engine = MechanismEngine()

    os.system("cls" if os.name == "nt" else "clear")
    _banner()
    print()
    w = _box(C_YELLOW(" 导入技能 "))
    _text_line(C_DIM("粘贴技能JSON定义，空行结束"))
    _text_line(C_DIM("格式示例:"))
    _text_line(C_DIM('  {"name": "my_skill", "description": "...",'))
    _text_line(C_DIM('   "trigger": "round_end", "condition": "True",'))
    _text_line(C_DIM('   "effect": "engine.add_log(\\"触发\\")"}'))
    _sep(w)

    lines = []
    print(f"{N2}  {C_DIM('粘贴JSON（输入空行结束）:')}")
    while True:
        print(f"{N2}  ", end="")
        line = input()
        if not line:
            break
        lines.append(line)

    if not lines:
        _padded(C_DIM("已取消"))
        _close_box(w)
        _pause()
        return

    json_text = "\n".join(lines)
    try:
        import json
        data = json.loads(json_text)
        if isinstance(data, list):
            count = 0
            for item in data:
                skill = temp_engine.register_skill_from_dict(item)
                if skill:
                    temp_engine.save_ai_skill(skill)
                    count += 1
            _padded(C_GREEN(f"成功导入 {count} 个技能"))
        else:
            skill = temp_engine.register_skill_from_dict(data)
            if skill:
                temp_engine.save_ai_skill(skill)
                _padded(C_GREEN(f"成功导入技能: {skill.name}"))
            else:
                _padded(C_RED("导入失败，请检查JSON格式"))
    except json.JSONDecodeError as e:
        _padded(C_RED(f"JSON解析失败: {e}"))
    except Exception as e:
        _padded(C_RED(f"导入异常: {e}"))

    _close_box(w)
    _pause()


def _player_defaults_menu():
    """玩家默认配置菜单"""
    settings = _load_settings()
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_YELLOW(" 玩家默认配置 "))

        thinking = settings.get("thinking", "disabled")
        show_reasoning = settings.get("show_reasoning", True)
        show_answer = settings.get("show_answer", True)
        default_num = settings.get("default_num_players", 5)

        print(f"{N2}  {C_DIM('思考模式:')}      {C_BOLD(thinking)}")
        r_icon = C_GREEN("开") if show_reasoning else C_RED("关")
        print(f"{N2}  {C_DIM('显示推理过程:')}   {r_icon}")
        a_icon = C_GREEN("开") if show_answer else C_RED("关")
        print(f"{N2}  {C_DIM('显示最终答案:')}   {a_icon}")
        print(f"{N2}  {C_DIM('默认专家人数:')}   {C_BOLD(str(default_num))}")
        _sep(w)
        print(f"{N2}  {C_CYAN(' [1] ')}  切换思考模式     {C_DIM(f'当前: {thinking}')}")
        print(f"{N2}  {C_GREEN(' [2] ')}  切换显示推理     {C_DIM(f'当前: {r_icon}')}")
        print(f"{N2}  {C_YELLOW(' [3] ')}  切换显示答案     {C_DIM(f'当前: {a_icon}')}")
        print(f"{N2}  {C_BLUE(' [4] ')}  设置默认人数")
        print(f"{N2}  {C_DIM(' [q] ')}  返回设置菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-4/q]')}: ").strip().lower()

        if choice == "1":
            modes = ["disabled", "auto", "enabled"]
            current = modes.index(thinking) if thinking in modes else 0
            settings["thinking"] = modes[(current + 1) % len(modes)]
            _save_settings(settings)
        elif choice == "2":
            settings["show_reasoning"] = not settings.get("show_reasoning", True)
            _save_settings(settings)
        elif choice == "3":
            settings["show_answer"] = not settings.get("show_answer", True)
            _save_settings(settings)
        elif choice == "4":
            print(f"\n{N2}  ", end="")
            try:
                n = input(f"{C_CYAN('默认人数')} {C_DIM('[5]')}: ").strip()
                if n:
                    settings["default_num_players"] = max(1, min(50, int(n)))
                    _save_settings(settings)
            except ValueError:
                pass
        elif choice == "q":
            return

        settings = _load_settings()


def _scheduler_params_menu():
    """调度器参数配置菜单"""
    settings = _load_settings()
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_BLUE(" 调度器参数 "))

        ef = settings.get("exploration_factor", 1.5)
        dw = settings.get("diversity_weight", 0.6)
        hw = settings.get("hunger_weight", 0.3)
        rt = settings.get("redundancy_threshold", 0.7)

        print(f"{N2}  {C_DIM('探索因子:')}       {C_BOLD(f'{ef:.1f}')}  {C_DIM('(越高越倾向探索性发言)')}")
        print(f"{N2}  {C_DIM('多样性权重:')}     {C_BOLD(f'{dw:.1f}')}  {C_DIM('(越高越倾向不同视角)')}")
        print(f"{N2}  {C_DIM('饥饿度权重:')}     {C_BOLD(f'{hw:.1f}')}  {C_DIM('(越高越倾向久未发言者)')}")
        print(f"{N2}  {C_DIM('冗余阈值:')}       {C_BOLD(f'{rt:.1f}')}  {C_DIM('(相似度超过此值视为冗余)')}")
        _sep(w)
        print(f"{N2}  {C_CYAN(' [1] ')}  探索因子     {C_DIM(f'当前: {ef:.1f}')}")
        print(f"{N2}  {C_GREEN(' [2] ')}  多样性权重   {C_DIM(f'当前: {dw:.1f}')}")
        print(f"{N2}  {C_YELLOW(' [3] ')}  饥饿度权重   {C_DIM(f'当前: {hw:.1f}')}")
        print(f"{N2}  {C_MAGENTA(' [4] ')}  冗余阈值     {C_DIM(f'当前: {rt:.1f}')}")
        print(f"{N2}  {C_DIM(' [q] ')}  返回设置菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-4/q]')}: ").strip().lower()

        if choice in ("1", "2", "3", "4"):
            keys = ["exploration_factor", "diversity_weight", "hunger_weight", "redundancy_threshold"]
            labels = ["探索因子", "多样性权重", "饥饿度权重", "冗余阈值"]
            idx = int(choice) - 1
            print(f"\n{N2}  ", end="")
            try:
                val = float(input(f"{C_CYAN(f'{labels[idx]}')} {C_DIM('[0.1-5.0]')}: ").strip())
                val = max(0.1, min(5.0, val))
                settings[keys[idx]] = val
                _save_settings(settings)
            except ValueError:
                pass
        elif choice == "q":
            return

        settings = _load_settings()


def _discussion_params_menu():
    """讨论参数配置菜单"""
    settings = _load_settings()
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_GREEN(" 讨论参数 "))

        mr = settings.get("max_rounds", 0)
        ct = settings.get("consensus_threshold", 0.85)
        st = settings.get("stall_threshold", 5)
        me = settings.get("max_essences", 500)
        dm = settings.get("default_num_players", 5)

        mr_label = C_BOLD("无限制") if mr == 0 else C_BOLD(str(mr))
        print(f"{N2}  {C_DIM('最大轮数:')}        {mr_label}  {C_DIM('(0=无限制)')}")
        print(f"{N2}  {C_DIM('共识阈值:')}        {C_BOLD(f'{ct:.0%}')}  {C_DIM('(高于此值建议结束)')}")
        print(f"{N2}  {C_DIM('僵持阈值:')}        {C_BOLD(str(st))}  {C_DIM('轮 (超过此值建议停止)')}")
        print(f"{N2}  {C_DIM('精华池上限:')}      {C_BOLD(str(me))}  {C_DIM('条 (超过此值建议结束)')}")
        print(f"{N2}  {C_DIM('默认专家人数:')}    {C_BOLD(str(dm))}")
        _sep(w)
        print(f"{N2}  {C_CYAN(' [1] ')}  最大轮数     {C_DIM(f'当前: {mr_label}')}")
        print(f"{N2}  {C_GREEN(' [2] ')}  共识阈值     {C_DIM(f'当前: {ct:.0%}')}")
        print(f"{N2}  {C_YELLOW(' [3] ')}  僵持阈值     {C_DIM(f'当前: {st} 轮')}")
        print(f"{N2}  {C_MAGENTA(' [4] ')}  精华池上限   {C_DIM(f'当前: {me} 条')}")
        print(f"{N2}  {C_BLUE(' [5] ')}  默认专家人数 {C_DIM(f'当前: {dm} 人')}")
        print(f"{N2}  {C_DIM(' [q] ')}  返回设置菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-5/q]')}: ").strip().lower()

        if choice == "1":
            print(f"\n{N2}  ", end="")
            try:
                n = int(input(f"{C_CYAN('最大轮数')} {C_DIM('[0=无限制]')}: ").strip())
                settings["max_rounds"] = max(0, n)
                _save_settings(settings)
            except ValueError:
                pass
        elif choice == "2":
            print(f"\n{N2}  ", end="")
            try:
                v = float(input(f"{C_CYAN('共识阈值')} {C_DIM('[0.5-1.0]')}: ").strip())
                settings["consensus_threshold"] = max(0.5, min(1.0, v))
                _save_settings(settings)
            except ValueError:
                pass
        elif choice == "3":
            print(f"\n{N2}  ", end="")
            try:
                n = int(input(f"{C_CYAN('僵持阈值')} {C_DIM('[3-20]')}: ").strip())
                settings["stall_threshold"] = max(3, min(20, n))
                _save_settings(settings)
            except ValueError:
                pass
        elif choice == "4":
            print(f"\n{N2}  ", end="")
            try:
                n = int(input(f"{C_CYAN('精华池上限')} {C_DIM('[100-2000]')}: ").strip())
                settings["max_essences"] = max(100, min(2000, n))
                _save_settings(settings)
            except ValueError:
                pass
        elif choice == "5":
            print(f"\n{N2}  ", end="")
            try:
                n = int(input(f"{C_CYAN('默认专家人数')} {C_DIM('[1-50]')}: ").strip())
                settings["default_num_players"] = max(1, min(50, n))
                _save_settings(settings)
            except ValueError:
                pass
        elif choice == "q":
            return

        settings = _load_settings()


def _tts_settings_menu():
    """语音输出（TTS）配置菜单"""
    from multimodal import get_tts, TTSDialog
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        _banner()
        print()
        w = _box(C_WHITE(" 语音输出设置 "))

        tts = get_tts()

        print(f"{N2}  {C_DIM('当前引擎:')}  {C_BOLD(tts.provider_name)}")
        print(f"{N2}  {C_DIM('状态:')}     {'✅ 已启用' if tts.enabled else '❌ 已禁用'}")
        if tts._provider == TTSProvider.EDGE_TTS:
            print(f"{N2}  {C_DIM('语音:')}     {C_BOLD(tts._edge_voice)}")
            print(f"{N2}  {C_DIM('语速:')}     {C_BOLD(tts._edge_rate)}")
        elif tts._provider == TTSProvider.PYTTSX3:
            print(f"{N2}  {C_DIM('语速:')}     {C_BOLD(tts._rate)}")
            print(f"{N2}  {C_DIM('音量:')}     {C_BOLD(int(tts._volume * 100))}%")
        _sep(w)
        print(f"{N2}  {C_CYAN(' [1] ')}  切换开关       {C_DIM(f'当前: {"启用" if tts.enabled else "禁用"}')}")
        print(f"{N2}  {C_GREEN(' [2] ')}  测试语音       {C_DIM('朗读测试文本')}")
        print(f"{N2}  {C_BLUE(' [3] ')}  切换 edge-tts  {C_DIM('微软免费 TTS（推荐）')}")
        print(f"{N2}  {C_DIM(' [q] ')}  返回设置菜单")
        _close_box(w)
        print()
        print(f"  {C_YELLOW('▸')}  ", end="")
        choice = input(f"{C_BOLD('请选择')} {C_DIM('[1-3/q]')}: ").strip().lower()

        if choice == "1":
            tts.enabled = not tts.enabled
            print(f"\n{N2}  {'✅ 已启用' if tts.enabled else '❌ 已禁用'}")
            _pause()
        elif choice == "2":
            print(f"\n{N2}  {C_DIM('正在测试...')}")
            tts.speak_sync("语音输出测试。如果听到声音，说明设置正确。", show_debug=True)
            print(f"{N2}  {C_GREEN('✓')} 测试完成")
            _pause()
        elif choice == "3":
            tts.switch_to_edge_tts()
            print(f"\n{N2}  {C_GREEN('✓')} 已切换到 edge-tts: {tts.provider_name} | 语音: {tts._edge_voice}")
            _pause()
        elif choice == "q":
            return


def _get_player_configs(num: int, settings: dict) -> list:
    """根据设置生成玩家配置列表"""
    app_cfg = _apply_settings(settings)
    thinking = settings.get("thinking", "disabled")
    show_reasoning = settings.get("show_reasoning", True)
    show_answer = settings.get("show_answer", True)
    model = app_cfg["model"]

    configs = []
    for i in range(1, num + 1):
        configs.append({
            "name": f"专家{i}",
            "model": model,
            "thinking": thinking,
            "show_reasoning": show_reasoning,
            "show_answer": show_answer,
        })
    return configs


# ═══════════════════════════════════════════════════════════════
#  SLSMDS 自编程模式（已禁用）
#  ═══════════════════════════════════════════════════════════════
#  自编程模式允许 AI 专家讨论并修改项目源码。
#  这是一个独立的主菜单级功能，与普通讨论完全分离。
#  如需启用，取消注释下方代码并恢复 prompt/self_program_*.txt 文件。
# ═══════════════════════════════════════════════════════════════

# ── 自编程提示词路径 ──
# SELF_PROGRAM_DISCUSS_PROMPT_PATH = "prompt/self_program_discussion_prompt.txt"
# SELF_PROGRAM_IMPLEMENT_PROMPT_PATH = "prompt/self_program_implement_prompt.txt"

# ── 自编程会话 ──
# def _self_programming_session(game):
#     """自编程模式主会话"""
#     from mechanism_skill import _empty_line, _box, _sep, _padded, _footer
#     _empty_line()
#     w = _box(C_MAGENTA(" 自编程模式 "))
#     _padded(f"专家: {game.num_players} 人  轮次: {game.round_count}")
#     _sep(w)
#     _padded(" [1]  输入新需求")
#     _padded(" [2]  自动生成需求（基于系统状态）")
#     _padded(" [q]  返回主菜单")
#     _sep(w)
#     choice = input(" 请选择 [1/2/q]: ").strip().lower()
#     if choice == 'q':
#         return
#     elif choice == '1':
#         requirement = input(" 请输入代码改进需求: ").strip()
#         if not requirement:
#             return
#         _run_self_programming_discussion(game, requirement)
#     elif choice == '2':
#         requirement = _generate_auto_requirement(game)
#         _padded(f"自动生成需求: {requirement}")
#         _run_self_programming_discussion(game, requirement)

# ── 自动生成需求 ──
# def _generate_auto_requirement(game):
#     """基于当前系统状态自动生成改进需求"""
#     state = {
#         "num_players": game.num_players,
#         "round_count": game.round_count,
#         "total_essences": len(game.essence_pool.items) if hasattr(game, 'essence_pool') else 0,
#     }
#     if hasattr(game, 'settings') and game.settings:
#         state.update(game.settings)
#     prompt = (
#         "你是一位系统架构师。请分析当前系统状态，提出一个代码改进需求。\n"
#         f"系统状态: {json.dumps(state, ensure_ascii=False)}\n"
#         "输出一段简短的改进需求描述（50字以内），直接输出文本即可。"
#     )
#     try:
#         from llm_client import LLMClient
#         client = LLMClient()
#         response, _ = client.chat(
#             [{"role": "user", "content": prompt}],
#             model=game.model_name if hasattr(game, 'model_name') else "deepseek-v4-flash",
#             thinking="disabled",
#             caller="自编程-需求生成",
#             show_reasoning=False, show_answer=False,
#         )
#         return response.strip() if response else "优化代码架构和讨论流程"
#     except Exception:
#         return "优化代码架构和讨论流程"

# ── 自编程讨论与实现 ──
# def _run_self_programming_discussion(game, requirement):
#     """
#     自编程讨论流程：
#     1. 专家们针对需求进行讨论
#     2. 汇总讨论结果，生成代码修改方案
#     3. 解析方案并应用修改（含回退机制）
#     """
#     from mechanism_skill import _empty_line, _box, _sep, _padded, _footer, _close_box
#     from llm_client import LLMClient
#     import traceback
#
#     client = LLMClient()
#     _empty_line()
#     w = _box(C_MAGENTA(" 自编程讨论 "))
#     _padded(f"需求: {C_BOLD(requirement)}")
#     _sep(w)
#
#     # 步骤 1: 专家讨论
#     _padded(C_DIM("阶段 1/3 — 专家讨论中..."))
#     discussions = []
#     for player in game.players:
#         if not player.alive:
#             continue
#         discuss_prompt = _read_file(SELF_PROGRAM_DISCUSS_PROMPT_PATH)
#         if not discuss_prompt:
#             discuss_prompt = "你是一位系统架构师。需求: {requirement}\n请给出你的修改建议。"
#         prompt = discuss_prompt.format(
#             self_name=player.name,
#             self_persona=player.persona,
#             requirement=requirement,
#             file_descriptions=_get_file_descriptions(),
#         )
#         try:
#             response, _ = client.chat(
#                 [{"role": "user", "content": prompt}],
#                 model=player.model_name,
#                 thinking="disabled",
#                 caller=f"自编程-{player.name}",
#                 show_reasoning=False, show_answer=False,
#             )
#             discussions.append({"name": player.name, "response": response})
#             _padded(f"{C_DIM(player.name)}: {response[:80]}...")
#         except Exception:
#             _padded(f"{C_RED(player.name)} 讨论失败")
#
#     # 步骤 2: 汇总方案
#     _sep(w)
#     _padded(C_DIM("阶段 2/3 — 汇总方案中..."))
#     summary_prompt = (
#         "你是一位资深系统架构师。以下是多位专家针对需求「{requirement}」的讨论。\n"
#         "请汇总所有人的意见，生成一个统一的代码修改方案。\n\n"
#         "专家意见:\n{discussions}\n\n"
#         "输出JSON格式:\n"
#         '{{"plan_summary": "方案概述", "changes": [{{"file": "文件路径", "action": "modify/create/delete", "content": "新内容"}}]}}'
#     ).format(
#         requirement=requirement,
#         discussions="\n".join(f"{d['name']}: {d['response']}" for d in discussions),
#     )
#     try:
#         summary, _ = client.chat(
#             [{"role": "user", "content": summary_prompt}],
#             model=game.players[0].model_name if game.players else "deepseek-v4-flash",
#             thinking="disabled",
#             caller="自编程-汇总",
#             show_reasoning=False, show_answer=False,
#         )
#         _padded(f"方案: {summary[:100]}...")
#     except Exception as e:
#         _padded(C_RED(f"汇总失败: {e}"))
#         _close_box(w)
#         return
#
#     # 步骤 3: 实现方案
#     _sep(w)
#     _padded(C_DIM("阶段 3/3 — 实现方案中..."))
#     try:
#         import json
#         plan = json.loads(summary)
#         changes = plan.get("changes", [])
#         if not changes:
#             _padded(C_YELLOW("方案中没有有效的修改"))
#             _close_box(w)
#             return
#
#         backup_files = {}
#         success = True
#         for change in changes:
#             filepath = change.get("file", "")
#             action = change.get("action", "modify")
#             content = change.get("content", "")
#             if not filepath:
#                 continue
#             # 备份原文件
#             if os.path.exists(filepath) and action in ("modify", "delete"):
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     backup_files[filepath] = f.read()
#             try:
#                 if action == "create":
#                     os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
#                     with open(filepath, "w", encoding="utf-8") as f:
#                         f.write(content)
#                     _padded(C_GREEN(f"创建: {filepath}"))
#                 elif action == "modify":
#                     with open(filepath, "w", encoding="utf-8") as f:
#                         f.write(content)
#                     _padded(C_GREEN(f"修改: {filepath}"))
#                 elif action == "delete":
#                     os.remove(filepath)
#                     _padded(C_GREEN(f"删除: {filepath}"))
#             except Exception as e:
#                 _padded(C_RED(f"操作失败: {filepath} — {e}"))
#                 success = False
#                 break
#
#         if success:
#             _padded(C_GREEN("自编程完成！"))
#         else:
#             _padded(C_YELLOW("操作失败，开始回退..."))
#             for filepath, content in backup_files.items():
#                 try:
#                     with open(filepath, "w", encoding="utf-8") as f:
#                         f.write(content)
#                     _padded(C_GREEN(f"回退: {filepath}"))
#                 except Exception as e:
#                     _padded(C_RED(f"回退失败: {filepath} — {e}"))
#     except json.JSONDecodeError:
#         _padded(C_RED("方案解析失败：AI返回格式异常"))
#     except Exception as e:
#         _padded(C_RED(f"实现失败: {e}"))
#         _padded(C_DIM(traceback.format_exc()))
#
#     _close_box(w)

# ── 文件描述 ──
# def _get_file_descriptions():
#     """生成项目文件描述列表"""
#     descriptions = {
#         "game.py": "主程序 + TUI，包含 Game 类和所有 TUI 函数",
#         "player.py": "玩家模块，定义 AI 专家参与者",
#         "llm_client.py": "LLM API 客户端，支持流式调用",
#         "mechanism_skill.py": "机制技能系统",
#         "essence_pool.py": "精华池",
#         "scheduler.py": "动态调度器",
#         "knowledge_base.py": "知识库",
#         "global_knowledge.py": "全局知识库",
#         "game_record.py": "游戏记录与检查点",
#         "observer.py": "观察者",
#         "cognitive_map_widget.py": "认知地图",
#         "replay_widget.py": "讨论回放",
#         "counterfactual_widget.py": "反事实推演",
#         "multimodal.py": "多模态扩展",
#     }
#     result = "项目文件列表:\n"
#     for fname, desc in descriptions.items():
#         exists = os.path.exists(fname)
#         status = "✓" if exists else "✗"
#         result += f"  [{status}] {fname} — {desc}\n"
#     return result

# ═══════════════════════════════════════════════════════════════


if __name__ == '__main__':
    _main_tui()





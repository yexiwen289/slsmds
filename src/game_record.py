"""
游戏记录模块 —— 集体智慧讨论的记录与报告生成

记录讨论过程中的所有数据：
- 发言记录
- 精华池状态变化
- 最终综合方案
- 控制台输出
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime
import json
import os
import sys
import io


class ConsoleCapture:
    def __init__(self, game_record: 'GameRecord'):
        self.game_record = game_record
        self._original_stdout = sys.stdout
        self._buffer = io.StringIO()

    def __enter__(self):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout

    def write(self, text):
        self._original_stdout.write(text)
        self._buffer.write(text)
        if text.strip():
            self.game_record.log_console(text.rstrip('\n'))

    def flush(self):
        self._original_stdout.flush()


@dataclass
class GameEvent:
    timestamp: str
    round_id: int
    event_type: str
    player_name: str
    data: Dict

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "round_id": self.round_id,
            "event_type": self.event_type,
            "player_name": self.player_name,
            "data": self.data
        }


def generate_game_id():
    import random
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = format(random.randint(0, 0xFFFF), '04x')
    return f"{timestamp}_{suffix}"


@dataclass
class DiscussionAction:
    """一轮讨论中的一条发言记录"""
    player_name: str
    speech: str
    key_insight: str = ""
    action: str = "new"  # new / refine / challenge
    discussion_thinking: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "player_name": self.player_name,
            "speech": self.speech,
            "key_insight": self.key_insight,
            "action": self.action,
            "discussion_thinking": self.discussion_thinking,
        }


@dataclass
class DiscussionRecord:
    """一轮讨论的完整记录"""
    round_id: int
    round_players: List[str]
    problem: str
    pool_summary: str
    speech_history: List[DiscussionAction] = field(default_factory=list)
    essences_added: List[Dict] = field(default_factory=list)
    pool_state_after: str = ""

    def to_dict(self) -> Dict:
        return {
            "round_id": self.round_id,
            "round_players": self.round_players,
            "problem": self.problem,
            "speech_history": [s.to_dict() for s in self.speech_history],
            "essences_added": self.essences_added,
            "pool_state_after": self.pool_state_after,
        }

    def add_speech_action(self, action: DiscussionAction) -> None:
        self.speech_history.append(action)

    def get_latest_round_info(self) -> str:
        return (
            f"第{self.round_id}轮讨论，参与玩家：{'、'.join(self.round_players)}，"
            f"讨论问题：{self.problem}"
        )

    def get_essence_status(self) -> str:
        if not self.essences_added:
            return "本轮未提炼出精华"
        return f"本轮提炼出{len(self.essences_added)}条精华"


@dataclass
class SynthesisRecord:
    """最终综合方案记录"""
    synthesizer: str
    solution_title: str
    summary: str
    final_conclusion: str

    def to_dict(self) -> Dict:
        return {
            "synthesizer": self.synthesizer,
            "solution_title": self.solution_title,
            "summary": self.summary,
            "final_conclusion": self.final_conclusion,
        }


class GameRecord:
    def __init__(self):
        self.game_id: str = generate_game_id()
        self.player_names: List[str] = []
        self.rounds: List[DiscussionRecord] = []
        self.synthesis_records: List[SynthesisRecord] = []
        self.final_solution: Optional[Dict] = None
        self.save_directory: str = "game_records"
        self.event_log: List[GameEvent] = []
        self.console_log: List[str] = []
        self.game_start_time: Optional[str] = None
        self.game_end_time: Optional[str] = None
        self.game_duration_seconds: float = 0

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def to_dict(self) -> Dict:
        return {
            "game_id": self.game_id,
            "player_names": self.player_names,
            "rounds": [round.to_dict() for round in self.rounds],
            "synthesis_records": [s.to_dict() for s in self.synthesis_records],
            "final_solution": self.final_solution,
            "event_log": [e.to_dict() for e in self.event_log],
            "console_log": self.console_log,
            "game_start_time": self.game_start_time,
            "game_end_time": self.game_end_time,
            "game_duration_seconds": self.game_duration_seconds,
        }

    def start_game(self, player_names: List[str]) -> None:
        self.player_names = player_names
        self.game_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_event("discussion_start", "SYSTEM", {"players": player_names})

    def _log_event(self, event_type: str, player_name: str, data: Dict) -> None:
        event = GameEvent(
            timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
            round_id=self._current_round_id(),
            event_type=event_type,
            player_name=player_name,
            data=data
        )
        self.event_log.append(event)

    def _current_round_id(self) -> int:
        return self.rounds[-1].round_id if self.rounds else 0

    def log_console(self, text: str) -> None:
        self.console_log.append(text)

    def start_round(self, round_id: int, round_players: List[str],
                    problem: str, pool_summary: str) -> None:
        round_record = DiscussionRecord(
            round_id=round_id,
            round_players=round_players,
            problem=problem,
            pool_summary=pool_summary,
        )
        self.rounds.append(round_record)
        self._log_event("round_start", "SYSTEM", {
            "round_id": round_id,
            "players": round_players,
            "problem": problem,
        })

    def record_speech(self, player_name: str, speech: str,
                      key_insight: str = "", action: str = "new",
                      discussion_thinking: str = None) -> None:
        current_round = self.get_current_round()
        if current_round:
            action_record = DiscussionAction(
                player_name=player_name,
                speech=speech,
                key_insight=key_insight,
                action=action,
                discussion_thinking=discussion_thinking,
            )
            current_round.add_speech_action(action_record)
            self._log_event("speech", player_name, {
                "speech": speech,
                "key_insight": key_insight,
                "action": action,
            })

    def record_persona(self, player_name: str, persona_name: str, persona: str) -> None:
        self._log_event("persona", player_name, {
            "persona_name": persona_name, "persona": persona
        })

    def record_essence_pool_state(self, round_id: int, pool_summary: str) -> None:
        current_round = self.get_current_round()
        if current_round:
            current_round.pool_state_after = pool_summary
        self._log_event("essence_pool_update", "SYSTEM", {
            "round_id": round_id,
            "summary_preview": pool_summary[:100],
        })

    def record_votes(self, voter_name: str, votes: List[Dict], round_id: int) -> None:
        """记录一位专家的投票"""
        self._log_event("vote", voter_name, {
            "round_id": round_id,
            "votes": votes,
            "vote_count": len(votes),
        })

    def record_debate(self, round_id: int, topic_essence_id: int,
                      attacker_name: str, defender_name: str,
                      attacker_argument: str, defender_argument: str,
                      attacker_concede: bool, defender_concede: bool) -> None:
        """记录一轮辩论"""
        self._log_event("debate", "SYSTEM", {
            "round_id": round_id,
            "topic_essence_id": topic_essence_id,
            "attacker": attacker_name,
            "defender": defender_name,
            "attacker_argument": attacker_argument[:200],
            "defender_argument": defender_argument[:200],
            "attacker_concede": attacker_concede,
            "defender_concede": defender_concede,
        })

    def record_synthesis(self, synthesizer: str, solution_title: str,
                         summary: str, final_conclusion: str) -> None:
        record = SynthesisRecord(
            synthesizer=synthesizer,
            solution_title=solution_title,
            summary=summary,
            final_conclusion=final_conclusion,
        )
        self.synthesis_records.append(record)
        self._log_event("synthesis", synthesizer, {
            "solution_title": solution_title,
            "summary": summary[:100],
        })

    def finish_game(self, final_solution: Dict) -> None:
        self.final_solution = final_solution
        self.game_end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_event("discussion_end", "SYSTEM", {
            "solution_title": final_solution.get("solution_title", ""),
        })

    def get_current_round(self) -> Optional[DiscussionRecord]:
        return self.rounds[-1] if self.rounds else None

    def get_latest_round_info(self) -> Optional[str]:
        current_round = self.get_current_round()
        return current_round.get_latest_round_info() if current_round else None

    def generate_final_report(self, player_stats_dict: Dict,
                              essence_pool: object,
                              problem: str) -> str:
        """
        生成最终讨论报告

        Args:
            player_stats_dict: 玩家统计信息
            essence_pool: EssencePool 实例
            problem: 讨论问题
        """
        from .essence_pool import EssencePool

        lines = []
        lines.append("=" * 60)
        lines.append("🧠 集体智慧讨论报告")
        lines.append("=" * 60)
        lines.append("")

        # 基本信息
        lines.append(f"🆔 讨论ID: {self.game_id}")
        lines.append(f"🕐 开始时间: {self.game_start_time or '未知'}")
        lines.append(f"🕐 结束时间: {self.game_end_time or '未知'}")
        if self.game_duration_seconds:
            minutes = self.game_duration_seconds / 60
            lines.append(f"⏱ 总耗时: {self.game_duration_seconds:.1f}s ({minutes:.1f}min)")
        lines.append(f"🔄 总讨论轮次: {len(self.rounds)}")
        lines.append(f"👥 参与专家: {', '.join(self.player_names)}")
        lines.append("")

        # 讨论问题
        lines.append("=" * 60)
        lines.append("📋 讨论问题")
        lines.append("=" * 60)
        lines.append(f"  {problem}")
        lines.append("")

        # 专家身份（人设）
        persona_events = [e for e in self.event_log if e.event_type == "persona"]
        if persona_events:
            lines.append("-" * 60)
            lines.append("🪪 参与专家身份资料")
            lines.append("-" * 60)
            for evt in persona_events:
                data = evt.data
                lines.append(f"  {evt.player_name} ({data.get('persona_name', '')}): {data.get('persona', '')}")
            lines.append("")

        # 玩家统计
        lines.append("-" * 60)
        lines.append("📊 专家参与统计")
        lines.append("-" * 60)
        for pname, stats in player_stats_dict.items():
            lines.append(f"\n  {pname}:")
            lines.append(f"    讨论发言: {stats.get('discussions_made', 0)}次")
            lines.append(f"    贡献精华: {stats.get('essences_contributed', 0)}条")
            lines.append(f"    深化精华: {stats.get('essences_refined', 0)}次")
            lines.append(f"    反驳精华: {stats.get('essences_challenged', 0)}次")
            lines.append(f"    引用精华: {stats.get('essences_cited', 0)}次")
        lines.append("")

        # 精华池最终状态
        lines.append("=" * 60)
        lines.append("🏆 精华池最终状态（按评分排序）")
        lines.append("=" * 60)
        if essence_pool.items:
            top_essences = essence_pool.get_top_essences(20)
            for i, item in enumerate(top_essences, 1):
                tags_str = f"[{', '.join(item.tags)}]" if item.tags else ""
                lines.append(f"\n  #{i} (ID:{item.id}) 评分:{item.score:.1f} {tags_str}")
                lines.append(f"    内容: {item.content}")
                lines.append(f"    贡献者: {item.contributor} (第{item.source_round}轮)")
                if item.parent_id:
                    parent = next((x for x in essence_pool.items if x.id == item.parent_id), None)
                    if parent:
                        lines.append(f"    继承自: #{item.parent_id} \"{parent.content[:40]}\"")
                relations = []
                if item.cited_by:
                    relations.append(f"被{len(item.cited_by)}人引用")
                if item.refined_by:
                    relations.append(f"被深化{len(item.refined_by)}次")
                if item.challenged_by:
                    relations.append(f"被{len(item.challenged_by)}人反驳")
                vote_info = []
                if item.approve_by:
                    vote_info.append(f"赞同{len(item.approve_by)}")
                if item.reject_by:
                    vote_info.append(f"反对{len(item.reject_by)}")
                if vote_info:
                    relations.append(" ".join(vote_info))
                if relations:
                    lines.append(f"    {' | '.join(relations)}")
            lines.append(f"\n  精华池总计: {len(essence_pool.items)} 条精华")
        else:
            lines.append("  无精华条目")
        lines.append("")

        # 投票统计
        vote_events = [e for e in self.event_log if e.event_type == "vote"]
        if vote_events:
            lines.append("-" * 60)
            lines.append("🗳️ 投票统计")
            lines.append("-" * 60)
            total_votes = sum(e.data.get("vote_count", 0) for e in vote_events)
            approve_total = 0
            reject_total = 0
            abstain_total = 0
            for e in vote_events:
                for v in e.data.get("votes", []):
                    vote = v.get("vote", "abstain")
                    if vote == "approve":
                        approve_total += 1
                    elif vote == "reject":
                        reject_total += 1
                    else:
                        abstain_total += 1
            lines.append(f"  总投票数: {total_votes} (赞同:{approve_total} 反对:{reject_total} 弃权:{abstain_total})")
            lines.append(f"  参与投票轮次: {len(set(e.round_id for e in vote_events))}轮")
            lines.append("")

        # 辩论统计
        debate_events = [e for e in self.event_log if e.event_type == "debate"]
        if debate_events:
            lines.append("-" * 60)
            lines.append("⚔️ 辩论统计")
            lines.append("-" * 60)
            lines.append(f"  总辩论场次: {len(debate_events)}")
            for e in debate_events:
                d = e.data
                lines.append(f"  第{d.get('round_id', '?')}轮 | 焦点:精华#{d.get('topic_essence_id', '?')} | "
                             f"挑战方:{d.get('attacker', '?')} vs 辩护方:{d.get('defender', '?')}")
            lines.append("")

        # 精华池演化历史
        lines.append("-" * 60)
        lines.append("📈 精华池演化历史")
        lines.append("-" * 60)
        evolution = essence_pool.get_evolution_summary()
        lines.append(f"  {evolution}")
        lines.append("")

        # 最终综合方案
        lines.append("=" * 60)
        lines.append("🎯 最终综合方案")
        lines.append("=" * 60)
        if self.final_solution:
            fs = self.final_solution
            lines.append(f"\n  标题: {fs.get('solution_title', '')}")
            lines.append(f"\n  摘要: {fs.get('summary', '')}")
            lines.append("")
            lines.append("  核心思想:")
            core_ideas = fs.get("core_ideas", [])
            if core_ideas:
                for i, idea in enumerate(core_ideas, 1):
                    if isinstance(idea, dict):
                        lines.append(f"\n  {i}. {idea.get('idea', '')}")
                        lines.append(f"     阐述: {idea.get('elaboration', '')}")
                        src_ids = idea.get('source_essence_ids', [])
                        if src_ids:
                            lines.append(f"     来源精华: {src_ids}")
                    else:
                        lines.append(f"\n  {i}. {idea}")
            else:
                lines.append("    （无详细核心思想记录）")
            lines.append("")

            key_insights = fs.get("key_insights", [])
            if key_insights:
                lines.append("  关键洞见:")
                for ki in key_insights:
                    if isinstance(ki, str):
                        lines.append(f"    - {ki}")
                    elif isinstance(ki, dict):
                        lines.append(f"    - {ki.get('insight', ki)}")

            divergence = fs.get("divergence_points", [])
            if divergence:
                lines.append("\n  分歧点:")
                for dp in divergence:
                    if isinstance(dp, str):
                        lines.append(f"    - {dp}")
                    elif isinstance(dp, dict):
                        lines.append(f"    - {dp.get('point', dp)}")

            lines.append(f"\n  最终结论: {fs.get('final_conclusion', '')}")
        else:
            lines.append("  无最终综合方案")
        lines.append("")

        # 各轮讨论详情
        lines.append("=" * 60)
        lines.append("📖 各轮讨论详情")
        lines.append("=" * 60)
        for rnd in self.rounds:
            lines.append("")
            lines.append(f"--- 第{rnd.round_id}轮讨论 ---")
            lines.append(f"  参与: {'、'.join(rnd.round_players)}")
            for action in rnd.speech_history:
                lines.append(f"  {action.player_name}: \"{action.speech[:80]}\"")
                if action.key_insight:
                    lines.append(f"    💡 核心见解: {action.key_insight}")
                action_label = {"new": "新观点", "refine": "深化", "challenge": "反驳"}
                lines.append(f"    [行动: {action_label.get(action.action, action.action)}]")
            if rnd.essences_added:
                lines.append(f"  🔍 本轮提炼精华:")
                for ess in rnd.essences_added:
                    lines.append(f"    [{ess.get('type', '论点')}] {ess.get('content', '')[:60]}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("🏁 报告结束")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_report(self, report: str) -> str:
        report_path = os.path.join(self.save_directory, f"{self.game_id}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        return report_path

    def save_console_log(self) -> str:
        log_path = os.path.join(self.save_directory, f"{self.game_id}_console.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for line in self.console_log:
                f.write(line + "\n")
        return log_path

    def to_checkpoint_dict(self) -> Dict:
        """序列化为断点保存格式"""
        return {
            "game_id": self.game_id,
            "player_names": self.player_names,
            "rounds": [r.to_dict() for r in self.rounds],
            "synthesis_records": [s.to_dict() for s in self.synthesis_records],
            "final_solution": self.final_solution,
            "event_log": [e.to_dict() for e in self.event_log],
            "console_log": self.console_log,
            "game_start_time": self.game_start_time,
            "game_end_time": self.game_end_time,
            "game_duration_seconds": self.game_duration_seconds,
        }

    @classmethod
    def from_checkpoint_dict(cls, data: Dict) -> 'GameRecord':
        """从断点数据恢复GameRecord"""
        record = cls.__new__(cls)
        record.game_id = data.get("game_id", generate_game_id())
        record.player_names = data.get("player_names", [])
        record.synthesis_records = []
        for sr in data.get("synthesis_records", []):
            record.synthesis_records.append(SynthesisRecord(
                synthesizer=sr.get("synthesizer", ""),
                solution_title=sr.get("solution_title", ""),
                summary=sr.get("summary", ""),
                final_conclusion=sr.get("final_conclusion", ""),
            ))
        record.final_solution = data.get("final_solution")
        record.event_log = []
        for ev in data.get("event_log", []):
            record.event_log.append(GameEvent(
                timestamp=ev.get("timestamp", ""),
                round_id=ev.get("round_id", 0),
                event_type=ev.get("event_type", ""),
                player_name=ev.get("player_name", ""),
                data=ev.get("data", {}),
            ))
        record.console_log = data.get("console_log", [])
        record.game_start_time = data.get("game_start_time")
        record.game_end_time = data.get("game_end_time")
        record.game_duration_seconds = data.get("game_duration_seconds", 0)
        record.save_directory = "game_records"

        # 恢复rounds
        record.rounds = []
        for rd in data.get("rounds", []):
            discussion_record = DiscussionRecord(
                round_id=rd["round_id"],
                round_players=rd.get("round_players", []),
                problem=rd.get("problem", ""),
                pool_summary=rd.get("pool_summary", ""),
                essences_added=rd.get("essences_added", []),
                pool_state_after=rd.get("pool_state_after", ""),
            )
            for sh in rd.get("speech_history", []):
                discussion_record.add_speech_action(DiscussionAction(
                    player_name=sh.get("player_name", ""),
                    speech=sh.get("speech", ""),
                    key_insight=sh.get("key_insight", ""),
                    action=sh.get("action", "new"),
                    discussion_thinking=sh.get("discussion_thinking"),
                ))
            record.rounds.append(discussion_record)

        if not os.path.exists(record.save_directory):
            os.makedirs(record.save_directory)

        return record
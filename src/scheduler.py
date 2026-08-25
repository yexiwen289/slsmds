"""
智能专家调度系统 —— 多臂赌博机（UCB）算法 + 多维评估 + 多样性约束

核心算法：
1. 每个专家是一个"摇臂"，有不确定的贡献价值
2. UCB（Upper Confidence Bound）平衡探索与利用
3. 多维评估：质量、新颖度、影响力、参与度、领域覆盖
4. 多样性约束：确保不同专业视角的专家都有发言机会
5. 自适应发言人数：随专家总数动态调整

扩展系统：
6. 多目标优化器（MultiObjectiveOptimizer）—— 平衡多样性、深度、公平性、新颖性
7. 专家画像系统（ExpertProfiler）—— 构建专家行为综合画像
8. 动态角色分配（DynamicRoleAssigner）—— 基于画像和讨论需求分配角色
9. 性能预测器（PerformancePredictor）—— 预测专家未来表现
10. 群体动力学建模（GroupDynamicsModeler）—— 建模群体互动模式
11. 讨论阶段检测（DiscussionPhaseDetector）—— 检测讨论阶段并优化调度
12. 自适应轮次分配（AdaptiveTurnAllocator）—— 动态分配发言轮次
13. 专业知识缺口填补（ExpertiseGapFiller）—— 检测并填补知识盲区
14. 干预调度（InterventionScheduler）—— 调度主持人干预
15. 学习型调度器（LearningScheduler）—— 从历史调度中学习改进
"""

import math
import random
import heapq
import itertools
import statistics
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Callable, Any, Union
from copy import deepcopy
from .player import Player


# 中文专业领域关键词库（用于计算专家相似度）
EXPERTISE_KEYWORDS = frozenset({
    "范畴论", "拓扑", "宇宙学", "量子", "数学", "物理", "逻辑", "代数",
    "几何", "信息", "计算", "因果", "时空", "对称", "结构", "模型",
    "公理", "证明", "定理", "函数", "集合", "群论", "同调", "层论",
    "概率", "统计", "力学", "电磁", "热力", "相对", "弦论", "圈量子",
    "非公理", "拓扑斯", "范畴", "态射", "函子", "自然变换", "极限",
    "余极限", "Heyting", "Grothendieck", "幂等", "自同构", "同伦",
    "哲学", "伦理", "社会", "经济", "政策", "环境", "生物", "化学",
    "认知", "心理", "语言", "工程", "系统", "控制", "网络", "神经",
    "计算", "算法", "数据", "学习", "优化", "博弈", "决策", "复杂",
    "混沌", "分形", "涌现", "自组织", "自适应", "演化", "遗传",
    "人工智能", "机器学习", "深度学习", "强化学习", "神经网络",
    "软件", "硬件", "架构", "设计", "安全", "隐私", "分布式",
    "密码学", "区块链", "量子里", "量子计算", "量子信息",
})


def _extract_expertise_tags(text: str) -> set:
    """从玩家的 persona 文本中提取专业领域标签"""
    if not text:
        return set()
    found = set()
    for kw in EXPERTISE_KEYWORDS:
        if kw in text:
            found.add(kw)
    return found


def _expertise_similarity(tags_a: set, tags_b: set) -> float:
    """计算两个专家之间的专业相似度（Jaccard 系数）"""
    if not tags_a or not tags_b:
        return 0.0
    intersection = tags_a & tags_b
    union = tags_a | tags_b
    return len(intersection) / len(union) if union else 0.0


def _calculate_n_speakers(total_alive: int) -> int:
    """
    自适应发言人数：根据存活专家总数动态决定每轮发言人数
    公式：min(max(3, ceil(total * 0.4)), total, 6)
    """
    if total_alive <= 3:
        return total_alive
    n = max(3, math.ceil(total_alive * 0.4))
    n = min(n, total_alive, 6)
    return n


class ExpertProfile:
    """单个专家的完整评估画像"""

    def __init__(self, name: str, persona: str, persona_name: str):
        self.name = name
        self.persona = persona
        self.persona_name = persona_name
        self.expertise_tags = _extract_expertise_tags(persona + persona_name)

        # 统计信息
        self.times_spoken = 0            # 发言次数
        self.last_spoke_round = 0        # 最后一次发言轮次
        self.rounds_since_last_spoke = 0 # 距离上次发言的轮数

        # 多维累积评分（每次发言后累加，取平均）
        self.quality_total = 0.0         # 质量分（精华评分）
        self.novelty_total = 0.0         # 新颖度分（new action 占比）
        self.influence_total = 0.0       # 影响力分（被引用次数）
        self.engagement_total = 0.0      # 参与度分（refine/challenge 行为）

    @property
    def mean_quality(self) -> float:
        return self.quality_total / max(self.times_spoken, 1)

    @property
    def mean_novelty(self) -> float:
        return self.novelty_total / max(self.times_spoken, 1)

    @property
    def mean_influence(self) -> float:
        return self.influence_total / max(self.times_spoken, 1)

    @property
    def mean_engagement(self) -> float:
        return self.engagement_total / max(self.times_spoken, 1)

    @property
    def composite_score(self) -> float:
        """加权综合得分"""
        return (
            self.mean_quality * 0.35 +
            self.mean_novelty * 0.25 +
            self.mean_influence * 0.25 +
            self.mean_engagement * 0.15
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "persona": self.persona,
            "persona_name": self.persona_name,
            "expertise_tags": list(self.expertise_tags),
            "times_spoken": self.times_spoken,
            "last_spoke_round": self.last_spoke_round,
            "rounds_since_last_spoke": self.rounds_since_last_spoke,
            "quality_total": self.quality_total,
            "novelty_total": self.novelty_total,
            "influence_total": self.influence_total,
            "engagement_total": self.engagement_total,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ExpertProfile':
        profile = cls.__new__(cls)
        profile.name = data["name"]
        profile.persona = data.get("persona", "")
        profile.persona_name = data.get("persona_name", "")
        profile.expertise_tags = set(data.get("expertise_tags", []))
        profile.times_spoken = data.get("times_spoken", 0)
        profile.last_spoke_round = data.get("last_spoke_round", 0)
        profile.rounds_since_last_spoke = data.get("rounds_since_last_spoke", 0)
        profile.quality_total = data.get("quality_total", 0.0)
        profile.novelty_total = data.get("novelty_total", 0.0)
        profile.influence_total = data.get("influence_total", 0.0)
        profile.engagement_total = data.get("engagement_total", 0.0)
        return profile


class ExpertScheduler:
    """
    智能专家调度器。

    核心算法（UCB 变体）：
        score = mean_composite + exploration_bonus + diversity_bonus + hunger_bonus

    其中：
        - mean_composite: 加权多维综合得分
        - exploration_bonus: C * sqrt(ln(N) / n_i)  (UCB 探索项)
        - diversity_bonus: 与已选专家最大相似度的负相关
        - hunger_bonus: 饥饿度 * 权重

    支持目标导向模式（goal_mode）：
        - balance: 默认，兼顾所有维度
        - converge: 加速收敛，倾向高共识专家
        - explore: 激发创新，倾向高争议/新颖专家
    """

    def __init__(self, exploration_factor: float = 1.5,
                 diversity_weight: float = 0.4,
                 hunger_weight: float = 0.5,
                 redundancy_threshold: float = 0.55,
                 goal_mode: str = "balance"):
        self.exploration_factor = exploration_factor
        self.diversity_weight = diversity_weight
        self.hunger_weight = hunger_weight
        self.redundancy_threshold = redundancy_threshold
        self.goal_mode = goal_mode

        self.round_count = 0
        # 累计共识度用于收敛模式判断
        self._consensus_history: List[float] = []
        self.profiles: Dict[str, ExpertProfile] = {}  # name -> profile

    def _get_profile(self, player: Player) -> ExpertProfile:
        """获取或创建专家的评估画像"""
        if player.name not in self.profiles:
            self.profiles[player.name] = ExpertProfile(
                name=player.name,
                persona=player.persona or "",
                persona_name=player.persona_name or "",
            )
        return self.profiles[player.name]

    def select_speakers(self, players: List[Player], round_count: int) -> List[Player]:
        """
        选择本轮发言的专家。

        第1轮：全部发言
        第2轮+：UCB 算法选择
        """
        self.round_count = round_count
        alive = [p for p in players if p.alive]

        # 确保所有存活玩家都有画像
        for p in alive:
            self._get_profile(p)

        # 第1轮：全部发言
        if round_count <= 1:
            return alive

        # 更新所有专家的画像（同步 round_since_last_spoke）
        for p in alive:
            profile = self._get_profile(p)
            profile.rounds_since_last_spoke = p.rounds_since_last_spoke

        # 计算每个专家的 UCB 分数
        scored: List[Tuple[Player, float]] = []
        for p in alive:
            profile = self._get_profile(p)
            ucb = self._ucb_score(profile, alive)
            scored.append((p, ucb))

        # 按 UCB 分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        # 自适应发言人数
        n_speakers = _calculate_n_speakers(len(alive))

        # 多样性约束选择
        selected = []
        for p, score in scored:
            if len(selected) >= n_speakers:
                break
            if not self._is_redundant(p, selected):
                selected.append(p)

        # 如果多样性约束导致人数不足，补充最高分专家
        if len(selected) < min(n_speakers, 3):
            for p, score in scored:
                if p not in selected:
                    selected.append(p)
                    if len(selected) >= min(n_speakers, 3):
                        break

        return selected

    def set_goal_mode(self, goal_mode: str) -> None:
        """动态切换目标导向模式"""
        valid = {"balance", "converge", "explore"}
        if goal_mode not in valid:
            raise ValueError(f"goal_mode 必须是 {valid} 之一，收到: {goal_mode}")
        self.goal_mode = goal_mode

    def _ucb_score(self, profile: ExpertProfile, all_alive: List[Player]) -> float:
        """计算 UCB 分数，根据 goal_mode 调整权重"""
        # 根据目标模式调整权重
        if self.goal_mode == "converge":
            quality_w = 0.50   # 质量权重更高
            novelty_w = 0.15   # 新颖权重降低
            influence_w = 0.20
            engagement_w = 0.15
            hunger_mult = 1.5  # 收敛模式优先让所有人发言
        elif self.goal_mode == "explore":
            quality_w = 0.20   # 质量权重降低
            novelty_w = 0.40   # 新颖权重更高
            influence_w = 0.20
            engagement_w = 0.20
            hunger_mult = 0.5  # 探索模式不急于让所有人发言
        else:  # balance
            quality_w = 0.35
            novelty_w = 0.25
            influence_w = 0.25
            engagement_w = 0.15
            hunger_mult = 1.0

        # 1. 综合得分（利用项）
        composite = (
            profile.mean_quality * quality_w +
            profile.mean_novelty * novelty_w +
            profile.mean_influence * influence_w +
            profile.mean_engagement * engagement_w
        )
        mean_score = composite if profile.times_spoken > 0 else 0.5

        # 2. UCB 探索项
        total_rounds = self.round_count
        if profile.times_spoken == 0:
            # 探索模式给新专家更高探索奖励
            exploration_bonus = self.exploration_factor * (3.0 if self.goal_mode == "explore" else 2.0)
        else:
            exploration_bonus = self.exploration_factor * math.sqrt(
                math.log(total_rounds + 1) / max(profile.times_spoken, 1)
            )

        # 3. 多样性奖励
        diversity_bonus = self._diversity_bonus(profile, all_alive)
        if self.goal_mode == "explore":
            diversity_bonus *= 1.5  # 探索模式鼓励多样性
        elif self.goal_mode == "converge":
            diversity_bonus *= 0.7  # 收敛模式降低多样性要求

        # 4. 饥饿度奖励
        hunger_bonus = min(profile.rounds_since_last_spoke, 5) * self.hunger_weight * hunger_mult

        return mean_score + exploration_bonus + diversity_bonus + hunger_bonus

    def _diversity_bonus(self, profile: ExpertProfile, all_alive: List[Player]) -> float:
        """
        多样性奖励：与所有存活专家的平均专业差异度。
        专家越独特，奖励越高（防止同类专家垄断发言权）。
        """
        if not all_alive:
            return 0.0

        total_sim = 0.0
        count = 0
        for p in all_alive:
            if p.name == profile.name:
                continue
            other = self._get_profile(p)
            sim = _expertise_similarity(profile.expertise_tags, other.expertise_tags)
            total_sim += sim
            count += 1

        avg_sim = total_sim / max(count, 1)
        # 相似度越低，多样性奖励越高
        return (1.0 - avg_sim) * self.diversity_weight

    def _is_redundant(self, player: Player, already_selected: List[Player]) -> bool:
        """
        判断某个专家是否与已选专家冗余（专业领域过度重叠）。
        只有当已选专家数量 >= 3 时才启用冗余过滤，保证基础多样性。
        """
        if len(already_selected) < 3:
            return False

        profile = self._get_profile(player)
        for selected in already_selected:
            sel_profile = self._get_profile(selected)
            sim = _expertise_similarity(profile.expertise_tags, sel_profile.expertise_tags)
            if sim > self.redundancy_threshold:
                return True
        return False

    def mark_as_spoken(self, player_names: List[str]):
        """标记本轮发言的专家，更新统计"""
        for name in player_names:
            if name in self.profiles:
                profile = self.profiles[name]
                profile.times_spoken += 1
                profile.last_spoke_round = self.round_count
                profile.rounds_since_last_spoke = 0

        # 更新所有未发言专家的饥饿度
        for name, profile in self.profiles.items():
            if name not in player_names:
                profile.rounds_since_last_spoke = self.round_count - profile.last_spoke_round

    def update_after_round(self, essence_pool, round_discussions: List[Dict]):
        """
        本轮结束后更新所有专家的多维评分。
        根据精华池引用关系和发言记录更新质量/新颖度/影响力/参与度。
        """
        # 1. 根据精华池更新质量和影响力
        for item in essence_pool.items:
            contributor = item.contributor
            if contributor in self.profiles:
                profile = self.profiles[contributor]
                # 质量：精华的基础评分
                profile.quality_total += max(item.score, 0.0)
                # 影响力：被引用的次数
                influence = len(item.cited_by) * 0.5 + len(item.refined_by) * 0.3
                profile.influence_total += min(influence, 3.0)

        # 2. 根据讨论记录更新新颖度和参与度
        for disc in round_discussions:
            name = disc.get("player_name", "")
            action = disc.get("action", "new")
            if name in self.profiles:
                profile = self.profiles[name]
                # 新颖度：提出新观点得高分
                profile.novelty_total += 1.0 if action == "new" else 0.3
                # 参与度：深化/反驳他人观点
                profile.engagement_total += 0.5 if action in ("refine", "challenge") else 0.1

    def select_debate_pair(self, players: List[Player],
                           essence_pool, round_count: int) -> Optional[Tuple]:
        """
        为本轮辩论挑选最对立的两方及辩论焦点。

        策略（按优先级）：
        1. 找到被反驳次数最多的精华 → challenger(任一反驳者) vs contributor
        2. 找到投票分歧最大的精华 → reject投票者 vs approve投票者/contributor
        3. 无明显对立时 → 选专业相似度最低的两名存活专家辩论评分最高精华

        受 goal_mode 影响：
        - converge: 降低辩论频率（只有争议度很高时才触发）
        - explore: 提高辩论频率（低争议度也可触发）
        """
        alive = [p for p in players if p.alive]
        alive_names = {p.name for p in alive}
        if len(alive) < 2:
            return None

        # 根据 goal_mode 设置争议度阈值
        if self.goal_mode == "converge":
            min_controversy = 3.0  # 收敛模式：高争议才辩论
        elif self.goal_mode == "explore":
            min_controversy = 0.5  # 探索模式：低争议也可辩论
        else:
            min_controversy = 1.0  # 平衡模式：默认阈值

        # 策略1：找被反驳最多的精华
        contested = []
        for item in essence_pool.items:
            if item.contributor not in alive_names:
                continue
            # 计算争议度 = 反驳数 + 反对票数*1.5
            controversy = len(item.challenged_by) + len(item.reject_by) * 1.5
            if controversy >= min_controversy:
                contested.append((controversy, item))

        if contested:
            contested.sort(key=lambda x: x[0], reverse=True)
            _, topic_item = contested[0]

            # attacker: 优先选反驳者中存活的，否则选反对票中存活的
            attacker_name = None
            for name in topic_item.challenged_by:
                if name in alive_names and name != topic_item.contributor:
                    attacker_name = name
                    break
            if not attacker_name:
                for name in topic_item.reject_by:
                    if name in alive_names and name != topic_item.contributor:
                        attacker_name = name
                        break

            if attacker_name:
                attacker = next(p for p in alive if p.name == attacker_name)
                defender = next(p for p in alive if p.name == topic_item.contributor)
                topic = {"id": topic_item.id, "content": topic_item.content,
                         "contributor": topic_item.contributor}
                return (attacker, defender, topic)

        # 策略3：无明显对立，选专业差异最大的两方辩论评分最高精华
        top_items = sorted(essence_pool.items, key=lambda x: x.score, reverse=True)
        topic_item = None
        for item in top_items:
            if item.contributor in alive_names:
                topic_item = item
                break

        if not topic_item:
            return None

        # 选专业相似度最低的两名存活专家
        defender = next((p for p in alive if p.name == topic_item.contributor), alive[0])
        candidates = [p for p in alive if p.name != defender.name]
        if not candidates:
            return None

        defender_profile = self._get_profile(defender)
        attacker = min(
            candidates,
            key=lambda p: _expertise_similarity(
                defender_profile.expertise_tags,
                self._get_profile(p).expertise_tags
            )
        )
        topic = {"id": topic_item.id, "content": topic_item.content,
                 "contributor": topic_item.contributor}
        return (attacker, defender, topic)

    def get_ranking(self) -> List[Tuple[str, float, str]]:
        """
        获取所有专家的排名列表。
        Returns: [(name, composite_score, summary_text), ...]
        """
        scored = []
        for name, profile in self.profiles.items():
            status = "now" if profile.rounds_since_last_spoke == 0 else f"wait{profile.rounds_since_last_spoke}"
            summary = (
                f"质量{profile.mean_quality:.1f} "
                f"新颖{profile.mean_novelty:.1f} "
                f"影响{profile.mean_influence:.1f} "
                f"参与{profile.mean_engagement:.1f} "
                f"发言{profile.times_spoken}次 {status}"
            )
            scored.append((name, profile.composite_score, summary))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── 动态行动计划 ──────────────────────────────────────────────────────────

    ACTION_TYPES = ["SPEECH", "FOLLOW_UP", "DEBATE", "SUMMARIZE", "PERSPECTIVE_SHIFT", "POLL"]

    def generate_action_plan(self, state: dict, players: List, essence_pool) -> List[dict]:
        """
        根据讨论状态生成动态行动计划。

        状态 → 动作映射：
          exploring      → 邀请沉默专家发言，降低发言门槛
          deep_debate    → 锁定热点话题，多轮交锋（A→B→A→B）
          converging     → 结构化提问，引导回答具体问题，加速投票
          stalled        → 魔鬼代言人，或建议用户换方向

        Args:
            state: 讨论状态报告字典（来自 Game._assess_state()）
            players: 所有玩家列表
            essence_pool: 精华池

        Returns:
            动作列表 [{"type": str, "players": [...], "topic": str, "rounds": int, ...}, ...]
        """
        phase = state.get("phase", "exploring")
        plan = []

        if phase == "exploring":
            plan = self._build_exploring_plan(state, players)
        elif phase == "deep_debate":
            plan = self._build_debate_plan(state, players, essence_pool)
        elif phase == "converging":
            plan = self._build_converging_plan(state, players, essence_pool)
        elif phase == "stalled":
            plan = self._build_stalled_plan(state, players, essence_pool)

        return plan

    def _build_exploring_plan(self, state: dict, players: List) -> List[dict]:
        """探索期行动计划：鼓励沉默专家发言，提出新视角"""
        plan = []
        silent = state.get("silent_players", [])
        dominant = state.get("dominant_players", [])

        # 先让沉默专家发言
        silent_players = [p for p in players if p.alive and p.name in silent]
        if silent_players:
            for p in silent_players[:2]:
                plan.append({
                    "type": "SPEECH",
                    "players": [p.name],
                    "topic": "请从你所擅长的专业视角提出新观点或补充现有讨论的盲区",
                    "rounds": 1,
                    "reason": f"邀请沉默专家 {p.name} 发言，增加观点多样性",
                })

        # 常规发言补充
        regular = [p for p in players if p.alive and p.name not in silent and p.name not in dominant]
        if regular:
            for p in regular[:3]:
                plan.append({
                    "type": "SPEECH",
                    "players": [p.name],
                    "topic": "请分享你的专业见解",
                    "rounds": 1,
                    "reason": "常规发言",
                })

        # 如果沉默专家不足，补充饥饿度高的专家
        if len(silent_players) < 2:
            hungry = sorted(
                [p for p in players if p.alive and p.name not in silent],
                key=lambda p: p.rounds_since_last_spoke, reverse=True
            )
            for p in hungry[:2]:
                if p.name not in [a["players"][0] for a in plan]:
                    plan.append({
                        "type": "SPEECH",
                        "players": [p.name],
                        "topic": "请分享你的专业见解",
                        "rounds": 1,
                        "reason": f"饥饿度({p.rounds_since_last_spoke}轮未发言)补充",
                    })

        return plan

    def _build_debate_plan(self, state: dict, players: List, essence_pool) -> List[dict]:
        """深入辩论期行动计划：锁定热点，多轮交锋"""
        plan = []
        hot_topics = state.get("hot_topics", [])

        if not hot_topics or not essence_pool.items:
            return self._build_exploring_plan(state, players)

        # 找争议最大的热点
        controversial = [t for t in hot_topics if t.get("reason") == "controversial"]
        target = controversial[0] if controversial else hot_topics[0]

        # 找持不同意见的专家
        target_item = None
        for item in essence_pool.items:
            if item.id == target["id"]:
                target_item = item
                break

        if target_item:
            # 赞成者 vs 反对者
            supporters = [p for p in players if p.alive and p.name in target_item.approve_by]
            opposers = [p for p in players if p.alive and p.name in target_item.reject_by]
            challengers = [p for p in players if p.alive and p.name in target_item.challenged_by]

            if supporters and (opposers or challengers):
                attacker = (opposers or challengers)[0]
                defender = supporters[0]
                plan.append({
                    "type": "DEBATE",
                    "players": [attacker.name, defender.name],
                    "topic": target["content"],
                    "rounds": 3,
                    "essence_id": target["id"],
                    "reason": f"针对争议精华 #{target['id']} 进行多轮辩论",
                })
            else:
                # 没有明确对立面，选饥饿度最高的专家发表意见
                hungry = sorted(
                    [p for p in players if p.alive],
                    key=lambda p: p.rounds_since_last_spoke, reverse=True
                )
                for p in hungry[:2]:
                    plan.append({
                        "type": "SPEECH",
                        "players": [p.name],
                        "topic": f"请就以下观点发表你的看法：{target['content'][:60]}",
                        "rounds": 1,
                        "reason": f"对热点话题发表意见",
                    })

        # 补充沉默专家发言
        silent = state.get("silent_players", [])
        for p in players:
            if p.alive and p.name in silent and not any(p.name in a["players"] for a in plan):
                plan.append({
                    "type": "SPEECH",
                    "players": [p.name],
                    "topic": "请从不同角度补充意见",
                    "rounds": 1,
                    "reason": f"沉默专家 {p.name} 补充发言",
                })

        return plan

    def _build_converging_plan(self, state: dict, players: List, essence_pool) -> List[dict]:
        """收敛期行动计划：结构化提问，引导回答具体问题，加速投票"""
        plan = []

        # 结构化收敛问题
        converge_questions = [
            "我们是否同意以下核心观点？请给出明确的同意或反对，并说明理由",
            "还有哪些关键反对意见未被充分讨论？",
            "如果必须总结当前讨论的 3 个核心结论，你会选择哪三个？",
        ]

        # 让饥饿度最高的 2 位专家回答收敛问题
        hungry = sorted(
            [p for p in players if p.alive],
            key=lambda p: p.rounds_since_last_spoke, reverse=True
        )
        for i, p in enumerate(hungry[:2]):
            plan.append({
                "type": "FOLLOW_UP",
                "players": [p.name],
                "topic": converge_questions[i % len(converge_questions)],
                "rounds": 1,
                "reason": f"收敛期结构化追问：{p.name}",
            })

        # 让得分最高的专家总结
        top = self.get_ranking()
        if top:
            top_name = top[0][0]
            top_player = next((p for p in players if p.alive and p.name == top_name), None)
            if top_player:
                plan.append({
                    "type": "SUMMARIZE",
                    "players": [top_name],
                    "topic": "请总结当前讨论的核心进展和达成的共识",
                    "rounds": 1,
                    "reason": f"由最高分专家 {top_name} 总结当前进展",
                })

        # 添加快速投票
        if essence_pool and essence_pool.items:
            pending = [it for it in essence_pool.items
                       if not it.approve_by and not it.reject_by]
            if pending:
                plan.append({
                    "type": "POLL",
                    "players": [p.name for p in players if p.alive],
                    "topic": f"对 {min(3, len(pending))} 条待投票精华进行快速投票",
                    "rounds": 1,
                    "essence_ids": [it.id for it in pending[:3]],
                    "reason": "收敛期加速投票",
                })

        return plan

    def _build_stalled_plan(self, state: dict, players: List, essence_pool) -> List[dict]:
        """僵持期行动计划：魔鬼代言人，强制换视角"""
        plan = []

        # 魔鬼代言人：让发言最多的专家从对立角度发言
        dominant = state.get("dominant_players", [])
        if dominant:
            devil = dominant[0]
            plan.append({
                "type": "PERSPECTIVE_SHIFT",
                "players": [devil],
                "topic": "请强制从你当前立场的完全对立面重新审视这个问题，提出 3 个你之前忽略的反方论点",
                "rounds": 1,
                "reason": f"魔鬼代言人：{devil} 强制换视角",
            })

        # 补充饥饿度高的专家
        hungry = sorted(
            [p for p in players if p.alive],
            key=lambda p: p.rounds_since_last_spoke, reverse=True
        )
        for p in hungry[:2]:
            if p.name not in [a["players"][0] for a in plan]:
                plan.append({
                    "type": "SPEECH",
                    "players": [p.name],
                    "topic": "请尝试从一个全新的角度分析当前问题，可以是任何之前未被提及的视角",
                    "rounds": 1,
                    "reason": f"僵持期引入新视角：{p.name}",
                })

        return plan

    # ── 扩展系统集成方法 ─────────────────────────────────────────────────────

    # 以下集成方法允许 ExpertScheduler 与扩展系统协同工作

    def integrate_optimizer(self, optimizer: 'MultiObjectiveOptimizer') -> None:
        """集成多目标优化器，用于调度决策的多目标权衡"""
        self._optimizer = optimizer

    def integrate_profiler(self, profiler: 'ExpertProfiler') -> None:
        """集成专家画像系统，提供更丰富的专家特征"""
        self._profiler = profiler

    def integrate_role_assigner(self, role_assigner: 'DynamicRoleAssigner') -> None:
        """集成动态角色分配器，为专家分配讨论角色"""
        self._role_assigner = role_assigner

    def integrate_predictor(self, predictor: 'PerformancePredictor') -> None:
        """集成性能预测器，预测专家未来表现"""
        self._predictor = predictor

    def integrate_dynamics_modeler(self, modeler: 'GroupDynamicsModeler') -> None:
        """集成群体动力学建模器，监控群体互动模式"""
        self._dynamics_modeler = modeler

    def integrate_phase_detector(self, detector: 'DiscussionPhaseDetector') -> None:
        """集成讨论阶段检测器，感知讨论阶段变化"""
        self._phase_detector = detector

    def integrate_turn_allocator(self, allocator: 'AdaptiveTurnAllocator') -> None:
        """集成自适应轮次分配器，优化发言轮次"""
        self._turn_allocator = allocator

    def integrate_gap_filler(self, gap_filler: 'ExpertiseGapFiller') -> None:
        """集成专业知识缺口填补器，检测知识盲区"""
        self._gap_filler = gap_filler

    def integrate_intervention_scheduler(self, scheduler: 'InterventionScheduler') -> None:
        """集成干预调度器，安排主持人干预"""
        self._intervention_scheduler = scheduler

    def integrate_learning_scheduler(self, learner: 'LearningScheduler') -> None:
        """集成学习型调度器，从历史中学习改进"""
        self._learner = learner

    def get_full_system_state(self) -> dict:
        """
        获取所有集成系统的完整状态摘要。
        仅在对应集成模块已注册时返回其状态。
        """
        state = {
            "scheduler": {
                "round_count": self.round_count,
                "goal_mode": self.goal_mode,
                "profiles_count": len(self.profiles),
                "exploration_factor": self.exploration_factor,
                "diversity_weight": self.diversity_weight,
                "hunger_weight": self.hunger_weight,
            }
        }
        if hasattr(self, '_optimizer'):
            state["optimizer"] = self._optimizer.get_weights()
        if hasattr(self, '_profiler'):
            state["profiler"] = {"profiles_count": len(self._profiler.profiles)}
        if hasattr(self, '_role_assigner'):
            state["role_assigner"] = self._role_assigner.get_role_distribution()
        if hasattr(self, '_predictor'):
            state["predictor"] = self._predictor.get_accuracy_stats()
        if hasattr(self, '_dynamics_modeler'):
            state["dynamics_modeler"] = self._dynamics_modeler.get_latest_patterns()
        if hasattr(self, '_phase_detector'):
            state["phase_detector"] = {"current_phase": self._phase_detector.current_phase}
        if hasattr(self, '_turn_allocator'):
            state["turn_allocator"] = self._turn_allocator.get_allocation_summary()
        if hasattr(self, '_gap_filler'):
            state["gap_filler"] = self._gap_filler.get_gap_summary()
        if hasattr(self, '_intervention_scheduler'):
            state["intervention_scheduler"] = self._intervention_scheduler.get_history_summary()
        if hasattr(self, '_learner'):
            state["learner"] = self._learner.get_learning_metrics()
        return state

    def to_dict(self) -> dict:
        return {
            "round_count": self.round_count,
            "profiles": {name: p.to_dict() for name, p in self.profiles.items()},
            "exploration_factor": self.exploration_factor,
            "diversity_weight": self.diversity_weight,
            "hunger_weight": self.hunger_weight,
            "redundancy_threshold": self.redundancy_threshold,
            "goal_mode": self.goal_mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ExpertScheduler':
        scheduler = cls(
            exploration_factor=data.get("exploration_factor", 1.5),
            diversity_weight=data.get("diversity_weight", 0.4),
            hunger_weight=data.get("hunger_weight", 0.5),
            redundancy_threshold=data.get("redundancy_threshold", 0.55),
            goal_mode=data.get("goal_mode", "balance"),
        )
        scheduler.round_count = data.get("round_count", 0)
        for name, pdata in data.get("profiles", {}).items():
            scheduler.profiles[name] = ExpertProfile.from_dict(pdata)
        return scheduler


# =============================================================================
# 扩展系统 1: MultiObjectiveOptimizer (多目标优化)
# =============================================================================

class MultiObjectiveOptimizer:
    """
    多目标优化器 —— 平衡多个调度目标。

    管理四个核心调度目标：
    - diversity:  最大化专家领域多样性
    - depth:      最大化讨论深度（高质量精华产出）
    - fairness:   确保所有专家获得公平发言机会
    - novelty:    优先考虑新颖/独特视角

    支持 Pareto 优化、加权目标评分、自适应权重调整和权衡分析。
    """

    OBJECTIVE_NAMES = ["diversity", "depth", "fairness", "novelty"]

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights: Dict[str, float] = weights or {
            "diversity": 0.30,
            "depth": 0.30,
            "fairness": 0.20,
            "novelty": 0.20,
        }
        self._normalize_weights()
        self._history: List[Dict[str, float]] = []  # 记录每次优化的目标分数
        self._pareto_front: List[Dict[str, float]] = []  # Pareto 前沿解集
        self._weight_adjustments: List[Dict[str, float]] = []
        self._adjustment_count = 0

    def _normalize_weights(self) -> None:
        """确保权重之和为 1.0"""
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    # ── 核心优化方法 ─────────────────────────────────────────────────────

    def optimize(self, candidates: List[Dict[str, float]],
                 previous_selections: Optional[List[str]] = None) -> List[int]:
        """
        对候选专家集合进行多目标优化排序。

        Args:
            candidates: 候选专家列表，每个元素为包含目标分数的字典
                        [{name, diversity, depth, fairness, novelty}, ...]
            previous_selections: 之前已选中的专家名称列表

        Returns:
            按综合评分降序排列的候选索引列表
        """
        if not candidates:
            return []

        scored: List[Tuple[float, int]] = []
        for idx, cand in enumerate(candidates):
            score = self.score_objectives(cand)
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked_indices = [idx for _, idx in scored]

        # 记录本次优化的目标分数
        record = {}
        if candidates:
            top_cand = candidates[ranked_indices[0]]
            for obj in self.OBJECTIVE_NAMES:
                record[obj] = top_cand.get(obj, 0.0)
        record["timestamp"] = len(self._history)
        self._history.append(record)

        # 更新 Pareto 前沿
        self._update_pareto_front(candidates)

        return ranked_indices

    def score_objectives(self, candidate: Dict[str, float]) -> float:
        """
        计算候选者的加权多目标综合得分。

        Args:
            candidate: 包含各目标分数的字典

        Returns:
            加权综合得分
        """
        total = 0.0
        for obj, weight in self.weights.items():
            value = candidate.get(obj, 0.0)
            total += value * weight
        return total

    # ── Pareto 优化 ──────────────────────────────────────────────────────

    def _update_pareto_front(self, candidates: List[Dict[str, float]]) -> None:
        """更新 Pareto 非支配解集"""
        if not candidates:
            return

        # 将候选者与现有 Pareto 前沿合并
        combined = self._pareto_front + candidates
        # 去重（基于目标值唯一性）
        unique = []
        seen = set()
        for c in combined:
            key = tuple(c.get(obj, 0.0) for obj in self.OBJECTIVE_NAMES)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # 计算非支配解
        self._pareto_front = []
        for i, c1 in enumerate(unique):
            dominated = False
            for j, c2 in enumerate(unique):
                if i == j:
                    continue
                # 检查 c2 是否支配 c1
                better_in_all = True
                strictly_better = False
                for obj in self.OBJECTIVE_NAMES:
                    v1 = c1.get(obj, 0.0)
                    v2 = c2.get(obj, 0.0)
                    if v2 < v1 - 1e-9:
                        better_in_all = False
                        break
                    if v2 > v1 + 1e-9:
                        strictly_better = True
                if better_in_all and strictly_better:
                    dominated = True
                    break
            if not dominated:
                self._pareto_front.append(c1)

        # 限制 Pareto 前沿大小，防止无限增长
        if len(self._pareto_front) > 50:
            # 按平均目标值排序，保留 top 50
            self._pareto_front.sort(
                key=lambda c: sum(c.get(obj, 0.0) for obj in self.OBJECTIVE_NAMES),
                reverse=True
            )
            self._pareto_front = self._pareto_front[:50]

    def get_pareto_front(self) -> List[Dict[str, float]]:
        """
        获取当前 Pareto 前沿解集。

        Returns:
            Pareto 非支配解列表
        """
        return list(self._pareto_front)

    def is_pareto_optimal(self, candidate: Dict[str, float]) -> bool:
        """
        检查一个候选解是否在 Pareto 前沿上。

        Args:
            candidate: 候选解的目标分数

        Returns:
            是否 Pareto 最优
        """
        for existing in self._pareto_front:
            dominated = True
            strictly_worse = False
            for obj in self.OBJECTIVE_NAMES:
                v_new = candidate.get(obj, 0.0)
                v_exist = existing.get(obj, 0.0)
                if v_new > v_exist + 1e-9:
                    dominated = False
                    break
                if v_new < v_exist - 1e-9:
                    strictly_worse = True
            if dominated and strictly_worse:
                return False
        return True

    # ── 权重调整 ─────────────────────────────────────────────────────────

    def adjust_weights(self, feedback: Dict[str, float]) -> Dict[str, float]:
        """
        根据反馈调整各目标权重。

        Args:
            feedback: 各目标的反馈评分（0.0 ~ 1.0），
                      低分表示该目标需要更多关注

        Returns:
            调整后的权重字典
        """
        self._adjustment_count += 1
        # 计算调整量：低反馈的目标获得更高权重
        adjustments = {}
        total_feedback = sum(feedback.values())
        if total_feedback > 0:
            for obj in self.OBJECTIVE_NAMES:
                fb = feedback.get(obj, 0.5)
                # 反馈越低，调整系数越大
                adjustments[obj] = (1.0 - fb) / (len(self.OBJECTIVE_NAMES) - 1 + 1e-9)
        else:
            for obj in self.OBJECTIVE_NAMES:
                adjustments[obj] = 0.0

        # 应用调整（平滑更新）
        learning_rate = 0.3
        for obj in self.OBJECTIVE_NAMES:
            self.weights[obj] = self.weights[obj] * (1 - learning_rate) + adjustments[obj] * learning_rate

        self._normalize_weights()
        self._weight_adjustments.append(dict(self.weights))
        return dict(self.weights)

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重设置"""
        return dict(self.weights)

    # ── 权衡分析 ─────────────────────────────────────────────────────────

    def get_tradeoffs(self, candidate_a: Dict[str, float],
                      candidate_b: Dict[str, float]) -> Dict[str, Any]:
        """
        分析两个候选方案之间的权衡关系。

        Args:
            candidate_a: 方案 A 的目标分数
            candidate_b: 方案 B 的目标分数

        Returns:
            权衡分析报告，包含各维度的优劣对比
        """
        tradeoffs = {
            "a_better": [],
            "b_better": [],
            "equal": [],
            "a_score": self.score_objectives(candidate_a),
            "b_score": self.score_objectives(candidate_b),
            "difference": 0.0,
        }
        for obj in self.OBJECTIVE_NAMES:
            va = candidate_a.get(obj, 0.0)
            vb = candidate_b.get(obj, 0.0)
            if va > vb + 1e-9:
                tradeoffs["a_better"].append({"objective": obj, "a_value": va, "b_value": vb})
            elif vb > va + 1e-9:
                tradeoffs["b_better"].append({"objective": obj, "a_value": va, "b_value": vb})
            else:
                tradeoffs["equal"].append({"objective": obj, "value": va})

        tradeoffs["difference"] = tradeoffs["a_score"] - tradeoffs["b_score"]
        return tradeoffs

    def get_optimization_history(self) -> List[Dict[str, float]]:
        """获取优化历史记录"""
        return list(self._history)

    def get_weight_adjustment_history(self) -> List[Dict[str, float]]:
        """获取权重调整历史"""
        return list(self._weight_adjustments)

    def to_dict(self) -> dict:
        return {
            "weights": dict(self.weights),
            "history_length": len(self._history),
            "pareto_front_size": len(self._pareto_front),
            "adjustment_count": self._adjustment_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MultiObjectiveOptimizer':
        optimizer = cls(weights=data.get("weights"))
        optimizer._history = data.get("history", [])
        # Pareto front 需要重建（因为存储的是引用）
        optimizer._pareto_front = data.get("pareto_front", [])
        optimizer._adjustment_count = data.get("adjustment_count", 0)
        return optimizer


# =============================================================================
# 扩展系统 2: ExpertProfiler (专家画像系统)
# =============================================================================

class ExpertProfiler:
    """
    专家画像系统 —— 构建专家行为综合画像。

    画像维度：
    - expertise:  专业领域覆盖范围和深度
    - style:      发言风格（分析型/创造型/批判型/综合型）
    - reliability: 贡献稳定性和可靠性
    - contribution: 贡献模式（发起者/回应者/挑战者/总结者）

    支持画像演变追踪、相似度比较和聚类查找。
    """

    # 发言风格类型
    STYLES = ["analytical", "creative", "critical", "synthesizing"]

    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}  # name -> 完整画像
        self._profile_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # 画像演变历史
        self._max_history = 50  # 每个画像最多保留的历史快照数

    # ── 画像构建与更新 ──────────────────────────────────────────────────

    def build_profile(self, name: str, discourse_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从行为记录构建专家画像。

        Args:
            name: 专家名称
            discourse_records: 该专家的历史发言记录列表

        Returns:
            完整的专家画像
        """
        if not discourse_records:
            return self._create_empty_profile(name)

        # 提取各维度特征
        expertise = self._analyze_expertise(name, discourse_records)
        style = self._analyze_style(discourse_records)
        reliability = self._analyze_reliability(discourse_records)
        contribution = self._analyze_contribution_pattern(discourse_records)

        profile = {
            "name": name,
            "expertise": expertise,
            "style": style,
            "reliability": reliability,
            "contribution": contribution,
            "summary": self._generate_summary(expertise, style, reliability, contribution),
            "record_count": len(discourse_records),
            "last_updated": len(self._profile_history.get(name, [])),
        }

        self.profiles[name] = profile
        self._save_history_snapshot(name, profile)
        return profile

    def _create_empty_profile(self, name: str) -> Dict[str, Any]:
        """创建空画像（无历史数据时使用）"""
        profile = {
            "name": name,
            "expertise": {"tags": [], "breadth": 0.0, "depth": 0.0},
            "style": {s: 0.0 for s in self.STYLES},
            "reliability": {"consistency": 0.5, "volatility": 0.0, "trend": "stable"},
            "contribution": {"initiator": 0.0, "responder": 0.0, "challenger": 0.0, "synthesizer": 0.0},
            "summary": "暂无足够数据构建画像",
            "record_count": 0,
            "last_updated": len(self._profile_history.get(name, [])),
        }
        self.profiles[name] = profile
        return profile

    def _analyze_expertise(self, name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析专业领域特征"""
        tags = set()
        for rec in records:
            text = rec.get("content", "") + rec.get("persona", "")
            extracted = _extract_expertise_tags(text)
            tags.update(extracted)

        # 计算广度：独特标签数
        breadth = min(len(tags) / 10.0, 1.0) if tags else 0.0

        # 计算深度：同一标签出现的频率集中度
        if tags:
            tag_counts = Counter()
            for rec in records:
                text = rec.get("content", "")
                for kw in EXPERTISE_KEYWORDS:
                    if kw in text:
                        tag_counts[kw] += 1
            total = sum(tag_counts.values())
            if total > 0:
                # 使用 Herfindahl 指数衡量集中度
                depth = sum((c / total) ** 2 for c in tag_counts.values())
                depth = min(depth * 3.0, 1.0)  # 归一化
            else:
                depth = 0.0
        else:
            depth = 0.0

        return {"tags": sorted(tags), "breadth": breadth, "depth": depth}

    def _analyze_style(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """分析发言风格"""
        style_scores = {s: 0.0 for s in self.STYLES}
        if not records:
            return style_scores

        for rec in records:
            action = rec.get("action", "new")
            content = rec.get("content", "")
            content_len = len(content)

            # analytical: 逻辑结构清晰，使用数据/证据
            if any(word in content for word in ["因为", "所以", "因此", "数据", "证据", "分析", "统计"]):
                style_scores["analytical"] += 1.0
            # creative: 提出新概念，类比，假设
            if any(word in content for word in ["假设", "想象", "类比", "可能", "或许", "创新", "全新"]):
                style_scores["creative"] += 1.0
            # critical: 质疑，反驳，指出不足
            if action == "challenge" or any(word in content for word in ["问题", "不足", "缺陷", "质疑", "反驳", "矛盾"]):
                style_scores["critical"] += 1.0
            # synthesizing: 总结，整合，连接不同观点
            if action == "refine" or any(word in content for word in ["总结", "整合", "综合", "结合", "联系", "融合"]):
                style_scores["synthesizing"] += 1.0

        # 归一化
        total = sum(style_scores.values()) or 1.0
        for s in self.STYLES:
            style_scores[s] = style_scores[s] / total
        return style_scores

    def _analyze_reliability(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析可靠性和稳定性"""
        if len(records) < 2:
            return {"consistency": 0.5, "volatility": 0.0, "trend": "stable"}

        # 计算质量得分的波动性
        quality_scores = []
        for rec in records:
            q = rec.get("quality", 0.0) or 0.0
            quality_scores.append(q)

        # 波动性 = 标准差 / 均值
        mean_q = statistics.mean(quality_scores) if quality_scores else 0.5
        if mean_q > 0 and len(quality_scores) > 1:
            volatility = statistics.stdev(quality_scores) / mean_q
            volatility = min(volatility, 1.0)
        else:
            volatility = 0.0

        # 一致性 = 1 - 波动性
        consistency = 1.0 - volatility

        # 趋势判断
        if len(quality_scores) >= 3:
            half = len(quality_scores) // 2
            first_half = statistics.mean(quality_scores[:half])
            second_half = statistics.mean(quality_scores[half:])
            if second_half > first_half + 0.1:
                trend = "improving"
            elif second_half < first_half - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {"consistency": consistency, "volatility": volatility, "trend": trend}

    def _analyze_contribution_pattern(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """分析贡献模式"""
        pattern = {"initiator": 0.0, "responder": 0.0, "challenger": 0.0, "synthesizer": 0.0}
        if not records:
            return pattern

        total = 0
        for rec in records:
            action = rec.get("action", "new")
            if action == "new":
                pattern["initiator"] += 1.0
            elif action == "refine":
                pattern["synthesizer"] += 1.0
            elif action == "challenge":
                pattern["challenger"] += 1.0
            else:
                pattern["responder"] += 1.0
            total += 1

        if total > 0:
            for k in pattern:
                pattern[k] /= total
        return pattern

    def _generate_summary(self, expertise: Dict, style: Dict,
                          reliability: Dict, contribution: Dict) -> str:
        """生成画像摘要文本"""
        parts = []
        # 专业领域
        tags = expertise.get("tags", [])
        if tags:
            parts.append(f"领域:{','.join(tags[:5])}")
        else:
            parts.append("领域:待定")

        # 主要风格
        main_style = max(style, key=style.get)
        style_names = {"analytical": "分析型", "creative": "创造型", "critical": "批判型", "synthesizing": "综合型"}
        parts.append(f"风格:{style_names.get(main_style, main_style)}")

        # 可靠性
        parts.append(f"可靠:{reliability.get('trend', 'stable')}")

        # 主要贡献模式
        main_contrib = max(contribution, key=contribution.get)
        contrib_names = {"initiator": "发起者", "responder": "回应者", "challenger": "挑战者", "synthesizer": "综合者"}
        parts.append(f"角色:{contrib_names.get(main_contrib, main_contrib)}")

        return " | ".join(parts)

    def _save_history_snapshot(self, name: str, profile: Dict[str, Any]) -> None:
        """保存画像历史快照"""
        snapshot = {
            "timestamp": len(self._profile_history[name]),
            "expertise_depth": profile["expertise"]["depth"],
            "expertise_breadth": profile["expertise"]["breadth"],
            "reliability_consistency": profile["reliability"]["consistency"],
            "style_dominant": max(profile["style"], key=profile["style"].get),
        }
        self._profile_history[name].append(snapshot)
        if len(self._profile_history[name]) > self._max_history:
            self._profile_history[name] = self._profile_history[name][-self._max_history:]

    # ── 画像查询 ────────────────────────────────────────────────────────

    def update_profile(self, name: str, new_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用新记录更新现有画像。

        Args:
            name: 专家名称
            new_records: 新增的发言记录

        Returns:
            更新后的画像
        """
        if name not in self.profiles:
            return self.build_profile(name, new_records)

        # 合并新旧记录
        existing = self.profiles[name]
        total_records = existing.get("record_count", 0)
        # 构建完整记录列表（模拟）
        all_records = list(new_records)
        # 为保持一致性，使用现有画像数据补充
        return self.build_profile(name, all_records)

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """获取指定专家的画像"""
        return self.profiles.get(name)

    # ── 画像比较 ────────────────────────────────────────────────────────

    def compare_profiles(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """
        比较两个专家的画像。

        Args:
            name_a: 专家 A 名称
            name_b: 专家 B 名称

        Returns:
            比较结果，包含各维度的相似度
        """
        pa = self.profiles.get(name_a)
        pb = self.profiles.get(name_b)
        if not pa or not pb:
            return {"error": "一个或两个专家不存在", "similarity": 0.0}

        # 专业相似度
        tags_a = set(pa["expertise"].get("tags", []))
        tags_b = set(pb["expertise"].get("tags", []))
        expertise_sim = _expertise_similarity(tags_a, tags_b)

        # 风格相似度（余弦相似度）
        style_a = pa["style"]
        style_b = pb["style"]
        dot = sum(style_a.get(s, 0.0) * style_b.get(s, 0.0) for s in self.STYLES)
        norm_a = math.sqrt(sum(v ** 2 for v in style_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in style_b.values()))
        style_sim = dot / (norm_a * norm_b + 1e-9)

        # 贡献模式相似度
        ca = pa["contribution"]
        cb = pb["contribution"]
        dot_c = sum(ca.get(k, 0.0) * cb.get(k, 0.0) for k in ca)
        norm_ca = math.sqrt(sum(v ** 2 for v in ca.values()))
        norm_cb = math.sqrt(sum(v ** 2 for v in cb.values()))
        contrib_sim = dot_c / (norm_ca * norm_cb + 1e-9)

        overall = (expertise_sim * 0.4 + style_sim * 0.3 + contrib_sim * 0.3)

        return {
            "name_a": name_a,
            "name_b": name_b,
            "overall_similarity": overall,
            "expertise_similarity": expertise_sim,
            "style_similarity": style_sim,
            "contribution_similarity": contrib_sim,
        }

    def find_similar(self, name: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        查找与指定专家最相似的专家。

        Args:
            name: 目标专家名称
            top_n: 返回的最相似专家数量

        Returns:
            [(相似专家名称, 相似度), ...] 按相似度降序排列
        """
        if name not in self.profiles:
            return []

        results = []
        for other_name in self.profiles:
            if other_name == name:
                continue
            comp = self.compare_profiles(name, other_name)
            if "error" not in comp:
                results.append((other_name, comp["overall_similarity"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def get_profile_evolution(self, name: str) -> List[Dict[str, Any]]:
        """获取指定专家的画像演变历史"""
        return list(self._profile_history.get(name, []))

    def to_dict(self) -> dict:
        return {
            "profiles": {
                name: {k: v for k, v in p.items() if k != "summary"}
                for name, p in self.profiles.items()
            },
            "profile_count": len(self.profiles),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ExpertProfiler':
        profiler = cls()
        profiler.profiles = data.get("profiles", {})
        return profiler


# =============================================================================
# 扩展系统 3: DynamicRoleAssigner (动态角色分配)
# =============================================================================

class DynamicRoleAssigner:
    """
    动态角色分配器 —— 基于专家画像和讨论需求分配角色。

    角色类型：
    - facilitator:     主持人/引导者，把控讨论方向
    - devil_advocate:  魔鬼代言人，提出反对意见
    - summarizer:      总结者，归纳讨论要点
    - explorer:        探索者，提出新方向
    - critic:          批评者，评估观点质量
    - bridge:          桥梁者，连接不同观点

    支持角色轮换、缺口填补和分布分析。
    """

    ROLES = ["facilitator", "devil_advocate", "summarizer", "explorer", "critic", "bridge"]

    # 角色与画风类型的最佳匹配
    ROLE_STYLE_MAP = {
        "facilitator": ["synthesizing", "analytical"],
        "devil_advocate": ["critical", "creative"],
        "summarizer": ["synthesizing", "analytical"],
        "explorer": ["creative", "analytical"],
        "critic": ["critical", "analytical"],
        "bridge": ["synthesizing", "creative"],
    }

    def __init__(self, rotation_interval: int = 3):
        self.rotation_interval = rotation_interval  # 每多少轮进行一次角色轮换
        self.current_roles: Dict[str, str] = {}  # name -> role
        self._role_history: List[Dict[str, str]] = []  # 每轮的角色分配记录
        self._round_count = 0
        self._role_expertise_map: Dict[str, Set[str]] = defaultdict(set)  # role -> expertise tags

    # ── 角色分配 ────────────────────────────────────────────────────────

    def assign_roles(self, profiles: Dict[str, Dict[str, Any]],
                     phase: str = "exploring",
                     force_rotate: bool = False) -> Dict[str, str]:
        """
        基于专家画像和当前阶段分配角色。

        Args:
            profiles: 专家画像字典 {name: profile}
            phase: 当前讨论阶段
            force_rotate: 是否强制轮换

        Returns:
            {专家名称: 角色} 映射
        """
        self._round_count += 1
        # 检查是否需要轮换
        if self.current_roles and not force_rotate:
            if self._round_count % self.rotation_interval != 0:
                return dict(self.current_roles)

        expert_names = list(profiles.keys())
        if len(expert_names) < len(self.ROLES):
            # 专家数量少于角色数，每人分配多角色（主角色 + 副角色）
            return self._assign_roles_insufficient(expert_names, profiles, phase)

        # 为每个角色选择最匹配的专家
        role_scores: Dict[str, List[Tuple[str, float]]] = {role: [] for role in self.ROLES}
        for name in expert_names:
            profile = profiles.get(name, {})
            style = profile.get("style", {})
            for role in self.ROLES:
                score = self._compute_role_fitness(name, profile, role, phase)
                role_scores[role].append((name, score))

        # 贪心分配：每个角色选最优且未被分配的人
        assigned: Dict[str, str] = {}
        assigned_names: Set[str] = set()
        # 按角色优先级排序（根据阶段调整）
        role_priority = self._get_role_priority(phase)
        for role in role_priority:
            candidates = sorted(role_scores[role], key=lambda x: x[1], reverse=True)
            for name, score in candidates:
                if name not in assigned_names:
                    assigned[name] = role
                    assigned_names.add(name)
                    # 更新角色-专业映射
                    tags = set(profile.get("expertise", {}).get("tags", []))
                    self._role_expertise_map[role].update(tags)
                    break

        # 如果还有未分配的专家，分配最适合的剩余角色
        remaining = [n for n in expert_names if n not in assigned_names]
        if remaining:
            all_roles = list(self.ROLES)
            for name in remaining:
                # 找当前分配最少的角色
                role_counts = Counter(assigned.values())
                least_role = min(all_roles, key=lambda r: role_counts.get(r, 0))
                assigned[name] = least_role

        self.current_roles = assigned
        self._role_history.append(dict(assigned))
        return dict(assigned)

    def _assign_roles_insufficient(self, names: List[str], profiles: Dict,
                                   phase: str) -> Dict[str, str]:
        """专家不足时的角色分配（每人承担多个角色）"""
        assigned: Dict[str, str] = {}
        role_priority = self._get_role_priority(phase)
        for i, name in enumerate(names):
            # 主角色
            primary_role = role_priority[i % len(role_priority)]
            assigned[name] = primary_role
        self.current_roles = assigned
        return dict(assigned)

    def _compute_role_fitness(self, name: str, profile: Dict[str, Any],
                              role: str, phase: str) -> float:
        """计算专家对某个角色的适配度"""
        style = profile.get("style", {})
        contribution = profile.get("contribution", {})
        reliability = profile.get("reliability", {})

        # 风格匹配
        preferred_styles = self.ROLE_STYLE_MAP.get(role, [])
        style_score = sum(style.get(s, 0.0) for s in preferred_styles) / max(len(preferred_styles), 1)

        # 贡献模式匹配
        contrib_scores = {
            "facilitator": contribution.get("synthesizer", 0.0),
            "devil_advocate": contribution.get("challenger", 0.0),
            "summarizer": contribution.get("synthesizer", 0.0),
            "explorer": contribution.get("initiator", 0.0),
            "critic": contribution.get("challenger", 0.0),
            "bridge": contribution.get("synthesizer", 0.0),
        }
        contrib_score = contrib_scores.get(role, 0.0)

        # 可靠性加分
        reliability_score = reliability.get("consistency", 0.5)

        # 根据阶段调整
        phase_multipliers = {
            "exploring": {"explorer": 1.5, "facilitator": 1.2},
            "deep_debate": {"devil_advocate": 1.5, "critic": 1.3},
            "converging": {"summarizer": 1.5, "bridge": 1.3},
            "stalled": {"explorer": 1.4, "devil_advocate": 1.3},
        }
        mult = phase_multipliers.get(phase, {}).get(role, 1.0)

        # 避免重复分配同一角色
        rotation_penalty = 0.0
        if name in self.current_roles and self.current_roles[name] == role:
            rotation_penalty = 0.2

        return (style_score * 0.4 + contrib_score * 0.3 + reliability_score * 0.3) * mult - rotation_penalty

    def _get_role_priority(self, phase: str) -> List[str]:
        """根据讨论阶段返回角色优先级"""
        priorities = {
            "exploring": ["explorer", "facilitator", "bridge", "critic", "devil_advocate", "summarizer"],
            "deep_debate": ["devil_advocate", "critic", "facilitator", "explorer", "bridge", "summarizer"],
            "converging": ["summarizer", "bridge", "facilitator", "critic", "explorer", "devil_advocate"],
            "stalled": ["explorer", "devil_advocate", "facilitator", "bridge", "critic", "summarizer"],
        }
        return priorities.get(phase, self.ROLES)

    # ── 角色轮换 ────────────────────────────────────────────────────────

    def rotate_roles(self, profiles: Dict[str, Dict[str, Any]],
                     phase: str = "exploring") -> Dict[str, str]:
        """
        强制轮换所有角色。

        Args:
            profiles: 专家画像
            phase: 当前讨论阶段

        Returns:
            轮换后的角色分配
        """
        return self.assign_roles(profiles, phase, force_rotate=True)

    # ── 缺口填补 ────────────────────────────────────────────────────────

    def fill_gaps(self, profiles: Dict[str, Dict[str, Any]],
                  missing_role: str, phase: str = "exploring") -> Optional[str]:
        """
        为特定角色缺口找到最佳填补专家。

        Args:
            profiles: 专家画像
            missing_role: 需要填补的角色
            phase: 当前讨论阶段

        Returns:
            最佳填补专家名称，若无可填补则返回 None
        """
        if missing_role not in self.ROLES:
            return None

        unassigned = [n for n in profiles if n not in self.current_roles]
        if not unassigned:
            # 所有专家都已分配角色，找最接近该角色的人
            candidates = list(profiles.keys())
        else:
            candidates = unassigned

        best_name = None
        best_score = -1.0
        for name in candidates:
            profile = profiles.get(name, {})
            score = self._compute_role_fitness(name, profile, missing_role, phase)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name:
            self.current_roles[best_name] = missing_role
        return best_name

    # ── 查询 ────────────────────────────────────────────────────────────

    def get_role_distribution(self) -> Dict[str, List[str]]:
        """
        获取当前角色分布。

        Returns:
            {角色: [专家名称列表]} 映射
        """
        distribution: Dict[str, List[str]] = {role: [] for role in self.ROLES}
        for name, role in self.current_roles.items():
            if role in distribution:
                distribution[role].append(name)
        return distribution

    def get_role(self, name: str) -> Optional[str]:
        """获取指定专家的当前角色"""
        return self.current_roles.get(name)

    def get_role_history(self) -> List[Dict[str, str]]:
        """获取角色分配历史"""
        return list(self._role_history)

    def to_dict(self) -> dict:
        return {
            "current_roles": dict(self.current_roles),
            "rotation_interval": self.rotation_interval,
            "round_count": self._round_count,
            "history_length": len(self._role_history),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DynamicRoleAssigner':
        assigner = cls(rotation_interval=data.get("rotation_interval", 3))
        assigner.current_roles = data.get("current_roles", {})
        assigner._round_count = data.get("round_count", 0)
        return assigner


# =============================================================================
# 扩展系统 4: PerformancePredictor (性能预测)
# =============================================================================

class PerformancePredictor:
    """
    性能预测器 —— 预测专家在后续轮次中的表现。

    基于以下因素进行预测：
    - 历史表现趋势（加权移动平均）
    - 当前话题与专家专业领域相关性
    - 近期表现趋势（近期上升/下降）
    - 参与度和活跃度

    提供置信度评分和预测准确性评估。
    """

    def __init__(self, window_size: int = 5, alpha: float = 0.3):
        self.window_size = window_size  # 移动平均窗口
        self.alpha = alpha  # 指数平滑系数
        self._performance_history: Dict[str, List[float]] = defaultdict(list)  # 历史表现
        self._predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # 预测记录
        self._accuracy_history: List[float] = []  # 预测准确率历史
        self._model_version = 0

    # ── 核心预测方法 ────────────────────────────────────────────────────

    def predict(self, name: str, current_profile: Dict[str, Any],
                topic_tags: Optional[Set[str]] = None,
                round_number: int = 0) -> Dict[str, Any]:
        """
        预测专家在下一轮的表现。

        Args:
            name: 专家名称
            current_profile: 当前专家画像
            topic_tags: 当前讨论话题的专业标签
            round_number: 当前轮次

        Returns:
            预测结果，包含预测分数、置信度和因素分解
        """
        # 1. 历史趋势预测
        history = self._performance_history.get(name, [])
        if len(history) >= 2:
            trend_score = self._exponential_smoothing(history)
            trend_confidence = min(len(history) / self.window_size, 1.0)
        elif len(history) == 1:
            trend_score = history[0]
            trend_confidence = 0.3
        else:
            # 无历史数据，使用画像中的 composite_score
            trend_score = 0.5
            trend_confidence = 0.1

        # 2. 话题相关性
        if topic_tags and "expertise" in current_profile:
            expert_tags = set(current_profile["expertise"].get("tags", []))
            relevance = _expertise_similarity(expert_tags, topic_tags)
        else:
            relevance = 0.5

        # 3. 近期趋势
        if len(history) >= 3:
            recent = history[-3:]
            recent_trend = "up" if recent[-1] > recent[0] else ("down" if recent[-1] < recent[0] else "stable")
            recent_slope = (recent[-1] - recent[0]) / max(len(recent), 1)
        else:
            recent_trend = "stable"
            recent_slope = 0.0

        # 4. 参与度调整
        engagement = 0.0
        if "contribution" in current_profile:
            contrib = current_profile["contribution"]
            engagement = sum(contrib.values()) / max(len(contrib), 1)

        # 综合预测
        predicted_score = (
            trend_score * 0.35 +
            relevance * 0.25 +
            engagement * 0.20 +
            max(0, recent_slope) * 0.10 +
            0.10  # 基础分
        )
        predicted_score = max(0.0, min(1.0, predicted_score))

        # 置信度
        confidence = (
            trend_confidence * 0.40 +
            min(len(history) / 10, 1.0) * 0.30 +
            (1.0 if topic_tags else 0.5) * 0.15 +
            0.15
        )
        confidence = max(0.0, min(1.0, confidence))

        prediction = {
            "name": name,
            "predicted_score": predicted_score,
            "confidence": confidence,
            "factors": {
                "trend_score": trend_score,
                "topic_relevance": relevance,
                "recent_trend": recent_trend,
                "recent_slope": recent_slope,
                "engagement": engagement,
            },
            "round_number": round_number,
        }

        self._predictions[name].append(prediction)
        return prediction

    def _exponential_smoothing(self, history: List[float]) -> float:
        """指数平滑预测"""
        if not history:
            return 0.5
        smoothed = history[0]
        for value in history[1:]:
            smoothed = self.alpha * value + (1 - self.alpha) * smoothed
        return smoothed

    # ── 预测查询 ────────────────────────────────────────────────────────

    def get_prediction(self, name: str, round_number: int = 0) -> Optional[Dict[str, Any]]:
        """
        获取指定专家的最新预测。

        Args:
            name: 专家名称
            round_number: 指定轮次（0 表示最新）

        Returns:
            预测结果，若无则返回 None
        """
        predictions = self._predictions.get(name, [])
        if not predictions:
            return None
        if round_number <= 0:
            return predictions[-1]
        for pred in reversed(predictions):
            if pred.get("round_number", 0) <= round_number:
                return pred
        return predictions[0]

    # ── 模型更新 ────────────────────────────────────────────────────────

    def update_model(self, name: str, actual_score: float, round_number: int) -> None:
        """
        使用实际表现更新预测模型。

        Args:
            name: 专家名称
            actual_score: 实际表现得分
            round_number: 当前轮次
        """
        self._performance_history[name].append(actual_score)

        # 限制历史长度
        if len(self._performance_history[name]) > self.window_size * 2:
            self._performance_history[name] = self._performance_history[name][-self.window_size * 2:]

        self._model_version += 1

        # 计算预测准确率
        predictions = self._predictions.get(name, [])
        if predictions:
            latest_pred = predictions[-1]
            pred_score = latest_pred.get("predicted_score", 0.5)
            error = abs(pred_score - actual_score)
            accuracy = max(0.0, 1.0 - error)
            self._accuracy_history.append(accuracy)
            # 记录实际结果
            latest_pred["actual_score"] = actual_score
            latest_pred["error"] = error
            latest_pred["accuracy"] = accuracy

    # ── 准确性评估 ──────────────────────────────────────────────────────

    def evaluate_accuracy(self) -> Dict[str, float]:
        """
        评估预测模型的整体准确性。

        Returns:
            准确率统计：均值、中位数、标准差、趋势
        """
        if not self._accuracy_history:
            return {"mean_accuracy": 0.0, "median_accuracy": 0.0, "std_accuracy": 0.0, "trend": "unknown"}

        mean_acc = statistics.mean(self._accuracy_history)
        median_acc = statistics.median(self._accuracy_history)
        std_acc = statistics.stdev(self._accuracy_history) if len(self._accuracy_history) > 1 else 0.0

        # 趋势判断
        if len(self._accuracy_history) >= 5:
            recent = self._accuracy_history[-5:]
            older = self._accuracy_history[-10:-5] if len(self._accuracy_history) >= 10 else self._accuracy_history[:-5]
            if older:
                trend = "improving" if statistics.mean(recent) > statistics.mean(older) + 0.05 else \
                        ("declining" if statistics.mean(recent) < statistics.mean(older) - 0.05 else "stable")
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "mean_accuracy": mean_acc,
            "median_accuracy": median_acc,
            "std_accuracy": std_acc,
            "trend": trend,
            "samples": len(self._accuracy_history),
        }

    def get_accuracy_stats(self) -> Dict[str, float]:
        """获取准确率统计的快捷方法"""
        return self.evaluate_accuracy()

    def get_prediction_history(self, name: str) -> List[Dict[str, Any]]:
        """获取指定专家的预测历史"""
        return list(self._predictions.get(name, []))

    def to_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "alpha": self.alpha,
            "model_version": self._model_version,
            "experts_tracked": len(self._performance_history),
            "predictions_made": sum(len(v) for v in self._predictions.values()),
            "accuracy": self.evaluate_accuracy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PerformancePredictor':
        predictor = cls(
            window_size=data.get("window_size", 5),
            alpha=data.get("alpha", 0.3),
        )
        predictor._model_version = data.get("model_version", 0)
        return predictor


# =============================================================================
# 扩展系统 5: GroupDynamicsModeler (群体动力学建模)
# =============================================================================

class GroupDynamicsModeler:
    """
    群体动力学建模器 —— 建模和分析群体互动模式。

    检测模式：
    - dominance:   某个专家主导讨论，压制其他声音
    - withdrawal:  部分专家退缩，参与度下降
    - alignment:   群体过度趋同，缺乏多样性
    - polarization: 群体分化成对立阵营
    - fragmentation: 讨论碎片化，缺乏连贯性

    测量群体凝聚力、互动密度和角色平衡度。
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size  # 分析窗口大小
        self._round_records: List[Dict[str, Any]] = []  # 每轮讨论记录
        self._pattern_history: List[Dict[str, Any]] = []  # 检测到的模式历史
        self._cohesion_history: List[float] = []  # 凝聚力历史
        self._interaction_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # 互动图

    # ── 核心建模方法 ────────────────────────────────────────────────────

    def model_dynamics(self, round_discussions: List[Dict[str, Any]],
                       round_number: int) -> Dict[str, Any]:
        """
        建模当前轮次的群体动力学。

        Args:
            round_discussions: 本轮讨论记录
            round_number: 当前轮次

        Returns:
            动力学分析报告
        """
        record = {
            "round": round_number,
            "discussions": round_discussions,
            "participants": self._extract_participants(round_discussions),
            "interactions": self._extract_interactions(round_discussions),
        }
        self._round_records.append(record)
        if len(self._round_records) > self.window_size * 2:
            self._round_records = self._round_records[-self.window_size * 2:]

        # 更新互动图
        self._update_interaction_graph(round_discussions)

        # 检测模式
        patterns = self.detect_patterns()
        # 测量凝聚力
        cohesion = self.measure_cohesion()
        self._cohesion_history.append(cohesion)

        return {
            "round": round_number,
            "patterns": patterns,
            "cohesion": cohesion,
            "participant_count": len(record["participants"]),
            "interaction_count": len(record["interactions"]),
        }

    def _extract_participants(self, discussions: List[Dict]) -> List[str]:
        """提取参与者列表"""
        participants = set()
        for d in discussions:
            name = d.get("player_name", "")
            if name:
                participants.add(name)
        return list(participants)

    def _extract_interactions(self, discussions: List[Dict]) -> List[Dict]:
        """提取互动关系"""
        interactions = []
        for d in discussions:
            ref = d.get("refers_to", "")
            if ref:
                interactions.append({
                    "from": d.get("player_name", ""),
                    "to": ref,
                    "type": d.get("action", "new"),
                })
        return interactions

    def _update_interaction_graph(self, discussions: List[Dict]) -> None:
        """更新互动图"""
        for d in discussions:
            from_name = d.get("player_name", "")
            to_name = d.get("refers_to", "")
            if from_name and to_name:
                self._interaction_graph[from_name][to_name] += 1

    # ── 模式检测 ────────────────────────────────────────────────────────

    def detect_patterns(self) -> Dict[str, Any]:
        """
        检测当前群体互动模式。

        Returns:
            检测到的模式及其严重程度
        """
        if len(self._round_records) < 2:
            return {
                "dominance": {"detected": False, "severity": 0.0, "details": {}},
                "withdrawal": {"detected": False, "severity": 0.0, "details": {}},
                "alignment": {"detected": False, "severity": 0.0, "details": {}},
                "polarization": {"detected": False, "severity": 0.0, "details": {}},
                "fragmentation": {"detected": False, "severity": 0.0, "details": {}},
            }

        patterns = {
            "dominance": self._detect_dominance(),
            "withdrawal": self._detect_withdrawal(),
            "alignment": self._detect_alignment(),
            "polarization": self._detect_polarization(),
            "fragmentation": self._detect_fragmentation(),
        }
        self._pattern_history.append(patterns)
        return patterns

    def _detect_dominance(self) -> Dict[str, Any]:
        """检测主导模式：某个专家发言占比过高"""
        recent = self._round_records[-min(self.window_size, len(self._round_records)):]
        speaker_counts: Dict[str, int] = Counter()
        total = 0
        for rec in recent:
            for d in rec.get("discussions", []):
                name = d.get("player_name", "")
                if name:
                    speaker_counts[name] += 1
                    total += 1

        if total == 0:
            return {"detected": False, "severity": 0.0, "details": {}}

        details = {}
        max_share = 0.0
        dominant_speaker = None
        for name, count in speaker_counts.items():
            share = count / total
            details[name] = share
            if share > max_share:
                max_share = share
                dominant_speaker = name

        # 阈值：单人发言占比超过 40% 视为主导
        detected = max_share > 0.4
        # 严重度：超过 40% 的部分线性增长
        severity = max(0.0, (max_share - 0.4) / 0.6) if detected else 0.0

        return {
            "detected": detected,
            "severity": min(severity, 1.0),
            "details": {
                "dominant_speaker": dominant_speaker,
                "max_share": max_share,
                "speaker_shares": details,
            },
        }

    def _detect_withdrawal(self) -> Dict[str, Any]:
        """检测退缩模式：部分专家参与度持续下降"""
        if len(self._round_records) < 3:
            return {"detected": False, "severity": 0.0, "details": {}}

        # 取最近三轮
        recent = self._round_records[-3:]
        # 统计每轮参与者
        participants_per_round = []
        for rec in recent:
            participants_per_round.append(set(rec.get("participants", [])))

        # 找持续缺席的专家
        all_participants = set()
        for p in participants_per_round:
            all_participants.update(p)

        withdrawn = []
        for expert in all_participants:
            # 检查是否在最近的轮次中逐渐缺席
            presence = [1 if expert in p_set else 0 for p_set in participants_per_round]
            if len(presence) >= 3 and presence[-1] == 0 and presence[-2] == 0:
                withdrawn.append(expert)

        detected = len(withdrawn) > 0
        severity = min(len(withdrawn) / 5.0, 1.0) if detected else 0.0

        return {
            "detected": detected,
            "severity": severity,
            "details": {"withdrawn_experts": withdrawn, "total_experts": len(all_participants)},
        }

    def _detect_alignment(self) -> Dict[str, Any]:
        """检测趋同模式：观点过度一致"""
        recent = self._round_records[-min(self.window_size, len(self._round_records)):]
        # 统计不同动作的比例
        action_counts = Counter()
        for rec in recent:
            for d in rec.get("discussions", []):
                action = d.get("action", "new")
                action_counts[action] += 1

        total = sum(action_counts.values())
        if total == 0:
            return {"detected": False, "severity": 0.0, "details": {}}

        # 缺少 challenge 动作表明趋同
        challenge_ratio = action_counts.get("challenge", 0) / total
        detected = challenge_ratio < 0.05
        severity = max(0.0, (0.05 - challenge_ratio) / 0.05) if detected else 0.0

        return {
            "detected": detected,
            "severity": min(severity, 1.0),
            "details": {
                "challenge_ratio": challenge_ratio,
                "action_distribution": dict(action_counts),
            },
        }

    def _detect_polarization(self) -> Dict[str, Any]:
        """检测极化模式：群体形成对立阵营"""
        # 基于互动图检测：如果存在两个互不（或极少）互动的子群
        # 简化实现：检查是否有专家之间完全没有互动
        if not self._interaction_graph:
            return {"detected": False, "severity": 0.0, "details": {}}

        experts = list(self._interaction_graph.keys())
        if len(experts) < 4:
            return {"detected": False, "severity": 0.0, "details": {}}

        # 检查互动密度
        total_possible = len(experts) * (len(experts) - 1)
        actual_interactions = 0
        for a in experts:
            for b in experts:
                if a != b and self._interaction_graph[a].get(b, 0) > 0:
                    actual_interactions += 1

        density = actual_interactions / max(total_possible, 1)
        detected = density < 0.3
        severity = max(0.0, (0.3 - density) / 0.3) if detected else 0.0

        return {
            "detected": detected,
            "severity": min(severity, 1.0),
            "details": {
                "interaction_density": density,
                "actual_interactions": actual_interactions,
                "total_possible": total_possible,
            },
        }

    def _detect_fragmentation(self) -> Dict[str, Any]:
        """检测碎片化：讨论话题频繁切换，缺乏连贯性"""
        if len(self._round_records) < 2:
            return {"detected": False, "severity": 0.0, "details": {}}

        recent = self._round_records[-2:]
        # 统计话题切换次数
        topic_changes = 0
        total_topics = 0
        prev_topics: Set[str] = set()
        for rec in recent:
            current_topics = set()
            for d in rec.get("discussions", []):
                topic = d.get("refers_to", "")
                if topic:
                    current_topics.add(topic)
            if prev_topics:
                new_topics = current_topics - prev_topics
                topic_changes += len(new_topics)
            total_topics += len(current_topics | prev_topics)
            prev_topics = current_topics

        if total_topics == 0:
            return {"detected": False, "severity": 0.0, "details": {}}

        change_rate = topic_changes / max(total_topics, 1)
        detected = change_rate > 0.6
        severity = max(0.0, (change_rate - 0.6) / 0.4) if detected else 0.0

        return {
            "detected": detected,
            "severity": min(severity, 1.0),
            "details": {"topic_change_rate": change_rate, "topic_changes": topic_changes},
        }

    # ── 凝聚力测量 ──────────────────────────────────────────────────────

    def measure_cohesion(self) -> float:
        """
        测量群体凝聚力。

        综合考虑：参与度均衡性、互动密度、观点多样性

        Returns:
            凝聚力分数 (0.0 ~ 1.0)
        """
        if len(self._round_records) < 2:
            return 0.5

        recent = self._round_records[-min(self.window_size, len(self._round_records)):]

        # 1. 参与度均衡性 (0~1)
        speaker_counts = Counter()
        for rec in recent:
            for d in rec.get("discussions", []):
                name = d.get("player_name", "")
                if name:
                    speaker_counts[name] += 1

        if not speaker_counts:
            return 0.5

        counts = list(speaker_counts.values())
        total = sum(counts)
        n = len(counts)
        if n <= 1:
            evenness = 1.0
        else:
            # 使用基尼系数的补数
            sorted_counts = sorted(counts)
            cumulative = 0
            gini_numerator = 0
            for i, c in enumerate(sorted_counts):
                cumulative += c
                gini_numerator += (i + 1) * c
            if total > 0 and n > 0:
                gini = (2 * gini_numerator) / (n * total) - (n + 1) / n
                evenness = 1.0 - abs(gini)
            else:
                evenness = 0.5

        # 2. 互动密度 (0~1)
        experts = list(speaker_counts.keys())
        total_possible = len(experts) * (len(experts) - 1)
        actual = 0
        for a in experts:
            for b in experts:
                if a != b and self._interaction_graph[a].get(b, 0) > 0:
                    actual += 1
        density = actual / max(total_possible, 1)

        # 3. 观点多样性 (0~1)
        challenge_count = 0
        total_actions = 0
        for rec in recent:
            for d in rec.get("discussions", []):
                total_actions += 1
                if d.get("action", "") == "challenge":
                    challenge_count += 1
        diversity = min(challenge_count / max(total_actions, 1) * 5, 1.0)

        # 综合
        cohesion = evenness * 0.4 + density * 0.3 + diversity * 0.3
        return max(0.0, min(1.0, cohesion))

    # ── 报告 ────────────────────────────────────────────────────────────

    def get_dynamics_report(self) -> Dict[str, Any]:
        """
        生成群体动力学综合报告。

        Returns:
            综合报告，包含所有检测到的模式、凝聚力趋势和互动摘要
        """
        patterns = self.detect_patterns()
        active_patterns = [
            {"pattern": name, "severity": info["severity"]}
            for name, info in patterns.items()
            if info["detected"]
        ]
        active_patterns.sort(key=lambda x: x["severity"], reverse=True)

        return {
            "active_patterns": active_patterns,
            "pattern_count": len(active_patterns),
            "cohesion": {
                "current": self._cohesion_history[-1] if self._cohesion_history else 0.5,
                "trend": self._get_cohesion_trend(),
                "history": list(self._cohesion_history[-10:]),
            },
            "interaction_graph_summary": {
                "nodes": len(self._interaction_graph),
                "edges": sum(len(targets) for targets in self._interaction_graph.values()),
            },
            "rounds_analyzed": len(self._round_records),
        }

    def _get_cohesion_trend(self) -> str:
        """判断凝聚力趋势"""
        if len(self._cohesion_history) < 3:
            return "stable"
        recent = self._cohesion_history[-3:]
        if recent[-1] > recent[0] + 0.05:
            return "improving"
        elif recent[-1] < recent[0] - 0.05:
            return "declining"
        return "stable"

    def get_latest_patterns(self) -> Dict[str, Any]:
        """获取最新检测到的模式"""
        return self._pattern_history[-1] if self._pattern_history else {}

    def to_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "rounds_recorded": len(self._round_records),
            "patterns_detected": len(self._pattern_history),
            "cohesion_history_length": len(self._cohesion_history),
            "interaction_graph_nodes": len(self._interaction_graph),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GroupDynamicsModeler':
        modeler = cls(window_size=data.get("window_size", 5))
        return modeler


# =============================================================================
# 扩展系统 6: DiscussionPhaseDetector (讨论阶段检测)
# =============================================================================

class DiscussionPhaseDetector:
    """
    讨论阶段检测器 —— 检测当前讨论阶段并推荐相应调度策略。

    阶段定义：
    - opening:    开场阶段，专家自我介绍和初步立场
    - exploration: 探索阶段，广泛提出观点和视角
    - deepening:  深化阶段，深入分析特定话题
    - convergence: 收敛阶段，寻找共识和总结
    - closing:    结束阶段，最终总结和方案形成

    提供阶段过渡预测、各阶段度量和干预建议。
    """

    PHASES = ["opening", "exploration", "deepening", "convergence", "closing"]

    # 阶段转换规则
    PHASE_TRANSITIONS = {
        "opening": "exploration",
        "exploration": "deepening",
        "deepening": "convergence",
        "convergence": "closing",
    }

    def __init__(self, min_rounds_per_phase: int = 2):
        self.min_rounds_per_phase = min_rounds_per_phase  # 每个阶段最少轮次
        self.current_phase: str = "opening"
        self._phase_history: List[Dict[str, Any]] = []  # 阶段历史
        self._phase_start_round: int = 0
        self._current_round: int = 0
        self._metrics_history: Dict[str, List[float]] = defaultdict(list)

    # ── 阶段检测 ────────────────────────────────────────────────────────

    def detect_phase(self, state: Dict[str, Any], round_number: int) -> str:
        """
        检测当前讨论阶段。

        Args:
            state: 讨论状态报告
            round_number: 当前轮次

        Returns:
            检测到的阶段名称
        """
        self._current_round = round_number
        phase_scores = self._compute_phase_scores(state)
        detected = max(phase_scores, key=phase_scores.get)

        # 记录度量
        for phase, score in phase_scores.items():
            self._metrics_history[phase].append(score)

        # 阶段转换检查
        if self._can_transition(detected, round_number):
            old_phase = self.current_phase
            self.current_phase = detected
            self._phase_start_round = round_number
            self._phase_history.append({
                "from": old_phase,
                "to": detected,
                "round": round_number,
                "scores": dict(phase_scores),
            })

        return self.current_phase

    def _compute_phase_scores(self, state: Dict[str, Any]) -> Dict[str, float]:
        """计算各阶段的匹配分数"""
        scores = {}
        phase = state.get("phase", "exploring")
        round_count = state.get("round_count", 0)
        essence_count = state.get("essence_count", 0)
        discussion_count = state.get("discussion_count", 0)
        consensus_level = state.get("consensus_level", 0.0)
        controversy_level = state.get("controversy_level", 0.0)
        silent_count = len(state.get("silent_players", []))
        hot_topics = state.get("hot_topics", [])
        total_players = state.get("total_players", 0)

        # opening 阶段：早期，讨论少，共识低
        opening_score = 0.0
        if round_count <= 2:
            opening_score = 1.0
        elif round_count <= 4:
            opening_score = 0.5
        if discussion_count < total_players * 2:
            opening_score += 0.2

        # exploration 阶段：讨论多，分歧大，新观点多
        exploration_score = 0.0
        if discussion_count > total_players * 2:
            exploration_score += 0.3
        if controversy_level > 0.3:
            exploration_score += 0.3
        if consensus_level < 0.4:
            exploration_score += 0.2
        if len(hot_topics) > 2:
            exploration_score += 0.2

        # deepening 阶段：精华多，有明确热点，讨论深入
        deepening_score = 0.0
        if essence_count > 3:
            deepening_score += 0.3
        if len(hot_topics) > 0:
            deepening_score += 0.2
        if discussion_count > total_players * 3:
            deepening_score += 0.2
        if controversy_level > 0.5:
            deepening_score += 0.3

        # convergence 阶段：共识上升，新观点减少，精华积累
        convergence_score = 0.0
        if consensus_level > 0.5:
            convergence_score += 0.3
        if controversy_level < 0.3:
            convergence_score += 0.3
        if essence_count > 5:
            convergence_score += 0.2
        if round_count > 5:
            convergence_score += 0.2

        # closing 阶段：高共识，低争议，多轮次
        closing_score = 0.0
        if consensus_level > 0.7:
            closing_score += 0.3
        if controversy_level < 0.15:
            closing_score += 0.3
        if round_count > 8:
            closing_score += 0.2
        if silent_count == 0 or silent_count == total_players:
            closing_score += 0.2

        return {
            "opening": opening_score,
            "exploration": exploration_score,
            "deepening": deepening_score,
            "convergence": convergence_score,
            "closing": closing_score,
        }

    def _can_transition(self, detected: str, round_number: int) -> bool:
        """检查是否可以进行阶段转换"""
        # 不能后退
        current_idx = self.PHASES.index(self.current_phase)
        detected_idx = self.PHASES.index(detected)

        if detected_idx < current_idx:
            return False

        # 检查最少轮次约束
        rounds_in_phase = round_number - self._phase_start_round
        if rounds_in_phase < self.min_rounds_per_phase:
            return False

        # 只允许前进一个阶段
        expected_next = self.PHASE_TRANSITIONS.get(self.current_phase)
        if detected == expected_next or detected == self.current_phase:
            return detected != self.current_phase

        # 允许跳过中间阶段（如果分数足够高）
        if detected_idx > current_idx + 1:
            # 检查中间阶段的分数是否也足够高
            for mid_phase in self.PHASES[current_idx + 1:detected_idx]:
                mid_scores = self._metrics_history.get(mid_phase, [])
                if mid_scores and mid_scores[-1] < 0.3:
                    return False

        return detected != self.current_phase

    # ── 阶段变化预测 ────────────────────────────────────────────────────

    def predict_phase_change(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测何时以及如何发生阶段转换。

        Args:
            state: 当前讨论状态

        Returns:
            预测结果，包含预计阶段、转换条件和时间估计
        """
        current_idx = self.PHASES.index(self.current_phase)
        if current_idx >= len(self.PHASES) - 1:
            return {
                "will_change": False,
                "next_phase": None,
                "reason": "已在最终阶段",
                "estimated_rounds": 0,
            }

        next_phase = self.PHASE_TRANSITIONS.get(self.current_phase)
        if not next_phase:
            return {"will_change": False, "next_phase": None, "reason": "无下一阶段"}

        # 计算转换条件满足度
        phase_scores = self._compute_phase_scores(state)
        next_score = phase_scores.get(next_phase, 0.0)
        current_score = phase_scores.get(self.current_phase, 0.0)

        # 强度比
        if current_score > 0:
            ratio = next_score / current_score
        else:
            ratio = next_score

        # 预计还需要多少轮
        if ratio >= 0.8:
            estimated_rounds = 1
        elif ratio >= 0.5:
            estimated_rounds = 2
        elif ratio >= 0.3:
            estimated_rounds = 3
        else:
            estimated_rounds = max(3, self.min_rounds_per_phase)

        will_change = ratio >= 0.7 and (self._current_round - self._phase_start_round) >= self.min_rounds_per_phase

        return {
            "will_change": will_change,
            "next_phase": next_phase,
            "current_phase": self.current_phase,
            "ratio": ratio,
            "estimated_rounds": estimated_rounds,
            "next_phase_score": next_score,
            "current_phase_score": current_score,
        }

    # ── 阶段度量 ────────────────────────────────────────────────────────

    def get_phase_metrics(self) -> Dict[str, Any]:
        """
        获取当前阶段的详细度量。

        Returns:
            当前阶段的度量数据
        """
        metrics = {}
        for phase in self.PHASES:
            history = self._metrics_history.get(phase, [])
            if history:
                metrics[phase] = {
                    "current": history[-1],
                    "mean": statistics.mean(history),
                    "max": max(history),
                    "trend": "up" if len(history) >= 2 and history[-1] > history[-2] else
                            ("down" if len(history) >= 2 and history[-1] < history[-2] else "stable"),
                    "samples": len(history),
                }
            else:
                metrics[phase] = {"current": 0.0, "mean": 0.0, "max": 0.0, "trend": "stable", "samples": 0}

        return {
            "current_phase": self.current_phase,
            "rounds_in_phase": self._current_round - self._phase_start_round,
            "metrics": metrics,
            "phase_history_count": len(self._phase_history),
        }

    # ── 干预建议 ────────────────────────────────────────────────────────

    def suggest_intervention(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        根据当前阶段建议合适的干预措施。

        Args:
            state: 当前讨论状态

        Returns:
            干预建议，若无需要则返回 None
        """
        phase = self.current_phase
        silent_count = len(state.get("silent_players", []))
        total_players = state.get("total_players", 0)
        controversy_level = state.get("controversy_level", 0.0)
        consensus_level = state.get("consensus_level", 0.0)
        essence_count = state.get("essence_count", 0)

        suggestions = {
            "opening": [
                {
                    "type": "clarify_goal",
                    "priority": "high",
                    "message": "明确讨论目标和规则，引导专家自我介绍和专业立场",
                    "condition": "phase == opening",
                },
                {
                    "type": "set_agenda",
                    "priority": "medium",
                    "message": "设置讨论议程，确保各领域覆盖",
                    "condition": "无预设议程",
                },
            ],
            "exploration": [
                {
                    "type": "encourage_silent",
                    "priority": "high",
                    "message": f"鼓励 {silent_count} 名沉默专家发言，增加观点多样性",
                    "condition": silent_count > 0,
                },
                {
                    "type": "broaden_perspective",
                    "priority": "medium",
                    "message": "引导专家从不同专业视角补充观点",
                    "condition": "讨论集中在少数领域",
                },
            ],
            "deepening": [
                {
                    "type": "focus_debate",
                    "priority": "high",
                    "message": "锁定争议最大的话题进行深入辩论",
                    "condition": controversy_level > 0.5,
                },
                {
                    "type": "structured_analysis",
                    "priority": "medium",
                    "message": "引导专家对核心观点进行结构化分析（前提-论据-结论）",
                    "condition": "讨论深度不足",
                },
            ],
            "convergence": [
                {
                    "type": "summarize_progress",
                    "priority": "high",
                    "message": "邀请专家总结当前共识和分歧",
                    "condition": consensus_level > 0.3,
                },
                {
                    "type": "vote_on_essence",
                    "priority": "medium",
                    "message": f"对 {essence_count} 条精华进行投票，加速收敛",
                    "condition": essence_count > 3,
                },
            ],
            "closing": [
                {
                    "type": "final_summary",
                    "priority": "high",
                    "message": "邀请专家做最终总结，形成讨论报告",
                    "condition": "接近结束",
                },
                {
                    "type": "evaluate_outcome",
                    "priority": "medium",
                    "message": "评估讨论成果，收集反馈",
                    "condition": "讨论结束",
                },
            ],
        }

        phase_suggestions = suggestions.get(phase, [])
        applicable = [s for s in phase_suggestions if s.get("condition", True) is not False]
        return {
            "phase": phase,
            "suggestions": applicable,
            "suggestion_count": len(applicable),
        }

    def to_dict(self) -> dict:
        return {
            "current_phase": self.current_phase,
            "phase_start_round": self._phase_start_round,
            "current_round": self._current_round,
            "phase_history": self._phase_history,
            "min_rounds_per_phase": self.min_rounds_per_phase,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DiscussionPhaseDetector':
        detector = cls(min_rounds_per_phase=data.get("min_rounds_per_phase", 2))
        detector.current_phase = data.get("current_phase", "opening")
        detector._phase_start_round = data.get("phase_start_round", 0)
        detector._current_round = data.get("current_round", 0)
        detector._phase_history = data.get("phase_history", [])
        return detector


# =============================================================================
# 扩展系统 7: AdaptiveTurnAllocator (自适应轮次分配)
# =============================================================================

class AdaptiveTurnAllocator:
    """
    自适应轮次分配器 —— 动态分配发言轮次，平衡参与度与价值。

    分配策略：
    - 基于专家画像和当前需求分配轮次
    - 平衡参与度，防止个别专家垄断发言
    - 处理沉默专家，鼓励其参与
    - 优先分配对当前讨论最有价值的专家

    提供轮次统计、分配报告和参与度平衡功能。
    """

    def __init__(self, max_turns_per_round: int = 3,
                 min_turns_per_expert: int = 1,
                 silence_threshold: int = 3):
        self.max_turns_per_round = max_turns_per_round
        self.min_turns_per_expert = min_turns_per_expert
        self.silence_threshold = silence_threshold  # 沉默判断阈值（轮次）
        self._turn_counts: Dict[str, int] = defaultdict(int)  # 总发言轮次
        self._round_allocations: List[Dict[str, int]] = []  # 每轮分配记录
        self._silence_history: Dict[str, int] = defaultdict(int)  # 连续沉默轮次
        self._total_rounds = 0

    # ── 核心分配方法 ────────────────────────────────────────────────────

    def allocate(self, experts: List[Dict[str, Any]],
                 profiles: Dict[str, Dict[str, Any]],
                 phase: str = "exploring",
                 max_turns: Optional[int] = None) -> Dict[str, int]:
        """
        为本轮分配发言轮次。

        Args:
            experts: 专家基本信息列表 [{name, rounds_since_last_spoke, ...}]
            profiles: 专家画像字典
            phase: 当前讨论阶段
            max_turns: 本轮最大总轮次（覆盖默认值）

        Returns:
            {专家名称: 分配轮次} 映射
        """
        self._total_rounds += 1
        max_turns = max_turns or self.max_turns_per_round

        # 计算每个专家的分配权重
        weights: Dict[str, float] = {}
        for expert in experts:
            name = expert.get("name", "")
            profile = profiles.get(name, {})
            weight = self._compute_weight(name, expert, profile, phase)
            weights[name] = weight

        # 按权重分配轮次
        total_weight = sum(weights.values()) or 1.0
        allocation: Dict[str, int] = {}

        # 先保证最低轮次
        for expert in experts:
            name = expert.get("name", "")
            allocation[name] = self.min_turns_per_expert

        # 剩余轮次按权重分配
        remaining = max_turns - sum(allocation.values())
        if remaining > 0:
            weighted_experts = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for name, weight in weighted_experts:
                if remaining <= 0:
                    break
                extra = max(1, int(weight / total_weight * remaining))
                extra = min(extra, remaining)
                allocation[name] = allocation.get(name, 0) + extra
                remaining -= extra

        # 确保不超过 max_turns
        total_allocated = sum(allocation.values())
        if total_allocated > max_turns:
            # 按权重降序，减少最后分配者的轮次
            names_sorted = sorted(weights.keys(), key=lambda n: weights[n], reverse=True)
            for name in reversed(names_sorted):
                if total_allocated <= max_turns:
                    break
                if allocation[name] > self.min_turns_per_expert:
                    reduction = min(allocation[name] - self.min_turns_per_expert, total_allocated - max_turns)
                    allocation[name] -= reduction
                    total_allocated -= reduction

        # 更新沉默计数
        for name in allocation:
            if allocation[name] > 0:
                self._silence_history[name] = 0
        # 未分配轮次的专家沉默计数增加
        allocated_names = set(allocation.keys())
        for expert in experts:
            name = expert.get("name", "")
            if name not in allocated_names or allocation[name] == 0:
                self._silence_history[name] += 1

        # 记录分配
        self._round_allocations.append(dict(allocation))
        for name, turns in allocation.items():
            self._turn_counts[name] += turns

        return allocation

    def _compute_weight(self, name: str, expert: Dict[str, Any],
                        profile: Dict[str, Any], phase: str) -> float:
        """
        计算专家的发言权重。
        """
        weight = 0.5  # 基础权重

        # 1. 饥饿度
        rounds_since = expert.get("rounds_since_last_spoke", 0)
        hunger_weight = min(rounds_since / 5.0, 1.0) * 0.3
        weight += hunger_weight

        # 2. 画像质量
        if "reliability" in profile:
            consistency = profile["reliability"].get("consistency", 0.5)
            weight += consistency * 0.2

        # 3. 阶段适配
        phase_multipliers = {
            "exploring": 0.2,
            "deepening": 0.15,
            "convergence": 0.1,
            "closing": 0.1,
        }
        weight += phase_multipliers.get(phase, 0.15)

        # 4. 沉默惩罚（沉默越久，权重越高）
        silence_rounds = self._silence_history.get(name, 0)
        if silence_rounds >= self.silence_threshold:
            weight += 0.3

        # 5. 参与度平衡（已发言越多，权重越低）
        total_turns = self._turn_counts.get(name, 0)
        if total_turns > 0:
            avg_turns = sum(self._turn_counts.values()) / max(len(self._turn_counts), 1)
            if total_turns > avg_turns * 1.5:
                weight -= 0.2  # 发言过多，降低权重

        return max(0.0, weight)

    # ── 轮次查询 ────────────────────────────────────────────────────────

    def get_turn_count(self, name: str) -> int:
        """获取指定专家的总发言轮次"""
        return self._turn_counts.get(name, 0)

    # ── 参与度平衡 ──────────────────────────────────────────────────────

    def balance(self, experts: List[Dict[str, Any]],
                profiles: Dict[str, Dict[str, Any]],
                target_balance: float = 0.3) -> Dict[str, int]:
        """
        专门用于平衡参与度的分配。

        Args:
            experts: 专家基本信息列表
            profiles: 专家画像
            target_balance: 目标平衡度（Gini 系数目标）

        Returns:
            平衡后的分配方案
        """
        # 检测当前不平衡度
        current_counts = [self._turn_counts.get(e.get("name", ""), 0) for e in experts]
        if not current_counts:
            return {}

        total = sum(current_counts)
        n = len(current_counts)
        if total == 0 or n <= 1:
            return self.allocate(experts, profiles, phase="balance")

        # 计算当前 Gini 系数
        sorted_counts = sorted(current_counts)
        cumulative = 0
        gini_num = 0
        for i, c in enumerate(sorted_counts):
            cumulative += c
            gini_num += (i + 1) * c
        gini = (2 * gini_num) / (n * total) - (n + 1) / n

        if gini <= target_balance:
            # 已经足够平衡
            return self.allocate(experts, profiles)

        # 需要平衡：优先分配给发言少的专家
        sorted_by_turns = sorted(experts, key=lambda e: self._turn_counts.get(e.get("name", ""), 0))
        allocation: Dict[str, int] = {}
        for i, expert in enumerate(sorted_by_turns):
            name = expert.get("name", "")
            if i < len(sorted_by_turns) // 2:
                allocation[name] = max(self.min_turns_per_expert, 2)
            else:
                allocation[name] = self.min_turns_per_expert

        self._round_allocations.append(dict(allocation))
        for name, turns in allocation.items():
            self._turn_counts[name] += turns

        return allocation

    # ── 沉默处理 ────────────────────────────────────────────────────────

    def handle_silence(self, experts: List[Dict[str, Any]],
                       profiles: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        识别并处理沉默专家。

        Args:
            experts: 专家基本信息列表
            profiles: 专家画像

        Returns:
            需要特别关注的沉默专家名称列表
        """
        silent_experts = []
        for expert in experts:
            name = expert.get("name", "")
            silence_rounds = self._silence_history.get(name, 0)
            rounds_since = expert.get("rounds_since_last_spoke", 0)
            effective_silence = max(silence_rounds, rounds_since)

            if effective_silence >= self.silence_threshold:
                silent_experts.append(name)

        return silent_experts

    # ── 报告 ────────────────────────────────────────────────────────────

    def get_allocation_report(self) -> Dict[str, Any]:
        """
        生成轮次分配报告。

        Returns:
            分配报告，包含各专家分配统计、参与度分析和平衡度
        """
        total_turns = sum(self._turn_counts.values())
        n_experts = len(self._turn_counts)

        if n_experts == 0:
            return {"error": "尚无分配数据"}

        # 参与度分析
        max_turns = max(self._turn_counts.values())
        min_turns = min(self._turn_counts.values())
        avg_turns = total_turns / n_experts

        # 标准差
        if n_experts > 1:
            variance = sum((c - avg_turns) ** 2 for c in self._turn_counts.values()) / n_experts
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0

        # 参与度分布
        distribution = {
            "high": sum(1 for c in self._turn_counts.values() if c > avg_turns * 1.2),
            "medium": sum(1 for c in self._turn_counts.values() if avg_turns * 0.8 <= c <= avg_turns * 1.2),
            "low": sum(1 for c in self._turn_counts.values() if c < avg_turns * 0.8),
        }

        return {
            "total_turns": total_turns,
            "total_rounds": self._total_rounds,
            "experts_count": n_experts,
            "avg_turns_per_expert": avg_turns,
            "max_turns": max_turns,
            "min_turns": min_turns,
            "std_dev": std_dev,
            "distribution": distribution,
            "balance_ratio": min_turns / max(max_turns, 1),
            "silent_experts_count": sum(1 for s in self._silence_history.values() if s >= self.silence_threshold),
            "per_expert": dict(self._turn_counts),
        }

    def get_allocation_summary(self) -> Dict[str, Any]:
        """获取分配摘要的快捷方法"""
        return {
            "total_turns": sum(self._turn_counts.values()),
            "experts_count": len(self._turn_counts),
            "rounds": self._total_rounds,
        }

    def to_dict(self) -> dict:
        return {
            "max_turns_per_round": self.max_turns_per_round,
            "min_turns_per_expert": self.min_turns_per_expert,
            "silence_threshold": self.silence_threshold,
            "total_rounds": self._total_rounds,
            "turn_counts": dict(self._turn_counts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AdaptiveTurnAllocator':
        allocator = cls(
            max_turns_per_round=data.get("max_turns_per_round", 3),
            min_turns_per_expert=data.get("min_turns_per_expert", 1),
            silence_threshold=data.get("silence_threshold", 3),
        )
        allocator._turn_counts = defaultdict(int, data.get("turn_counts", {}))
        allocator._total_rounds = data.get("total_rounds", 0)
        return allocator


# =============================================================================
# 扩展系统 8: ExpertiseGapFiller (专业知识缺口填补)
# =============================================================================

class ExpertiseGapFiller:
    """
    专业知识缺口填补器 —— 检测讨论中的知识盲区并建议填补方案。

    核心功能：
    - 检测讨论中缺失的专业领域
    - 评估已有专家可覆盖的领域
    - 建议外部知识注入
    - 对缺口进行优先级排序

    支持缺口报告、领域覆盖分析和知识注入建议。
    """

    # 知识领域大类（用于缺口检测）
    DOMAIN_CATEGORIES = {
        "formal_sciences": {"数学", "逻辑", "范畴论", "代数", "几何", "拓扑", "概率", "统计", "计算", "算法"},
        "natural_sciences": {"物理", "化学", "生物", "宇宙学", "量子", "力学", "电磁", "热力", "相对", "弦论", "量子计算"},
        "social_sciences": {"哲学", "伦理", "社会", "经济", "政策", "心理", "认知", "语言"},
        "engineering": {"工程", "系统", "控制", "网络", "软件", "硬件", "架构", "设计", "安全", "分布式"},
        "data_science": {"数据", "学习", "优化", "人工智能", "机器学习", "深度学习", "神经网络", "强化学习"},
        "emerging": {"密码学", "区块链", "量子里", "量子信息", "复杂", "混沌", "涌现", "自组织"},
    }

    def __init__(self):
        self._known_gaps: Dict[str, Dict[str, Any]] = {}  # domain -> gap info
        self._gap_history: List[Dict[str, Any]] = []  # 缺口检测历史
        self._injections: List[Dict[str, Any]] = []  # 知识注入记录
        self._coverage_history: List[Dict[str, float]] = []  # 领域覆盖历史

    # ── 缺口检测 ────────────────────────────────────────────────────────

    def detect_gaps(self, expert_tags: Dict[str, Set[str]],
                    discussion_topics: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        检测当前专家团队的知识缺口。

        Args:
            expert_tags: {专家名称: 专业标签集合}
            discussion_topics: 当前讨论涉及的话题标签

        Returns:
            缺口检测结果，包含各领域的覆盖情况和缺失程度
        """
        # 计算各领域覆盖
        domain_coverage: Dict[str, Dict[str, Any]] = {}
        for domain, domain_tags in self.DOMAIN_CATEGORIES.items():
            # 专家可覆盖的该领域标签
            covered_tags = set()
            for tags in expert_tags.values():
                covered_tags.update(tags & domain_tags)

            # 该领域总标签数
            total_tags = len(domain_tags)
            # 覆盖标签数
            covered_count = len(covered_tags & domain_tags)
            # 缺失标签
            missing_tags = domain_tags - covered_tags

            coverage = covered_count / max(total_tags, 1)

            domain_coverage[domain] = {
                "coverage": coverage,
                "covered_tags": sorted(covered_tags & domain_tags),
                "missing_tags": sorted(missing_tags),
                "missing_count": len(missing_tags),
                "total_tags": total_tags,
            }

        # 讨论话题覆盖分析
        topic_gaps = set()
        if discussion_topics:
            all_expert_tags = set()
            for tags in expert_tags.values():
                all_expert_tags.update(tags)
            topic_gaps = discussion_topics - all_expert_tags

        # 综合结果
        gaps = {
            "domain_coverage": domain_coverage,
            "topic_gaps": sorted(topic_gaps),
            "overall_coverage": statistics.mean([d["coverage"] for d in domain_coverage.values()]),
            "most_lacking": self._find_most_lacking(domain_coverage),
            "timestamp": len(self._gap_history),
        }

        # 保存缺口信息
        for domain, info in domain_coverage.items():
            self._known_gaps[domain] = info

        self._gap_history.append(gaps)
        self._coverage_history.append({d: info["coverage"] for d, info in domain_coverage.items()})
        return gaps

    def _find_most_lacking(self, domain_coverage: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """找出最缺乏的领域"""
        lacking = []
        for domain, info in domain_coverage.items():
            if info["coverage"] < 0.5:
                lacking.append({
                    "domain": domain,
                    "coverage": info["coverage"],
                    "missing_tags": info["missing_tags"],
                    "missing_count": info["missing_count"],
                })
        lacking.sort(key=lambda x: x["coverage"])
        return lacking

    # ── 缺口填补 ────────────────────────────────────────────────────────

    def fill_gaps(self, expert_tags: Dict[str, Set[str]],
                  discussion_topics: Optional[Set[str]] = None,
                  top_n: int = 3) -> List[Dict[str, Any]]:
        """
        检测缺口并生成填补建议。

        Args:
            expert_tags: {专家名称: 专业标签集合}
            discussion_topics: 当前讨论话题标签
            top_n: 返回的优先填补建议数量

        Returns:
            填补建议列表，按优先级排序
        """
        gaps = self.detect_gaps(expert_tags, discussion_topics)
        suggestions = self._prioritize_gaps(gaps, discussion_topics)
        return suggestions[:top_n]

    def _prioritize_gaps(self, gaps: Dict[str, Any],
                         discussion_topics: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """对缺口进行优先级排序"""
        suggestions = []
        domain_coverage = gaps.get("domain_coverage", {})

        # 1. 讨论话题相关缺口优先
        if discussion_topics:
            for domain, info in domain_coverage.items():
                domain_tags = self.DOMAIN_CATEGORIES.get(domain, set())
                topic_overlap = len(discussion_topics & domain_tags)
                if topic_overlap > 0 and info["coverage"] < 0.6:
                    suggestions.append({
                        "domain": domain,
                        "priority": "high",
                        "coverage": info["coverage"],
                        "missing_tags": info["missing_tags"],
                        "reason": f"领域与当前讨论话题相关（重叠 {topic_overlap} 个标签）",
                    })

        # 2. 覆盖极低的领域
        for domain, info in domain_coverage.items():
            if info["coverage"] < 0.3:
                # 避免重复
                if not any(s["domain"] == domain for s in suggestions):
                    suggestions.append({
                        "domain": domain,
                        "priority": "high" if info["coverage"] < 0.2 else "medium",
                        "coverage": info["coverage"],
                        "missing_tags": info["missing_tags"],
                        "reason": f"领域覆盖率仅 {info['coverage']:.0%}，存在显著知识缺口",
                    })

        # 3. 中等覆盖但缺口数量大的领域
        for domain, info in domain_coverage.items():
            if 0.3 <= info["coverage"] < 0.6 and info["missing_count"] > 3:
                if not any(s["domain"] == domain for s in suggestions):
                    suggestions.append({
                        "domain": domain,
                        "priority": "medium",
                        "coverage": info["coverage"],
                        "missing_tags": info["missing_tags"],
                        "reason": f"领域有 {info['missing_count']} 个未覆盖标签",
                    })

        # 4. 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: (priority_order.get(s["priority"], 3), s["coverage"]))
        return suggestions

    # ── 知识注入建议 ────────────────────────────────────────────────────

    def suggest_expertise(self, missing_tags: List[str],
                          context: Optional[str] = None) -> Dict[str, Any]:
        """
        针对缺失的知识领域提出注入建议。

        Args:
            missing_tags: 缺失的专业标签列表
            context: 当前讨论上下文

        Returns:
            知识注入建议
        """
        if not missing_tags:
            return {"has_suggestions": False, "suggestions": []}

        suggestions = []
        for tag in missing_tags[:5]:  # 最多处理 5 个
            # 确定所属领域
            domain = None
            for d, tags in self.DOMAIN_CATEGORIES.items():
                if tag in tags:
                    domain = d
                    break
            domain = domain or "unknown"

            # 生成注入建议
            suggestion = {
                "tag": tag,
                "domain": domain,
                "injection_type": "external",
                "suggested_action": f"引入 {tag} 领域的专业知识或参考资料",
                "rationale": f"当前讨论中缺少 {tag} 视角，可能导致分析不全面",
                "relevance": 0.8,
            }
            suggestions.append(suggestion)

        return {
            "has_suggestions": True,
            "suggestion_count": len(suggestions),
            "suggestions": suggestions,
            "context": context,
        }

    # ── 报告 ────────────────────────────────────────────────────────────

    def get_gap_report(self, expert_tags: Dict[str, Set[str]],
                       discussion_topics: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        生成完整的专业知识缺口报告。

        Args:
            expert_tags: {专家名称: 专业标签集合}
            discussion_topics: 当前讨论话题标签

        Returns:
            完整的缺口分析报告
        """
        gaps = self.detect_gaps(expert_tags, discussion_topics)
        suggestions = self._prioritize_gaps(gaps, discussion_topics)

        # 覆盖趋势
        coverage_trend = "stable"
        if len(self._coverage_history) >= 3:
            recent = self._coverage_history[-3:]
            avg_first = statistics.mean(recent[0].values())
            avg_last = statistics.mean(recent[-1].values())
            if avg_last > avg_first + 0.05:
                coverage_trend = "improving"
            elif avg_last < avg_first - 0.05:
                coverage_trend = "declining"

        return {
            "overall_coverage": gaps["overall_coverage"],
            "coverage_trend": coverage_trend,
            "domain_coverage": gaps["domain_coverage"],
            "topic_gaps": gaps["topic_gaps"],
            "most_lacking": gaps["most_lacking"],
            "priority_suggestions": suggestions,
            "total_gaps_detected": len(gaps["most_lacking"]),
            "history_length": len(self._gap_history),
        }

    def get_gap_summary(self) -> Dict[str, Any]:
        """获取缺口摘要"""
        if not self._gap_history:
            return {"coverage": 0.0, "gaps": [], "history_length": 0}
        latest = self._gap_history[-1]
        return {
            "coverage": latest["overall_coverage"],
            "gaps": latest["most_lacking"],
            "history_length": len(self._gap_history),
        }

    def to_dict(self) -> dict:
        return {
            "known_gaps": {k: {"coverage": v.get("coverage", 0), "missing_count": v.get("missing_count", 0)}
                           for k, v in self._known_gaps.items()},
            "gap_history_length": len(self._gap_history),
            "injections_count": len(self._injections),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ExpertiseGapFiller':
        filler = cls()
        filler._known_gaps = data.get("known_gaps", {})
        filler._injections = data.get("injections", [])
        return filler


# =============================================================================
# 扩展系统 9: InterventionScheduler (干预调度)
# =============================================================================

class InterventionScheduler:
    """
    干预调度器 —— 安排和优化主持人干预。

    干预类型：
    - redirect:      引导讨论回到正轨
    - encourage:     鼓励参与
    - challenge:     提出挑战性问题
    - summarize:     总结进展
    - clarify:       澄清模糊点
    - probe:         深入追问
    - reframe:       重新框架化问题
    - mediate:       调解分歧

    支持干预时机的优化选择和干预效果追踪。
    """

    INTERVENTION_TYPES = [
        "redirect", "encourage", "challenge", "summarize",
        "clarify", "probe", "reframe", "mediate",
    ]

    def __init__(self, min_interval: int = 3, max_interventions_per_round: int = 2):
        self.min_interval = min_interval  # 最小干预间隔（轮次）
        self.max_interventions_per_round = max_interventions_per_round
        self._last_intervention_round: int = 0
        self._intervention_history: List[Dict[str, Any]] = []
        self._effectiveness_scores: Dict[str, List[float]] = defaultdict(list)
        self._total_interventions = 0

    # ── 干预调度 ────────────────────────────────────────────────────────

    def schedule_intervention(self, state: Dict[str, Any],
                              round_number: int,
                              patterns: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        调度新一轮的干预。

        Args:
            state: 当前讨论状态
            round_number: 当前轮次
            patterns: 群体动力学检测到的模式（可选）

        Returns:
            干预计划，若不需要干预则返回 None
        """
        # 检查最小间隔
        if round_number - self._last_intervention_round < self.min_interval:
            # 但紧急情况可以覆盖
            if not self._is_urgent(state, patterns):
                return None

        # 检查本轮已干预次数
        round_interventions = [i for i in self._intervention_history
                               if i.get("round", 0) == round_number]
        if len(round_interventions) >= self.max_interventions_per_round:
            return None

        # 选择干预类型
        intervention_type = self.select_type(state, patterns)
        if not intervention_type:
            return None

        # 找到最佳时机
        optimal_time = self.find_optimal_time(state, round_number)

        intervention = {
            "type": intervention_type,
            "round": round_number,
            "optimal_time": optimal_time,
            "reason": self._get_intervention_reason(intervention_type, state, patterns),
            "target": self._get_intervention_target(intervention_type, state),
            "urgency": "high" if self._is_urgent(state, patterns) else "normal",
        }

        self._intervention_history.append(intervention)
        self._last_intervention_round = round_number
        self._total_interventions += 1
        return intervention

    def _is_urgent(self, state: Dict[str, Any],
                   patterns: Optional[Dict[str, Any]] = None) -> bool:
        """判断是否需要紧急干预"""
        # 检查僵持状态
        if state.get("phase") == "stalled":
            return True

        # 检查严重的主导模式
        if patterns:
            dominance = patterns.get("dominance", {})
            if dominance.get("detected") and dominance.get("severity", 0) > 0.7:
                return True
            withdrawal = patterns.get("withdrawal", {})
            if withdrawal.get("detected") and withdrawal.get("severity", 0) > 0.7:
                return True

        return False

    def _get_intervention_reason(self, intervention_type: str,
                                  state: Dict[str, Any],
                                  patterns: Optional[Dict[str, Any]] = None) -> str:
        """生成干预原因"""
        reasons = {
            "redirect": "讨论偏离主题，需要引导回到正轨",
            "encourage": "部分专家参与度不足，需要鼓励发言",
            "challenge": "观点趋同，需要引入挑战性视角",
            "summarize": "讨论进展丰富，需要阶段性总结",
            "clarify": "存在模糊或歧义观点，需要澄清",
            "probe": "观点表面化，需要深入追问",
            "reframe": "讨论陷入僵局，需要重新框架化问题",
            "mediate": "存在激烈分歧，需要调解",
        }
        base = reasons.get(intervention_type, "常规干预")

        # 添加模式相关信息
        if patterns:
            for pattern_name, pattern_info in patterns.items():
                if pattern_info.get("detected"):
                    base += f"（检测到{pattern_name}模式）"
                    break
        return base

    def _get_intervention_target(self, intervention_type: str,
                                  state: Dict[str, Any]) -> Optional[str]:
        """确定干预目标"""
        if intervention_type == "encourage":
            silent = state.get("silent_players", [])
            if silent:
                return silent[0]
        elif intervention_type == "challenge":
            dominant = state.get("dominant_players", [])
            if dominant:
                return dominant[0]
        return None

    # ── 干预类型选择 ────────────────────────────────────────────────────

    def select_type(self, state: Dict[str, Any],
                    patterns: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        根据当前状态选择最合适的干预类型。

        Args:
            state: 当前讨论状态
            patterns: 群体动力学模式

        Returns:
            选择的干预类型，若无合适类型返回 None
        """
        type_scores: Dict[str, float] = {t: 0.0 for t in self.INTERVENTION_TYPES}

        phase = state.get("phase", "exploring")
        silent_count = len(state.get("silent_players", []))
        dominant_count = len(state.get("dominant_players", []))
        total_players = state.get("total_players", 0)
        controversy_level = state.get("controversy_level", 0.0)
        consensus_level = state.get("consensus_level", 0.0)

        # 基于阶段评分
        phase_scores = {
            "exploring": {"encourage": 0.8, "probe": 0.6, "clarify": 0.5},
            "deep_debate": {"challenge": 0.8, "probe": 0.7, "mediate": 0.6, "redirect": 0.4},
            "converging": {"summarize": 0.9, "clarify": 0.6, "redirect": 0.5},
            "stalled": {"reframe": 0.9, "redirect": 0.7, "challenge": 0.6},
        }
        for t, score in phase_scores.get(phase, {}).items():
            type_scores[t] += score

        # 基于模式评分
        if patterns:
            if patterns.get("dominance", {}).get("detected"):
                type_scores["redirect"] += 0.6
                type_scores["encourage"] += 0.5
            if patterns.get("withdrawal", {}).get("detected"):
                type_scores["encourage"] += 0.8
                type_scores["probe"] += 0.4
            if patterns.get("alignment", {}).get("detected"):
                type_scores["challenge"] += 0.8
                type_scores["reframe"] += 0.5
            if patterns.get("polarization", {}).get("detected"):
                type_scores["mediate"] += 0.8
                type_scores["summarize"] += 0.5
            if patterns.get("fragmentation", {}).get("detected"):
                type_scores["redirect"] += 0.7
                type_scores["summarize"] += 0.6

        # 基于状态评分
        if silent_count > total_players * 0.3:
            type_scores["encourage"] += 0.5
        if dominant_count > 0:
            type_scores["redirect"] += 0.4
        if controversy_level > 0.7:
            type_scores["mediate"] += 0.6
            type_scores["clarify"] += 0.4
        if consensus_level > 0.6:
            type_scores["summarize"] += 0.5

        # 效果历史调整（效果好的类型加分）
        for t in self.INTERVENTION_TYPES:
            scores = self._effectiveness_scores.get(t, [])
            if scores:
                avg_effectiveness = statistics.mean(scores)
                type_scores[t] += avg_effectiveness * 0.3

        # 选择最高分
        best_type = max(type_scores, key=type_scores.get)
        if type_scores[best_type] > 0.3:
            return best_type
        return None

    # ── 时机优化 ────────────────────────────────────────────────────────

    def find_optimal_time(self, state: Dict[str, Any],
                          round_number: int) -> Dict[str, Any]:
        """
        找到干预的最佳时机。

        Args:
            state: 当前讨论状态
            round_number: 当前轮次

        Returns:
            最佳时机分析
        """
        # 分析当前轮次中已发言的专家数
        silent_count = len(state.get("silent_players", []))
        total_players = state.get("total_players", 0)
        active_ratio = (total_players - silent_count) / max(total_players, 1)

        # 分析讨论连续性
        discussion_count = state.get("discussion_count", 0)
        expected_discussions = total_players * 2

        timings = {
            "immediate": False,
            "after_current_speaker": False,
            "next_round": False,
            "reasoning": "",
        }

        # 紧急情况：立即干预
        if self._is_urgent(state, None):
            timings["immediate"] = True
            timings["reasoning"] = "紧急情况，需要立即干预"
            return timings

        # 活跃度低：等当前发言者说完
        if active_ratio < 0.5:
            timings["after_current_speaker"] = True
            timings["reasoning"] = "参与度低，等当前发言结束后干预"
            return timings

        # 讨论量不足：下一轮再干预
        if discussion_count < expected_discussions * 0.5:
            timings["next_round"] = True
            timings["reasoning"] = "讨论量不足，下一轮再干预"
            return timings

        # 默认：立即干预
        timings["immediate"] = True
        timings["reasoning"] = "适宜时机，可以立即干预"
        return timings

    # ── 效果追踪 ────────────────────────────────────────────────────────

    def record_effectiveness(self, intervention_type: str,
                             effectiveness: float) -> None:
        """
        记录干预效果。

        Args:
            intervention_type: 干预类型
            effectiveness: 效果评分 (0.0 ~ 1.0)
        """
        effectiveness = max(0.0, min(1.0, effectiveness))
        self._effectiveness_scores[intervention_type].append(effectiveness)

    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """获取干预历史"""
        return list(self._intervention_history)

    def get_history_summary(self) -> Dict[str, Any]:
        """获取干预历史摘要"""
        return {
            "total_interventions": self._total_interventions,
            "types_used": dict(Counter(i["type"] for i in self._intervention_history)),
            "last_intervention_round": self._last_intervention_round,
        }

    def to_dict(self) -> dict:
        return {
            "min_interval": self.min_interval,
            "max_interventions_per_round": self.max_interventions_per_round,
            "total_interventions": self._total_interventions,
            "last_intervention_round": self._last_intervention_round,
            "effectiveness": {
                t: {
                    "mean": statistics.mean(scores) if scores else 0.0,
                    "count": len(scores),
                }
                for t, scores in self._effectiveness_scores.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InterventionScheduler':
        scheduler = cls(
            min_interval=data.get("min_interval", 3),
            max_interventions_per_round=data.get("max_interventions_per_round", 2),
        )
        scheduler._total_interventions = data.get("total_interventions", 0)
        scheduler._last_intervention_round = data.get("last_intervention_round", 0)
        return scheduler


# =============================================================================
# 扩展系统 10: LearningScheduler (学习型调度器)
# =============================================================================

class LearningScheduler:
    """
    学习型调度器 —— 从历史调度结果中学习，持续改进调度策略。

    学习机制：
    - 记录每次调度的结果和效果
    - 跟踪关键调度指标
    - 基于历史表现调整策略参数
    - 支持策略的渐进式优化

    提供策略学习、指标追踪和策略调整功能。
    """

    def __init__(self, learning_rate: float = 0.1,
                 exploration_rate: float = 0.2,
                 discount_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate  # 探索率（Epsilon-greedy）
        self.discount_factor = discount_factor  # 折扣因子（Q-learning）
        self._policy: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._metrics_history: List[Dict[str, float]] = []
        self._outcomes: List[Dict[str, Any]] = []
        self._total_learn_iterations = 0
        self._strategy_adjustments: List[Dict[str, Any]] = []

    # ── 核心学习方法 ────────────────────────────────────────────────────

    def learn(self, state: Dict[str, Any], action: str,
              reward: float, next_state: Optional[Dict[str, Any]] = None) -> None:
        """
        从一次调度决策中学习。

        使用 Q-learning 更新规则：
            Q(s, a) = Q(s, a) + lr * (reward + discount * max(Q(s', a')) - Q(s, a))

        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态（可选）
        """
        self._total_learn_iterations += 1

        # 提取状态特征
        state_key = self._encode_state(state)
        next_state_key = self._encode_state(next_state) if next_state else state_key

        # 当前 Q 值
        current_q = self._policy[state_key].get(action, 0.0)

        # 未来最大 Q 值
        if next_state:
            next_q_values = self._policy[next_state_key].values()
            max_next_q = max(next_q_values) if next_q_values else 0.0
        else:
            max_next_q = 0.0

        # Q-learning 更新
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self._policy[state_key][action] = new_q

        # 记录结果
        self._outcomes.append({
            "iteration": self._total_learn_iterations,
            "state_key": state_key,
            "action": action,
            "reward": reward,
            "old_q": current_q,
            "new_q": new_q,
        })

    def _encode_state(self, state: Optional[Dict[str, Any]]) -> str:
        """将状态编码为策略键"""
        if not state:
            return "default"

        phase = state.get("phase", "unknown")
        consensus = "high" if state.get("consensus_level", 0) > 0.6 else \
                    ("low" if state.get("consensus_level", 0) < 0.3 else "medium")
        controversy = "high" if state.get("controversy_level", 0) > 0.6 else \
                      ("low" if state.get("controversy_level", 0) < 0.3 else "medium")
        silent = "many" if len(state.get("silent_players", [])) > 2 else \
                 ("few" if len(state.get("silent_players", [])) > 0 else "none")

        return f"{phase}|{consensus}|{controversy}|{silent}"

    # ── 策略更新 ────────────────────────────────────────────────────────

    def update_policy(self, feedback: Dict[str, float]) -> Dict[str, Any]:
        """
        根据反馈更新调度策略。

        Args:
            feedback: 各维度的反馈评分 {dimension: score}

        Returns:
            策略更新摘要
        """
        # 分析反馈，确定需要调整的方向
        adjustments = {}
        for dimension, score in feedback.items():
            if score < 0.3:
                # 该维度表现差，需要加大关注
                adjustments[dimension] = "increase"
            elif score > 0.7:
                # 该维度表现好，可以保持
                adjustments[dimension] = "maintain"
            else:
                adjustments[dimension] = "monitor"

        # 调整探索率（根据反馈趋势）
        if self._metrics_history:
            recent_rewards = [o.get("reward", 0) for o in self._outcomes[-10:]]
            if recent_rewards:
                avg_reward = statistics.mean(recent_rewards)
                if avg_reward < 0.3:
                    # 效果差，增加探索
                    self.exploration_rate = min(0.5, self.exploration_rate + 0.05)
                elif avg_reward > 0.7:
                    # 效果好，减少探索
                    self.exploration_rate = max(0.05, self.exploration_rate - 0.02)

        adjustment_record = {
            "iteration": self._total_learn_iterations,
            "feedback": feedback,
            "adjustments": adjustments,
            "new_exploration_rate": self.exploration_rate,
        }
        self._strategy_adjustments.append(adjustment_record)

        return {
            "adjustments": adjustments,
            "exploration_rate": self.exploration_rate,
            "policy_size": len(self._policy),
        }

    # ── 策略查询 ────────────────────────────────────────────────────────

    def get_best_action(self, state: Dict[str, Any],
                        available_actions: List[str]) -> Tuple[str, float]:
        """
        获取当前状态下最优动作。

        Args:
            state: 当前状态
            available_actions: 可用动作列表

        Returns:
            (最优动作, 期望Q值)
        """
        state_key = self._encode_state(state)
        state_policy = self._policy[state_key]

        # Epsilon-greedy 探索
        if random.random() < self.exploration_rate:
            action = random.choice(available_actions)
            return action, state_policy.get(action, 0.0)

        # 选择 Q 值最高的动作
        best_action = available_actions[0]
        best_q = -float('inf')
        for action in available_actions:
            q = state_policy.get(action, 0.0)
            if q > best_q:
                best_q = q
                best_action = action

        return best_action, best_q

    # ── 指标追踪 ────────────────────────────────────────────────────────

    def get_learning_metrics(self) -> Dict[str, Any]:
        """
        获取学习指标摘要。

        Returns:
            学习指标，包括策略大小、平均奖励、学习进度等
        """
        if not self._outcomes:
            return {
                "iterations": 0,
                "policy_size": 0,
                "avg_reward": 0.0,
                "exploration_rate": self.exploration_rate,
                "learning_rate": self.learning_rate,
                "convergence": 0.0,
            }

        recent_outcomes = self._outcomes[-20:]
        avg_reward = statistics.mean([o.get("reward", 0) for o in recent_outcomes])

        # 收敛度：最近 Q 值变化幅度
        if len(self._outcomes) >= 10:
            recent_qs = [o.get("new_q", 0) for o in self._outcomes[-10:]]
            q_variance = statistics.variance(recent_qs) if len(recent_qs) > 1 else 1.0
            convergence = max(0.0, 1.0 - min(q_variance, 1.0))
        else:
            convergence = 0.0

        return {
            "iterations": self._total_learn_iterations,
            "policy_size": len(self._policy),
            "avg_reward": avg_reward,
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "convergence": convergence,
            "strategy_adjustments": len(self._strategy_adjustments),
        }

    # ── 策略调整 ────────────────────────────────────────────────────────

    def adjust_strategy(self, target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        根据目标指标调整策略参数。

        Args:
            target_metrics: 目标指标 {metric_name: target_value}

        Returns:
            调整后的策略参数
        """
        adjustments = {}

        for metric, target in target_metrics.items():
            if metric == "exploration_rate":
                adjustment = target - self.exploration_rate
                self.exploration_rate = max(0.05, min(0.5, target))
                adjustments["exploration_rate"] = self.exploration_rate
            elif metric == "learning_rate":
                self.learning_rate = max(0.01, min(0.5, target))
                adjustments["learning_rate"] = self.learning_rate
            elif metric == "discount_factor":
                self.discount_factor = max(0.5, min(0.99, target))
                adjustments["discount_factor"] = self.discount_factor

        self._strategy_adjustments.append({
            "iteration": self._total_learn_iterations,
            "type": "manual_adjustment",
            "target_metrics": target_metrics,
            "result": adjustments,
        })

        return {
            "adjusted": adjustments,
            "current_params": {
                "exploration_rate": self.exploration_rate,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
            },
        }

    def get_policy_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        policy_summary = {}
        for state_key, actions in self._policy.items():
            if actions:
                best_action = max(actions, key=actions.get)
                policy_summary[state_key] = {
                    "best_action": best_action,
                    "best_q": actions[best_action],
                    "action_count": len(actions),
                }
        return {
            "policy_entries": len(policy_summary),
            "policy_summary": policy_summary,
            "total_learn_iterations": self._total_learn_iterations,
        }

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "discount_factor": self.discount_factor,
            "total_learn_iterations": self._total_learn_iterations,
            "policy_size": len(self._policy),
            "outcomes_count": len(self._outcomes),
            "strategy_adjustments_count": len(self._strategy_adjustments),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LearningScheduler':
        learner = cls(
            learning_rate=data.get("learning_rate", 0.1),
            exploration_rate=data.get("exploration_rate", 0.2),
            discount_factor=data.get("discount_factor", 0.9),
        )
        learner._total_learn_iterations = data.get("total_learn_iterations", 0)
        return learner
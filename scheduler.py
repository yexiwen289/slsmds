"""
智能专家调度系统 —— 多臂赌博机（UCB）算法 + 多维评估 + 多样性约束

核心算法：
1. 每个专家是一个"摇臂"，有不确定的贡献价值
2. UCB（Upper Confidence Bound）平衡探索与利用
3. 多维评估：质量、新颖度、影响力、参与度、领域覆盖
4. 多样性约束：确保不同专业视角的专家都有发言机会
5. 自适应发言人数：随专家总数动态调整
"""

import math
from typing import List, Dict, Optional, Tuple
from player import Player


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
            status = "🎤" if profile.rounds_since_last_spoke == 0 else f"⏳{profile.rounds_since_last_spoke}轮"
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
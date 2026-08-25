"""
精华池（Essence Pool）—— 集体讨论的核心智慧沉淀机制

核心概念：
- 所有 AI 玩家围绕一个开放性问题进行多轮讨论
- 每一轮讨论后，系统从发言中提炼出"精华"（关键见解、论据、创新点）
- 精华被收集到"精华池"中，供后续轮次参考和深化
- 玩家可以基于精华池中的已有精华进行反驳、深化、补充
- 最终，从精华池中综合生成最终解决方案

精华评分机制：
- 每个精华条目的初始分数由提炼者（LLM）给出
- 后续轮次中，其他玩家可对精华进行"引用"（+1）、"深化"（+2）、"反驳"（-1）
- 高质量精华自然沉淀，低质量精华逐渐沉底

扩展系统：
- 知识图谱：从精华中提取实体和关系，构建概念层次结构
- 时序演化分析：追踪精华随时间演变的模式
- 矛盾检测：识别精华之间的冲突和矛盾
- 自动摘要：多级别精华摘要生成
- 质量预测：基于特征预测精华质量
- 主题建模：将精华聚类为主题
- 生命周期管理：管理精华从诞生到归档的完整生命周期
- 交叉授粉追踪：追踪精华之间的影响链
- 共识强度测量：评估共识达成程度
- 空白分析：识别讨论中的缺失主题
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Set
from collections import defaultdict, Counter, deque
import datetime
import math
import random
import statistics
import json
import re


@dataclass
class EssenceItem:
    """精华池中的一条精华"""
    id: int
    content: str                     # 精华内容
    contributor: str                 # 贡献者（原始提出者）
    source_round: int                # 来源轮次
    round: int                       # 添加/更新的轮次
    score: float = 0.0               # 当前评分
    parent_id: Optional[int] = None  # 父精华ID（用于追踪深化/反驳关系）
    tags: List[str] = field(default_factory=list)  # 标签：["论点", "论据", "创新点", "反驳", "深化"]
    cited_by: List[str] = field(default_factory=list)  # 引用过的玩家
    refined_by: List[str] = field(default_factory=list)  # 深化过的玩家
    challenged_by: List[str] = field(default_factory=list)  # 反驳过的玩家
    approve_by: List[str] = field(default_factory=list)  # 投赞同票的玩家
    reject_by: List[str] = field(default_factory=list)  # 投反对票的玩家
    abstain_by: List[str] = field(default_factory=list)  # 弃权的玩家
    vote_reasons: List[dict] = field(default_factory=list)  # 投票理由记录
    clarifications: List[dict] = field(default_factory=list)  # 澄清记录：[{question, answer, asker, round}]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "contributor": self.contributor,
            "source_round": self.source_round,
            "round": self.round,
            "score": self.score,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "cited_by": self.cited_by,
            "refined_by": self.refined_by,
            "challenged_by": self.challenged_by,
            "approve_by": self.approve_by,
            "reject_by": self.reject_by,
            "abstain_by": self.abstain_by,
            "vote_reasons": self.vote_reasons,
            "clarifications": self.clarifications,
        }


class EssencePool:
    """精华池——管理所有精华条目的生命周期"""

    def __init__(self):
        self.items: List[EssenceItem] = []
        self._next_id: int = 1
        self._history: List[dict] = []  # 操作历史

    def add_essence(self, content: str, contributor: str, round_id: int,
                    parent_id: Optional[int] = None,
                    tags: Optional[List[str]] = None,
                    score: float = 1.0) -> EssenceItem:
        """添加一条精华到池中"""
        item = EssenceItem(
            id=self._next_id,
            content=content,
            contributor=contributor,
            source_round=round_id,
            round=round_id,
            score=score,
            parent_id=parent_id,
            tags=tags or [],
        )
        self._next_id += 1
        self.items.append(item)
        self._history.append({
            "action": "add",
            "item_id": item.id,
            "content": content[:50],
            "contributor": contributor,
            "round": round_id,
        })
        return item

    def cite_essence(self, item_id: int, player_name: str, round_id: int) -> bool:
        """引用一条精华（增加评分）"""
        item = self._get_item(item_id)
        if not item or player_name in item.cited_by:
            return False
        item.cited_by.append(player_name)
        item.score += 1.0
        item.round = round_id
        self._history.append({
            "action": "cite",
            "item_id": item_id,
            "player": player_name,
            "round": round_id,
        })
        return True

    def refine_essence(self, item_id: int, new_content: str,
                       player_name: str, round_id: int) -> EssenceItem:
        """基于某条精华提出深化版本（增加评分，生成子条目）"""
        parent = self._get_item(item_id)
        if not parent:
            raise ValueError(f"精华 #{item_id} 不存在")

        child = self.add_essence(
            content=new_content,
            contributor=player_name,
            round_id=round_id,
            parent_id=item_id,
            tags=parent.tags + ["深化"],
            score=2.0,
        )
        parent.refined_by.append(player_name)
        parent.score += 2.0
        parent.round = round_id
        self._history.append({
            "action": "refine",
            "item_id": item_id,
            "child_id": child.id,
            "player": player_name,
            "round": round_id,
        })
        return child

    def challenge_essence(self, item_id: int, player_name: str, round_id: int) -> bool:
        """反驳一条精华（降低评分）"""
        item = self._get_item(item_id)
        if not item or player_name in item.challenged_by:
            return False
        item.challenged_by.append(player_name)
        item.score -= 1.0
        item.round = round_id
        self._history.append({
            "action": "challenge",
            "item_id": item_id,
            "player": player_name,
            "round": round_id,
        })
        return True

    def vote_essence(self, item_id: int, player_name: str,
                     vote: str, reason: str, round_id: int) -> bool:
        """
        对一条精华投票（approve/reject/abstain）。
        - approve: 评分 +1.0
        - reject: 评分 -1.5（反对的惩罚强于反驳，因为是全员评估）
        - abstain: 不影响评分
        同一玩家对同一精华只能投一次票。
        """
        item = self._get_item(item_id)
        if not item:
            return False
        # 检查是否已投过票
        all_voted = set(item.approve_by) | set(item.reject_by) | set(item.abstain_by)
        if player_name in all_voted:
            return False

        if vote == "approve":
            item.approve_by.append(player_name)
            item.score += 1.0
        elif vote == "reject":
            item.reject_by.append(player_name)
            item.score -= 1.5
        else:  # abstain
            item.abstain_by.append(player_name)

        item.vote_reasons.append({
            "player": player_name,
            "vote": vote,
            "reason": reason[:80],
            "round": round_id,
        })
        self._history.append({
            "action": "vote",
            "item_id": item_id,
            "player": player_name,
            "vote": vote,
            "round": round_id,
        })
        return True

    def get_vote_summary(self, item_id: int) -> dict:
        """获取某条精华的投票摘要"""
        item = self._get_item(item_id)
        if not item:
            return {}
        return {
            "approve": len(item.approve_by),
            "reject": len(item.reject_by),
            "abstain": len(item.abstain_by),
            "net": len(item.approve_by) - len(item.reject_by),
        }

    def add_clarification(self, item_id: int, question: str, answer: str,
                          asker: str, round_id: int) -> bool:
        """对一条精华追加澄清问答记录（不修改评分，仅补充上下文）"""
        item = self._get_item(item_id)
        if not item:
            return False
        item.clarifications.append({
            "question": question[:200],
            "answer": answer[:500],
            "asker": asker,
            "round": round_id,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        item.round = round_id
        self._history.append({
            "action": "clarify",
            "item_id": item_id,
            "asker": asker,
            "round": round_id,
        })
        return True

    def calculate_consensus(self, total_players: int, goal_mode: str = "balance") -> dict:
        """
        计算当前讨论的共识度指标。

        综合考虑：
        - 投票一致性：精华的 approve/(approve+reject) 比例均值
        - 反驳密度：被反驳精华占比（越高共识越低）
        - 弃权比例：高弃权表示观点不清晰

        受 goal_mode 影响：
        - converge: 降低"高共识"门槛（加速收敛）
        - explore: 提高"高共识"门槛（鼓励继续辩论）
        - balance: 默认阈值

        返回：
          level: "high" | "medium" | "low" | "assessing"
          score: 0.0-1.0
          suggested_action: str
          details: 详细统计
        """
        if not self.items or total_players <= 0:
            return {
                "level": "assessing",
                "score": 0.0,
                "suggested_action": "继续讨论，正在收集观点",
                "details": {"reason": "尚无精华或玩家"},
            }

        # 仅统计本轮之前已投票的精华（source_round < current）
        voted_items = [it for it in self.items
                       if (it.approve_by or it.reject_by or it.abstain_by)]
        challenged_count = sum(1 for it in self.items if it.challenged_by)

        # 冷启动：投票数据不足时标记为"评估中"，不误导用户
        if not voted_items:
            sample_count = len(self.items)
            if sample_count < 3:
                action = "继续讨论，观点收集中"
            else:
                action = "即将进入投票阶段"
            return {
                "level": "assessing",
                "score": 0.5,
                "suggested_action": action,
                "details": {"reason": f"已有 {sample_count} 条精华，等待第一轮投票结果"},
            }

        # 根据 goal_mode 设置共识阈值
        if goal_mode == "converge":
            high_threshold = 0.55  # 收敛模式：更容易达到高共识
            medium_threshold = 0.30
        elif goal_mode == "explore":
            high_threshold = 0.80  # 探索模式：更难达到高共识
            medium_threshold = 0.45
        else:  # balance
            high_threshold = 0.70
            medium_threshold = 0.40

        # 1. 投票一致性：每条精华的 approve 比例
        approve_ratios = []
        abstain_ratios = []
        for it in voted_items:
            total_votes = len(it.approve_by) + len(it.reject_by) + len(it.abstain_by)
            if total_votes == 0:
                continue
            approve_ratios.append(len(it.approve_by) / total_votes)
            abstain_ratios.append(len(it.abstain_by) / total_votes)

        avg_approve = sum(approve_ratios) / len(approve_ratios) if approve_ratios else 0.0
        avg_abstain = sum(abstain_ratios) / len(abstain_ratios) if abstain_ratios else 0.0

        # 2. 反驳密度
        challenge_ratio = challenged_count / len(self.items)

        # 3. 综合评分：approve 占比 - 反驳惩罚 - 弃权惩罚
        score = avg_approve - 0.3 * challenge_ratio - 0.2 * avg_abstain
        score = max(0.0, min(1.0, score))

        if score >= high_threshold:
            level = "high"
            if goal_mode == "converge":
                suggested = "共识已达成，建议输出方案"
            elif goal_mode == "explore":
                suggested = "探索充分，可转向收敛或输出方案"
            else:
                suggested = "可以总结"
        elif score >= medium_threshold:
            level = "medium"
            suggested = "寻求共识"
        else:
            level = "low"
            suggested = "继续讨论"

        return {
            "level": level,
            "score": round(score, 3),
            "suggested_action": suggested,
            "details": {
                "avg_approve_ratio": round(avg_approve, 3),
                "avg_abstain_ratio": round(avg_abstain, 3),
                "challenge_ratio": round(challenge_ratio, 3),
                "voted_items": len(voted_items),
                "total_items": len(self.items),
                "goal_mode": goal_mode,
            },
        }

    def get_top_essences(self, n: int = 5) -> List[EssenceItem]:
        """获取评分最高的 N 条精华"""
        sorted_items = sorted(self.items, key=lambda x: x.score, reverse=True)
        return sorted_items[:n]

    def get_essences_by_round(self, round_id: int) -> List[EssenceItem]:
        """获取指定轮次添加的精华"""
        return [item for item in self.items if item.source_round == round_id]

    def get_essence_tree(self, item_id: int) -> List[EssenceItem]:
        """获取某条精华及其所有子精华（深化链）"""
        result = []
        item = self._get_item(item_id)
        if not item:
            return result
        result.append(item)
        for child in self.items:
            if child.parent_id == item_id:
                result.extend(self.get_essence_tree(child.id))
        return result

    def get_pool_summary(self, top_n: int = 5, compressed: bool = False) -> str:
        """生成精华池摘要文本（用于大模型上下文）"""
        if not self.items:
            return "（空）" if compressed else "（精华池为空，尚无已提炼的见解）"

        top = self.get_top_essences(top_n)
        if compressed:
            from .compression import compress_essence_pool
            return compress_essence_pool(
                [e.to_dict() for e in top], max_items=top_n
            )

        lines = [
            "=" * 50,
            "📋 精华池当前状态（按评分排序）",
            "=" * 50,
        ]
        for i, item in enumerate(top, 1):
            tags_str = f"[{', '.join(item.tags)}]" if item.tags else ""
            lines.append(
                f"  #{i} (ID:{item.id}) 评分:{item.score:.1f} {tags_str}\n"
                f"     内容: {item.content}\n"
                f"     贡献者: {item.contributor} (第{item.source_round}轮)\n"
                f"     引用:{len(item.cited_by)}人 深化:{len(item.refined_by)}次 反驳:{len(item.challenged_by)}次 澄清:{len(item.clarifications)}次"
            )
        lines.append(f"\n  精华池总计: {len(self.items)} 条精华")
        lines.append("=" * 50)
        return "\n".join(lines)

    def get_controversy_map(self) -> dict:
        """
        构建争议地图：精华之间的关系链。

        返回：
          {
            "chains": [                          # 深化链（支持→支持）
                {"root": EssenceItem, "children": [EssenceItem, ...]},
                ...
            ],
            "challenges": [                      # 反驳关系
                {"target": EssenceItem, "challenger_names": [name,...], "detail_str": str},
                ...
            ],
            "clarifications": [                  # 澄清记录
                {"target": EssenceItem, "count": int, "detail_str": str},
                ...
            ],
            "voted_items": int,                  # 已投票精华数
            "total_items": int,
          }
        """
        # 1) 构建 parent → children 映射（深化链）
        roots = []
        children_map = {}
        for it in self.items:
            if it.parent_id is None:
                roots.append(it)
            else:
                children_map.setdefault(it.parent_id, []).append(it)

        chains = []
        for root in roots:
            # BFS 展开子节点
            children = []
            stack = [root.id]
            while stack:
                pid = stack.pop()
                for ch in children_map.get(pid, []):
                    children.append(ch)
                    stack.append(ch.id)
            if children:
                chains.append({"root": root, "children": children})

        # 2) 找出被反驳/挑战的精华
        challenges = []
        for it in self.items:
            if it.challenged_by:
                # 找到挑战它的精华（parent_id 链 + 反驳标签）
                challengers = [ch for ch in self.items
                               if it.id in ch.parent_id and any("反驳" in t for t in ch.tags)]
                detail = f"{it.contributor} 提出 #{it.id} (评分 {it.score:.1f})，被 {', '.join(it.challenged_by)} 反驳"
                if challengers:
                    detail += f"；反方观点: {challengers[0].content[:60]}..."
                challenges.append({
                    "target": it,
                    "challenger_names": list(it.challenged_by),
                    "detail_str": detail,
                })
        # 按挑战人数降序
        challenges.sort(key=lambda x: len(x["challenger_names"]), reverse=True)

        # 3) 被澄清的精华
        clarifications = []
        for it in self.items:
            if it.clarifications:
                questions = "; ".join(c["question"][:30] for c in it.clarifications[:2])
                detail = f"{it.contributor} 的 #{it.id} 被澄清 {len(it.clarifications)} 次，追问: {questions}"
                clarifications.append({
                    "target": it,
                    "count": len(it.clarifications),
                    "detail_str": detail,
                })
        clarifications.sort(key=lambda x: x["count"], reverse=True)

        voted = sum(1 for it in self.items
                    if (it.approve_by or it.reject_by or it.abstain_by))
        return {
            "chains": chains,
            "challenges": challenges,
            "clarifications": clarifications,
            "voted_items": voted,
            "total_items": len(self.items),
        }

    def get_all_essences_text(self) -> str:
        """获取所有精华的完整文本（用于最终综合）"""
        if not self.items:
            return "（无精华条目）"

        sorted_items = sorted(self.items, key=lambda x: x.score, reverse=True)
        lines = []
        for item in sorted_items:
            tags_str = f"[{', '.join(item.tags)}]" if item.tags else ""
            lines.append(
                f"  - [{item.id}] {item.content} "
                f"(贡献者:{item.contributor}, 评分:{item.score:.1f}, {tags_str})"
            )
        return "\n".join(lines)

    def get_evolution_summary(self) -> str:
        """获取精华演化历史摘要"""
        if not self._history:
            return "无演化历史"

        lines = ["精华池演化历史："]
        for h in self._history:
            if h["action"] == "add":
                lines.append(f"  第{h['round']}轮 | {h['contributor']} 提出: \"{h['content']}\"")
            elif h["action"] == "refine":
                lines.append(f"  第{h['round']}轮 | {h['player']} 深化 #{h['item_id']}")
            elif h["action"] == "cite":
                lines.append(f"  第{h['round']}轮 | {h['player']} 引用 #{h['item_id']}")
            elif h["action"] == "challenge":
                lines.append(f"  第{h['round']}轮 | {h['player']} 反驳 #{h['item_id']}")
        return "\n".join(lines)

    def _get_item(self, item_id: int) -> Optional[EssenceItem]:
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def remove_essence(self, item_id: int) -> bool:
        """手动删除一条精华"""
        item = self._get_item(item_id)
        if not item:
            return False
        self.items.remove(item)
        self._history.append({
            "action": "remove",
            "item_id": item_id,
            "content": item.content[:50],
        })
        return True

    def update_essence(self, item_id: int, content: str = None, score: float = None) -> bool:
        """手动更新一条精华的内容或评分"""
        item = self._get_item(item_id)
        if not item:
            return False
        if content is not None:
            item.content = content
        if score is not None:
            item.score = score
        self._history.append({
            "action": "update",
            "item_id": item_id,
            "content": item.content[:50],
            "score": item.score,
        })
        return True

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "history": self._history,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EssencePool':
        """从字典恢复精华池状态"""
        pool = cls()
        pool._next_id = data.get("next_id", 1)
        pool._history = data.get("history", [])
        for item_data in data.get("items", []):
            item = EssenceItem(
                id=item_data["id"],
                content=item_data["content"],
                contributor=item_data["contributor"],
                source_round=item_data["source_round"],
                round=item_data.get("round", item_data["source_round"]),
                score=item_data.get("score", 0.0),
                parent_id=item_data.get("parent_id"),
                tags=item_data.get("tags", []),
                cited_by=item_data.get("cited_by", []),
                refined_by=item_data.get("refined_by", []),
                challenged_by=item_data.get("challenged_by", []),
                approve_by=item_data.get("approve_by", []),
                reject_by=item_data.get("reject_by", []),
                abstain_by=item_data.get("abstain_by", []),
                vote_reasons=item_data.get("vote_reasons", []),
                clarifications=item_data.get("clarifications", []),
            )
            pool.items.append(item)
        return pool


# ============================================================================
# 扩展系统 1: EssenceKnowledgeGraph (知识图谱)
# ============================================================================

@dataclass
class EntityNode:
    """知识图谱中的实体节点"""
    name: str
    entity_type: str  # concept, method, person, domain, etc.
    frequency: int = 1
    first_seen_round: int = 0
    last_seen_round: int = 0
    related_essence_ids: List[int] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationEdge:
    """知识图谱中的关系边"""
    source: str
    target: str
    relation_type: str  # is_a, part_of, supports, contradicts, refines, etc.
    weight: float = 1.0
    essence_ids: List[int] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


class EssenceKnowledgeGraph:
    """
    精华知识图谱 —— 从精华条目中提取实体和关系，构建概念层次结构。

    功能：
    - 实体提取：从精华内容中识别关键概念、方法、领域等实体
    - 关系构建：分析实体之间的语义关系
    - 概念层次：构建 is_a / part_of 层次结构
    - 交叉引用：追踪精华之间的跨引用关系
    """

    # 常见中文停用词
    STOP_WORDS: Set[str] = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
        "它", "们", "那", "些", "为", "与", "及", "或", "但", "而",
        "从", "以", "对", "被", "把", "向", "让", "给", "用", "能",
        "可以", "这个", "那个", "什么", "怎么", "如何", "为什么", "因为",
        "所以", "但是", "如果", "虽然", "而且", "或者", "不过", "然而",
    }

    # 关系类型关键词对
    RELATION_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
        "is_a": [("是一种", "是"), ("属于", "归为"), ("是一种类型的", "类型")],
        "part_of": [("包含", "包括"), ("由...组成", "组成部分"), ("由...构成", "构成")],
        "supports": [("支持", "支撑"), ("基于", "根据"), ("有助于", "促进")],
        "contradicts": [("矛盾", "相反"), ("反对", "不认同"), ("不同于", "区别于")],
        "refines": [("深化", "细化"), ("补充", "完善"), ("改进", "提升")],
        "causes": [("导致", "引发"), ("造成", "产生"), ("使得", "促使")],
        "depends_on": [("依赖于", "取决于"), ("需要", "要求"), ("前提是", "先决条件")],
    }

    def __init__(self):
        self.entities: Dict[str, EntityNode] = {}  # name -> EntityNode
        self.relations: List[RelationEdge] = []
        self._entity_relation_map: Dict[str, List[int]] = defaultdict(list)  # entity_name -> relation indices
        self._cross_references: Dict[int, List[int]] = defaultdict(list)  # essence_id -> [related essence_ids]
        self._built: bool = False

    def extract_entities(self, essence: EssenceItem, lang: str = "zh") -> List[str]:
        """
        从一条精华中提取实体。

        策略：
        - 中文：基于标点/引号分割 + 长度过滤 + 停用词过滤
        - 英文：基于空格分割 + 首字母大写单词 + 长度过滤
        返回提取到的实体名称列表。
        """
        content = essence.content
        extracted: List[str] = []

        if lang == "zh":
            # 中文实体提取：寻找引号内的概念、双字以上词汇
            quoted = re.findall(r'["""「」『』【】《》]([^""「」『』【】《》]{2,20})["""「」『』【】《》]', content)
            extracted.extend(quoted)

            # 提取书名号内的概念
            angle = re.findall(r'[《》]([^《》]{2,20})[《》]', content)
            extracted.extend(angle)

            # 按常见分隔符分词
            segments = re.split(r'[，。！？、；：,\.!\?;:\s()（）\[\]]', content)
            for seg in segments:
                seg = seg.strip()
                if len(seg) >= 2 and len(seg) <= 20 and seg not in self.STOP_WORDS:
                    # 避免纯数字或纯标点
                    if not re.match(r'^[\d\s%#@]+$', seg):
                        extracted.append(seg)
        else:
            # 英文实体提取：大写单词、引号内容
            quoted = re.findall(r'[""]([^""]{2,50})[""]', content)
            extracted.extend(quoted)
            words = re.findall(r'[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*', content)
            extracted.extend(words)
            # 普通名词短语
            for w in re.findall(r'[a-zA-Z]{4,}', content):
                if w.lower() not in {"this", "that", "with", "from", "have", "what", "which", "where", "there"}:
                    extracted.append(w)

        # 去重
        unique_names = list(dict.fromkeys(extracted))

        # 更新实体节点
        now = datetime.datetime.now()
        for name in unique_names:
            if name in self.entities:
                self.entities[name].frequency += 1
                self.entities[name].last_seen_round = essence.round
                if essence.id not in self.entities[name].related_essence_ids:
                    self.entities[name].related_essence_ids.append(essence.id)
            else:
                self.entities[name] = EntityNode(
                    name=name,
                    entity_type=self._infer_entity_type(name),
                    frequency=1,
                    first_seen_round=essence.source_round,
                    last_seen_round=essence.round,
                    related_essence_ids=[essence.id],
                )

        self._built = False
        return unique_names

    def _infer_entity_type(self, name: str) -> str:
        """推断实体类型"""
        # 方法/方法论：以"法"、"论"、"式"结尾
        if re.search(r'[法论式]$', name):
            return "method"
        # 理论/主义：以"主义"、"理论"、"学说"结尾
        if re.search(r'(主义|理论|学说|原理)$', name):
            return "theory"
        # 系统/框架：以"系统"、"框架"、"体系"结尾
        if re.search(r'(系统|框架|体系|平台)$', name):
            return "system"
        # 领域/学科
        if re.search(r'(学|术|科|工程|技术)$', name):
            return "domain"
        # 默认：概念
        return "concept"

    def extract_relations(self, essence: EssenceItem, lang: str = "zh") -> List[RelationEdge]:
        """
        从一条精华中提取实体之间的关系。

        通过分析精华内容中的关系关键词模式，识别实体之间的语义关系。
        返回新发现的关系列表。
        """
        content = essence.content
        found_entities = [e for e in self.entities.values()
                          if essence.id in e.related_essence_ids]
        entity_names = [e.name for e in found_entities]

        if len(entity_names) < 2:
            return []

        new_relations: List[RelationEdge] = []
        existing_pairs = set()

        for rel_type, patterns in self.RELATION_PATTERNS.items():
            for pattern_words in patterns:
                for kw in pattern_words:
                    if kw in content:
                        # 找到实体对
                        for i, src in enumerate(entity_names):
                            src_pos = content.find(src)
                            if src_pos < 0:
                                continue
                            for j, tgt in enumerate(entity_names):
                                if i == j or tgt == src:
                                    continue
                                tgt_pos = content.find(tgt)
                                if tgt_pos < 0:
                                    continue
                                pair_key = f"{src}|{rel_type}|{tgt}"
                                if pair_key in existing_pairs:
                                    continue
                                # 检查关键词是否在实体之间
                                kw_pos = content.find(kw)
                                if min(src_pos, tgt_pos) < kw_pos < max(src_pos, tgt_pos):
                                    existing_pairs.add(pair_key)
                                    edge = RelationEdge(
                                        source=src,
                                        target=tgt,
                                        relation_type=rel_type,
                                        weight=1.0,
                                        essence_ids=[essence.id],
                                    )
                                    new_relations.append(edge)
                                    self.relations.append(edge)
                                    self._entity_relation_map[src].append(len(self.relations) - 1)
                                    self._entity_relation_map[tgt].append(len(self.relations) - 1)
                                    break

        self._built = False
        return new_relations

    def build_graph(self) -> Dict[str, Any]:
        """
        构建完整知识图谱，包括：
        - 实体统计
        - 关系网络
        - 概念层次树
        - 交叉引用网络

        返回图谱的完整结构化表示。
        """
        # 1. 构建概念层次树
        hierarchy: Dict[str, List[str]] = defaultdict(list)
        for rel in self.relations:
            if rel.relation_type == "is_a":
                hierarchy[rel.target].append(rel.source)
            elif rel.relation_type == "part_of":
                hierarchy[rel.source].append(rel.target)

        # 2. 构建交叉引用网络
        cross_refs: Dict[int, List[int]] = {}
        for eid in self._cross_references:
            cross_refs[eid] = list(self._cross_references[eid])

        # 3. 自动发现缺失的交叉引用
        # 如果两个精华共享多个实体，推定它们相关
        shared_entity_threshold = 2
        essence_ids = set()
        for e in self.entities.values():
            for eid in e.related_essence_ids:
                essence_ids.add(eid)

        eid_list = sorted(essence_ids)
        for i in range(len(eid_list)):
            for j in range(i + 1, len(eid_list)):
                eid1, eid2 = eid_list[i], eid_list[j]
                shared = []
                for e in self.entities.values():
                    if eid1 in e.related_essence_ids and eid2 in e.related_essence_ids:
                        shared.append(e.name)
                if len(shared) >= shared_entity_threshold:
                    if eid2 not in self._cross_references[eid1]:
                        self._cross_references[eid1].append(eid2)
                    if eid1 not in self._cross_references[eid2]:
                        self._cross_references[eid2].append(eid1)

        self._built = True

        return {
            "entities": {
                name: {
                    "type": node.entity_type,
                    "frequency": node.frequency,
                    "first_seen": node.first_seen_round,
                    "last_seen": node.last_seen_round,
                    "related_essences": len(node.related_essence_ids),
                }
                for name, node in self.entities.items()
            },
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.relation_type,
                    "weight": r.weight,
                    "essence_ids": r.essence_ids,
                }
                for r in self.relations
            ],
            "hierarchy": dict(hierarchy),
            "cross_references": {str(k): v for k, v in cross_refs.items()},
            "stats": {
                "total_entities": len(self.entities),
                "total_relations": len(self.relations),
                "total_cross_references": sum(len(v) for v in self._cross_references.values()),
            },
        }

    def get_central_concepts(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        获取核心概念（按中心度排序）。

        中心度计算基于：
        - 实体频率
        - 关联关系数量
        - 关联精华数量
        """
        concept_scores: List[Tuple[str, float]] = []
        for name, node in self.entities.items():
            relation_count = len(self._entity_relation_map.get(name, []))
            centrality = (
                0.4 * (node.frequency / max(1, max(e.frequency for e in self.entities.values()))) +
                0.3 * (relation_count / max(1, max(len(self._entity_relation_map.get(n, [])) for n in self.entities))) +
                0.3 * (len(node.related_essence_ids) / max(1, max(len(e.related_essence_ids) for e in self.entities.values())))
            )
            concept_scores.append((name, centrality))

        concept_scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "name": name,
                "centrality": round(score, 4),
                "type": self.entities[name].entity_type,
                "frequency": self.entities[name].frequency,
                "relation_count": len(self._entity_relation_map.get(name, [])),
                "essence_count": len(self.entities[name].related_essence_ids),
            }
            for name, score in concept_scores[:top_n]
        ]

    def find_related_essences(self, essence_id: int, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        查找与给定精华相关的其他精华（基于知识图谱）。

        使用 BFS 遍历实体共享图和交叉引用网络。
        """
        target_essence = None
        # 找涉及的实体
        related_entities = []
        for node in self.entities.values():
            if essence_id in node.related_essence_ids:
                related_entities.append(node)

        if not related_entities:
            return []

        # BFS 收集相关精华
        visited_essences: Set[int] = {essence_id}
        queue: deque = deque([(essence_id, 0)])
        related: List[Dict[str, Any]] = []

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # 通过实体共享找关联
            for node in self.entities.values():
                if current_id in node.related_essence_ids:
                    for neighbor_id in node.related_essence_ids:
                        if neighbor_id not in visited_essences:
                            visited_essences.add(neighbor_id)
                            shared_entities = [e.name for e in self.entities.values()
                                               if current_id in e.related_essence_ids
                                               and neighbor_id in e.related_essence_ids]
                            related.append({
                                "essence_id": neighbor_id,
                                "depth": depth + 1,
                                "shared_entities": shared_entities,
                                "relation": "shared_entity",
                            })
                            queue.append((neighbor_id, depth + 1))

            # 通过交叉引用找关联
            for neighbor_id in self._cross_references.get(current_id, []):
                if neighbor_id not in visited_essences:
                    visited_essences.add(neighbor_id)
                    related.append({
                        "essence_id": neighbor_id,
                        "depth": depth + 1,
                        "shared_entities": [],
                        "relation": "cross_reference",
                    })
                    queue.append((neighbor_id, depth + 1))

        # 去重（按 essence_id）
        seen = set()
        unique_related = []
        for r in related:
            if r["essence_id"] not in seen:
                seen.add(r["essence_id"])
                unique_related.append(r)

        return unique_related

    def get_concept_path(self, concept_a: str, concept_b: str) -> List[Dict[str, Any]]:
        """
        查找两个概念之间的路径（BFS 最短路径）。

        返回路径上的关系序列。
        """
        if concept_a not in self.entities or concept_b not in self.entities:
            return []

        # 构建邻接表
        adj: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        for idx, rel in enumerate(self.relations):
            adj[rel.source].append((rel.target, rel.relation_type, idx))
            adj[rel.target].append((rel.source, rel.relation_type, idx))

        # BFS
        visited: Set[str] = {concept_a}
        queue: deque = deque([(concept_a, [])])

        while queue:
            current, path = queue.popleft()
            if current == concept_b:
                return path

            for neighbor, rel_type, rel_idx in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [{
                        "from": current,
                        "to": neighbor,
                        "relation": rel_type,
                        "relation_index": rel_idx,
                    }]
                    queue.append((neighbor, new_path))

        return []

    def to_dict(self) -> dict:
        return {
            "entities": {name: {
                "name": node.name,
                "type": node.entity_type,
                "frequency": node.frequency,
                "first_seen_round": node.first_seen_round,
                "last_seen_round": node.last_seen_round,
                "related_essence_ids": node.related_essence_ids,
                "attributes": node.attributes,
            } for name, node in self.entities.items()},
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "weight": r.weight,
                    "essence_ids": r.essence_ids,
                }
                for r in self.relations
            ],
            "cross_references": {str(k): v for k, v in self._cross_references.items()},
        }


# ============================================================================
# 扩展系统 2: TemporalEvolutionAnalyzer (时序演化分析)
# ============================================================================

@dataclass
class EvolutionPoint:
    """时序演化数据点"""
    round: int
    score: float
    citation_count: int
    challenge_count: int
    refine_count: int
    clarification_count: int
    entity_count: int


@dataclass
class LifecycleStage:
    """概念生命周期阶段"""
    stage: str  # birth, growth, maturity, decay, dormant
    start_round: int
    confidence: float  # 0.0 - 1.0


class TemporalEvolutionAnalyzer:
    """
    时序演化分析器 —— 追踪精华如何随时间演变。

    功能：
    - 逐轮追踪精华评分和引用变化
    - 检测概念的生命周期（诞生/成长/成熟/衰退）
    - 测量主题漂移（thematic drift）
    - 识别趋势和模式
    """

    def __init__(self):
        self._evolution_history: Dict[int, List[EvolutionPoint]] = defaultdict(list)
        self._round_items: Dict[int, List[int]] = defaultdict(list)  # round -> essence ids
        self._stage_cache: Dict[int, LifecycleStage] = {}
        self._drift_history: List[Dict[str, Any]] = []

    def record_round(self, pool: EssencePool, round_id: int) -> None:
        """
        记录一轮的精华状态快照。

        为池中每条精华生成 EvolutionPoint 并存储。
        """
        self._round_items[round_id] = [item.id for item in pool.items]

        for item in pool.items:
            point = EvolutionPoint(
                round=round_id,
                score=item.score,
                citation_count=len(item.cited_by),
                challenge_count=len(item.challenged_by),
                refine_count=len(item.refined_by),
                clarification_count=len(item.clarifications),
                entity_count=0,  # 由外部知识图谱填充
            )
            self._evolution_history[item.id].append(point)

    def analyze_evolution(self, essence_id: int) -> Dict[str, Any]:
        """
        分析单条精华的演化轨迹。

        返回：
        - score_trajectory: 评分变化序列
        - trend: 趋势方向 (+1 上升, 0 平稳, -1 下降)
        - volatility: 波动性 (标准差)
        - peak_round: 最高评分轮次
        - current_stage: 当前生命周期阶段
        """
        points = self._evolution_history.get(essence_id, [])
        if not points:
            return {
                "essence_id": essence_id,
                "score_trajectory": [],
                "trend": 0,
                "volatility": 0.0,
                "peak_round": None,
                "current_stage": "unknown",
                "message": "无演化数据",
            }

        points.sort(key=lambda p: p.round)
        scores = [p.score for p in points]
        trajectory = [{"round": p.round, "score": p.score} for p in points]

        # 趋势：线性回归斜率
        if len(scores) >= 2:
            x = list(range(len(scores)))
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(scores)
            sum_xy = sum(xi * yi for xi, yi in zip(x, scores))
            sum_xx = sum(xi * xi for xi in x)
            denom = n * sum_xx - sum_x * sum_x
            if denom != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denom
            else:
                slope = 0.0
            trend = 1 if slope > 0.02 else (-1 if slope < -0.02 else 0)
        else:
            trend = 0

        # 波动性
        volatility = statistics.stdev(scores) if len(scores) >= 2 else 0.0

        # 峰值
        max_score = max(scores)
        peak_idx = scores.index(max_score)
        peak_round = points[peak_idx].round if points else None

        # 生命周期阶段
        stage = self._determine_stage(essence_id, points, trend)

        return {
            "essence_id": essence_id,
            "score_trajectory": trajectory,
            "trend": trend,
            "volatility": round(volatility, 4),
            "peak_round": peak_round,
            "peak_score": max_score,
            "current_score": scores[-1] if scores else 0.0,
            "current_stage": stage.stage,
            "stage_confidence": round(stage.confidence, 3),
            "data_points": len(points),
        }

    def _determine_stage(self, essence_id: int, points: List[EvolutionPoint],
                         trend: int) -> LifecycleStage:
        """根据演化数据确定生命周期阶段"""
        if len(points) < 2:
            stage = LifecycleStage(stage="birth", start_round=points[0].round, confidence=0.5)
            self._stage_cache[essence_id] = stage
            return stage

        scores = [p.score for p in points]
        recent = scores[-3:] if len(scores) >= 3 else scores
        first_round = points[0].round
        last_round = points[-1].round

        # 诞生：刚出现，评分低但引用在增长
        if len(points) <= 2 and trend >= 0:
            stage = LifecycleStage(stage="birth", start_round=first_round, confidence=0.7)
            self._stage_cache[essence_id] = stage
            return stage

        # 成长：评分快速上升，引用增多
        if trend > 0 and scores[-1] > scores[0] * 1.2:
            confidence = min(0.9, 0.5 + 0.1 * (scores[-1] - scores[0]) / max(1, scores[0]))
            stage = LifecycleStage(stage="growth", start_round=first_round, confidence=confidence)
            self._stage_cache[essence_id] = stage
            return stage

        # 成熟：评分稳定，波动小
        if len(recent) >= 3 and statistics.stdev(recent) < 0.1 * max(1, sum(recent) / len(recent)):
            confidence = 0.8
            stage = LifecycleStage(stage="maturity", start_round=first_round, confidence=confidence)
            self._stage_cache[essence_id] = stage
            return stage

        # 衰退：评分持续下降
        if trend < 0 and len(scores) >= 3:
            decline_rate = (scores[0] - scores[-1]) / max(1, scores[0])
            confidence = min(0.9, 0.3 + decline_rate)
            stage = LifecycleStage(stage="decay", start_round=first_round, confidence=confidence)
            self._stage_cache[essence_id] = stage
            return stage

        # 休眠：长期无变化
        if len(points) >= 3 and all(p.score == points[-1].score for p in points[-3:]):
            stage = LifecycleStage(stage="dormant", start_round=first_round, confidence=0.6)
            self._stage_cache[essence_id] = stage
            return stage

        # 默认：保持上次判断
        if essence_id in self._stage_cache:
            return self._stage_cache[essence_id]
        return LifecycleStage(stage="birth", start_round=first_round, confidence=0.3)

    def detect_trends(self, pool: EssencePool, window: int = 3) -> List[Dict[str, Any]]:
        """
        检测整体趋势。

        分析所有精华在最近 window 轮内的变化模式。
        返回趋势列表，包括：
        - rising_stars: 快速上升的精华
        - fading_ideas: 衰退的精华
        - stable_concepts: 稳定的精华
        """
        if not pool.items:
            return []

        rising_stars: List[Dict[str, Any]] = []
        fading_ideas: List[Dict[str, Any]] = []
        stable_concepts: List[Dict[str, Any]] = []

        for item in pool.items:
            analysis = self.analyze_evolution(item.id)
            if analysis["current_stage"] == "unknown":
                continue

            if analysis["trend"] > 0 and analysis["volatility"] < 0.5:
                rising_stars.append({
                    "essence_id": item.id,
                    "content": item.content[:80],
                    "current_score": analysis["current_score"],
                    "peak_score": analysis["peak_score"],
                    "stage": analysis["current_stage"],
                })
            elif analysis["trend"] < 0:
                fading_ideas.append({
                    "essence_id": item.id,
                    "content": item.content[:80],
                    "current_score": analysis["current_score"],
                    "peak_score": analysis["peak_score"],
                    "stage": analysis["current_stage"],
                })
            else:
                stable_concepts.append({
                    "essence_id": item.id,
                    "content": item.content[:80],
                    "current_score": analysis["current_score"],
                    "stage": analysis["current_stage"],
                })

        rising_stars.sort(key=lambda x: x["current_score"], reverse=True)
        fading_ideas.sort(key=lambda x: x["current_score"])
        stable_concepts.sort(key=lambda x: x["current_score"], reverse=True)

        return [
            {"category": "rising_stars", "count": len(rising_stars), "items": rising_stars[:5]},
            {"category": "fading_ideas", "count": len(fading_ideas), "items": fading_ideas[:5]},
            {"category": "stable_concepts", "count": len(stable_concepts), "items": stable_concepts[:5]},
        ]

    def get_lifecycle(self, essence_id: int) -> LifecycleStage:
        """获取指定精华的生命周期阶段"""
        if essence_id in self._stage_cache:
            return self._stage_cache[essence_id]
        points = self._evolution_history.get(essence_id, [])
        if not points:
            return LifecycleStage(stage="unknown", start_round=0, confidence=0.0)
        return self._determine_stage(essence_id, points, 0)

    def measure_drift(self, pool: EssencePool, round_a: int, round_b: int) -> Dict[str, Any]:
        """
        测量两个轮次之间的主题漂移。

        通过比较两个轮次中精华内容的语义重叠程度来评估漂移大小。
        """
        items_a = [item for item in pool.items if item.source_round == round_a]
        items_b = [item for item in pool.items if item.source_round == round_b]

        if not items_a or not items_b:
            return {
                "drift_score": 0.0,
                "round_a": round_a,
                "round_b": round_b,
                "message": "数据不足",
            }

        # 提取关键词
        def extract_keywords(items: List[EssenceItem]) -> Set[str]:
            keywords: Set[str] = set()
            for item in items:
                words = re.split(r'[，。！？、；：,\.!\?;:\s()（）\[\]""「」]', item.content)
                for w in words:
                    w = w.strip()
                    if len(w) >= 2:
                        keywords.add(w)
            return keywords

        kw_a = extract_keywords(items_a)
        kw_b = extract_keywords(items_b)

        if not kw_a or not kw_b:
            return {"drift_score": 0.0, "round_a": round_a, "round_b": round_b, "message": "关键词不足"}

        # Jaccard 相似度
        intersection = kw_a & kw_b
        union = kw_a | kw_b
        jaccard = len(intersection) / len(union) if union else 0.0

        # 漂移 = 1 - 相似度
        drift = 1.0 - jaccard

        # 新增/消失的关键词
        new_keywords = kw_b - kw_a
        lost_keywords = kw_a - kw_b

        result = {
            "drift_score": round(drift, 4),
            "jaccard_similarity": round(jaccard, 4),
            "round_a": round_a,
            "round_b": round_b,
            "round_a_items": len(items_a),
            "round_b_items": len(items_b),
            "new_keywords_count": len(new_keywords),
            "lost_keywords_count": len(lost_keywords),
            "new_keywords_sample": list(new_keywords)[:10],
            "lost_keywords_sample": list(lost_keywords)[:10],
        }

        self._drift_history.append(result)
        return result

    def get_evolution_timeline(self, pool: EssencePool) -> List[Dict[str, Any]]:
        """
        获取精华池的完整演化时间线。

        按轮次汇总：精华数量、平均评分、新概念、活跃度等。
        """
        rounds: Set[int] = set()
        for item in pool.items:
            rounds.add(item.source_round)
            rounds.add(item.round)

        timeline = []
        for rnd in sorted(rounds):
            round_items = [item for item in pool.items if item.source_round <= rnd <= item.round]
            if not round_items:
                continue

            scores = [item.score for item in round_items]
            new_items = [item for item in pool.items if item.source_round == rnd]

            stage_counts: Dict[str, int] = defaultdict(int)
            for item in round_items:
                stage = self.get_lifecycle(item.id)
                stage_counts[stage.stage] += 1

            timeline.append({
                "round": rnd,
                "total_essences": len(round_items),
                "new_essences": len(new_items),
                "avg_score": round(statistics.mean(scores), 3),
                "max_score": max(scores),
                "min_score": min(scores),
                "stage_distribution": dict(stage_counts),
            })

        return timeline

    def to_dict(self) -> dict:
        return {
            "evolution_history": {
                str(eid): [
                    {"round": p.round, "score": p.score, "citations": p.citation_count,
                     "challenges": p.challenge_count, "refines": p.refine_count}
                    for p in points
                ]
                for eid, points in self._evolution_history.items()
            },
            "stage_cache": {
                str(eid): {"stage": s.stage, "start_round": s.start_round, "confidence": s.confidence}
                for eid, s in self._stage_cache.items()
            },
            "drift_history": self._drift_history,
        }


# ============================================================================
# 扩展系统 3: ContradictionDetector (矛盾检测)
# ============================================================================

@dataclass
class ConflictPair:
    """一对矛盾的精华"""
    essence_a_id: int
    essence_b_id: int
    topic: str
    severity: float  # 0.0 - 1.0
    reason: str
    round_detected: int
    resolution_status: str = "unresolved"  # unresolved, resolved, acknowledged


class ContradictionDetector:
    """
    矛盾检测器 —— 识别精华之间的冲突和矛盾。

    功能：
    - 检测同一主题下立场相反的精华
    - 评估矛盾严重程度
    - 生成矛盾报告和网络图
    - 追踪矛盾解决状态
    """

    # 立场对立关键词对
    OPPOSITION_KEYWORDS: List[Tuple[List[str], List[str]]] = [
        (["支持", "赞成", "同意", "肯定", "认可", "赞同"], ["反对", "否定", "不认同", "质疑", "反驳", "不赞成"]),
        (["应该", "必须", "需要", "有必要"], ["不应该", "不必", "无需", "没有必要"]),
        (["是", "确实", "无疑", "显然"], ["不是", "并非", "未必", "不一定"]),
        (["好", "有利", "优势", "好处", "积极"], ["坏", "不利", "劣势", "坏处", "消极"]),
        (["增加", "提高", "上升", "增长"], ["减少", "降低", "下降", "缩减"]),
        (["重要", "关键", "核心", "必要"], ["次要", "边缘", "无关", "不必要"]),
        (["可行", "可能", "可以实现"], ["不可行", "不可能", "无法实现"]),
    ]

    def __init__(self):
        self.conflict_pairs: List[ConflictPair] = []
        self._resolved_pairs: List[ConflictPair] = []
        self._topic_cache: Dict[str, List[int]] = defaultdict(list)  # topic -> essence_ids

    def detect(self, essences: List[EssenceItem], round_id: int) -> List[ConflictPair]:
        """
        在精华列表中检测矛盾对。

        遍历所有精华对，通过关键词分析和立场检测识别矛盾。
        返回新发现的矛盾对列表。
        """
        if len(essences) < 2:
            return []

        # 为每条精华提取主题和立场
        essence_topics: Dict[int, List[str]] = {}
        essence_stances: Dict[int, str] = {}

        for item in essences:
            topics = self._extract_topics(item.content)
            essence_topics[item.id] = topics
            stance = self._detect_stance(item.content)
            essence_stances[item.id] = stance
            for t in topics:
                self._topic_cache[t].append(item.id)

        # 检查已存在的对
        existing_pairs: Set[Tuple[int, int]] = set()
        for cp in self.conflict_pairs:
            existing_pairs.add((cp.essence_a_id, cp.essence_b_id))
            existing_pairs.add((cp.essence_b_id, cp.essence_a_id))

        new_conflicts: List[ConflictPair] = []
        for i in range(len(essences)):
            for j in range(i + 1, len(essences)):
                a, b = essences[i], essences[j]
                pair_key = (a.id, b.id)
                if pair_key in existing_pairs:
                    continue

                # 找共同主题
                common_topics = set(essence_topics.get(a.id, [])) & set(essence_topics.get(b.id, []))
                if not common_topics:
                    # 如果没有明显的共同主题，检查是否有父-子关系中的对立
                    if b.parent_id == a.id and any("反驳" in t for t in b.tags):
                        common_topics = {"(反驳关系)"}
                    elif a.parent_id == b.id and any("反驳" in t for t in a.tags):
                        common_topics = {"(反驳关系)"}
                    else:
                        continue

                # 计算矛盾严重度
                severity, reason = self._calculate_severity(a, b, common_topics, essence_stances)

                if severity > 0.3:
                    cp = ConflictPair(
                        essence_a_id=a.id,
                        essence_b_id=b.id,
                        topic=", ".join(sorted(common_topics)[:3]),
                        severity=round(severity, 3),
                        reason=reason,
                        round_detected=round_id,
                    )
                    new_conflicts.append(cp)
                    self.conflict_pairs.append(cp)
                    existing_pairs.add(pair_key)

        return new_conflicts

    def _extract_topics(self, content: str) -> List[str]:
        """从内容中提取主题词"""
        topics: List[str] = []
        # 寻找引号内的概念
        quoted = re.findall(r'["""「」『』【】《》]([^""「」『』【】《》]{2,15})["""「」『』【】《》]', content)
        topics.extend(quoted)

        # 提取关键名词短语
        segments = re.split(r'[，。！？、；：,\.!\?;:\s]', content)
        for seg in segments:
            seg = seg.strip()
            # 2-6个字的名词短语
            if 2 <= len(seg) <= 6 and not re.match(r'^[\d\s%#@\.]+$', seg):
                topics.append(seg)

        # 去重，保留出现频率最高的
        topic_counter = Counter(topics)
        return [t for t, _ in topic_counter.most_common(5)]

    def _detect_stance(self, content: str) -> str:
        """检测内容立场: positive, negative, neutral"""
        pos_score = 0
        neg_score = 0

        for pos_words, neg_words in self.OPPOSITION_KEYWORDS:
            for pw in pos_words:
                if pw in content:
                    pos_score += 1
            for nw in neg_words:
                if nw in content:
                    neg_score += 1

        if pos_score > neg_score:
            return "positive"
        elif neg_score > pos_score:
            return "negative"
        return "neutral"

    def _calculate_severity(self, a: EssenceItem, b: EssenceItem,
                            common_topics: Set[str],
                            stances: Dict[int, str]) -> Tuple[float, str]:
        """
        计算一对精华的矛盾严重度 (0.0 - 1.0)。

        因素：
        - 立场对立程度
        - 共同主题数量
        - 双方评分（高评分矛盾更严重）
        - 反驳标签
        """
        stance_a = stances.get(a.id, "neutral")
        stance_b = stances.get(b.id, "neutral")

        # 立场对立
        stance_opposition = 0.0
        if (stance_a == "positive" and stance_b == "negative") or \
           (stance_a == "negative" and stance_b == "positive"):
            stance_opposition = 0.5
        elif stance_a != stance_b:
            stance_opposition = 0.2

        # 主题重叠度
        topic_overlap = min(1.0, len(common_topics) / 3.0)

        # 反驳标签加分
        tag_opposition = 0.0
        if any("反驳" in t for t in a.tags) or any("反驳" in t for t in b.tags):
            tag_opposition = 0.3
        if b.parent_id == a.id and any("反驳" in t for t in b.tags):
            tag_opposition = 0.4
        if a.parent_id == b.id and any("反驳" in t for t in a.tags):
            tag_opposition = 0.4

        # 评分因子（高评分矛盾更严重）
        score_factor = (a.score + b.score) / 20.0
        score_factor = min(0.2, score_factor)

        severity = stance_opposition + topic_overlap * 0.3 + tag_opposition + score_factor
        severity = min(1.0, severity)

        # 生成原因
        reasons = []
        if stance_opposition > 0:
            reasons.append("立场对立")
        if tag_opposition > 0.3:
            reasons.append("直接反驳关系")
        if common_topics:
            reasons.append(f"共同主题: {', '.join(sorted(common_topics)[:2])}")

        reason = "；".join(reasons) if reasons else "检测到观点差异"

        return severity, reason

    def find_conflict_pairs(self, topic: Optional[str] = None,
                            min_severity: float = 0.0) -> List[ConflictPair]:
        """
        查找矛盾对，可按主题和最低严重度过滤。

        返回匹配的 ConflictPair 列表。
        """
        results = self.conflict_pairs
        if topic:
            results = [cp for cp in results if topic.lower() in cp.topic.lower()]
        if min_severity > 0:
            results = [cp for cp in results if cp.severity >= min_severity]
        return sorted(results, key=lambda cp: cp.severity, reverse=True)

    def get_contradiction_network(self) -> Dict[str, Any]:
        """
        构建矛盾网络图。

        返回：
        - nodes: 精华节点列表
        - edges: 矛盾边列表
        - stats: 统计信息
        """
        node_ids: Set[int] = set()
        for cp in self.conflict_pairs:
            node_ids.add(cp.essence_a_id)
            node_ids.add(cp.essence_b_id)

        # 计算每个节点的矛盾度
        node_conflict_count: Dict[int, int] = defaultdict(int)
        node_avg_severity: Dict[int, List[float]] = defaultdict(list)
        for cp in self.conflict_pairs:
            node_conflict_count[cp.essence_a_id] += 1
            node_conflict_count[cp.essence_b_id] += 1
            node_avg_severity[cp.essence_a_id].append(cp.severity)
            node_avg_severity[cp.essence_b_id].append(cp.severity)

        nodes = [
            {
                "essence_id": eid,
                "conflict_count": node_conflict_count.get(eid, 0),
                "avg_severity": round(statistics.mean(node_avg_severity.get(eid, [0])), 3),
            }
            for eid in node_ids
        ]

        edges = [
            {
                "source": cp.essence_a_id,
                "target": cp.essence_b_id,
                "severity": cp.severity,
                "topic": cp.topic,
                "status": cp.resolution_status,
            }
            for cp in self.conflict_pairs
        ]

        unresolved = sum(1 for cp in self.conflict_pairs if cp.resolution_status == "unresolved")

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_conflicts": len(self.conflict_pairs),
                "unresolved": unresolved,
                "resolved": len(self.conflict_pairs) - unresolved,
                "max_severity": max((cp.severity for cp in self.conflict_pairs), default=0.0),
                "avg_severity": round(
                    statistics.mean([cp.severity for cp in self.conflict_pairs]), 3
                ) if self.conflict_pairs else 0.0,
            },
        }

    def resolve_conflict(self, essence_a_id: int, essence_b_id: int) -> bool:
        """
        标记一对矛盾为已解决。
        """
        for cp in self.conflict_pairs:
            if (cp.essence_a_id == essence_a_id and cp.essence_b_id == essence_b_id) or \
               (cp.essence_a_id == essence_b_id and cp.essence_b_id == essence_a_id):
                cp.resolution_status = "resolved"
                self._resolved_pairs.append(cp)
                self.conflict_pairs.remove(cp)
                return True
        return False

    def get_conflict_report(self) -> str:
        """生成矛盾分析报告文本"""
        if not self.conflict_pairs:
            return "未检测到矛盾。"

        lines = [
            "=" * 50,
            "矛盾检测报告",
            "=" * 50,
        ]

        # 按主题分组
        topic_groups: Dict[str, List[ConflictPair]] = defaultdict(list)
        for cp in self.conflict_pairs:
            topic_groups[cp.topic].append(cp)

        for topic, pairs in sorted(topic_groups.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"\n主题: {topic} ({len(pairs)} 对矛盾)")
            for cp in pairs[:5]:
                status = "[已解决]" if cp.resolution_status == "resolved" else "[未解决]"
                lines.append(
                    f"  {status} #{cp.essence_a_id} <-> #{cp.essence_b_id} "
                    f"严重度: {cp.severity:.2f} | {cp.reason}"
                )

        lines.append(f"\n总计: {len(self.conflict_pairs)} 对矛盾")
        unresolved = sum(1 for cp in self.conflict_pairs if cp.resolution_status == "unresolved")
        lines.append(f"未解决: {unresolved} 对")
        lines.append("=" * 50)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "conflict_pairs": [
                {
                    "essence_a_id": cp.essence_a_id,
                    "essence_b_id": cp.essence_b_id,
                    "topic": cp.topic,
                    "severity": cp.severity,
                    "reason": cp.reason,
                    "round_detected": cp.round_detected,
                    "status": cp.resolution_status,
                }
                for cp in self.conflict_pairs
            ],
            "resolved_pairs": [
                {
                    "essence_a_id": cp.essence_a_id,
                    "essence_b_id": cp.essence_b_id,
                    "topic": cp.topic,
                }
                for cp in self._resolved_pairs
            ],
        }


# ============================================================================
# 扩展系统 4: AutoSummarizer (自动摘要)
# ============================================================================

class AutoSummarizer:
    """
    自动摘要器 —— 多级别精华摘要生成。

    功能：
    - 单条精华摘要（压缩/提炼）
    - 主题簇摘要（同一主题下的精华汇总）
    - 全池摘要（精华池的整体概括）
    - 关键发现提取
    - 执行摘要（面向决策者）
    """

    # 摘要最大长度（字符数）
    SINGLE_MAX_LENGTH = 150
    CLUSTER_MAX_LENGTH = 500
    POOL_MAX_LENGTH = 1500
    EXECUTIVE_MAX_LENGTH = 800

    def __init__(self):
        self._summary_cache: Dict[str, str] = {}

    def summarize(self, essence: EssenceItem, max_length: int = None) -> str:
        """
        生成单条精华的摘要。

        自动压缩精华内容，保留核心信息。
        """
        if max_length is None:
            max_length = self.SINGLE_MAX_LENGTH

        cache_key = f"single_{essence.id}"
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        content = essence.content

        # 如果内容本身就很短，直接返回
        if len(content) <= max_length:
            self._summary_cache[cache_key] = content
            return content

        # 压缩策略：保留关键信息
        # 1. 提取引号内的内容（高优先级）
        quoted = re.findall(r'["""「」『』【】]([^""「」『』【】"]{3,})["""「」『』【】"]', content)

        # 2. 提取首句
        sentences = re.split(r'(?<=[。！？\.!\?])', content)
        first_sentence = sentences[0].strip() if sentences else ""

        # 3. 构建摘要
        summary_parts = []
        remaining = max_length

        if quoted:
            for q in quoted:
                if len(q) + 3 <= remaining:
                    summary_parts.append(f"「{q}」")
                    remaining -= len(q) + 3

        if remaining > 10 and first_sentence:
            if len(first_sentence) <= remaining:
                summary_parts.append(first_sentence)
            else:
                summary_parts.append(first_sentence[:remaining - 3] + "...")

        # 如果还没有任何内容，截取开头
        if not summary_parts:
            summary_parts.append(content[:max_length - 3] + "...")

        summary = " ".join(summary_parts)
        self._summary_cache[cache_key] = summary
        return summary

    def summarize_cluster(self, essences: List[EssenceItem], topic: str,
                          max_length: int = None) -> str:
        """
        生成主题簇的摘要。

        汇总同一主题下的所有精华，提取共识和分歧。
        """
        if max_length is None:
            max_length = self.CLUSTER_MAX_LENGTH

        if not essences:
            return f"主题「{topic}」下无精华条目。"

        cache_key = f"cluster_{topic}_{len(essences)}"
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        # 按评分排序
        sorted_items = sorted(essences, key=lambda x: x.score, reverse=True)

        # 统计
        total = len(sorted_items)
        avg_score = statistics.mean([it.score for it in sorted_items])
        top_score = sorted_items[0].score
        contributors = list(set(it.contributor for it in sorted_items))

        # 分歧检测
        positive_stance = sum(1 for it in sorted_items
                              if any(w in it.content for w in ["支持", "赞成", "同意", "肯定", "好", "有利"]))
        negative_stance = sum(1 for it in sorted_items
                              if any(w in it.content for w in ["反对", "否定", "质疑", "反驳", "坏", "不利"]))

        # 提取关键内容
        summaries = []
        remaining = max_length

        header = f"主题「{topic}」: {total} 条精华, 平均评分 {avg_score:.1f}, {len(contributors)} 位贡献者"
        if len(header) + 5 <= remaining:
            summaries.append(header)
            remaining -= len(header) + 1

        # 共识/分歧说明
        if positive_stance > negative_stance * 2:
            consensus_note = f"【倾向共识】正面立场占优 ({positive_stance}:{negative_stance})"
        elif negative_stance > positive_stance * 2:
            consensus_note = f"【存在分歧】负面立场占优 ({negative_stance}:{positive_stance})"
        else:
            consensus_note = f"【观点多元】正反立场接近 ({positive_stance}:{negative_stance})"

        if len(consensus_note) + 3 <= remaining:
            summaries.append(consensus_note)
            remaining -= len(consensus_note) + 1

        # 顶部精华摘要
        for item in sorted_items[:3]:
            if remaining < 20:
                break
            item_summary = self.summarize(item, max_length=80)
            line = f"#{item.id}({item.score:.1f}): {item_summary}"
            if len(line) + 3 <= remaining:
                summaries.append(line)
                remaining -= len(line) + 1

        result = "\n".join(summaries)
        self._summary_cache[cache_key] = result
        return result

    def summarize_pool(self, pool: EssencePool, max_length: int = None) -> str:
        """
        生成精华池的整体摘要。

        综合所有精华，呈现讨论的整体图景。
        """
        if max_length is None:
            max_length = self.POOL_MAX_LENGTH

        if not pool.items:
            return "精华池为空。"

        cache_key = f"pool_{len(pool.items)}_{pool._next_id}"
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        items = pool.items
        total = len(items)
        scores = [it.score for it in items]
        avg_score = statistics.mean(scores)
        contributors = list(set(it.contributor for it in items))
        tags_used = set()
        for it in items:
            tags_used.update(it.tags)

        # 按评分分组
        high_quality = [it for it in items if it.score >= 3.0]
        medium_quality = [it for it in items if 1.0 <= it.score < 3.0]
        low_quality = [it for it in items if it.score < 1.0]

        # 按来源轮次分组
        round_groups: Dict[int, List[EssenceItem]] = defaultdict(list)
        for it in items:
            round_groups[it.source_round].append(it)

        # 构建摘要
        parts = []
        remaining = max_length

        # 头部统计
        header = (
            f"精华池摘要: 共 {total} 条精华, {len(contributors)} 位贡献者, "
            f"平均评分 {avg_score:.2f}, 覆盖 {len(round_groups)} 轮讨论"
        )
        if len(header) <= remaining:
            parts.append(header)
            remaining -= len(header) + 2

        # 质量分布
        quality_line = (
            f"质量分布: 高质量({len(high_quality)}) / 中等({len(medium_quality)}) / 低质量({len(low_quality)})"
        )
        if len(quality_line) <= remaining:
            parts.append(quality_line)
            remaining -= len(quality_line) + 2

        # 标签分布
        tag_line = f"标签分布: {', '.join(sorted(tags_used)[:8])}" if tags_used else ""
        if tag_line and len(tag_line) <= remaining:
            parts.append(tag_line)
            remaining -= len(tag_line) + 2

        # 各轮次进展
        parts.append("\n轮次进展:")
        remaining -= 6
        for rnd in sorted(round_groups.keys()):
            if remaining < 20:
                break
            rnd_items = round_groups[rnd]
            rnd_top = max(rnd_items, key=lambda x: x.score)
            line = (
                f"  第{rnd}轮: {len(rnd_items)} 条精华, "
                f"最佳: #{rnd_top.id}({rnd_top.score:.1f})「{rnd_top.content[:30]}...」"
            )
            if len(line) + 2 <= remaining:
                parts.append(line)
                remaining -= len(line) + 1

        # 顶部精华
        parts.append("\n关键精华:")
        remaining -= 6
        for item in sorted(items, key=lambda x: x.score, reverse=True)[:3]:
            if remaining < 20:
                break
            line = f"  #{item.id}({item.score:.1f}) {item.content[:50]}..."
            if len(line) + 2 <= remaining:
                parts.append(line)
                remaining -= len(line) + 1

        result = "\n".join(parts)
        self._summary_cache[cache_key] = result
        return result

    def extract_key_findings(self, pool: EssencePool, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        提取关键发现。

        从精华池中提取最重要的发现和见解，基于：
        - 评分
        - 引用次数
        - 深化次数
        - 反驳次数（说明有讨论价值）
        """
        findings: List[Dict[str, Any]] = []

        for item in pool.items:
            # 综合重要性评分
            importance = (
                0.4 * (item.score / max(1, max(it.score for it in pool.items))) +
                0.2 * (len(item.cited_by) / max(1, max(len(it.cited_by) for it in pool.items))) +
                0.2 * (len(item.refined_by) / max(1, max(len(it.refined_by) for it in pool.items))) +
                0.1 * (len(item.challenged_by) / max(1, max(len(it.challenged_by) for it in pool.items))) +
                0.1 * (len(item.clarifications) / max(1, max(len(it.clarifications) for it in pool.items)))
            )

            if importance > 0:
                findings.append({
                    "essence_id": item.id,
                    "content": item.content,
                    "importance": round(importance, 3),
                    "score": item.score,
                    "citations": len(item.cited_by),
                    "refinements": len(item.refined_by),
                    "challenges": len(item.challenged_by),
                    "contributor": item.contributor,
                    "round": item.source_round,
                })

        findings.sort(key=lambda x: x["importance"], reverse=True)
        return findings[:top_n]

    def generate_executive_summary(self, pool: EssencePool,
                                    max_length: int = None) -> str:
        """
        生成执行摘要（面向决策者）。

        简洁、重点突出，包含：
        - 讨论概况
        - 核心发现（3-5点）
        - 共识状态
        - 建议行动
        """
        if max_length is None:
            max_length = self.EXECUTIVE_MAX_LENGTH

        if not pool.items:
            return "【执行摘要】尚无讨论内容。"

        # 概况
        total = len(pool.items)
        contributors = set(it.contributor for it in pool.items)
        rounds = set(it.source_round for it in pool.items)

        # 核心发现
        findings = self.extract_key_findings(pool, top_n=3)
        findings_text = []
        for f in findings:
            findings_text.append(f"  - {f['content'][:60]}... (评分 {f['score']:.1f})")

        # 共识状态
        consensus = pool.calculate_consensus(len(contributors))

        # 建议行动
        if consensus["level"] == "high":
            action = "建议整理最终方案，准备输出"
        elif consensus["level"] == "medium":
            action = "仍有分歧领域，建议聚焦关键争议点深入讨论"
        elif consensus["level"] == "low":
            action = "建议主持人引导讨论方向，寻找共识基础"
        else:
            action = "继续收集更多观点"

        summary = (
            f"【执行摘要】\n"
            f"讨论概况: {total} 条精华, {len(contributors)} 位参与者, {len(rounds)} 轮讨论\n\n"
            f"核心发现:\n" + "\n".join(findings_text) + "\n\n"
            f"共识水平: {consensus['level']} (评分: {consensus['score']:.2f})\n\n"
            f"建议行动: {action}"
        )

        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."

        return summary

    def clear_cache(self) -> None:
        """清除摘要缓存"""
        self._summary_cache.clear()

    def to_dict(self) -> dict:
        return {
            "cache_size": len(self._summary_cache),
        }


# ============================================================================
# 扩展系统 5: QualityPredictor (质量预测)
# ============================================================================

@dataclass
class PredictionResult:
    """预测结果"""
    essence_id: int
    predicted_score: float
    actual_score: Optional[float]
    confidence: float  # 0.0 - 1.0
    features: Dict[str, float]


class QualityPredictor:
    """
    质量预测器 —— 基于精华特征预测其质量。

    功能：
    - 特征提取：从精华中提取多维特征
    - 质量预测：基于历史数据预测精华的最终评分
    - 高潜力识别：早期发现高潜力精华
    - 准确率评估：追踪预测准确率

    预测模型使用加权线性回归，基于以下特征：
    - 内容长度（字符数）
    - 实体数量（知识图谱中的实体）
    - 标签多样性
    - 贡献者历史评分
    - 轮次（早期 vs 后期）
    - 父精华评分（如果有父精华）
    - 引用数
    - 澄清数
    """

    # 特征权重（通过训练或手动设定）
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "content_length_norm": 0.05,
        "entity_count": 0.10,
        "tag_diversity": 0.15,
        "contributor_avg_score": 0.20,
        "round_weight": 0.05,
        "parent_score": 0.10,
        "citation_rate": 0.15,
        "clarification_count": 0.05,
        "has_tags": 0.10,
        "is_refinement": 0.05,
    }

    def __init__(self):
        self.weights: Dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        self._predictions: List[PredictionResult] = []
        self._feature_cache: Dict[int, Dict[str, float]] = {}
        self._contributor_history: Dict[str, List[float]] = defaultdict(list)
        self._training_data: List[Tuple[Dict[str, float], float]] = []

    def predict_quality(self, essence: EssenceItem, pool: EssencePool,
                        knowledge_graph: Optional[EssenceKnowledgeGraph] = None) -> PredictionResult:
        """
        预测一条精华的质量（最终评分）。

        基于提取的特征和当前权重计算预测评分。
        """
        features = self._extract_features(essence, pool, knowledge_graph)
        predicted = self._compute_prediction(features)

        # 置信度：基于训练数据量
        data_points = len(self._training_data)
        confidence = min(0.9, 0.3 + 0.02 * data_points)

        result = PredictionResult(
            essence_id=essence.id,
            predicted_score=round(predicted, 3),
            actual_score=essence.score,
            confidence=round(confidence, 3),
            features=features,
        )
        self._predictions.append(result)
        return result

    def _extract_features(self, essence: EssenceItem, pool: EssencePool,
                          knowledge_graph: Optional[EssenceKnowledgeGraph]) -> Dict[str, float]:
        """提取精华特征向量"""
        if essence.id in self._feature_cache:
            return self._feature_cache[essence.id]

        features: Dict[str, float] = {}

        # 1. 内容长度（归一化）
        features["content_length_norm"] = min(1.0, len(essence.content) / 200.0)

        # 2. 实体数量
        entity_count = 0
        if knowledge_graph:
            for node in knowledge_graph.entities.values():
                if essence.id in node.related_essence_ids:
                    entity_count += 1
        features["entity_count"] = min(1.0, entity_count / 5.0)

        # 3. 标签多样性
        features["tag_diversity"] = min(1.0, len(set(essence.tags)) / 3.0)

        # 4. 贡献者历史平均评分
        hist = self._contributor_history.get(essence.contributor, [])
        features["contributor_avg_score"] = min(1.0, statistics.mean(hist) / 5.0) if hist else 0.3

        # 5. 轮次权重（早期观点可能更有价值）
        max_round = max((it.source_round for it in pool.items), default=1)
        features["round_weight"] = 1.0 - (essence.source_round / max(1, max_round))

        # 6. 父精华评分
        parent_score = 0.0
        if essence.parent_id is not None:
            parent = pool._get_item(essence.parent_id)
            if parent:
                parent_score = parent.score
        features["parent_score"] = min(1.0, parent_score / 5.0)

        # 7. 引用率（早期被引用的信号）
        total_citations = len(essence.cited_by)
        features["citation_rate"] = min(1.0, total_citations / 3.0)

        # 8. 澄清数
        features["clarification_count"] = min(1.0, len(essence.clarifications) / 2.0)

        # 9. 是否有标签
        features["has_tags"] = 1.0 if essence.tags else 0.0

        # 10. 是否为深化内容
        features["is_refinement"] = 1.0 if essence.parent_id is not None else 0.0

        self._feature_cache[essence.id] = features
        return features

    def _compute_prediction(self, features: Dict[str, float]) -> float:
        """基于特征和权重计算预测评分"""
        score = 0.0
        for feat_name, feat_value in features.items():
            weight = self.weights.get(feat_name, 0.0)
            score += weight * feat_value * 5.0  # 放大到 5 分制
        return max(0.0, min(5.0, score))

    def train(self, pool: EssencePool,
              knowledge_graph: Optional[EssenceKnowledgeGraph] = None) -> Dict[str, Any]:
        """
        训练预测模型。

        基于池中现有精华的实际评分，优化特征权重。
        使用简单梯度下降。
        """
        if len(pool.items) < 3:
            return {"status": "insufficient_data", "samples": len(pool.items)}

        # 准备训练数据
        self._training_data.clear()
        for item in pool.items:
            features = self._extract_features(item, pool, knowledge_graph)
            self._training_data.append((features, item.score))

        # 更新贡献者历史
        for item in pool.items:
            self._contributor_history[item.contributor].append(item.score)

        # 梯度下降优化权重
        learning_rate = 0.01
        epochs = 100
        losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            for features, actual_score in self._training_data:
                predicted = self._compute_prediction(features)
                error = predicted - actual_score
                total_loss += error * error

                # 梯度更新
                for feat_name, feat_value in features.items():
                    if feat_name in self.weights:
                        grad = 2 * error * feat_value * 5.0
                        self.weights[feat_name] -= learning_rate * grad

            # 确保权重非负
            for k in self.weights:
                self.weights[k] = max(0.0, self.weights[k])

            # 归一化权重
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for k in self.weights:
                    self.weights[k] /= total_weight

            losses.append(total_loss / max(1, len(self._training_data)))

        return {
            "status": "trained",
            "samples": len(self._training_data),
            "final_loss": round(losses[-1], 6),
            "initial_loss": round(losses[0], 6),
            "epochs": epochs,
            "weights": dict(self.weights),
        }

    def get_high_potential(self, pool: EssencePool,
                           knowledge_graph: Optional[EssenceKnowledgeGraph] = None,
                           top_n: int = 5) -> List[Dict[str, Any]]:
        """
        识别高潜力精华（预测评分 > 当前评分 且 差距较大）。

        用于早期发现那些可能被低估的精华。
        """
        candidates: List[Dict[str, Any]] = []

        for item in pool.items:
            # 只考虑早期精华（评分尚未稳定）
            if len(item.cited_by) + len(item.refined_by) + len(item.challenged_by) >= 5:
                continue

            prediction = self.predict_quality(item, pool, knowledge_graph)
            gap = prediction.predicted_score - (prediction.actual_score or 0.0)

            if gap > 0.3 and prediction.confidence > 0.3:
                candidates.append({
                    "essence_id": item.id,
                    "content": item.content[:60],
                    "current_score": prediction.actual_score,
                    "predicted_score": prediction.predicted_score,
                    "potential_gap": round(gap, 3),
                    "confidence": prediction.confidence,
                    "contributor": item.contributor,
                })

        candidates.sort(key=lambda x: x["potential_gap"], reverse=True)
        return candidates[:top_n]

    def evaluate_accuracy(self) -> Dict[str, Any]:
        """
        评估预测准确率。

        比较预测评分与实际评分，计算误差指标。
        """
        if not self._predictions:
            return {"status": "no_predictions", "mae": 0.0, "rmse": 0.0}

        errors = []
        for p in self._predictions:
            if p.actual_score is not None:
                errors.append(p.predicted_score - p.actual_score)

        if not errors:
            return {"status": "no_actual_scores", "mae": 0.0, "rmse": 0.0}

        abs_errors = [abs(e) for e in errors]
        mae = statistics.mean(abs_errors)
        rmse = math.sqrt(statistics.mean([e * e for e in errors]))

        # 准确率：在正负0.5范围内的比例
        within_05 = sum(1 for e in abs_errors if e <= 0.5)
        accuracy = within_05 / len(abs_errors) if abs_errors else 0.0

        return {
            "status": "evaluated",
            "total_predictions": len(self._predictions),
            "total_errors": len(errors),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "accuracy_within_05": round(accuracy, 4),
            "max_error": round(max(abs_errors), 4),
            "min_error": round(min(abs_errors), 4),
        }

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "predictions": [
                {
                    "essence_id": p.essence_id,
                    "predicted": p.predicted_score,
                    "actual": p.actual_score,
                    "confidence": p.confidence,
                }
                for p in self._predictions[-20:]  # 只保留最近20条
            ],
            "training_samples": len(self._training_data),
            "contributors_tracked": len(self._contributor_history),
        }


# ============================================================================
# 扩展系统 6: TopicModeler (主题建模)
# ============================================================================

@dataclass
class TopicCluster:
    """主题簇"""
    name: str
    essence_ids: List[int]
    keywords: List[str]
    avg_score: float
    first_round: int
    last_round: int
    size: int
    coherence: float  # 主题连贯性 0.0 - 1.0


class TopicModeler:
    """
    主题建模器 —— 将精华聚类为主题并追踪其演变。

    功能：
    - 基于关键词和标签的精华聚类
    - 主题摘要和统计
    - 新兴主题检测
    - 主题演化追踪
    - 主题关联分析
    """

    # 主题关键词模型
    TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "技术方案": ["技术", "方案", "实现", "架构", "设计", "系统", "平台", "框架", "算法", "接口"],
        "理论分析": ["理论", "原理", "概念", "模型", "范式", "机制", "规律", "定理", "假设", "推论"],
        "实践应用": ["实践", "应用", "案例", "场景", "落地", "部署", "实施", "运营", "测试", "验证"],
        "风险评估": ["风险", "安全", "隐患", "威胁", "漏洞", "攻击", "防御", "保护", "隐私", "合规"],
        "资源管理": ["资源", "成本", "效率", "优化", "性能", "扩展", "容量", "预算", "人力", "时间"],
        "用户体验": ["用户", "体验", "界面", "交互", "可用性", "易用", "设计", "反馈", "满意度", "需求"],
        "战略规划": ["战略", "规划", "愿景", "目标", "路线图", "里程碑", "优先级", "投资", "ROI", "市场"],
        "合作沟通": ["协作", "沟通", "团队", "组织", "流程", "角色", "职责", "协调", "对齐", "同步"],
        "创新突破": ["创新", "突破", "颠覆", "变革", "新兴", "前沿", "探索", "实验", "原型", "POC"],
        "伦理法规": ["伦理", "法规", "法律", "政策", "监管", "合规", "道德", "责任", "透明", "公平"],
    }

    def __init__(self):
        self.topics: Dict[str, TopicCluster] = {}
        self._topic_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._essence_topic_map: Dict[int, str] = {}  # essence_id -> topic_name
        self._keyword_freq: Dict[str, int] = Counter()

    def model_topics(self, essences: List[EssenceItem]) -> Dict[str, TopicCluster]:
        """
        对所有精华进行主题建模。

        基于关键词匹配和标签分析，将精华聚类到预定义主题中。
        返回主题名称 -> TopicCluster 的映射。
        """
        if not essences:
            return {}

        # 更新关键词频率
        for item in essences:
            content = item.content
            for word in re.split(r'[，。！？、；：,\.!\?;:\s()（）\[\]""「」]', content):
                word = word.strip()
                if len(word) >= 2:
                    self._keyword_freq[word] += 1

        # 为每条精华分配主题
        topic_essences: Dict[str, List[EssenceItem]] = defaultdict(list)
        for item in essences:
            topic = self._assign_topic(item)
            topic_essences[topic].append(item)
            self._essence_topic_map[item.id] = topic

        # 构建 TopicCluster
        for topic_name, items in topic_essences.items():
            scores = [it.score for it in items]
            rounds = [it.source_round for it in items]
            keywords = self._extract_topic_keywords(items, topic_name)

            cluster = TopicCluster(
                name=topic_name,
                essence_ids=[it.id for it in items],
                keywords=keywords,
                avg_score=round(statistics.mean(scores), 3) if scores else 0.0,
                first_round=min(rounds) if rounds else 0,
                last_round=max(rounds) if rounds else 0,
                size=len(items),
                coherence=self._calculate_coherence(items, topic_name, keywords),
            )
            self.topics[topic_name] = cluster

        return self.topics

    def _assign_topic(self, essence: EssenceItem) -> str:
        """
        为精华分配最匹配的主题。

        基于关键词匹配得分，选择得分最高的主题。
        如果没有匹配，归为"其他"。
        """
        content = essence.content
        best_topic = "其他"
        best_score = 0.0

        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = 0.0
            for kw in keywords:
                if kw in content:
                    score += 1.0
            # 标签匹配加分
            for tag in essence.tags:
                if tag in topic:
                    score += 2.0
            # 归一化
            score /= max(1, len(keywords))

            if score > best_score:
                best_score = score
                best_topic = topic

        return best_topic

    def _extract_topic_keywords(self, items: List[EssenceItem], topic: str) -> List[str]:
        """提取主题的关键词"""
        # 从预定义关键词中选取
        predefined = self.TOPIC_KEYWORDS.get(topic, [])
        if predefined:
            return predefined[:5]

        # 从内容中提取高频词
        word_counter: Counter = Counter()
        for item in items:
            words = re.split(r'[，。！？、；：,\.!\?;:\s()（）\[\]""「」]', item.content)
            for w in words:
                w = w.strip()
                if len(w) >= 2:
                    word_counter[w] += 1

        return [w for w, _ in word_counter.most_common(5)]

    def _calculate_coherence(self, items: List[EssenceItem], topic: str, keywords: List[str]) -> float:
        """计算主题连贯性"""
        if not items or not keywords:
            return 0.0

        # 关键词覆盖度
        matched_count = 0
        for item in items:
            for kw in keywords:
                if kw in item.content:
                    matched_count += 1
                    break

        coverage = matched_count / len(items) if items else 0.0

        # 标签一致性
        tag_consistency = 0.0
        relevant_tags = {"论点", "论据", "深化", "反驳", "创新点"}
        for item in items:
            if any(t in item.tags for t in relevant_tags):
                tag_consistency += 1.0
        tag_consistency /= len(items) if items else 1.0

        return round((coverage * 0.6 + tag_consistency * 0.4), 3)

    def get_topic_summary(self, topic_name: str) -> Dict[str, Any]:
        """获取指定主题的摘要"""
        cluster = self.topics.get(topic_name)
        if not cluster:
            return {"topic": topic_name, "status": "not_found"}

        return {
            "topic": topic_name,
            "size": cluster.size,
            "avg_score": cluster.avg_score,
            "keywords": cluster.keywords,
            "coherence": cluster.coherence,
            "rounds": f"{cluster.first_round} - {cluster.last_round}",
            "essence_count": len(cluster.essence_ids),
        }

    def detect_emerging(self, current_round: int, window: int = 2) -> List[Dict[str, Any]]:
        """
        检测新兴主题。

        基于最近 window 轮中精华的主题分布变化，识别可能正在兴起的新主题。
        """
        emerging: List[Dict[str, Any]] = []

        for topic_name, cluster in self.topics.items():
            # 检查该主题是否在最近几轮中出现
            recent_essences = [
                eid for eid in cluster.essence_ids
                if self._get_essence_round(eid) >= current_round - window
            ]

            if not recent_essences:
                continue

            # 新兴主题特征：
            # 1. 较新出现（first_round 接近当前）
            # 2. 增长速度快
            # 3. 关键词新颖

            age = current_round - cluster.first_round

            if age <= window and cluster.size >= 2:
                # 计算增长率
                history = self._topic_history.get(topic_name, [])
                growth_rate = 0.0
                if len(history) >= 2:
                    prev_size = history[-2].get("size", 0)
                    curr_size = history[-1].get("size", 0)
                    if prev_size > 0:
                        growth_rate = (curr_size - prev_size) / prev_size

                emerging.append({
                    "topic": topic_name,
                    "size": cluster.size,
                    "avg_score": cluster.avg_score,
                    "age_rounds": age,
                    "growth_rate": round(growth_rate, 3),
                    "keywords": cluster.keywords[:3],
                    "coherence": cluster.coherence,
                })

        emerging.sort(key=lambda x: x["growth_rate"], reverse=True)
        return emerging

    def _get_essence_round(self, essence_id: int) -> int:
        """获取精华的轮次（辅助方法）"""
        # 从 essence_topic_map 反向查找轮次信息
        # 实际轮次存储在 TopicCluster 中
        for cluster in self.topics.values():
            if essence_id in cluster.essence_ids:
                return cluster.last_round
        return 0

    def track_evolution(self, round_id: int) -> Dict[str, Any]:
        """
        记录当前轮次的主题分布快照，用于后续演化追踪。

        返回当前轮次的主题统计。
        """
        snapshot: Dict[str, Any] = {
            "round": round_id,
            "topics": {},
            "total_essences": 0,
        }

        for topic_name, cluster in self.topics.items():
            round_essences = len(cluster.essence_ids)
            snapshot["topics"][topic_name] = {
                "size": round_essences,
                "avg_score": cluster.avg_score,
                "coherence": cluster.coherence,
            }
            snapshot["total_essences"] += round_essences

            self._topic_history[topic_name].append({
                "round": round_id,
                "size": round_essences,
                "avg_score": cluster.avg_score,
            })

        return snapshot

    def get_topic_evolution(self, topic_name: str) -> List[Dict[str, Any]]:
        """
        获取指定主题的演化历史。

        返回每轮的主题大小和平均评分变化。
        """
        return list(self._topic_history.get(topic_name, []))

    def get_all_topic_summaries(self) -> List[Dict[str, Any]]:
        """获取所有主题的摘要列表"""
        summaries = []
        for topic_name, cluster in sorted(
            self.topics.items(), key=lambda x: x[1].size, reverse=True
        ):
            summaries.append({
                "topic": topic_name,
                "essence_count": cluster.size,
                "avg_score": cluster.avg_score,
                "keywords": cluster.keywords,
                "coherence": cluster.coherence,
                "round_range": f"{cluster.first_round} - {cluster.last_round}",
            })
        return summaries

    def to_dict(self) -> dict:
        return {
            "topics": {
                name: {
                    "essence_ids": cluster.essence_ids,
                    "keywords": cluster.keywords,
                    "avg_score": cluster.avg_score,
                    "size": cluster.size,
                    "coherence": cluster.coherence,
                    "first_round": cluster.first_round,
                    "last_round": cluster.last_round,
                }
                for name, cluster in self.topics.items()
            },
            "topic_history": dict(self._topic_history),
            "essence_topic_map": {str(k): v for k, v in self._essence_topic_map.items()},
        }


# ============================================================================
# 扩展系统 7: EssenceLifecycleManager (生命周期管理)
# ============================================================================

@dataclass
class LifecycleRecord:
    """生命周期记录"""
    essence_id: int
    stage: str  # new, developing, mature, decaying, archived
    entered_at_round: int
    previous_stage: Optional[str]
    reason: str
    confidence: float


class EssenceLifecycleManager:
    """
    精华生命周期管理器 —— 管理精华从诞生到归档的完整生命周期。

    阶段定义：
    - new: 新提交，尚在观察期
    - developing: 正在深化和讨论中
    - mature: 已成熟，评分稳定且较高
    - decaying: 评分下降，关注度减少
    - archived: 已归档，不再活跃参与讨论

    自动操作：
    - 低质量精华自动归档
    - 高质量精华自动提升
    - 根据活性调整生命周期
    """

    # 生命周期阈值
    ARCHIVE_SCORE_THRESHOLD = 0.5  # 低于此评分可归档
    PROMOTE_SCORE_THRESHOLD = 3.0  # 高于此评分可提升
    DECAY_ROUNDS_THRESHOLD = 3     # 连续多少轮无变化视为衰退
    NEW_OBSERVATION_ROUNDS = 2     # 新精华的观察轮次

    def __init__(self):
        self.records: Dict[int, LifecycleRecord] = {}
        self._stage_items: Dict[str, List[int]] = defaultdict(list)
        self._transition_history: List[Dict[str, Any]] = []
        self._archived_items: List[int] = []

    def update_lifecycle(self, pool: EssencePool, round_id: int) -> List[LifecycleRecord]:
        """
        更新所有精华的生命周期状态。

        为池中每条精华评估当前阶段，必要时进行阶段转换。
        返回所有更新的记录。
        """
        updated: List[LifecycleRecord] = []

        for item in pool.items:
            current_stage = self._get_current_stage(item.id)
            new_stage, reason, confidence = self._evaluate_stage(item, pool, round_id)

            if new_stage != current_stage:
                record = LifecycleRecord(
                    essence_id=item.id,
                    stage=new_stage,
                    entered_at_round=round_id,
                    previous_stage=current_stage or "new",
                    reason=reason,
                    confidence=round(confidence, 3),
                )
                self.records[item.id] = record
                self._transition_history.append({
                    "essence_id": item.id,
                    "from": current_stage,
                    "to": new_stage,
                    "round": round_id,
                    "reason": reason,
                })
                updated.append(record)

            # 更新阶段索引
            self._stage_items[new_stage].append(item.id)

        # 去重
        for stage in self._stage_items:
            self._stage_items[stage] = list(dict.fromkeys(self._stage_items[stage]))

        return updated

    def _get_current_stage(self, essence_id: int) -> Optional[str]:
        """获取当前生命周期阶段"""
        if essence_id in self.records:
            return self.records[essence_id].stage
        if essence_id in self._archived_items:
            return "archived"
        return None

    def _evaluate_stage(self, item: EssenceItem, pool: EssencePool,
                        round_id: int) -> Tuple[str, str, float]:
        """
        评估一条精华应该处于哪个生命周期阶段。

        考虑因素：
        - 评分
        - 评分变化趋势
        - 活跃度（引用/深化/反驳次数）
        - 存在时间
        """
        age = round_id - item.source_round

        # 归档检查：评分极低或已被大量反驳
        if item.score <= self.ARCHIVE_SCORE_THRESHOLD and age >= self.NEW_OBSERVATION_ROUNDS:
            return "archived", f"评分 {item.score:.1f} 低于归档阈值 {self.ARCHIVE_SCORE_THRESHOLD}", 0.8

        # 新提交
        if age < self.NEW_OBSERVATION_ROUNDS:
            return "new", f"新提交，观察期 第{age}/{self.NEW_OBSERVATION_ROUNDS} 轮", 0.6

        # 活跃度检查
        total_activity = len(item.cited_by) + len(item.refined_by) + len(item.challenged_by)
        recent_activity = item.round >= round_id - 1  # 是否在最近一轮有活动

        # 成熟：高评分 + 活跃
        if item.score >= self.PROMOTE_SCORE_THRESHOLD and total_activity >= 2:
            return "mature", f"评分 {item.score:.1f} >= 提升阈值, 活跃度 {total_activity}", 0.85

        # 发展中：有一定的活跃度和评分
        if total_activity > 0 and item.score >= 1.0:
            return "developing", f"活跃度 {total_activity}, 评分 {item.score:.1f}", 0.7

        # 衰退：低活跃度 + 长时间未更新
        if age >= self.DECAY_ROUNDS_THRESHOLD and not recent_activity and total_activity <= 1:
            return "decaying", f"连续 {age} 轮无显著活动", 0.65

        # 默认保持当前阶段
        current = self._get_current_stage(item.id)
        if current:
            return current, "保持当前阶段", 0.5

        return "new", "初始分配", 0.5

    def archive(self, pool: EssencePool, item_id: int, round_id: int) -> bool:
        """
        手动归档一条精华。

        将精华移出活跃池，但保留历史记录。
        """
        item = pool._get_item(item_id)
        if not item:
            return False

        # 记录归档
        self.records[item_id] = LifecycleRecord(
            essence_id=item_id,
            stage="archived",
            entered_at_round=round_id,
            previous_stage=self._get_current_stage(item_id) or "unknown",
            reason="手动归档",
            confidence=1.0,
        )
        self._archived_items.append(item_id)

        # 从活跃阶段列表中移除
        for stage in self._stage_items:
            if item_id in self._stage_items[stage]:
                self._stage_items[stage].remove(item_id)

        self._transition_history.append({
            "essence_id": item_id,
            "from": self._get_current_stage(item_id) or "unknown",
            "to": "archived",
            "round": round_id,
            "reason": "手动归档",
        })
        return True

    def promote(self, pool: EssencePool, item_id: int, round_id: int) -> bool:
        """
        手动提升一条精华为成熟状态。

        用于人工干预，将有价值的精华提前标记为成熟。
        """
        item = pool._get_item(item_id)
        if not item:
            return False

        self.records[item_id] = LifecycleRecord(
            essence_id=item_id,
            stage="mature",
            entered_at_round=round_id,
            previous_stage=self._get_current_stage(item_id) or "unknown",
            reason="手动提升",
            confidence=1.0,
        )
        self._stage_items["mature"].append(item_id)

        self._transition_history.append({
            "essence_id": item_id,
            "from": self._get_current_stage(item_id) or "unknown",
            "to": "mature",
            "round": round_id,
            "reason": "手动提升",
        })
        return True

    def get_active(self, pool: EssencePool) -> List[EssenceItem]:
        """
        获取当前活跃的精华（非归档状态）。

        返回按评分排序的活跃精华列表。
        """
        active_items = []
        for item in pool.items:
            if item.id not in self._archived_items:
                stage = self._get_current_stage(item.id)
                if stage and stage != "archived":
                    active_items.append(item)
                else:
                    # 未记录但不在归档列表中，视为活跃
                    active_items.append(item)

        return sorted(active_items, key=lambda x: x.score, reverse=True)

    def get_stage_summary(self) -> Dict[str, Any]:
        """
        获取生命周期阶段统计摘要。

        返回各阶段的精华数量和比例。
        """
        total = sum(len(items) for items in self._stage_items.values()) + len(self._archived_items)
        if total == 0:
            return {"total": 0, "stages": {}}

        summary: Dict[str, Any] = {
            "total": total,
            "stages": {},
            "transitions": len(self._transition_history),
        }

        for stage in ["new", "developing", "mature", "decaying", "archived"]:
            count = len(self._stage_items.get(stage, []))
            if stage == "archived":
                count = len(self._archived_items)
            summary["stages"][stage] = {
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0.0,
            }

        return summary

    def get_transition_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取生命周期转换历史"""
        return self._transition_history[-limit:]

    def to_dict(self) -> dict:
        return {
            "records": {
                str(eid): {
                    "stage": r.stage,
                    "entered_at": r.entered_at_round,
                    "previous_stage": r.previous_stage,
                    "reason": r.reason,
                    "confidence": r.confidence,
                }
                for eid, r in self.records.items()
            },
            "stage_items": {str(k): v for k, v in self._stage_items.items()},
            "archived_items": self._archived_items,
            "transitions": self._transition_history[-50:],
        }


# ============================================================================
# 扩展系统 8: CrossPollinationTracker (交叉授粉追踪)
# ============================================================================

@dataclass
class InfluenceLink:
    """影响链接"""
    source_id: int
    target_id: int
    influence_type: str  # citation, refinement, challenge, clarification
    strength: float  # 0.0 - 1.0
    detected_round: int
    evidence: str


@dataclass
class InnovationChain:
    """创新链"""
    essence_ids: List[int]
    start_round: int
    end_round: int
    length: int
    total_influence: float
    description: str


class CrossPollinationTracker:
    """
    交叉授粉追踪器 —— 追踪精华之间的影响和借力关系。

    功能：
    - 追踪精华之间的影响链（谁影响了谁）
    - 测量影响强度
    - 识别创新链（系列相关的精华）
    - 分析知识传播路径
    """

    def __init__(self):
        self.influence_links: List[InfluenceLink] = []
        self._source_outgoing: Dict[int, List[int]] = defaultdict(list)  # source_id -> [link_indices]
        self._target_incoming: Dict[int, List[int]] = defaultdict(list)  # target_id -> [link_indices]
        self._innovation_chains: List[InnovationChain] = []

    def track_influence(self, pool: EssencePool, round_id: int) -> List[InfluenceLink]:
        """
        追踪当前轮次中所有精华之间的影响关系。

        分析：
        - 引用关系（cited_by）
        - 深化关系（refined_by）
        - 反驳关系（challenged_by）
        - 澄清关系（clarifications）
        - 父-子关系（parent_id）
        """
        new_links: List[InfluenceLink] = []

        for item in pool.items:
            # 父-子关系：子受父影响
            if item.parent_id is not None:
                parent = pool._get_item(item.parent_id)
                if parent:
                    # 避免重复
                    if not self._link_exists(item.parent_id, item.id, "refinement"):
                        evidence = f"#{item.id} 深化自 #{item.parent_id}"
                        link = InfluenceLink(
                            source_id=item.parent_id,
                            target_id=item.id,
                            influence_type="refinement",
                            strength=self._calculate_refinement_strength(parent, item),
                            detected_round=round_id,
                            evidence=evidence,
                        )
                        new_links.append(link)
                        self._add_link(link)

            # 引用关系：被引用的精华影响了引用者
            # 注意：引用者是玩家，这里我们通过追踪引用来推断精华间的影响
            if item.cited_by:
                # 查找同一轮次中引用者提出的其他精华
                same_round_items = [it for it in pool.items
                                    if it.source_round == round_id
                                    and it.contributor in item.cited_by
                                    and it.id != item.id]
                for related in same_round_items:
                    if not self._link_exists(item.id, related.id, "citation"):
                        link = InfluenceLink(
                            source_id=item.id,
                            target_id=related.id,
                            influence_type="citation",
                            strength=0.4,
                            detected_round=round_id,
                            evidence=f"{item.contributor} 引用 #{item.id} 后提出 #{related.id}",
                        )
                        new_links.append(link)
                        self._add_link(link)

            # 反驳关系：被反驳的精华影响了反驳者
            if item.challenged_by:
                same_round_refutations = [it for it in pool.items
                                          if it.source_round == round_id
                                          and it.contributor in item.challenged_by
                                          and it.parent_id == item.id]
                for refutation in same_round_refutations:
                    if not self._link_exists(item.id, refutation.id, "challenge"):
                        link = InfluenceLink(
                            source_id=item.id,
                            target_id=refutation.id,
                            influence_type="challenge",
                            strength=0.6,
                            detected_round=round_id,
                            evidence=f"#{refutation.id} 反驳 #{item.id}",
                        )
                        new_links.append(link)
                        self._add_link(link)

        # 检测跨主题影响
        self._detect_cross_topic_influence(pool, round_id, new_links)

        return new_links

    def _link_exists(self, source_id: int, target_id: int, influence_type: str) -> bool:
        """检查链接是否已存在"""
        for idx in self._source_outgoing.get(source_id, []):
            link = self.influence_links[idx]
            if link.target_id == target_id and link.influence_type == influence_type:
                return True
        return False

    def _add_link(self, link: InfluenceLink) -> None:
        """添加链接到索引"""
        idx = len(self.influence_links)
        self.influence_links.append(link)
        self._source_outgoing[link.source_id].append(idx)
        self._target_incoming[link.target_id].append(idx)

    def _calculate_refinement_strength(self, parent: EssenceItem, child: EssenceItem) -> float:
        """计算深化关系的影响强度"""
        score_ratio = child.score / max(1.0, parent.score)
        tag_overlap = len(set(parent.tags) & set(child.tags)) / max(1, len(set(parent.tags) | set(child.tags)))
        strength = 0.5 + 0.3 * min(2.0, score_ratio) + 0.2 * tag_overlap
        return min(1.0, strength)

    def _detect_cross_topic_influence(self, pool: EssencePool, round_id: int,
                                       new_links: List[InfluenceLink]) -> None:
        """
        检测跨主题影响。

        当一个精华的内容关键词与另一个精华高度重叠时，推定存在影响。
        """
        if len(pool.items) < 2:
            return

        items_by_contributor: Dict[str, List[EssenceItem]] = defaultdict(list)
        for item in pool.items:
            items_by_contributor[item.contributor].append(item)

        for contributor, items in items_by_contributor.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if self._link_exists(a.id, b.id, "cross_topic") or \
                       self._link_exists(b.id, a.id, "cross_topic"):
                        continue

                    # 计算内容重叠
                    words_a = set(re.split(r'[，。！？、；：,\.!\?;:\s]', a.content))
                    words_b = set(re.split(r'[，。！？、；：,\.!\?;:\s]', b.content))
                    overlap = words_a & words_b
                    overlap_ratio = len(overlap) / max(1, min(len(words_a), len(words_b)))

                    if overlap_ratio > 0.3:
                        # 按时间顺序确定方向
                        if a.source_round < b.source_round:
                            src, tgt = a, b
                        else:
                            src, tgt = b, a

                        link = InfluenceLink(
                            source_id=src.id,
                            target_id=tgt.id,
                            influence_type="cross_topic",
                            strength=round(overlap_ratio, 3),
                            detected_round=round_id,
                            evidence=f"内容重叠 {overlap_ratio:.0%} (贡献者: {contributor})",
                        )
                        new_links.append(link)
                        self._add_link(link)

    def get_influence_network(self, essence_id: Optional[int] = None) -> Dict[str, Any]:
        """
        获取影响网络。

        如果指定 essence_id，返回该精华的影响子图；
        否则返回完整影响网络。
        """
        if essence_id is not None:
            # 子图：该精华的出边和入边
            relevant_indices = set(
                self._source_outgoing.get(essence_id, []) +
                self._target_incoming.get(essence_id, [])
            )
            links = [self.influence_links[i] for i in relevant_indices]
        else:
            links = self.influence_links

        # 统计节点
        node_ids: Set[int] = set()
        out_degree: Dict[int, int] = defaultdict(int)
        in_degree: Dict[int, int] = defaultdict(int)
        total_influence: Dict[int, float] = defaultdict(float)

        for link in links:
            node_ids.add(link.source_id)
            node_ids.add(link.target_id)
            out_degree[link.source_id] += 1
            in_degree[link.target_id] += 1
            total_influence[link.source_id] += link.strength
            total_influence[link.target_id] += link.strength * 0.5

        nodes = [
            {
                "essence_id": eid,
                "out_degree": out_degree.get(eid, 0),
                "in_degree": in_degree.get(eid, 0),
                "total_influence": round(total_influence.get(eid, 0.0), 3),
            }
            for eid in node_ids
        ]

        edges = [
            {
                "source": link.source_id,
                "target": link.target_id,
                "type": link.influence_type,
                "strength": round(link.strength, 3),
                "evidence": link.evidence,
            }
            for link in links
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_links": len(links),
                "total_nodes": len(nodes),
                "avg_strength": round(
                    statistics.mean([l.strength for l in links]), 3
                ) if links else 0.0,
            },
        }

    def find_innovation_chains(self, pool: EssencePool, min_length: int = 2) -> List[InnovationChain]:
        """
        发现创新链。

        创新链是一系列相互影响的精华，形成一条知识演进路径。
        使用 DFS 从每个精华出发，寻找最长路径。
        """
        if not self.influence_links:
            return []

        # 构建邻接表
        adj: Dict[int, List[Tuple[int, float, str]]] = defaultdict(list)
        for link in self.influence_links:
            adj[link.source_id].append((link.target_id, link.strength, link.influence_type))

        chains: List[InnovationChain] = []
        visited_edges: Set[Tuple[int, int]] = set()

        def dfs(current_id: int, path: List[int], depth: int, total_strength: float):
            """深度优先搜索创新链"""
            if depth >= min_length:
                # 构造成链
                essence_ids = list(path)
                start_round = min(
                    (pool._get_item(eid).source_round for eid in essence_ids if pool._get_item(eid)),
                    default=0
                )
                end_round = max(
                    (pool._get_item(eid).source_round for eid in essence_ids if pool._get_item(eid)),
                    default=0
                )

                # 生成描述
                descriptions = []
                for i in range(len(essence_ids) - 1):
                    eid1, eid2 = essence_ids[i], essence_ids[i + 1]
                    item1 = pool._get_item(eid1)
                    item2 = pool._get_item(eid2)
                    if item1 and item2:
                        descriptions.append(
                            f"#{eid1}({item1.content[:20]}...) -> #{eid2}({item2.content[:20]}...)"
                        )

                chain = InnovationChain(
                    essence_ids=essence_ids,
                    start_round=start_round,
                    end_round=end_round,
                    length=len(essence_ids),
                    total_influence=round(total_strength, 3),
                    description=" | ".join(descriptions),
                )
                chains.append(chain)

            if depth >= 5:  # 最大深度限制
                return

            for neighbor, strength, _ in adj.get(current_id, []):
                edge_key = (current_id, neighbor)
                if edge_key not in visited_edges and neighbor not in path:
                    visited_edges.add(edge_key)
                    dfs(neighbor, path + [neighbor], depth + 1, total_strength + strength)

        # 从每个节点出发
        all_nodes = set()
        for link in self.influence_links:
            all_nodes.add(link.source_id)
            all_nodes.add(link.target_id)

        for node in sorted(all_nodes):
            dfs(node, [node], 1, 0.0)

        # 按长度和影响力排序
        chains.sort(key=lambda c: (c.length, c.total_influence), reverse=True)
        self._innovation_chains = chains[:10]  # 保留最长最好的10条

        return self._innovation_chains

    def get_influence_statistics(self) -> Dict[str, Any]:
        """获取影响网络统计"""
        if not self.influence_links:
            return {"status": "no_links"}

        link_types = Counter(l.influence_type for l in self.influence_links)
        strengths = [l.strength for l in self.influence_links]

        return {
            "total_links": len(self.influence_links),
            "link_types": dict(link_types),
            "avg_strength": round(statistics.mean(strengths), 3) if strengths else 0.0,
            "max_strength": round(max(strengths), 3) if strengths else 0.0,
            "min_strength": round(min(strengths), 3) if strengths else 0.0,
            "innovation_chains_found": len(self._innovation_chains),
            "longest_chain_length": max((c.length for c in self._innovation_chains), default=0),
        }

    def to_dict(self) -> dict:
        return {
            "influence_links": [
                {
                    "source_id": l.source_id,
                    "target_id": l.target_id,
                    "type": l.influence_type,
                    "strength": l.strength,
                    "detected_round": l.detected_round,
                    "evidence": l.evidence,
                }
                for l in self.influence_links
            ],
            "innovation_chains": [
                {
                    "essence_ids": c.essence_ids,
                    "length": c.length,
                    "total_influence": c.total_influence,
                    "description": c.description[:100],
                }
                for c in self._innovation_chains
            ],
        }


# ============================================================================
# 扩展系统 9: ConsensusStrengthMeter (共识强度测量)
# ============================================================================

@dataclass
class ConsensusMetric:
    """共识度量指标"""
    topic: str
    score: float  # 0.0 - 1.0
    agreement_level: str  # strong, moderate, weak, none
    participant_count: int
    agree_count: int
    disagree_count: int
    abstain_count: int
    key_points: List[str]


class ConsensusStrengthMeter:
    """
    共识强度测量器 —— 评估各主题的共识达成程度。

    功能：
    - 逐主题测量共识强度
    - 追踪共识随时间演变
    - 生成共识热力图
    - 识别分歧点和共识点
    """

    # 共识等级阈值
    STRONG_CONSENSUS_THRESHOLD = 0.75
    MODERATE_CONSENSUS_THRESHOLD = 0.55
    WEAK_CONSENSUS_THRESHOLD = 0.35

    def __init__(self):
        self.metrics: Dict[str, ConsensusMetric] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._topic_participants: Dict[str, Set[str]] = defaultdict(set)

    def measure_consensus(self, pool: EssencePool, topic: str,
                          essence_ids: List[int]) -> ConsensusMetric:
        """
        测量指定主题的共识强度。

        基于：
        - 投票赞成/反对比例
        - 反驳密度
        - 弃权率
        - 参与者多样性
        """
        if not essence_ids:
            return ConsensusMetric(
                topic=topic, score=0.0, agreement_level="none",
                participant_count=0, agree_count=0, disagree_count=0,
                abstain_count=0, key_points=[],
            )

        items = [pool._get_item(eid) for eid in essence_ids if pool._get_item(eid)]
        if not items:
            return ConsensusMetric(
                topic=topic, score=0.0, agreement_level="none",
                participant_count=0, agree_count=0, disagree_count=0,
                abstain_count=0, key_points=[],
            )

        # 统计投票
        total_approve = sum(len(it.approve_by) for it in items)
        total_reject = sum(len(it.reject_by) for it in items)
        total_abstain = sum(len(it.abstain_by) for it in items)
        total_votes = total_approve + total_reject + total_abstain

        # 参与者
        participants: Set[str] = set()
        for it in items:
            participants.update(it.approve_by)
            participants.update(it.reject_by)
            participants.update(it.abstain_by)
            participants.add(it.contributor)
        self._topic_participants[topic].update(participants)

        # 反驳统计
        challenged_items = sum(1 for it in items if it.challenged_by)
        challenge_ratio = challenged_items / len(items) if items else 0.0

        # 共识计算
        if total_votes == 0:
            # 无投票数据，基于评分和内容评估
            avg_score = statistics.mean([it.score for it in items])
            score_consistency = 1.0 - statistics.stdev([it.score for it in items]) / max(1.0, avg_score)
            score = min(0.6, max(0.1, score_consistency))
        else:
            approve_ratio = total_approve / total_votes if total_votes > 0 else 0.0
            abstain_ratio = total_abstain / total_votes if total_votes > 0 else 0.0
            score = approve_ratio - 0.3 * challenge_ratio - 0.15 * abstain_ratio
            score = max(0.0, min(1.0, score))

        # 共识等级
        if score >= self.STRONG_CONSENSUS_THRESHOLD:
            agreement_level = "strong"
        elif score >= self.MODERATE_CONSENSUS_THRESHOLD:
            agreement_level = "moderate"
        elif score >= self.WEAK_CONSENSUS_THRESHOLD:
            agreement_level = "weak"
        else:
            agreement_level = "none"

        # 关键点
        key_points = self._extract_key_points(items)

        metric = ConsensusMetric(
            topic=topic,
            score=round(score, 3),
            agreement_level=agreement_level,
            participant_count=len(participants),
            agree_count=total_approve,
            disagree_count=total_reject,
            abstain_count=total_abstain,
            key_points=key_points,
        )
        self.metrics[topic] = metric

        # 记录历史
        self._history[topic].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "score": metric.score,
            "level": metric.agreement_level,
            "participants": metric.participant_count,
        })

        return metric

    def _extract_key_points(self, items: List[EssenceItem]) -> List[str]:
        """提取关键共识/分歧点"""
        key_points: List[str] = []

        # 高评分精华（共识点）
        high_score = sorted(items, key=lambda x: x.score, reverse=True)[:2]
        for it in high_score:
            key_points.append(f"共识点: {it.content[:50]}... (评分 {it.score:.1f})")

        # 被反驳的精华（分歧点）
        challenged = [it for it in items if it.challenged_by]
        for it in challenged[:2]:
            challengers = ", ".join(it.challenged_by[:3])
            key_points.append(f"分歧点: {it.content[:40]}... (被 {challengers} 反驳)")

        return key_points[:4]

    def get_agreement_map(self, pool: EssencePool,
                          topics: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        生成共识热力图。

        返回所有主题的共识强度分布，以可视化格式呈现。
        """
        agreement_map: Dict[str, Any] = {
            "topics": {},
            "overall_score": 0.0,
            "strongest_topic": None,
            "weakest_topic": None,
        }

        scores = []
        strongest = ("", 0.0)
        weakest = ("", 1.0)

        for topic, essence_ids in topics.items():
            metric = self.measure_consensus(pool, topic, essence_ids)
            agreement_map["topics"][topic] = {
                "score": metric.score,
                "level": metric.agreement_level,
                "participants": metric.participant_count,
                "agree": metric.agree_count,
                "disagree": metric.disagree_count,
                "abstain": metric.abstain_count,
                "key_points": metric.key_points,
            }
            scores.append(metric.score)

            if metric.score > strongest[1]:
                strongest = (topic, metric.score)
            if metric.score < weakest[1]:
                weakest = (topic, metric.score)

        agreement_map["overall_score"] = round(statistics.mean(scores), 3) if scores else 0.0
        agreement_map["strongest_topic"] = {"topic": strongest[0], "score": strongest[1]}
        agreement_map["weakest_topic"] = {"topic": weakest[0], "score": weakest[1]}

        return agreement_map

    def track_consensus_evolution(self, topic: str) -> List[Dict[str, Any]]:
        """
        追踪指定主题的共识演变。

        返回随时间变化的共识分数序列。
        """
        return list(self._history.get(topic, []))

    def get_overall_consensus(self, pool: EssencePool,
                              topic_modeler: TopicModeler) -> Dict[str, Any]:
        """
        获取整体共识评估。

        综合所有主题的共识状态，给出整体评估。
        """
        if not topic_modeler.topics:
            return {"status": "no_topics", "overall_score": 0.0, "level": "none"}

        topic_essence_map = {
            name: cluster.essence_ids
            for name, cluster in topic_modeler.topics.items()
        }

        agreement_map = self.get_agreement_map(pool, topic_essence_map)
        scores = [t["score"] for t in agreement_map["topics"].values()]

        if not scores:
            return {"status": "no_data", "overall_score": 0.0, "level": "none"}

        overall = statistics.mean(scores)

        if overall >= self.STRONG_CONSENSUS_THRESHOLD:
            level = "strong"
        elif overall >= self.MODERATE_CONSENSUS_THRESHOLD:
            level = "moderate"
        elif overall >= self.WEAK_CONSENSUS_THRESHOLD:
            level = "weak"
        else:
            level = "none"

        return {
            "status": "measured",
            "overall_score": round(overall, 3),
            "level": level,
            "topic_count": len(agreement_map["topics"]),
            "strongest_area": agreement_map["strongest_topic"],
            "weakest_area": agreement_map["weakest_topic"],
            "details": agreement_map["topics"],
        }

    def to_dict(self) -> dict:
        return {
            "metrics": {
                topic: {
                    "score": m.score,
                    "level": m.agreement_level,
                    "participants": m.participant_count,
                    "agree": m.agree_count,
                    "disagree": m.disagree_count,
                    "abstain": m.abstain_count,
                }
                for topic, m in self.metrics.items()
            },
            "history": dict(self._history),
            "topic_participants": {
                topic: list(participants)
                for topic, participants in self._topic_participants.items()
            },
        }


# ============================================================================
# 扩展系统 10: GapAnalyzer (空白分析)
# ============================================================================

@dataclass
class TopicGap:
    """主题空白"""
    missing_topic: str
    related_keywords: List[str]
    suggested_questions: List[str]
    priority: float  # 0.0 - 1.0
    evidence: str


class GapAnalyzer:
    """
    空白分析器 —— 识别讨论中的缺失领域和未探索方向。

    功能：
    - 识别缺失主题（与预定义主题列表对比）
    - 分析讨论覆盖率
    - 建议探索方向
    - 生成空白报告
    """

    # 预期覆盖的主题领域（用于比较）
    EXPECTED_TOPICS: Dict[str, List[str]] = {
        "实施可行性": ["实施", "部署", "开发", "工期", "资源", "技术栈", "迁移"],
        "成本效益": ["成本", "预算", "ROI", "收益", "投入", "性价比", "节约"],
        "风险管理": ["风险", "安全", "隐私", "合规", "法律", "伦理", "监管"],
        "用户体验": ["用户", "体验", "界面", "可用性", "易用性", "满意度"],
        "技术架构": ["架构", "设计", "扩展性", "性能", "可用性", "维护"],
        "运营维护": ["运营", "维护", "监控", "运维", "支持", "迭代"],
        "市场竞争": ["竞争", "市场", "定位", "差异化", "优势", "份额"],
        "团队能力": ["团队", "技能", "培训", "招聘", "知识", "协作"],
        "时间规划": ["时间", "计划", "里程碑", "阶段", "截止", "周期"],
        "质量标准": ["质量", "测试", "标准", "指标", "度量", "验证"],
        "创新机会": ["创新", "突破", "专利", "研发", "前沿", "探索"],
        "利益相关方": ["利益", "相关方", "客户", "合作伙伴", "股东", "社区"],
    }

    def __init__(self):
        self.gaps: List[TopicGap] = []
        self._covered_topics: Set[str] = set()
        self._coverage_history: List[Dict[str, Any]] = []

    def analyze_gaps(self, pool: EssencePool,
                     topic_modeler: Optional[TopicModeler] = None) -> List[TopicGap]:
        """
        分析讨论中的空白。

        对比当前讨论覆盖的主题与预期主题列表，
        识别未被充分讨论的领域。
        """
        if not pool.items:
            # 如果无精华，所有主题都是空白
            for topic, keywords in self.EXPECTED_TOPICS.items():
                gap = TopicGap(
                    missing_topic=topic,
                    related_keywords=keywords,
                    suggested_questions=self._generate_questions(topic, keywords),
                    priority=0.5,
                    evidence="尚无讨论内容",
                )
                self.gaps.append(gap)
            return self.gaps

        # 收集已覆盖的主题
        all_content = " ".join(it.content for it in pool.items)
        for topic, keywords in self.EXPECTED_TOPICS.items():
            match_count = sum(1 for kw in keywords if kw in all_content)
            if match_count >= 2:
                self._covered_topics.add(topic)

        # 如果有 TopicModeler，补充已有主题
        if topic_modeler:
            for topic_name in topic_modeler.topics:
                self._covered_topics.add(topic_name)

        # 识别缺失主题
        all_topics = set(self.EXPECTED_TOPICS.keys())
        if topic_modeler:
            all_topics.update(topic_modeler.topics.keys())

        missing_topics = all_topics - self._covered_topics

        # 计算优先级
        self.gaps.clear()
        for topic in missing_topics:
            keywords = self.EXPECTED_TOPICS.get(topic, [])
            priority = self._calculate_gap_priority(topic, pool)

            # 检查是否部分覆盖
            if topic in self.EXPECTED_TOPICS:
                match_count = sum(1 for kw in self.EXPECTED_TOPICS[topic] if kw in all_content)
                coverage_ratio = match_count / max(1, len(self.EXPECTED_TOPICS[topic]))
                if coverage_ratio > 0.3:
                    # 部分覆盖，优先级降低
                    priority *= (1.0 - coverage_ratio)

            gap = TopicGap(
                missing_topic=topic,
                related_keywords=keywords,
                suggested_questions=self._generate_questions(topic, keywords),
                priority=round(priority, 3),
                evidence=f"当前讨论中未发现与「{topic}」相关的关键词",
            )
            self.gaps.append(gap)

        # 按优先级排序
        self.gaps.sort(key=lambda g: g.priority, reverse=True)

        # 记录历史
        self._coverage_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "covered_count": len(self._covered_topics),
            "total_expected": len(self.EXPECTED_TOPICS),
            "gap_count": len(self.gaps),
            "coverage_ratio": round(len(self._covered_topics) / max(1, len(self.EXPECTED_TOPICS)), 3),
        })

        return self.gaps

    def _calculate_gap_priority(self, topic: str, pool: EssencePool) -> float:
        """
        计算空白优先级。

        考虑因素：
        - 主题与当前讨论的相关性（通过关键词重叠判断）
        - 讨论的深度（轮次数）
        - 参与者数量
        """
        keywords = self.EXPECTED_TOPICS.get(topic, [])
        if not keywords:
            return 0.3

        # 相关性：检查当前讨论是否涉及相关概念
        all_content = " ".join(it.content for it in pool.items)
        relevance = 0.0
        for kw in keywords:
            for char in kw:
                if char in all_content:
                    relevance += 0.1
                    break
        relevance = min(1.0, relevance)

        # 讨论深度因子：轮次越多，越需要覆盖该主题
        rounds = set(it.source_round for it in pool.items)
        depth_factor = min(1.0, len(rounds) / 10.0)

        # 参与者因子
        participants = set(it.contributor for it in pool.items)
        participant_factor = min(1.0, len(participants) / 5.0)

        # 综合优先级
        priority = 0.4 * relevance + 0.3 * depth_factor + 0.3 * participant_factor
        return max(0.1, min(1.0, priority))

    def _generate_questions(self, topic: str, keywords: List[str]) -> List[str]:
        """
        生成探索性问题，引导讨论填补空白。

        为每个缺失主题生成 2-3 个引导性问题。
        """
        question_templates: Dict[str, List[str]] = {
            "实施可行性": [
                f"当前的「{topic}」方案在实施层面面临哪些具体挑战？",
                f"需要哪些资源和技术支持来确保「{topic}」的顺利落地？",
                f"「{topic}」的实施时间线应该如何规划？",
            ],
            "成本效益": [
                f"「{topic}」的预期成本和收益如何？",
                f"是否有更经济高效的替代方案？",
                f"长期来看，「{topic}」的投资回报率如何？",
            ],
            "风险管理": [
                f"在「{topic}」方面存在哪些潜在风险？",
                f"如何建立有效的风险缓解机制？",
                f"「{topic}」的合规和法律要求是什么？",
            ],
            "用户体验": [
                f"目标用户对「{topic}」的核心需求是什么？",
                f"如何衡量和优化「{topic}」的用户体验？",
                f"「{topic}」在用户界面设计上有什么最佳实践？",
            ],
            "技术架构": [
                f"「{topic}」的技术架构设计原则是什么？",
                f"如何确保「{topic}」架构的可扩展性和可维护性？",
                f"「{topic}」的关键技术选型考虑因素有哪些？",
            ],
            "运营维护": [
                f"「{topic}」的日常运营和监控策略是什么？",
                f"如何进行「{topic}」的持续改进和迭代？",
                f"「{topic}」的运维团队需要具备哪些能力？",
            ],
            "市场竞争": [
                f"在「{topic}」领域的市场格局如何？",
                f"我们的「{topic}」方案相比竞品有何独特优势？",
                f"如何通过「{topic}」建立差异化竞争力？",
            ],
            "团队能力": [
                f"执行「{topic}」需要哪些核心技能和团队配置？",
                f"当前团队在「{topic}」方面的能力缺口是什么？",
                f"如何通过培训或招聘补齐「{topic}」的能力短板？",
            ],
            "时间规划": [
                f"「{topic}」的关键里程碑和时间节点如何设定？",
                f"「{topic}」的分阶段实施计划是什么？",
                f"如何管理「{topic}」的时间进度风险？",
            ],
            "质量标准": [
                f"「{topic}」的质量标准应该如何定义？",
                f"如何建立「{topic}」的质量度量和验证体系？",
                f"「{topic}」的最佳实践和行业标准是什么？",
            ],
            "创新机会": [
                f"在「{topic}」领域有哪些创新机会尚未被发掘？",
                f"如何将前沿技术应用于「{topic}」？",
                f"「{topic}」的突破性可能来自哪些方向？",
            ],
            "利益相关方": [
                f"「{topic}」涉及哪些主要利益相关方？",
                f"如何管理和平衡各利益相关方的期望？",
                f"「{topic}」对客户和合作伙伴的具体影响是什么？",
            ],
        }

        questions = question_templates.get(topic, [
            f"关于「{topic}」有哪些关键考虑因素？",
            f"「{topic}」的现状和挑战是什么？",
            f"如何优化和改进「{topic}」方面的工作？",
        ])

        # 如果有关键词，融入问题
        if keywords:
            kw_sample = keywords[:3]
            questions.append(
                f"讨论中提到了「{', '.join(kw_sample)}」等概念，如何从「{topic}」角度系统分析？"
            )

        return questions[:3]

    def get_missing_topics(self, min_priority: float = 0.0) -> List[TopicGap]:
        """
        获取缺失主题列表。

        可按最低优先级过滤。
        """
        if min_priority > 0:
            return [g for g in self.gaps if g.priority >= min_priority]
        return list(self.gaps)

    def suggest_exploration(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        建议探索方向。

        基于空白分析结果，推荐优先级最高的探索方向，
        每个方向附带引导性问题。
        """
        if not self.gaps:
            return [{"message": "当前讨论覆盖全面，暂无明确的探索方向建议"}]

        sorted_gaps = sorted(self.gaps, key=lambda g: g.priority, reverse=True)
        suggestions = []

        for gap in sorted_gaps[:top_n]:
            suggestions.append({
                "missing_topic": gap.missing_topic,
                "priority": gap.priority,
                "related_keywords": gap.related_keywords,
                "suggested_questions": gap.suggested_questions,
                "rationale": (
                    f"当前讨论在「{gap.missing_topic}」方面存在空白。"
                    f"建议从以下角度引导讨论: {gap.suggested_questions[0]}"
                ),
            })

        return suggestions

    def get_coverage_report(self) -> str:
        """
        生成覆盖率报告文本。

        包含：
        - 总体覆盖率
        - 已覆盖主题列表
        - 缺失主题列表（高优先级）
        - 探索建议
        """
        total = len(self.EXPECTED_TOPICS)
        covered = len(self._covered_topics)
        coverage_ratio = covered / total if total > 0 else 0.0

        lines = [
            "=" * 50,
            "讨论空白分析报告",
            "=" * 50,
            f"\n总体覆盖率: {covered}/{total} ({coverage_ratio:.1%})",
        ]

        if coverage_ratio >= 0.8:
            lines.append("评价: 讨论覆盖全面")
        elif coverage_ratio >= 0.5:
            lines.append("评价: 部分领域需要补充")
        else:
            lines.append("评价: 大量主题尚未覆盖，建议拓展讨论范围")

        # 已覆盖
        if self._covered_topics:
            lines.append(f"\n已覆盖主题 ({len(self._covered_topics)}):")
            for topic in sorted(self._covered_topics):
                lines.append(f"  [+] {topic}")

        # 缺失主题
        missing = [g for g in self.gaps if g.priority >= 0.3]
        if missing:
            lines.append(f"\n高优先级缺失主题 ({len(missing)}):")
            for gap in missing[:5]:
                lines.append(f"  [-] {gap.missing_topic} (优先级: {gap.priority:.2f})")
                for q in gap.suggested_questions[:2]:
                    lines.append(f"      - {q}")

        if self.gaps:
            lines.append("\n建议探索方向:")
            for suggestion in self.suggest_exploration(top_n=3):
                lines.append(f"  * {suggestion['rationale']}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "gaps": [
                {
                    "missing_topic": g.missing_topic,
                    "related_keywords": g.related_keywords,
                    "suggested_questions": g.suggested_questions,
                    "priority": g.priority,
                    "evidence": g.evidence,
                }
                for g in self.gaps
            ],
            "covered_topics": sorted(self._covered_topics),
            "coverage_history": self._coverage_history,
        }
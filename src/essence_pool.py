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
"""

from dataclasses import dataclass, field
from typing import List, Optional
import datetime


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
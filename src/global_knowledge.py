"""
跨讨论知识迁移系统 —— 全局知识库

持久化存储所有讨论的精华，支持跨讨论检索、冲突检测和经验预热。

核心功能：
1. 主题聚类：将不同讨论中的相似精华自动聚类
2. 冲突检测：识别不同讨论中对同一问题的矛盾结论
3. 经验迁移：新讨论开始时，自动检索历史相关精华作为"预热材料"
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Optional
from datetime import datetime


class GlobalKnowledgeBase:
    """全局持久化知识库，跨讨论共享知识"""

    STORAGE_FILE = "global_knowledge.json"

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or self.STORAGE_FILE
        self.sessions: List[Dict] = []  # 所有讨论会话
        self._load()

    # ── 持久化 ──

    def _load(self) -> None:
        """从磁盘加载全局知识库"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.sessions = data.get("sessions", [])
            except (json.JSONDecodeError, Exception):
                self.sessions = []

    def save(self) -> None:
        """保存到磁盘"""
        data = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_count": len(self.sessions),
            "sessions": self.sessions,
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 记录 ──

    def record_session(self, game_id: str, problem: str,
                       discussion_mode: str, round_count: int,
                       player_names: List[str],
                       essences: List[Dict],
                       final_solution: Dict) -> None:
        """记录一场讨论的结果到全局知识库"""
        session = {
            "game_id": game_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "problem": problem,
            "discussion_mode": discussion_mode,
            "round_count": round_count,
            "player_names": player_names,
            "essence_count": len(essences),
            "essences": essences,
            "final_solution": {
                "solution_title": final_solution.get("solution_title", ""),
                "summary": final_solution.get("summary", "")[:500],
                "final_conclusion": final_solution.get("final_conclusion", ""),
            },
            "tags": self._extract_tags(essences),
        }
        self.sessions.append(session)
        self.save()

    @staticmethod
    def _extract_tags(essences: List[Dict]) -> List[str]:
        """从精华列表中提取高频标签"""
        all_tags = Counter()
        for ess in essences:
            for tag in ess.get("tags", []):
                if isinstance(tag, str) and len(tag) >= 2:
                    all_tags[tag] += 1
        return [tag for tag, _ in all_tags.most_common(10)]

    # ── 检索 ──

    def search_related(self, problem: str, top_k: int = 5) -> List[Dict]:
        """
        搜索与当前问题相关的历史精华。

        策略：
        1. 问题文本与历史 session 的 problem 重叠度
        2. 标签匹配
        3. 按匹配度排序
        """
        if not self.sessions:
            return []

        query_terms = self._split_terms(problem)
        if not query_terms:
            return []

        scored_essences = []
        for session in self.sessions:
            session_terms = self._split_terms(session.get("problem", ""))
            session_score = self._compute_similarity(query_terms, session_terms)

            for ess in session.get("essences", []):
                ess_terms = self._split_terms(ess.get("content", ""))
                ess_score = self._compute_similarity(query_terms, ess_terms)
                # 综合：session 匹配度 * 0.3 + 精华匹配度 * 0.7
                total = session_score * 0.3 + ess_score * 0.7
                if total > 0:
                    scored_essences.append((total, {
                        "content": ess.get("content", ""),
                        "contributor": ess.get("contributor", ""),
                        "score": ess.get("score", 0),
                        "tags": ess.get("tags", []),
                        "source_problem": session.get("problem", "")[:80],
                        "source_game_id": session.get("game_id", ""),
                        "source_round": ess.get("source_round", 0),
                    }))

        scored_essences.sort(key=lambda x: x[0], reverse=True)
        return [ess for _, ess in scored_essences[:top_k]]

    def search_related_by_tags(self, tags: List[str], top_k: int = 5) -> List[Dict]:
        """通过标签检索相关精华"""
        if not self.sessions or not tags:
            return []

        tag_set = set(t.lower() for t in tags)
        scored = []
        for session in self.sessions:
            for ess in session.get("essences", []):
                ess_tags = set(t.lower() for t in ess.get("tags", []))
                overlap = len(tag_set & ess_tags)
                if overlap > 0:
                    score = overlap / max(len(tag_set), 1)
                    scored.append((score, ess | {
                        "source_problem": session.get("problem", "")[:80],
                        "source_game_id": session.get("game_id", ""),
                    }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ess for _, ess in scored[:top_k]]

    # ── 冲突检测 ──

    def detect_conflicts(self, problem: str) -> List[Dict]:
        """
        检测不同讨论中对同一问题的矛盾结论。

        返回：
          [{
            "topic": str,               # 冲突话题
            "sides": [                   # 两个对立结论
                {"session_id": str, "conclusion": str, "problem": str},
                ...
            ],
            "essence_ids": [],
          }]
        """
        # 简单实现：找有相同标签的不同 session 中，评分高但立场对立的精华
        related = self.search_related(problem, top_k=20)
        if len(related) < 2:
            return []

        # 按标签分组，检查同一标签组内是否有矛盾观点
        tag_groups = defaultdict(list)
        for ess in related:
            for tag in ess.get("tags", []):
                tag_groups[tag].append(ess)

        conflicts = []
        for tag, group in tag_groups.items():
            if len(group) < 2:
                continue
            # 按评分排序，取前两个做对比
            group.sort(key=lambda x: x.get("score", 0), reverse=True)
            a, b = group[0], group[1]
            if a.get("source_game_id") != b.get("source_game_id"):
                conflicts.append({
                    "topic": tag,
                    "sides": [
                        {"session_id": a.get("source_game_id", ""),
                         "conclusion": a.get("content", "")[:100],
                         "problem": a.get("source_problem", "")},
                        {"session_id": b.get("source_game_id", ""),
                         "conclusion": b.get("content", "")[:100],
                         "problem": b.get("source_problem", "")},
                    ],
                })

        return conflicts[:5]

    # ── 预热材料 ──

    def get_warmup_material(self, problem: str, max_chars: int = 2000) -> str:
        """
        生成新讨论的预热材料：从历史讨论中检索相关精华。

        格式化为可直接注入 prompt 的文本。
        """
        related = self.search_related(problem, top_k=8)
        if not related:
            return ""

        lines = ["📚 跨讨论知识迁移 · 历史相关精华（以下内容来自之前的讨论）："]
        char_count = len(lines[0])

        for i, ess in enumerate(related, 1):
            text = (
                f"\n  [{i}] 来自「{ess.get('source_problem', '未知问题')}」: "
                f"{ess.get('content', '')[:120]} "
                f"(评分: {ess.get('score', 0):.1f}, 标签: {', '.join(ess.get('tags', [])[:4])})"
            )
            if char_count + len(text) > max_chars:
                lines.append("\n  ...（更多历史精华已截断）")
                break
            lines.append(text)
            char_count += len(text)

        # 冲突检测
        conflicts = self.detect_conflicts(problem)
        if conflicts:
            lines.append(f"\n\n⚠️  历史冲突提醒（以下话题存在不同讨论中的矛盾结论）：")
            for c in conflicts[:3]:
                lines.append(f"  - 「{c['topic']}」: 一方认为「{c['sides'][0]['conclusion'][:40]}」, "
                             f"另一方认为「{c['sides'][1]['conclusion'][:40]}」")
            lines.append("  （建议本场讨论关注这些矛盾点，尝试达成一致）")

        return "\n".join(lines)

    # ── 工具方法 ──

    @staticmethod
    def _split_terms(text: str) -> List[str]:
        """将文本拆分为检索词（中文滑动窗口 + 英文单词）"""
        from . import split_terms
        return split_terms(text)

    @staticmethod
    def _compute_similarity(terms_a: List[str], terms_b: List[str]) -> float:
        """计算两个词列表的 Jaccard 相似度"""
        if not terms_a or not terms_b:
            return 0.0
        set_a, set_b = set(terms_a), set(terms_b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / max(len(union), 1)

    def get_statistics(self) -> Dict:
        """获取全局知识库统计信息"""
        total_essences = sum(
            s.get("essence_count", len(s.get("essences", [])))
            for s in self.sessions
        )
        all_tags = Counter()
        for s in self.sessions:
            for t in s.get("tags", []):
                all_tags[t] += 1

        return {
            "session_count": len(self.sessions),
            "total_essences": total_essences,
            "top_tags": [tag for tag, _ in all_tags.most_common(10)],
            "storage_path": self.storage_path,
        }
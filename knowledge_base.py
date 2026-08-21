"""
知识库系统 —— 内存知识库 + 标签匹配搜索

所有讨论发言和精华见解存储在内存中，专家可以按需搜索，
避免每次都将全量上下文注入 prompt，大幅节省 token。

搜索策略（简单稳定，向下兼容）：
1. 精华条目携带LLM提炼的关键词 tags（精华生成阶段已生成）
2. 发言条目用内容分词 + 子串匹配
3. 按"标签命中 + 内容子串匹配 + 新鲜度 + 精华评分"综合排序
"""

import re
from collections import Counter
from typing import List, Dict


class KnowledgeBase:
    """内存知识库，标签+子串混合搜索"""

    def __init__(self):
        self.discussions: List[Dict] = []
        self.essences: List[Dict] = []
        self._items: List[Dict] = []

    def add_discussion(self, round_id: int, player_name: str,
                       speech: str, key_insight: str, action: str) -> None:
        """添加一条讨论发言"""
        insight_part = f" 核心见解: {key_insight}" if key_insight else ""
        full_text = f"第{round_id}轮 {player_name}发言: {speech}{insight_part}"
        item = {
            "type": "discussion",
            "round": round_id,
            "player": player_name,
            "speech": speech,
            "insight": key_insight,
            "action": action,
            "text": full_text,
            "search_blob": f"{player_name} {speech} {key_insight}".lower(),
        }
        self.discussions.append(item)
        self._items.append(item)

    def add_essence(self, essence_item) -> None:
        """添加一条精华见解（携带LLM提炼的tags，用于搜索匹配）"""
        tags = list(essence_item.tags or [])
        first_tag = tags[0] if tags else "论点"
        full_text = (
            f"[精华#{essence_item.id} {first_tag}] "
            f"{essence_item.content} "
            f"(贡献者: {essence_item.contributor}, 评分: {essence_item.score:.1f})"
        )
        # 把 tags 也拼进 search_blob 便于子串匹配
        tag_str = " ".join(tags)
        item = {
            "type": "essence",
            "id": essence_item.id,
            "text": full_text,
            "content": essence_item.content,
            "contributor": essence_item.contributor,
            "score": essence_item.score,
            "essence_type": first_tag,
            "round": getattr(essence_item, 'source_round',
                             getattr(essence_item, 'round_id', 0)),
            "tags": tags,
            "search_blob": (
                f"{essence_item.content} {essence_item.contributor} "
                f"{tag_str} {' '.join(tags)}"
            ).lower(),
        }
        self.essences.append(item)
        self._items.append(item)

    @staticmethod
    def _split_terms(query: str) -> List[str]:
        """把查询拆成若干子串（中文2-4字滑动 + 英文单词），用于子串匹配"""
        query = (query or "").lower().strip()
        if not query:
            return []
        terms = set()
        # 英文单词
        for w in re.findall(r'[a-zA-Z]{3,}', query):
            terms.add(w)
        # 中文短语：去掉空格标点后的连续中文用2字和3字滑动
        chinese_segments = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        for seg in chinese_segments:
            if len(seg) <= 4:
                terms.add(seg)
            else:
                for size in (2, 3, 4):
                    for i in range(len(seg) - size + 1):
                        terms.add(seg[i:i + size])
        return [t for t in terms if len(t) >= 2]

    def search(self, query: str, top_k: int = 5,
               exclude_player: str = "") -> List[Dict]:
        """标签+子串混合搜索，简单稳定"""
        if not self._items or not query:
            return []

        terms = self._split_terms(query)
        scores = []

        for idx, item in enumerate(self._items):
            if exclude_player and item.get("player") == exclude_player:
                continue

            score = 0.0

            # ── 1) 标签精确匹配（精华才有 tags；权重最高）──
            tags = item.get("tags") or []
            for tag in tags:
                t_lower = tag.lower()
                # 标签出现在查询里 → 强加分
                if t_lower in query.lower():
                    score += 3.0
                # 查询词包含在标签里 → 加分
                for term in terms:
                    if term in t_lower:
                        score += 1.5

            # ── 2) 内容子串匹配（讨论发言和精华都走这个）──
            blob = item.get("search_blob", "")
            if blob and terms:
                matches = sum(1 for term in terms if term in blob)
                # 匹配比例越高分越高
                score += matches * 0.8

            # ── 3) 新鲜度加分 ──
            round_num = item.get("round", 0) or 0
            score += round_num * 0.05

            # ── 4) 精华评分加权 ──
            if item.get("type") == "essence":
                score += item.get("score", 0) * 0.15

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [self._items[idx] for idx, _ in scores[:top_k]]

    def search_by_persona(self, persona: str, top_k: int = 5) -> List[Dict]:
        """根据专家人设自动搜索相关上下文"""
        return self.search(persona, top_k=top_k)

    def _collect_all_tags(self) -> List[str]:
        """从所有精华tags和高频讨论词汇中提取话题展示标签"""
        all_tags = Counter()
        for item in self.essences:
            for t in item.get("tags", []):
                clean = re.sub(r'[\s\W]+', '', t)
                if clean and len(clean) >= 2:
                    all_tags[clean] += 2  # 精华tags权重高

        # 兜底：如果精华tags还少，从内容词频抽
        if sum(all_tags.values()) < 10:
            for item in self._items:
                content = item.get("content", "") or item.get("speech", "") or ""
                for seg in re.findall(r'[\u4e00-\u9fff]{2,4}', content):
                    all_tags[seg] += 1

        return [tag for tag, _ in all_tags.most_common(15)]

    def get_index_text(self) -> str:
        """生成知识库的紧凑索引（用于注入prompt）"""
        n_discussions = len(self.discussions)
        n_essences = len(self.essences)
        if n_discussions == 0 and n_essences == 0:
            return "（知识库为空，尚无讨论记录）"

        top_topics = self._collect_all_tags()

        lines = [
            f"📚 知识库总览：",
            f"  讨论发言: {n_discussions} 条 | 精华见解: {n_essences} 条",
        ]
        if top_topics:
            lines.append(f"  覆盖话题: {'、'.join(top_topics)}")

        if self.essences:
            sorted_ess = sorted(self.essences,
                                key=lambda x: x.get("score", 0), reverse=True)
            lines.append(f"  🏆 精华精华 ({len(sorted_ess)} 条):")
            for ess in sorted_ess[:3]:
                content = ess['content'][:60]
                lines.append(f"    #[{ess['id']}] {content}... ({ess['score']:.1f}分)")
        return "\n".join(lines)

    def format_search_results(self, results: List[Dict],
                              max_chars: int = 1500) -> str:
        """格式化搜索结果，限制字符数"""
        if not results:
            return "（未找到相关结果）"

        lines = []
        char_count = 0
        for item in results:
            if item["type"] == "discussion":
                text = (f"[第{item['round']}轮 {item['player']}] "
                        f"{item['speech'][:120]}")
                if item.get("insight"):
                    text += f"\n  💡 {item['insight']}"
            else:
                text = (f"[精华#{item['id']} {item['essence_type']} "
                        f"评分:{item['score']:.1f}] {item['content'][:120]}")
                tags = item.get("tags") or []
                if tags:
                    text += f"\n  🏷️ {', '.join(tags[:6])}"

            if char_count + len(text) > max_chars:
                lines.append("  ...（更多结果已截断）")
                break
            lines.append(f"  {text}")
            char_count += len(text)

        return "\n".join(lines)
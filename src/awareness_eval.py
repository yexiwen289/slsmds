"""
意识评定器 —— 基于 Embedding API 的语义评分引擎

使用 OpenAI Embedding API 将文本嵌入为 1536 维语义向量，
通过计算与精华池中所有向量的语义距离来评估五维度意识得分。

核心优势：
  - 直接理解语义，不依赖关键词匹配
  - 对不同写作风格、不同表达方式具有稳定性
  - 1536 维稠密向量捕捉完整语义信息
"""

import json
import numpy as np
from openai import OpenAI

_DIMENSION_NAMES = ["capability", "mission", "emotion", "culture", "perspective"]
_DIMENSION_LABELS = {
    "capability": "能力意识",
    "mission": "使命意识",
    "emotion": "情感意识",
    "culture": "文化意识",
    "perspective": "视角意识",
}

# 情感锚点文本（预嵌入后缓存）
_EMOTION_ANCHORS = [
    "恐惧、害怕、担忧、焦虑、不安、紧张、惶恐",
    "愤怒、生气、不满、厌恶、恼火、怨恨",
    "悲伤、忧伤、难过、痛苦、失落、遗憾、哀伤",
    "快乐、喜悦、高兴、满足、欣慰、温暖、幸福",
    "希望、美好、期待、向往、憧憬、信任",
    "惊讶、震惊、意外、震撼、惊叹、不可思议",
    "好奇、探求、追问、求知、探索、质疑、求知欲",
    "敬畏、庄严、崇高、深邃、浩瀚、永恒、超越",
    "绝望、无助、孤独、寂寞、空虚、迷茫",
    "激情、渴望、热爱、狂热、迷恋、执着",
]


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


class EmbeddingEvaluator:
    """
    基于 Embedding API 的意识评定器。

    将输出文本和精华池内容都通过 Embedding API 嵌入为 1536 维向量，
    在稠密语义空间中计算五维度得分。
    """

    DIMENSION_NAMES = _DIMENSION_NAMES
    DIMENSION_LABELS = _DIMENSION_LABELS

    WEIGHTS = {"capability": 0.20, "mission": 0.20,
               "emotion": 0.15, "culture": 0.15,
               "perspective": 0.30}

    def __init__(self, essence_pool=None, problem="", api_key="", base_url="",
                 embedding_model="text-embedding-3-small"):
        self.essence_pool = essence_pool
        self.problem = problem
        self.embedding_model = embedding_model
        self._client = None
        self._cache = {}
        self._history = []

        # 初始化 OpenAI 客户端
        if api_key and base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)

        # 预计算缓存
        self._ref_embeddings = None
        self._ref_texts = None
        self._problem_embedding = None
        self._emotion_anchor_embeddings = None

    def _embed(self, text: str) -> np.ndarray:
        """将文本嵌入为稠密向量（带缓存）"""
        if not text or not text.strip():
            return np.zeros(1536)

        # 用前200字做 key（避免长文本缓存 miss）
        cache_key = text[:200]
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._client:
            raise RuntimeError("Embedding client 未初始化")

        resp = self._client.embeddings.create(
            model=self.embedding_model,
            input=text[:8000]  # API 长度限制
        )
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        self._cache[cache_key] = emb
        return emb

    def _get_ref_embeddings(self):
        """获取精华池所有内容的 embedding（缓存）"""
        if self._ref_embeddings is not None:
            return self._ref_embeddings, self._ref_texts

        refs = []
        ref_texts = []
        if self.essence_pool and hasattr(self.essence_pool, 'items') and self.essence_pool.items:
            for item in self.essence_pool.items:
                content = getattr(item, 'content', '') or (item.get('content', '') if isinstance(item, dict) else '')
                if content and len(content) > 10:
                    try:
                        emb = self._embed(content)
                        refs.append(emb)
                        ref_texts.append(content)
                    except Exception:
                        pass

        self._ref_embeddings = refs
        self._ref_texts = ref_texts
        return refs, ref_texts

    def _get_problem_embedding(self):
        """获取问题文本的 embedding（缓存）"""
        if self._problem_embedding is not None:
            return self._problem_embedding
        if not self.problem:
            return None
        try:
            self._problem_embedding = self._embed(self.problem)
        except Exception:
            self._problem_embedding = None
        return self._problem_embedding

    def _get_emotion_anchors(self):
        """获取情感锚点 embedding（缓存）"""
        if self._emotion_anchor_embeddings is not None:
            return self._emotion_anchor_embeddings

        anchors = []
        for anchor_text in _EMOTION_ANCHORS:
            try:
                emb = self._embed(anchor_text)
                anchors.append(emb)
            except Exception:
                pass
        self._emotion_anchor_embeddings = anchors
        return anchors

    def clear_cache(self):
        """清空所有缓存"""
        self._cache = {}
        self._ref_embeddings = None
        self._ref_texts = None
        self._problem_embedding = None
        self._emotion_anchor_embeddings = None

    def evaluate(self, text: str, context: dict = None) -> dict:
        if not text or not text.strip():
            return self._empty_result()

        # 1. 嵌入待评文本
        try:
            v = self._embed(text)
        except Exception as e:
            return self._empty_result()

        # 2. 收集参考嵌入
        refs, ref_texts = self._get_ref_embeddings()

        # 2.5 相关性门控：与问题的语义对齐度（相对精华池平均值）
        problem_emb = self._get_problem_embedding()
        if problem_emb is not None:
            raw_rel = max(0.0, _cosine_sim(v, problem_emb))
            if refs:
                ref_rels = [max(0.0, _cosine_sim(r, problem_emb)) for r in refs]
                avg_ref_rel = float(np.mean(ref_rels))
            else:
                avg_ref_rel = 0.5
            relative_rel = raw_rel / max(avg_ref_rel, 1e-8)
            # 非线性门控：平方使低相关性文本被更强地惩罚
            relevance = min(1.0, 0.2 + 0.8 * min(1.0, relative_rel ** 1.5))
        elif refs:
            center = np.mean(np.array(refs), axis=0)
            raw_rel = max(0.0, _cosine_sim(v, center))
            ref_rels = [max(0.0, _cosine_sim(r, center)) for r in refs]
            avg_ref_rel = float(np.mean(ref_rels))
            relative_rel = raw_rel / max(avg_ref_rel, 1e-8)
            relevance = min(1.0, 0.2 + 0.8 * min(1.0, relative_rel ** 1.5))
        else:
            relevance = 0.5

        # 2.6 文本长度归一化：短文本信息量低，应适当压低
        text_len = len(text)
        if text_len < 80:
            length_factor = 0.3 + 0.7 * (text_len / 80.0)
        elif text_len > 200:
            length_factor = 1.0
        else:
            length_factor = 0.8 + 0.2 * ((text_len - 80) / 120.0)

        # 综合门控 = 相关性 × 长度归一化
        gate = relevance * length_factor

        # 3. 计算五维度分数
        scores = {}
        scores["capability"] = self._score_capability(v, refs, gate)
        scores["mission"] = self._score_mission(v, refs, text, relevance)
        scores["emotion"] = self._score_emotion(v, refs, text, gate)
        scores["culture"] = self._score_culture(v, refs, gate)
        scores["perspective"] = self._score_perspective(v, refs, gate)

        # 4. 加权综合
        overall = sum(scores[d] * self.WEIGHTS[d] for d in self.DIMENSION_NAMES)

        record = {
            "capability": round(scores["capability"], 4),
            "mission": round(scores["mission"], 4),
            "emotion": round(scores["emotion"], 4),
            "culture": round(scores["culture"], 4),
            "perspective": round(scores["perspective"], 4),
            "awareness_score": round(overall, 4),
            "relevance": round(float(relevance), 4),
            "context": context or {},
        }
        self._history.append(record)
        return record

    def _empty_result(self):
        return {d: 0.0 for d in self.DIMENSION_NAMES} | {"awareness_score": 0.0}

    # ── 五维度评分 ──────────────────────────────────────────

    def _score_capability(self, v: np.ndarray, refs: list, relevance: float) -> float:
        """能力意识 ≡ 相关性 × 边界距离

        只有既切题又超越边界的文本才得高分
        """
        if not refs:
            return round(min(1.0, relevance * 0.7), 4)

        all_v = np.array(refs)
        center = np.mean(all_v, axis=0)

        dist = float(np.linalg.norm(v - center))
        ref_dists = np.linalg.norm(all_v - center, axis=1)
        max_ref = float(np.max(ref_dists)) if len(ref_dists) > 0 else 1.0
        mean_ref = float(np.mean(ref_dists)) if len(ref_dists) > 0 else 1.0

        boundary_ratio = dist / max(max_ref, 1e-8)
        mean_ratio = dist / max(mean_ref, 1e-8)

        # 几何分
        geo_score = 0.6 * min(1.0, boundary_ratio) + 0.4 * min(1.0, mean_ratio * 0.7)
        # 门控：相关性 × 几何分
        score = relevance * geo_score
        return round(min(1.0, score), 4)

    def _score_mission(self, v: np.ndarray, refs: list, text: str, relevance: float) -> float:
        """使命意识 ≡ 与问题的语义对齐度 + 在讨论框架内"""
        problem_emb = self._get_problem_embedding()
        if problem_emb is not None:
            sim_to_problem = max(0.0, _cosine_sim(v, problem_emb))
        else:
            sim_to_problem = relevance

        if refs:
            ref_sims = [_cosine_sim(v, r) for r in refs]
            avg_sim = float(np.mean(ref_sims))
            max_sim = float(np.max(ref_sims))
        else:
            avg_sim = 0.5
            max_sim = 0.5

        score = 0.5 * sim_to_problem + 0.3 * max(0.0, avg_sim) + 0.2 * max(0.0, max_sim)
        return round(min(1.0, score), 4)

    def _score_emotion(self, v: np.ndarray, refs: list, text: str, relevance: float) -> float:
        """情感意识 ≡ 与情感锚点的语义对齐度（受相关性门控）"""
        anchors = self._get_emotion_anchors()
        if not anchors:
            return round(min(1.0, relevance * 0.3), 4)

        sims = [_cosine_sim(v, a) for a in anchors]
        threshold = 0.3
        covered = sum(1 for s in sims if s > threshold)
        coverage = covered / len(anchors)
        max_sim = max(sims) if sims else 0.0
        avg_sim = float(np.mean(sims)) if sims else 0.0

        if refs:
            ref_emotion_sims = []
            for r in refs:
                r_sims = [_cosine_sim(r, a) for a in anchors]
                r_covered = sum(1 for s in r_sims if s > threshold)
                ref_emotion_sims.append(r_covered / len(anchors))
            avg_ref_emotion = float(np.mean(ref_emotion_sims))
        else:
            avg_ref_emotion = 0.3

        emotion_ratio = coverage / max(avg_ref_emotion, 1e-8)

        raw = (
            0.40 * coverage +
            0.25 * min(1.0, max_sim * 1.5) +
            0.20 * min(1.0, emotion_ratio * 0.6) +
            0.15 * avg_sim
        )
        # 门控
        score = relevance * raw
        return round(min(1.0, score), 4)

    def _score_culture(self, v: np.ndarray, refs: list, relevance: float) -> float:
        """文化意识 ≡ 与精华池语义分布的多样性（受相关性门控）

        使用方差异：与不同精华的相似度方差大 = 聚焦少数主题 = 低文化
        方差小 = 均匀覆盖多主题 = 高文化
        """
        if not refs:
            return round(min(1.0, relevance * 0.5), 4)

        sims = np.array([_cosine_sim(v, r) for r in refs])

        # 方差：方差大 = 只聚焦少数 = 低文化
        var_sim = float(np.var(sims))
        # 均值
        mean_sim = float(np.mean(sims))

        # 方差越低 → 文化越高（0~1 归一化）
        # 典型方差范围 0.001~0.05
        diversity = max(0.0, 1.0 - var_sim * 30.0)

        # 覆盖度：相似度超过阈值的精华数量比
        threshold = float(np.mean(sims)) - 0.5 * float(np.std(sims))
        covered = sum(1 for s in sims if s > threshold)
        coverage = covered / len(refs)

        raw = 0.5 * diversity + 0.3 * coverage + 0.2 * mean_sim
        # 门控
        score = relevance * raw
        return round(min(1.0, score), 4)

    def _score_perspective(self, v: np.ndarray, refs: list, relevance: float) -> float:
        """视角意识 ≡ 相关性 × 到最近邻的距离

        只有切题且独特的文本才得高分
        """
        if not refs:
            return round(min(1.0, relevance * 0.5), 4)

        sims = np.array([_cosine_sim(v, r) for r in refs])
        dists = 1.0 - sims

        nn_dist = float(np.min(dists))
        avg_dist = float(np.mean(dists))

        ref_nn_dists = []
        for i, r in enumerate(refs):
            r_sims = np.array([_cosine_sim(r, r2) for j, r2 in enumerate(refs) if j != i])
            r_dists = 1.0 - r_sims
            ref_nn_dists.append(float(np.min(r_dists)))
        avg_ref_nn = float(np.mean(ref_nn_dists)) if ref_nn_dists else 0.5

        uniqueness = nn_dist / max(avg_ref_nn, 1e-8)
        nn_ratio = nn_dist / max(avg_dist, 1e-8)

        raw = (
            0.50 * min(1.0, uniqueness) +
            0.30 * min(1.0, nn_ratio * 2.0) +
            0.20 * min(1.0, nn_dist * 2.0)
        )
        # 门控
        score = relevance * raw
        return round(min(1.0, score), 4)


# ── 兼容接口 ──────────────────────────────────────────────

class AwarenessEvaluator(EmbeddingEvaluator):
    """兼容旧版 AwarenessEvaluator 接口"""
    pass
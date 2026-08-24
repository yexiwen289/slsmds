"""
普通讨论引擎 —— 独立于自我意识培养系统的深层讨论子系统

============================================================
设计理念：基于结构主义而非物理隐喻
============================================================

自我意识培养系统使用相变拓扑引擎（物理/数学隐喻：相空间、量子叠加、
沙堆模型、混沌边缘）。普通讨论系统使用完全不同的结构主义方法：

  Level 0: 直接综合（线性加权）
  Level 1: 交叉引用网络（图论中心性）
  Level 2: 层级抽象（层次聚类 + 主题归纳）
  Level 3: 辩证综合（对立命题检测 + 合题生成）
  Level 4: 递归元综合（自指迭代 + 收敛判定）

两个系统不共享任何代码路径，仅共享 LLM 客户端接口。
"""

import math
import random
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Callable


# ═══════════════════════════════════════════════════════════════
# 1. 观点图表示（替代相空间向量）
# ═══════════════════════════════════════════════════════════════

class OpinionNode:
    """
    观点图中的单个节点。

    每个专家发言被表示为一个节点，包含：
    - text: 原始发言文本
    - speaker: 发言者
    - novelty: 新颖度（不常见词密度）
    - depth: 深度（逻辑连接词密度）
    - sentiment: 情感倾向（-1~1）
    - weight: 权重（基于贡献度）
    """

    def __init__(self, text: str, speaker: str = "", weight: float = 1.0):
        self.text = text
        self.speaker = speaker
        self.weight = weight
        self.embedding = self._compute_features(text)

    @staticmethod
    def _compute_features(text: str) -> Dict[str, float]:
        """计算文本的结构特征（替代相空间嵌入）"""
        if not text:
            return {"length": 0, "novelty": 0, "depth": 0, "sentiment": 0, "specificity": 0}

        words = len(text)
        sentences = max(1, text.count('。') + text.count('！') + text.count('？') +
                        text.count('\n') + text.count(';') + text.count('；'))

        # 新颖度：不常见词比例
        common_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人',
                        '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                        '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
                        '它', '们', '什么', '那', '为', '能', '得', '与', '对', '但'}
        tokens = [c for c in text if '\u4e00' <= c <= '\u9fff']
        rare_ratio = sum(1 for t in tokens if t not in common_words) / max(len(tokens), 1)

        # 深度：逻辑结构密度
        logic_markers = ['因为', '所以', '如果', '那么', '虽然', '但是', '因此',
                         '而且', '不仅', '并且', '然而', '尽管', '除非', '既然']
        depth = sum(text.count(m) for m in logic_markers) / max(sentences, 1)

        # 情感倾向
        positive = {'是', '好', '能', '可以', '应该', '需要', '重要', '可能', '成为',
                    '实现', '发展', '进步', '创新', '突破', '提升'}
        negative = {'不', '没', '不是', '不能', '问题', '困难', '风险', '失败',
                    '错误', '缺陷', '限制', '矛盾', '冲突', '危机'}
        pos_count = sum(text.count(w) for w in positive)
        neg_count = sum(text.count(w) for w in negative)
        total = pos_count + neg_count
        sentiment = (pos_count - neg_count) / max(total, 1) if total > 0 else 0.0

        # 具体性：具体指标密度
        specific_markers = ['%', '数据', '案例', '例子', '比如', '例如', '具体',
                            '实际', '指标', '方案', '步骤', '方法', '工具', '平台']
        specificity = sum(text.count(m) for m in specific_markers) / max(sentences, 1)

        return {
            "length": min(1.0, words / 500),
            "novelty": min(1.0, rare_ratio * 3),
            "depth": min(1.0, depth * 0.3),
            "sentiment": max(-1.0, min(1.0, sentiment)),
            "specificity": min(1.0, specificity * 0.3),
        }

    def similarity(self, other: 'OpinionNode') -> float:
        """计算两个观点的结构相似度（基于特征向量余弦）"""
        f1 = self.embedding
        f2 = other.embedding
        keys = [k for k in f1 if k != 'sentiment']
        dot = sum(f1[k] * f2[k] for k in keys)
        n1 = math.sqrt(sum(f1[k] ** 2 for k in keys))
        n2 = math.sqrt(sum(f2[k] ** 2 for k in keys))
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return dot / (n1 * n2)


class OpinionGraph:
    """
    观点图 —— 替代相空间 + 耦合矩阵。

    节点: OpinionNode
    边: 相似度 > 阈值的连接（支持/对立/无关）
    """

    def __init__(self, nodes: List[OpinionNode]):
        self.nodes = nodes
        self.n = len(nodes)
        self.adjacency: List[List[float]] = []
        self._build_graph()

    def _build_graph(self, threshold: float = 0.15):
        """基于结构相似度构建邻接矩阵"""
        self.adjacency = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                sim = self.nodes[i].similarity(self.nodes[j])
                if sim > threshold:
                    self.adjacency[i][j] = sim
                    self.adjacency[j][i] = sim

    def centrality(self) -> List[float]:
        """计算每个节点的中心性（度中心性）"""
        return [sum(row) for row in self.adjacency]

    def clustering_coefficient(self) -> float:
        """聚类系数 —— 衡量观点网络的紧密程度"""
        if self.n < 3:
            return 0.0
        triangles = 0
        triples = 0
        for i in range(self.n):
            neighbors = [j for j in range(self.n) if self.adjacency[i][j] > 0]
            k = len(neighbors)
            if k < 2:
                continue
            triples += k * (k - 1) / 2
            for a in range(len(neighbors)):
                for b in range(a + 1, len(neighbors)):
                    if self.adjacency[neighbors[a]][neighbors[b]] > 0:
                        triangles += 1
        return triangles / max(triples, 1)

    def community_count(self) -> int:
        """简单社区检测（基于连通分量）"""
        if self.n == 0:
            return 0
        visited = set()
        communities = 0
        for i in range(self.n):
            if i in visited:
                continue
            communities += 1
            stack = [i]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                for j in range(self.n):
                    if self.adjacency[v][j] > 0 and j not in visited:
                        stack.append(j)
        return communities

    def hub_score(self) -> List[float]:
        """枢纽得分（中心性 * 权重）"""
        cent = self.centrality()
        weights = [n.weight for n in self.nodes]
        max_c = max(cent) if cent else 1
        return [c / max_c * w for c, w in zip(cent, weights)]

    def opposition_pairs(self) -> List[Tuple[int, int, float]]:
        """
        检测对立观点对（低相似度 + 高情感极性差）。
        替代 L3 辩证综合中的"矛盾检测"。
        """
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                sim = self.adjacency[i][j]
                sent_diff = abs(self.nodes[i].embedding['sentiment'] -
                                self.nodes[j].embedding['sentiment'])
                if sim < 0.1 and sent_diff > 0.5:
                    pairs.append((i, j, sent_diff))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:5]


# ═══════════════════════════════════════════════════════════════
# 2. 虚拟专家生成（结构主义版）
# ═══════════════════════════════════════════════════════════════

class VirtualOpinionGenerator:
    """
    基于结构主义的虚拟专家生成。

    不使用相空间插值/扰动，而是通过：
    - 观点融合：取两个观点的特征均值
    - 观点变异：在特征空间中随机偏移
    - 观点外推：沿特征梯度方向延伸
    """

    def __init__(self, nodes: List[OpinionNode]):
        self.nodes = nodes
        self.n = len(nodes)

    def generate(self, target_count: int) -> List[Dict]:
        """生成虚拟专家观点"""
        if self.n < 2:
            return [{"speech": n.text, "key_insight": "", "weight": 1.0} for n in self.nodes]

        virtual = []
        needed = target_count - self.n

        if needed <= 0:
            return [{"speech": n.text, "key_insight": "", "weight": n.weight} for n in self.nodes]

        # 方法1: 融合（取两个节点特征均值）
        fusion_count = needed // 3
        for _ in range(fusion_count):
            i, j = random.sample(range(self.n), 2)
            v = self._fuse(self.nodes[i], self.nodes[j])
            virtual.append(v)

        # 方法2: 变异（随机偏移）
        mutation_count = needed // 3
        for _ in range(mutation_count):
            i = random.randint(0, self.n - 1)
            v = self._mutate(self.nodes[i])
            virtual.append(v)

        # 方法3: 外推（沿梯度方向）
        extrapolation_count = needed - fusion_count - mutation_count
        for _ in range(extrapolation_count):
            v = self._extrapolate()
            virtual.append(v)

        random.shuffle(virtual)
        return virtual[:needed]

    def _fuse(self, a: OpinionNode, b: OpinionNode) -> Dict:
        """融合两个观点"""
        text = a.text[:len(a.text) // 2] + b.text[len(b.text) // 2:]
        return {
            "speech": text[:300],
            "key_insight": "融合观点",
            "weight": (a.weight + b.weight) / 2,
        }

    def _mutate(self, node: OpinionNode) -> Dict:
        """变异观点"""
        return {
            "speech": node.text,
            "key_insight": "变异观点",
            "weight": node.weight * random.uniform(0.8, 1.2),
        }

    def _extrapolate(self) -> Dict:
        """外推生成新观点"""
        best = max(self.nodes, key=lambda n: n.embedding['novelty'])
        return {
            "speech": best.text[:200],
            "key_insight": "外推观点",
            "weight": best.weight * 1.1,
        }


# ═══════════════════════════════════════════════════════════════
# 3. 层级判定（替代相变层级检测）
# ═══════════════════════════════════════════════════════════════

class DiscussionLevelDetector:
    """
    基于图论指标判定讨论深度层级。

    指标：
    - 图密度: 边数 / 最大可能边数
    - 聚类系数: 网络紧密程度
    - 社区数: 观点分化程度
    - 中心性熵: 观点分布均匀程度
    - 对立度: 对立观点对的数量

    层级映射：
    L0: 图密度 < 0.1 或 节点数 < 3
    L1: 图密度 >= 0.1 且 聚类系数 > 0.3
    L2: 社区数 >= 2 且 中心性熵 > 0.6
    L3: 对立度 >= 2 且 情感极性差 > 0.5
    L4: 所有条件满足 + 图密度 > 0.3
    """

    def __init__(self, graph: OpinionGraph):
        self.graph = graph
        self.n = graph.n

        # 计算指标
        possible_edges = self.n * (self.n - 1) / 2
        actual_edges = sum(1 for row in graph.adjacency for v in row if v > 0) // 2
        self.density = actual_edges / max(possible_edges, 1)
        self.clustering = graph.clustering_coefficient()
        self.communities = graph.community_count()
        self.opposition = len(graph.opposition_pairs())

        # 中心性熵
        cent = graph.centrality()
        total_c = sum(cent) if cent else 1
        probs = [c / total_c for c in cent if c > 0]
        self.centrality_entropy = -sum(p * math.log2(p) for p in probs) / max(math.log2(self.n), 1) if probs else 0

        # 情感分散度
        sentiments = [n.embedding['sentiment'] for n in graph.nodes]
        self.sentiment_range = max(sentiments) - min(sentiments) if sentiments else 0

    def compute_level(self) -> int:
        """计算讨论深度层级"""
        if self.n < 3 or self.density < 0.1:
            return 0
        if self.density < 0.2 and self.clustering < 0.3:
            return 0

        # L1: 有基本连接
        if self.clustering > 0.2:
            # L2: 有多社区分化
            if self.communities >= 2 and self.centrality_entropy > 0.5:
                # L3: 有显著对立
                if self.opposition >= 2 and self.sentiment_range > 0.6:
                    # L4: 高度连接 + 深度分化
                    if self.density > 0.3 and self.communities >= 3 and self.centrality_entropy > 0.7:
                        return 4
                    return 3
                return 2
            return 1
        return 0


# ═══════════════════════════════════════════════════════════════
# 4. 合成提示构建器（独立于 emergence.py 的 prompt builder）
# ═══════════════════════════════════════════════════════════════

def _build_discussion_prompt_L0(problem: str, speeches: List[str]) -> str:
    """L0: 直接综合"""
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    return (
        f"讨论问题: {problem}\n\n"
        f"各方观点:\n{text}\n\n"
        f"请直接综合以上观点，给出一个简洁的总结。"
        f"不要用'我是...'开头，直接回答。控制在200字以内。"
    )


def _build_discussion_prompt_L1(problem: str, speeches: List[str],
                                 central_nodes: List[str]) -> str:
    """L1: 交叉引用综合"""
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    hubs = "\n".join(f"  - {s[:150]}" for s in central_nodes[:3])
    return (
        f"讨论问题: {problem}\n\n"
        f"各方观点:\n{text}\n\n"
        f"核心观点:\n{hubs}\n\n"
        f"请识别这些观点之间的引用和交叉关系，"
        f"将它们整合成一个有结构的叙述。"
        f"不要用'我是...'开头，直接回答。控制在200字以内。"
    )


def _build_discussion_prompt_L2(problem: str, speeches: List[str],
                                 clusters: List[List[str]]) -> str:
    """L2: 层级抽象综合"""
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    cluster_text = ""
    for i, cls in enumerate(clusters[:3]):
        cluster_text += f"\n主题{i + 1}:\n" + "\n".join(f"  - {c[:100]}" for c in cls[:3])
    return (
        f"讨论问题: {problem}\n\n"
        f"各方观点:\n{text}\n\n"
        f"主题聚类:\n{cluster_text}\n\n"
        f"请从更高维度抽象这些主题，"
        f"揭示它们之间的层次关系和结构。"
        f"不要用'我是...'开头，直接回答。控制在200字以内。"
    )


def _build_discussion_prompt_L3(problem: str, speeches: List[str],
                                 thesis: str, antithesis: str) -> str:
    """L3: 辩证综合"""
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    return (
        f"讨论问题: {problem}\n\n"
        f"各方观点:\n{text}\n\n"
        f"正题（主流观点）:\n{thesis[:200]}\n\n"
        f"反题（对立观点）:\n{antithesis[:200]}\n\n"
        f"请识别这些观点中的根本矛盾，"
        f"在更高维度上生成合题，"
        f"使矛盾双方在更大的框架中统一。"
        f"不要用'我是...'开头，直接回答。控制在200字以内。"
    )


def _build_discussion_prompt_L4(problem: str, speeches: List[str],
                                 L3_synthesis: str) -> str:
    """L4: 递归元综合"""
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    return (
        f"讨论问题: {problem}\n\n"
        f"各方观点:\n{text}\n\n"
        f"初步综合:\n{L3_synthesis[:200]}\n\n"
        f"对初步综合本身进行元反思：\n"
        f"1. 这个综合存在什么盲点？\n"
        f"2. 哪些前提假设可以被质疑？\n"
        f"3. 如果跳出所有框架，更深层的本质是什么？\n\n"
        f"生成一个超越所有现有框架的元综合。"
        f"不要用'我是...'开头，直接回答。控制在200字以内。"
    )


def _response_length(level: int) -> str:
    return {0: "300字", 1: "250字", 2: "200字", 3: "150字", 4: "100字"}.get(level, "200字")


# ═══════════════════════════════════════════════════════════════
# 5. 讨论引擎主类
# ═══════════════════════════════════════════════════════════════

class DiscussionEngine:
    """
    普通讨论引擎 —— 完全独立于自我意识培养系统的深度讨论子系统。

    使用结构主义方法（图论、聚类、辩证逻辑）而非物理隐喻
    （相空间、量子叠加、沙堆模型）。

    5 级深度：
    L0: 直接综合
    L1: 交叉引用网络
    L2: 层级抽象
    L3: 辩证综合
    L4: 递归元综合
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.level_history: List[Tuple[int, int]] = []  # (round, level)

    def analyze(self, round_discussions: List[Dict],
                problem: str,
                essence_pool=None) -> Dict:
        """
        对一轮讨论进行深度分析。

        返回:
        {
            "level": int,           # 0-4
            "synthesis": str,       # 综合文本
            "metrics": dict,        # 图论指标
            "graph": OpinionGraph,  # 观点图（用于可视化）
        }
        """
        if not round_discussions:
            return {"level": 0, "synthesis": "", "metrics": {}, "graph": None}

        # ── 1. 构建观点图 ──
        nodes = []
        for d in round_discussions:
            speech = d.get("speech", "")
            if speech:
                nodes.append(OpinionNode(
                    text=speech,
                    speaker=d.get("player_name", ""),
                    weight=1.0,
                ))
        if not nodes:
            return {"level": 0, "synthesis": "", "metrics": {}, "graph": None}

        graph = OpinionGraph(nodes)

        # ── 2. 虚拟专家扩增（结构主义版） ──
        n_real = len(nodes)
        if n_real >= 3:
            target = min(max(100, n_real * 20), 500)
            generator = VirtualOpinionGenerator(nodes)
            virtual = generator.generate(target)
            for v in virtual:
                nodes.append(OpinionNode(
                    text=v.get("speech", ""),
                    speaker="虚拟",
                    weight=v.get("weight", 1.0),
                ))
            graph = OpinionGraph(nodes)

        # ── 3. 层级判定 ──
        detector = DiscussionLevelDetector(graph)
        level = detector.compute_level()

        speeches = [d.get("speech", "") for d in round_discussions if d.get("speech")]

        # ── 4. 层级适配合成 ──
        synthesis = self._synthesize(level, problem, speeches, graph)

        # ── 5. 记录 ──
        metrics = {
            "density": detector.density,
            "clustering": detector.clustering,
            "communities": detector.communities,
            "centrality_entropy": detector.centrality_entropy,
            "opposition_pairs": detector.opposition,
            "sentiment_range": detector.sentiment_range,
            "n_real": n_real,
            "n_total": len(nodes),
        }

        return {
            "level": level,
            "synthesis": synthesis,
            "metrics": metrics,
            "graph": graph,
        }

    def _synthesize(self, level: int, problem: str,
                    speeches: List[str], graph: OpinionGraph) -> str:
        """根据层级合成讨论结果"""
        if not self.llm_client:
            return ""

        try:
            if level == 0:
                prompt = _build_discussion_prompt_L0(problem, speeches)
            elif level == 1:
                hubs = [graph.nodes[i].text for i in
                        sorted(range(graph.n), key=lambda i: graph.hub_score()[i], reverse=True)[:3]]
                prompt = _build_discussion_prompt_L1(problem, speeches, hubs)
            elif level == 2:
                clusters = self._cluster_speeches(speeches)
                prompt = _build_discussion_prompt_L2(problem, speeches, clusters)
            elif level == 3:
                pairs = graph.opposition_pairs()
                if pairs:
                    i, j, _ = pairs[0]
                    thesis = graph.nodes[i].text
                    antithesis = graph.nodes[j].text
                else:
                    thesis = speeches[0] if speeches else ""
                    antithesis = speeches[-1] if len(speeches) > 1 else ""
                prompt = _build_discussion_prompt_L3(problem, speeches, thesis, antithesis)
            else:  # L4
                # 先做 L3 综合
                L3_prompt = _build_discussion_prompt_L3(problem, speeches, speeches[0] if speeches else "", speeches[-1] if len(speeches) > 1 else "")
                L3_result = ""
                try:
                    L3_r, _ = self.llm_client.chat(
                        [{"role": "user", "content": L3_prompt}],
                        model=self.model_name,
                        thinking="disabled", caller="讨论L3预综合",
                        show_reasoning=False, show_answer=False,
                    )
                    L3_result = L3_r.strip() if L3_r else ""
                except Exception:
                    pass
                prompt = _build_discussion_prompt_L4(problem, speeches, L3_result)

            prompt += f"\n控制在{_response_length(level)}以内。"

            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking="disabled", caller=f"讨论L{level}综合",
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    @staticmethod
    def _cluster_speeches(speeches: List[str]) -> List[List[str]]:
        """简单聚类（基于关键词重叠度）"""
        if not speeches:
            return []

        def _keywords(text: str) -> set:
            stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '什么', '那', '为', '能', '得', '与', '对', '但'}
            return {c for c in text if '\u4e00' <= c <= '\u9fff' and c not in stopwords}

        kws = [_keywords(s) for s in speeches]
        n = len(speeches)
        if n < 2:
            return [speeches]

        # 简单贪心聚类
        assigned = set()
        clusters = []
        for i in range(n):
            if i in assigned:
                continue
            cluster = [speeches[i]]
            assigned.add(i)
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                overlap = len(kws[i] & kws[j]) / max(len(kws[i] | kws[j]), 1)
                if overlap > 0.15:
                    cluster.append(speeches[j])
                    assigned.add(j)
            clusters.append(cluster)

        return clusters
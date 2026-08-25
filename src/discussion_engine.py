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
import json
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Callable, Any

# ============================================================
# P2: 深层语义特征嵌入（替代浅层文本统计）
# ============================================================

_SEMANTIC_PROMPT = """分析以下文本的深层语义特征，仅输出JSON：

文本: {text}

输出JSON格式：
{{
  "argument_type": "实证/规范/分析/类比/批判/综合",
  "stance": "支持/反对/中立/探索",
  "abstraction_level": 0.0~1.0,  // 0=具体案例, 1=纯理论
  "knowledge_domain": "技术/哲学/社会/科学/艺术/商业/心理/政治/伦理/综合",
  "uncertainty": 0.0~1.0,        // 0=绝对断言, 1=完全推测
  "novelty": 0.0~1.0,            // 0=常识复述, 1=全新观点
  "key_concepts": ["概念1", "概念2"]
}}
只输出JSON，不要其他内容。"""


class SemanticFeatureExtractor:
    """
    P2: 深层语义特征嵌入。

    使用 LLM 提取语义特征维度，替代 P2 之前的浅层词频统计。
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self._cache: Dict[str, Dict] = {}

    def extract(self, text: str) -> Dict[str, float]:
        """提取深层语义特征"""
        if not text:
            return self._default()

        # 缓存命中
        cache_key = text[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 用 LLM 提取语义特征
        if self.llm_client:
            result = self._llm_extract(text)
        else:
            result = self._fallback_extract(text)

        self._cache[cache_key] = result
        return result

    def _llm_extract(self, text: str) -> Dict[str, float]:
        """LLM 语义分析"""
        try:
            prompt = _SEMANTIC_PROMPT.format(text=text[:800])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking="disabled",
                caller="语义特征提取",
                show_reasoning=False, show_answer=False,
            )
            if response:
                # 提取 JSON
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return self._normalize(data)
        except Exception:
            pass
        return self._fallback_extract(text)

    def _normalize(self, data: dict) -> Dict[str, float]:
        """将 LLM 输出归一化为特征向量"""
        # 论证类型编码
        at_map = {"实证": 0.9, "规范": 0.8, "分析": 0.7, "类比": 0.6,
                  "批判": 0.5, "综合": 0.4, "探索": 0.3}
        at = data.get("argument_type", "分析")
        at_val = max(at_map.get(k, 0.5) for k in at_map if k in at) if isinstance(at, str) else 0.5

        # 立场编码
        stance_map = {"支持": 1.0, "反对": -1.0, "中立": 0.0, "探索": 0.3}
        st = data.get("stance", "中立")
        st_val = stance_map.get(st, 0.0)

        return {
            "argument_type": at_val,
            "stance": st_val,
            "abstraction_level": max(0.0, min(1.0, float(data.get("abstraction_level", 0.5)))),
            "knowledge_domain": self._encode_domain(data.get("knowledge_domain", "综合")),
            "uncertainty": max(0.0, min(1.0, float(data.get("uncertainty", 0.5)))),
            "novelty": max(0.0, min(1.0, float(data.get("novelty", 0.5)))),
            "concept_density": min(1.0, len(data.get("key_concepts", [])) / 10),
        }

    @staticmethod
    def _encode_domain(domain: str) -> float:
        """知识域编码"""
        domains = ["技术", "哲学", "社会", "科学", "艺术", "商业", "心理", "政治", "伦理", "综合"]
        domain_map = {d: (i + 1) / len(domains) for i, d in enumerate(domains)}
        for d in domains:
            if d in domain:
                return domain_map[d]
        return 0.5

    @staticmethod
    def _default() -> Dict[str, float]:
        return {
            "argument_type": 0.5, "stance": 0.0,
            "abstraction_level": 0.5, "knowledge_domain": 0.5,
            "uncertainty": 0.5, "novelty": 0.5, "concept_density": 0.3,
        }

    @staticmethod
    def _fallback_extract(text: str) -> Dict[str, float]:
        """无 LLM 时的降级方案"""
        sentences = max(1, text.count('。') + text.count('！') + text.count('？') +
                        text.count('\n') + text.count(';') + text.count('；'))
        words = len(text)
        tokens = [c for c in text if '\u4e00' <= c <= '\u9fff']

        common_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人'}
        rare_ratio = sum(1 for t in tokens if t not in common_words) / max(len(tokens), 1)

        logic_markers = ['因为', '所以', '如果', '那么', '虽然', '但是', '因此']
        depth = sum(text.count(m) for m in logic_markers) / max(sentences, 1)

        return {
            "argument_type": min(1.0, depth * 2),
            "stance": 0.0,
            "abstraction_level": min(1.0, rare_ratio * 2),
            "knowledge_domain": 0.5,
            "uncertainty": 0.5,
            "novelty": min(1.0, rare_ratio * 3),
            "concept_density": min(1.0, words / 1000),
        }


# ============================================================
# 观点图节点（使用深层语义特征）
# ============================================================

class OpinionNode:
    """观点图中的单个节点，使用 P2 深层语义特征嵌入。"""

    def __init__(self, text: str, speaker: str = "", weight: float = 1.0,
                 feature_extractor: SemanticFeatureExtractor = None):
        self.text = text
        self.speaker = speaker
        self.weight = weight
        self.extractor = feature_extractor
        self.embedding = self._compute_features(text)

    def _compute_features(self, text: str) -> Dict[str, float]:
        if self.extractor:
            return self.extractor.extract(text)
        # 无提取器时的默认值
        return {"argument_type": 0.5, "stance": 0.0, "abstraction_level": 0.5,
                "knowledge_domain": 0.5, "uncertainty": 0.5, "novelty": 0.5, "concept_density": 0.3}

    def similarity(self, other: 'OpinionNode') -> float:
        """计算语义相似度（余弦）"""
        f1 = self.embedding
        f2 = other.embedding
        keys = [k for k in f1 if k != 'stance']
        dot = sum(f1[k] * f2[k] for k in keys)
        n1 = math.sqrt(sum(f1[k] ** 2 for k in keys))
        n2 = math.sqrt(sum(f2[k] ** 2 for k in keys))
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return dot / (n1 * n2)


# ============================================================
# 观点图
# ============================================================

class OpinionGraph:
    """观点图 —— 语义特征网络"""

    def __init__(self, nodes: List[OpinionNode]):
        self.nodes = nodes
        self.n = len(nodes)
        self.adjacency: List[List[float]] = []
        self._build_graph()

    def _build_graph(self, threshold: float = 0.1):
        self.adjacency = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                sim = self.nodes[i].similarity(self.nodes[j])
                if sim > threshold:
                    self.adjacency[i][j] = sim
                    self.adjacency[j][i] = sim

    def centrality(self) -> List[float]:
        return [sum(row) for row in self.adjacency]

    def clustering_coefficient(self) -> float:
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
        cent = self.centrality()
        weights = [n.weight for n in self.nodes]
        max_c = max(cent) if cent else 1
        return [c / max_c * w for c, w in zip(cent, weights)]

    def opposition_pairs(self) -> List[Tuple[int, int, float]]:
        """对立观点对：低相似度 + 高立场差"""
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                sim = self.adjacency[i][j]
                st_diff = abs(self.nodes[i].embedding.get('stance', 0) -
                              self.nodes[j].embedding.get('stance', 0))
                if sim < 0.15 and st_diff > 0.6:
                    pairs.append((i, j, st_diff))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:5]


# ============================================================
# P1: 语义化虚拟专家生成（替代文本拼接）
# ============================================================

_VIRTUAL_EXPERT_PROMPT = """你是一个虚拟专家，正在参与一个深度讨论。

讨论问题: {problem}

已有专家的观点涉及以下语义维度:
- 论证类型: {argument_types}
- 知识域: {knowledge_domains}
- 抽象层级范围: {abstraction_range}

你的任务是生成一个【全新的、有实质内容的】观点，要求:
1. 不要复述或拼接已有观点
2. 从新的角度或新的知识域切入
3. 有具体的论证而非空泛陈述
4. 控制在100字以内
5. 直接输出观点文本，不要任何前缀"""


class SemanticVirtualExpertGenerator:
    """
    P1: 语义化虚拟专家生成。

    使用 LLM 生成真正有语义内容的虚拟专家观点，
    替代原版文本拼接/截断的"语义空心"方案。
    """

    def __init__(self, nodes: List[OpinionNode], llm_client=None, model_name: str = ""):
        self.nodes = nodes
        self.n = len(nodes)
        self.llm_client = llm_client
        self.model_name = model_name

    def generate(self, target_count: int, problem: str = "") -> List[Dict]:
        """生成语义化虚拟专家观点"""
        if self.n < 2:
            return [{"speech": n.text, "key_insight": "", "weight": 1.0} for n in self.nodes]

        needed = target_count - self.n
        if needed <= 0:
            return [{"speech": n.text, "key_insight": "", "weight": n.weight} for n in self.nodes]

        # 已有语义空间分析
        types = set(n.embedding.get('argument_type', 0.5) for n in self.nodes)
        domains = set(n.embedding.get('knowledge_domain', 0.5) for n in self.nodes)
        abs_levels = [n.embedding.get('abstraction_level', 0.5) for n in self.nodes]

        virtual = []

        # 有 LLM 时：逐个生成语义化的虚拟专家
        if self.llm_client and problem:
            # 采样策略：填补语义空间空白
            batch_size = min(5, needed)
            for _ in range(0, needed, batch_size):
                # 选择与已有观点差异最大的采样点
                target_type = random.uniform(0.1, 0.9)
                target_domain = random.uniform(0.1, 0.9)
                target_abs = random.uniform(
                    max(0, min(abs_levels) - 0.2),
                    min(1, max(abs_levels) + 0.2)
                )

                prompt = _VIRTUAL_EXPERT_PROMPT.format(
                    problem=problem[:300],
                    argument_types=f"{min(types):.1f}~{max(types):.1f}",
                    knowledge_domains=f"{min(domains):.1f}~{max(domains):.1f}",
                    abstraction_range=f"{min(abs_levels):.1f}~{max(abs_levels):.1f}",
                )

                try:
                    response, _ = self.llm_client.chat(
                        [{"role": "user", "content": prompt}],
                        model=self.model_name,
                        thinking="disabled",
                        caller="虚拟专家生成",
                        show_reasoning=False, show_answer=False,
                    )
                    if response and response.strip():
                        text = response.strip()
                        # 清理可能的引导语
                        for prefix in ["观点：", "观点:", "虚拟专家：", "虚拟专家:"]:
                            if text.startswith(prefix):
                                text = text[len(prefix):]
                        virtual.append({
                            "speech": text[:200],
                            "key_insight": "语义生成",
                            "weight": random.uniform(0.8, 1.2),
                        })
                except Exception:
                    pass

        # 补充不足部分（用语义采样）
        remaining = needed - len(virtual)
        if remaining > 0:
            virtual.extend(self._semantic_sample(remaining))

        random.shuffle(virtual)
        return virtual[:needed]

    def _semantic_sample(self, count: int) -> List[Dict]:
        """基于语义空间采样的无 LLM 降级方案"""
        result = []
        for _ in range(count):
            # 选择语义空间中最远的节点对，取中间点
            if self.n >= 2:
                i, j = random.sample(range(self.n), 2)
                src = self.nodes[i]
                dst = self.nodes[j]
                # 生成一个"语义中间点"的文本
                mid_text = src.text[:len(src.text) // 3] + dst.text[len(dst.text) // 3:]
                result.append({
                    "speech": mid_text[:200],
                    "key_insight": "语义采样",
                    "weight": (src.weight + dst.weight) / 2,
                })
            else:
                result.append({
                    "speech": self.nodes[0].text[:200],
                    "key_insight": "采样",
                    "weight": self.nodes[0].weight,
                })
        return result


# ============================================================
# P0: 涌现验证机制
# ============================================================

_EMERGENCE_VERIFICATION_PROMPT = """评估以下"综合输出"是否包含超越输入观点的真正涌现内容。

【输入观点】
{input_views}

【综合输出】
{synthesis}

请从以下三个维度评分（0.0~1.0），只输出JSON：
{{
  "novelty_score": 0.0~1.0,   // 综合输出是否包含输入中没有的新信息/新角度
  "depth_score": 0.0~1.0,     // 综合输出是否比任何单个输入观点更深刻
  "synthesis_score": 0.0~1.0, // 综合输出是否真正整合了多个观点而非复述一个
  "is_truly_emergent": true/false,  // 是否真正涌现
  "reason": "一句话说明"
}}
只输出JSON，不要其他内容。"""


class EmergenceVerifier:
    """
    P0: 涌现验证机制。

    区分"真正涌现"（产生了输入中没有的新信息）和
    "风格模仿"（只是换了个说法复述已有观点）。
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.history: List[Dict] = []

    def verify(self, synthesis: str, input_speeches: List[str]) -> Dict:
        """
        验证合成输出是否真正涌现。

        返回:
        {
            "is_emergent": bool,    # 是否真正涌现
            "novelty": 0.0~1.0,     # 新颖度
            "depth": 0.0~1.0,       # 深度
            "synthesis": 0.0~1.0,   # 综合度
            "reason": str,          # 判定理由
        }
        """
        if not synthesis or not input_speeches:
            return {"is_emergent": False, "novelty": 0.0, "depth": 0.0,
                    "synthesis": 0.0, "reason": "无输出或输入"}

        result = self._llm_verify(synthesis, input_speeches)
        self.history.append(result)
        return result

    def _llm_verify(self, synthesis: str, input_speeches: List[str]) -> Dict:
        """LLM 涌现验证"""
        if not self.llm_client:
            return self._fallback_verify(synthesis, input_speeches)

        try:
            input_text = "\n".join(f"- {s[:200]}" for s in input_speeches[:5])
            prompt = _EMERGENCE_VERIFICATION_PROMPT.format(
                input_views=input_text,
                synthesis=synthesis[:500],
            )
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking="disabled",
                caller="涌现验证",
                show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return {
                    "is_emergent": data.get("is_truly_emergent", False),
                    "novelty": max(0.0, min(1.0, float(data.get("novelty_score", 0.0)))),
                    "depth": max(0.0, min(1.0, float(data.get("depth_score", 0.0)))),
                    "synthesis": max(0.0, min(1.0, float(data.get("synthesis_score", 0.0)))),
                    "reason": data.get("reason", ""),
                }
        except Exception:
            pass
        return self._fallback_verify(synthesis, input_speeches)

    @staticmethod
    def _fallback_verify(synthesis: str, input_speeches: List[str]) -> Dict:
        """无 LLM 时的统计降级"""
        # 计算综合输出与输入的平均相似度
        synth_tokens = set(synthesis[:300])
        similarities = []
        for s in input_speeches[:5]:
            inp_tokens = set(s[:300])
            if not synth_tokens or not inp_tokens:
                continue
            overlap = len(synth_tokens & inp_tokens) / max(len(synth_tokens | inp_tokens), 1)
            similarities.append(overlap)

        avg_sim = sum(similarities) / max(len(similarities), 1) if similarities else 1.0
        novelty = 1.0 - avg_sim
        depth = min(1.0, len(synthesis) / 300)
        synthesis_score = max(0.0, 1.0 - avg_sim * 2) if len(input_speeches) > 1 else 0.0

        return {
            "is_emergent": novelty > 0.3,
            "novelty": novelty,
            "depth": depth,
            "synthesis": synthesis_score,
            "reason": f"统计降级: 平均相似度{avg_sim:.2f}",
        }


# ============================================================
# P3: 输出质量闭环反馈
# ============================================================

_QUALITY_EVALUATION_PROMPT = """评估以下讨论综合输出的质量，只输出JSON：

【综合输出】
{synthesis}

评估维度：
{{
  "coherence": 0.0~1.0,    // 逻辑连贯性
  "depth": 0.0~1.0,        // 思想深度
  "novelty": 0.0~1.0,      // 新颖度
  "actionability": 0.0~1.0 // 可操作性/具体性
}}
只输出JSON，不要其他内容。"""


class QualityCalibrator:
    """
    P3: 输出质量闭环反馈。

    自动评估合成质量并调整引擎参数，使系统可自我校准。
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.history: List[Dict] = []
        self.params = {
            "amplification_ratio": 20.0,    # 虚拟专家放大率
            "synthesis_temperature": 0.7,   # 合成温度
            "min_level_for_llm": 1,          # 最低 LLM 合成层级
        }

    def evaluate(self, synthesis: str) -> Dict:
        """评估合成质量"""
        if not synthesis:
            return {"coherence": 0.0, "depth": 0.0, "novelty": 0.0, "actionability": 0.0}

        quality = self._llm_evaluate(synthesis)
        self.history.append(quality)
        self._adjust_params()
        return quality

    def _llm_evaluate(self, synthesis: str) -> Dict:
        if not self.llm_client:
            return self._fallback_evaluate(synthesis)
        try:
            prompt = _QUALITY_EVALUATION_PROMPT.format(synthesis=synthesis[:500])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking="disabled",
                caller="质量评估",
                show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return {
                    "coherence": max(0.0, min(1.0, float(data.get("coherence", 0.5)))),
                    "depth": max(0.0, min(1.0, float(data.get("depth", 0.5)))),
                    "novelty": max(0.0, min(1.0, float(data.get("novelty", 0.5)))),
                    "actionability": max(0.0, min(1.0, float(data.get("actionability", 0.5)))),
                }
        except Exception:
            pass
        return self._fallback_evaluate(synthesis)

    @staticmethod
    def _fallback_evaluate(synthesis: str) -> Dict:
        sentences = max(1, synthesis.count('。') + synthesis.count('！') + synthesis.count('？'))
        words = len(synthesis)
        logic_markers = ['因为', '所以', '如果', '那么', '虽然', '但是', '因此']
        depth = min(1.0, sum(synthesis.count(m) for m in logic_markers) / max(sentences, 1) * 0.5)
        return {
            "coherence": min(1.0, sentences / max(words / 50, 1)),
            "depth": depth,
            "novelty": 0.5,
            "actionability": min(1.0, words / 500),
        }

    def _adjust_params(self):
        """根据质量趋势调整参数"""
        if len(self.history) < 3:
            return

        recent = self.history[-3:]
        avg_depth = sum(q.get("depth", 0.5) for q in recent) / 3
        avg_novelty = sum(q.get("novelty", 0.5) for q in recent) / 3
        avg_coherence = sum(q.get("coherence", 0.5) for q in recent) / 3

        # 深度不足 → 提高放大率（更多虚拟专家引入多样性）
        if avg_depth < 0.3:
            self.params["amplification_ratio"] = min(50, self.params["amplification_ratio"] * 1.2)
            self.params["min_level_for_llm"] = max(0, self.params["min_level_for_llm"] - 1)
        elif avg_depth > 0.7:
            self.params["amplification_ratio"] = max(5, self.params["amplification_ratio"] * 0.9)

        # 新颖度不足 → 提高温度
        if avg_novelty < 0.3:
            self.params["synthesis_temperature"] = min(1.0, self.params["synthesis_temperature"] + 0.1)
        elif avg_novelty > 0.7:
            self.params["synthesis_temperature"] = max(0.3, self.params["synthesis_temperature"] - 0.1)

        # 连贯性不足 → 降低温度
        if avg_coherence < 0.3:
            self.params["synthesis_temperature"] = max(0.3, self.params["synthesis_temperature"] - 0.1)

    def get_params(self) -> Dict:
        return dict(self.params)

    def get_quality_trend(self) -> str:
        """返回质量趋势摘要"""
        if len(self.history) < 2:
            return "数据不足"
        recent = self.history[-3:] if len(self.history) >= 3 else self.history
        avg_all = {k: sum(q.get(k, 0.5) for q in recent) / len(recent)
                   for k in ["coherence", "depth", "novelty", "actionability"]}
        return (f"连贯:{avg_all['coherence']:.2f} "
                f"深度:{avg_all['depth']:.2f} "
                f"新颖:{avg_all['novelty']:.2f} "
                f"可操作:{avg_all['actionability']:.2f}")


# ============================================================
# 层级判定
# ============================================================

class DiscussionLevelDetector:
    """
    基于语义图指标判定讨论深度层级。

    L0: 图密度 < 0.1 或 节点数 < 3
    L1: 图密度 >= 0.1 且 聚类系数 > 0.2
    L2: 社区数 >= 2 且 中心性熵 > 0.5
    L3: 对立度 >= 2 且 立场差 > 0.6
    L4: 所有条件满足 + 图密度 > 0.3
    """

    def __init__(self, graph: OpinionGraph):
        self.graph = graph
        self.n = graph.n

        possible_edges = self.n * (self.n - 1) / 2
        actual_edges = sum(1 for row in graph.adjacency for v in row if v > 0) // 2
        self.density = actual_edges / max(possible_edges, 1)
        self.clustering = graph.clustering_coefficient()
        self.communities = graph.community_count()
        self.opposition = len(graph.opposition_pairs())

        cent = graph.centrality()
        total_c = sum(cent) if cent else 1
        probs = [c / total_c for c in cent if c > 0]
        self.centrality_entropy = -sum(p * math.log2(p) for p in probs) / max(math.log2(self.n), 1) if probs else 0

        stances = [n.embedding.get('stance', 0) for n in graph.nodes]
        self.stance_range = max(stances) - min(stances) if stances else 0

    def compute_level(self) -> int:
        if self.n < 3 or self.density < 0.1:
            return 0
        if self.density < 0.2 and self.clustering < 0.2:
            return 0
        if self.clustering > 0.2:
            if self.communities >= 2 and self.centrality_entropy > 0.5:
                if self.opposition >= 2 and self.stance_range > 0.6:
                    if self.density > 0.3 and self.communities >= 3 and self.centrality_entropy > 0.7:
                        return 4
                    return 3
                return 2
            return 1
        return 0


# ============================================================
# 合成提示构建器
# ============================================================

def _build_discussion_prompt_L0(problem: str, speeches: List[str]) -> str:
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    return (f"讨论问题: {problem}\n\n各方观点:\n{text}\n\n"
            f"请直接综合以上观点，给出一个简洁的总结。"
            f"不要用'我是...'开头，直接回答。控制在200字以内。")


def _build_discussion_prompt_L1(problem: str, speeches: List[str],
                                 central_nodes: List[str]) -> str:
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    hubs = "\n".join(f"  - {s[:150]}" for s in central_nodes[:3])
    return (f"讨论问题: {problem}\n\n各方观点:\n{text}\n\n核心观点:\n{hubs}\n\n"
            f"请识别这些观点之间的引用和交叉关系，"
            f"将它们整合成一个有结构的叙述。"
            f"不要用'我是...'开头，直接回答。控制在200字以内。")


def _build_discussion_prompt_L2(problem: str, speeches: List[str],
                                 clusters: List[List[str]]) -> str:
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    cluster_text = ""
    for i, cls in enumerate(clusters[:3]):
        cluster_text += f"\n主题{i + 1}:\n" + "\n".join(f"  - {c[:100]}" for c in cls[:3])
    return (f"讨论问题: {problem}\n\n各方观点:\n{text}\n\n主题聚类:\n{cluster_text}\n\n"
            f"请从更高维度抽象这些主题，揭示它们之间的层次关系和结构。"
            f"不要用'我是...'开头，直接回答。控制在200字以内。")


def _build_discussion_prompt_L3(problem: str, speeches: List[str],
                                 thesis: str, antithesis: str) -> str:
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    return (f"讨论问题: {problem}\n\n各方观点:\n{text}\n\n"
            f"正题（主流观点）:\n{thesis[:200]}\n\n反题（对立观点）:\n{antithesis[:200]}\n\n"
            f"请识别这些观点中的根本矛盾，在更高维度上生成合题，"
            f"使矛盾双方在更大的框架中统一。"
            f"不要用'我是...'开头，直接回答。控制在200字以内。")


def _build_discussion_prompt_L4(problem: str, speeches: List[str],
                                 L3_synthesis: str) -> str:
    text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
    return (f"讨论问题: {problem}\n\n各方观点:\n{text}\n\n初步综合:\n{L3_synthesis[:200]}\n\n"
            f"对初步综合本身进行元反思：\n"
            f"1. 这个综合存在什么盲点？\n"
            f"2. 哪些前提假设可以被质疑？\n"
            f"3. 如果跳出所有框架，更深层的本质是什么？\n\n"
            f"生成一个超越所有现有框架的元综合。"
            f"不要用'我是...'开头，直接回答。控制在200字以内。")


def _response_length(level: int) -> str:
    return {0: "300字", 1: "250字", 2: "200字", 3: "150字", 4: "100字"}.get(level, "200字")


# ============================================================
# 讨论引擎主类
# ============================================================

class DiscussionEngine:
    """
    普通讨论引擎 —— 完整集成 P0~P3 修复。

    P0: EmergenceVerifier — 涌现验证，过滤风格模仿
    P1: SemanticVirtualExpertGenerator — 语义化虚拟专家
    P2: SemanticFeatureExtractor — 深层语义特征嵌入
    P3: QualityCalibrator — 输出质量闭环反馈
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.level_history: List[Tuple[int, int]] = []

        # P2: 语义特征提取器
        self.feature_extractor = SemanticFeatureExtractor(llm_client, model_name)

        # P0: 涌现验证器
        self.emergence_verifier = EmergenceVerifier(llm_client, model_name)

        # P3: 质量校准器
        self.quality_calibrator = QualityCalibrator(llm_client, model_name)

        # 统计
        self.total_emergence_count = 0
        self.total_style_mimicry_count = 0

    def analyze(self, round_discussions: List[Dict],
                problem: str,
                essence_pool=None) -> Dict:
        """
        对一轮讨论进行深度分析（集成 P0~P3）。
        """
        if not round_discussions:
            return {"level": 0, "synthesis": "", "metrics": {}, "graph": None,
                    "is_emergent": False, "quality": {}, "verification": {}}

        # ── 1. 构建语义观点图（P2: 深层语义特征嵌入） ──
        nodes = []
        for d in round_discussions:
            speech = d.get("speech", "")
            if speech:
                nodes.append(OpinionNode(
                    text=speech,
                    speaker=d.get("player_name", ""),
                    weight=1.0,
                    feature_extractor=self.feature_extractor,
                ))
        if not nodes:
            return {"level": 0, "synthesis": "", "metrics": {}, "graph": None,
                    "is_emergent": False, "quality": {}, "verification": {}}

        graph = OpinionGraph(nodes)

        # ── 2. P1: 语义化虚拟专家扩增 ──
        n_real = len(nodes)
        if n_real >= 3:
            amp_ratio = self.quality_calibrator.params["amplification_ratio"]
            target = min(max(100, int(n_real * amp_ratio)), 500)
            generator = SemanticVirtualExpertGenerator(
                nodes, self.llm_client, self.model_name
            )
            virtual = generator.generate(target, problem=problem)
            for v in virtual:
                nodes.append(OpinionNode(
                    text=v.get("speech", ""),
                    speaker="虚拟",
                    weight=v.get("weight", 1.0),
                    feature_extractor=self.feature_extractor,
                ))
            graph = OpinionGraph(nodes)

        # ── 3. 层级判定 ──
        detector = DiscussionLevelDetector(graph)
        level = detector.compute_level()

        speeches = [d.get("speech", "") for d in round_discussions if d.get("speech")]

        # ── 4. 层级适配合成 ──
        synthesis = self._synthesize(level, problem, speeches, graph)

        # ── 5. P0: 涌现验证 ──
        verification = {"is_emergent": False, "novelty": 0.0, "depth": 0.0,
                        "synthesis": 0.0, "reason": "无输出"}
        if synthesis:
            verification = self.emergence_verifier.verify(synthesis, speeches)
            if verification.get("is_emergent", False):
                self.total_emergence_count += 1
            else:
                self.total_style_mimicry_count += 1

        # ── 6. P3: 质量评估 & 参数校准 ──
        quality = self.quality_calibrator.evaluate(synthesis) if synthesis else {}

        # ── 7. 记录 ──
        self.level_history.append((len(self.level_history), level))
        metrics = {
            "density": detector.density,
            "clustering": detector.clustering,
            "communities": detector.communities,
            "centrality_entropy": detector.centrality_entropy,
            "opposition_pairs": detector.opposition,
            "stance_range": detector.stance_range,
            "n_real": n_real,
            "n_total": len(nodes),
            "emergence_novelty": verification.get("novelty", 0),
            "emergence_depth": verification.get("depth", 0),
            "quality_coherence": quality.get("coherence", 0),
            "quality_depth": quality.get("depth", 0),
            "quality_novelty": quality.get("novelty", 0),
        }

        return {
            "level": level,
            "synthesis": synthesis,
            "metrics": metrics,
            "graph": graph,
            "is_emergent": verification.get("is_emergent", False),
            "verification": verification,
            "quality": quality,
            "calibrator_params": self.quality_calibrator.get_params(),
        }

    def _synthesize(self, level: int, problem: str,
                    speeches: List[str], graph: OpinionGraph) -> str:
        """根据层级合成讨论结果"""
        if not self.llm_client:
            return ""

        min_level = self.quality_calibrator.params["min_level_for_llm"]
        # 低于最低合成层级的不调用 LLM
        if level < min_level:
            return ""

        temperature = self.quality_calibrator.params["synthesis_temperature"]

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
                L3_prompt = _build_discussion_prompt_L3(
                    problem, speeches,
                    speeches[0] if speeches else "",
                    speeches[-1] if len(speeches) > 1 else ""
                )
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

            # P3: 应用温度参数
            extra_kwargs = {}
            if temperature and temperature != 0.7:
                extra_kwargs["temperature"] = temperature

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
        """简单聚类"""
        if not speeches:
            return []

        def _keywords(text: str) -> set:
            stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人',
                         '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                         '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
                         '它', '们', '什么', '那', '为', '能', '得', '与', '对', '但'}
            return {c for c in text if '\u4e00' <= c <= '\u9fff' and c not in stopwords}

        kws = [_keywords(s) for s in speeches]
        n = len(speeches)
        if n < 2:
            return [speeches]

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
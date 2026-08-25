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

扩展模块：
  - 论证挖掘系统 (ArgumentationMining)
  - 叙事分析系统 (NarrativeAnalysis)
  - 共识测量系统 (ConsensusMeasurement)
  - 认知偏差检测 (BiasDetector)
  - 知识图谱提取 (DiscussionKnowledgeGraph)
  - 讨论质量评估 (QualityAssessor)
  - 参与者角色检测 (RoleDetector)
  - 时序模式分析 (TemporalPatternAnalyzer)
  - 多视角分析 (MultiPerspectiveAnalyzer)
  - 话语结构分析 (DiscourseAnalyzer)
"""

import math
import random
import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Callable, Any, Set
from dataclasses import dataclass, field

# ============================================================
# P2: 深层语义特征嵌入（替代浅层文本统计）
# ============================================================

_SEMANTIC_PROMPT = """分析以下文本的深层语义特征，仅输出JSON：

文本: {text}

输出JSON格式：
{{
  "argument_type": "实证/规范/分析/类比/批判/综合",
  "stance": "支持/反对/中立/探索",
  "abstraction_level": 0.0~1.0,
  "knowledge_domain": "技术/哲学/社会/科学/艺术/商业/心理/政治/伦理/综合",
  "uncertainty": 0.0~1.0,
  "novelty": 0.0~1.0,
  "key_concepts": ["概念1", "概念2"]
}}
只输出JSON，不要其他内容。"""


class SemanticFeatureExtractor:
    """P2: 深层语义特征嵌入"""

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self._cache: Dict[str, Dict] = {}

    def extract(self, text: str) -> Dict[str, float]:
        if not text:
            return self._default()
        cache_key = text[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.llm_client:
            result = self._llm_extract(text)
        else:
            result = self._fallback_extract(text)
        self._cache[cache_key] = result
        return result

    def _llm_extract(self, text: str) -> Dict[str, float]:
        try:
            prompt = _SEMANTIC_PROMPT.format(text=text[:800])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="语义特征提取", show_reasoning=False, show_answer=False,
            )
            if response:
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
        at_map = {"实证": 0.9, "规范": 0.8, "分析": 0.7, "类比": 0.6,
                  "批判": 0.5, "综合": 0.4, "探索": 0.3}
        at = data.get("argument_type", "分析")
        at_val = max(at_map.get(k, 0.5) for k in at_map if k in at) if isinstance(at, str) else 0.5
        stance_map = {"支持": 1.0, "反对": -1.0, "中立": 0.0, "探索": 0.3}
        st = data.get("stance", "中立")
        st_val = stance_map.get(st, 0.0)
        return {
            "argument_type": at_val, "stance": st_val,
            "abstraction_level": max(0.0, min(1.0, float(data.get("abstraction_level", 0.5)))),
            "knowledge_domain": self._encode_domain(data.get("knowledge_domain", "综合")),
            "uncertainty": max(0.0, min(1.0, float(data.get("uncertainty", 0.5)))),
            "novelty": max(0.0, min(1.0, float(data.get("novelty", 0.5)))),
            "concept_density": min(1.0, len(data.get("key_concepts", [])) / 10),
        }

    @staticmethod
    def _encode_domain(domain: str) -> float:
        domains = ["技术", "哲学", "社会", "科学", "艺术", "商业", "心理", "政治", "伦理", "综合"]
        dm = {d: (i + 1) / len(domains) for i, d in enumerate(domains)}
        for d in domains:
            if d in domain:
                return dm[d]
        return 0.5

    @staticmethod
    def _default() -> Dict[str, float]:
        return {"argument_type": 0.5, "stance": 0.0, "abstraction_level": 0.5,
                "knowledge_domain": 0.5, "uncertainty": 0.5, "novelty": 0.5, "concept_density": 0.3}

    @staticmethod
    def _fallback_extract(text: str) -> Dict[str, float]:
        sentences = max(1, text.count('。') + text.count('！') + text.count('？') +
                        text.count('\n') + text.count(';') + text.count('；'))
        tokens = [c for c in text if '\u4e00' <= c <= '\u9fff']
        common_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人'}
        rare_ratio = sum(1 for t in tokens if t not in common_words) / max(len(tokens), 1)
        logic_markers = ['因为', '所以', '如果', '那么', '虽然', '但是', '因此']
        depth = sum(text.count(m) for m in logic_markers) / max(sentences, 1)
        return {"argument_type": min(1.0, depth * 2), "stance": 0.0,
                "abstraction_level": min(1.0, rare_ratio * 2),
                "knowledge_domain": 0.5, "uncertainty": 0.5,
                "novelty": min(1.0, rare_ratio * 3), "concept_density": min(1.0, len(text) / 1000)}


# ============================================================
# 观点图节点
# ============================================================

class OpinionNode:
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
        return {"argument_type": 0.5, "stance": 0.0, "abstraction_level": 0.5,
                "knowledge_domain": 0.5, "uncertainty": 0.5, "novelty": 0.5, "concept_density": 0.3}

    def similarity(self, other: 'OpinionNode') -> float:
        f1 = self.embedding
        f2 = other.embedding
        keys = [k for k in f1 if k != 'stance']
        dot = sum(f1[k] * f2[k] for k in keys)
        n1 = math.sqrt(sum(f1[k] ** 2 for k in keys))
        n2 = math.sqrt(sum(f2[k] ** 2 for k in keys))
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return dot / (n1 * n2)


class OpinionGraph:
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
# 论证挖掘系统 (Argumentation Mining)
# ============================================================

_ARGUMENTATION_PROMPT = """分析以下文本的论证结构，只输出JSON：

文本: {text}

输出格式：
{{
  "claim": "核心主张",
  "premises": ["前提1", "前提2"],
  "conclusion": "结论",
  "argument_type": "演绎/归纳/类比/因果/权威/经验",
  "strength": 0.0~1.0,
  "fallacies": ["谬误类型"] 或 []
}}
只输出JSON。"""


class Argument:
    """单个论证的结构化表示"""
    def __init__(self, claim: str = "", premises: List[str] = None,
                 conclusion: str = "", arg_type: str = "未知",
                 strength: float = 0.5, fallacies: List[str] = None,
                 source_text: str = "", speaker: str = ""):
        self.claim = claim
        self.premises = premises or []
        self.conclusion = conclusion
        self.arg_type = arg_type
        self.strength = strength
        self.fallacies = fallacies or []
        self.source_text = source_text
        self.speaker = speaker

    def to_dict(self) -> dict:
        return {"claim": self.claim, "premises": self.premises,
                "conclusion": self.conclusion, "type": self.arg_type,
                "strength": self.strength, "fallacies": self.fallacies,
                "speaker": self.speaker}

    def is_valid(self) -> bool:
        return bool(self.claim) and len(self.premises) > 0


class ArgumentationMining:
    """
    论证挖掘系统。

    从讨论文本中提取论证结构：
    - 主张 (Claim) 识别
    - 前提 (Premise) 提取
    - 论证类型分类
    - 谬误检测
    - 论证强度评估
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name

    def analyze(self, text: str, speaker: str = "") -> Argument:
        """分析单段文本的论证结构"""
        if not text.strip():
            return Argument(source_text=text, speaker=speaker)

        if self.llm_client:
            arg = self._llm_analyze(text, speaker)
        else:
            arg = self._rule_analyze(text, speaker)

        arg.source_text = text
        return arg

    def analyze_all(self, speeches: List[Dict]) -> List[Argument]:
        """批量分析多段发言"""
        return [self.analyze(s.get("speech", ""), s.get("player_name", ""))
                for s in speeches if s.get("speech")]

    def _llm_analyze(self, text: str, speaker: str) -> Argument:
        try:
            prompt = _ARGUMENTATION_PROMPT.format(text=text[:600])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="论证挖掘", show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return Argument(
                    claim=data.get("claim", ""),
                    premises=data.get("premises", []),
                    conclusion=data.get("conclusion", ""),
                    arg_type=data.get("argument_type", "未知"),
                    strength=max(0.0, min(1.0, float(data.get("strength", 0.5)))),
                    fallacies=data.get("fallacies", []),
                    source_text=text, speaker=speaker,
                )
        except Exception:
            pass
        return self._rule_analyze(text, speaker)

    @staticmethod
    def _rule_analyze(text: str, speaker: str) -> Argument:
        """基于规则的论证分析（降级方案）"""
        premises = []
        conclusion = ""
        claim = text[:80] if len(text) > 80 else text

        # 找前提标记
        premise_markers = ['因为', '由于', '基于', '根据', '考虑到', 'given that', 'since']
        for marker in premise_markers:
            if marker in text:
                idx = text.index(marker)
                end = text.find('。', idx)
                if end > idx:
                    premises.append(text[idx:end + 1])

        # 找结论标记
        conclusion_markers = ['因此', '所以', '综上所述', 'thus', 'therefore', 'consequently']
        for marker in conclusion_markers:
            if marker in text:
                idx = text.index(marker)
                end = text.find('。', idx)
                if end > idx:
                    conclusion = text[idx:end + 1]
                break

        if not premises and not conclusion:
            # 如果没有明显的论证结构，假设整段是一个主张
            claim = text[:100]

        # 谬误检测（简单规则）
        fallacies = []
        certainty_phrases = ['绝对', '一定', '肯定', '毫无疑问', 'always', 'never', 'obviously']
        for p in certainty_phrases:
            if p in text:
                fallacies.append("过度断言")
                break

        return Argument(claim=claim, premises=premises, conclusion=conclusion,
                        arg_type="规则分析", strength=0.5 if premises else 0.3,
                        fallacies=fallacies, source_text=text, speaker=speaker)

    def build_argument_network(self, arguments: List[Argument]) -> Dict:
        """构建论证网络：支持/反对关系"""
        supports = []
        opposes = []
        for i, a1 in enumerate(arguments):
            for j, a2 in enumerate(arguments):
                if i >= j:
                    continue
                # 检查前提是否包含对方的主张
                for p in a1.premises:
                    if a2.claim and (a2.claim[:20] in p or p[:20] in a2.claim):
                        supports.append((i, j, "支持"))
                for p in a2.premises:
                    if a1.claim and (a1.claim[:20] in p or p[:20] in a1.claim):
                        supports.append((i, j, "支持"))
        return {
            "arguments": [a.to_dict() for a in arguments],
            "supports": supports,
            "total": len(arguments),
            "avg_strength": sum(a.strength for a in arguments) / max(len(arguments), 1),
            "fallacy_count": sum(len(a.fallacies) for a in arguments),
        }


# ============================================================
# 谬误检测系统 (Fallacy Detection)
# ============================================================

_FALLACY_PROMPT = """检测以下文本中的逻辑谬误，只输出JSON：

文本: {text}

输出格式（只输出JSON）：
{{
  "fallacies": [
    {{
      "type": "稻草人/滑坡谬误/虚假两难/诉诸权威/诉诸情感/人身攻击/循环论证/以偏概全/轻率归纳/事后归因/无真值",
      "evidence": "文本中对应的内容",
      "severity": 0.0~1.0
    }}
  ]
}}
如果没有谬误，输出 {{"fallacies": []}}
只输出JSON。"""


class FallacyDetector:
    """
    谬误检测系统。

    识别讨论中的常见逻辑谬误：
    - 稻草人谬误、滑坡谬误、虚假两难、诉诸权威
    - 诉诸情感、人身攻击、循环论证、以偏概全
    - 轻率归纳、事后归因、无真值、过度概括
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name

    def detect(self, text: str) -> List[Dict]:
        """检测单段文本中的谬误"""
        if not text.strip():
            return []

        if self.llm_client:
            return self._llm_detect(text)
        return self._rule_detect(text)

    def detect_all(self, speeches: List[Dict]) -> List[Dict]:
        """批量检测"""
        all_fallacies = []
        for s in speeches:
            text = s.get("speech", "")
            if text:
                fallacies = self.detect(text)
                for f in fallacies:
                    f["speaker"] = s.get("player_name", "")
                    f["text_snippet"] = text[:80]
                all_fallacies.extend(fallacies)
        return all_fallacies

    def _llm_detect(self, text: str) -> List[Dict]:
        try:
            prompt = _FALLACY_PROMPT.format(text=text[:600])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="谬误检测", show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return data.get("fallacies", [])
        except Exception:
            pass
        return self._rule_detect(text)

    @staticmethod
    def _rule_detect(text: str) -> List[Dict]:
        """基于规则的谬误检测"""
        fallacies = []

        # 诉诸情感
        emotion_words = ['可怕', '恐怖', '令人震惊', 'horrible', 'terrible', 'shocking']
        if any(w in text for w in emotion_words):
            fallacies.append({"type": "诉诸情感", "evidence": "使用情绪化语言",
                              "severity": 0.4})

        # 过度断言
        absolute_words = ['绝对', '永远', '从不', '所有', 'always', 'never', 'everyone']
        if any(w in text for w in absolute_words):
            fallacies.append({"type": "过度概括", "evidence": "使用绝对化词语",
                              "severity": 0.5})

        # 人身攻击
        ad_hominem = re.search(r'(你|你[们]?)\s*(就|是|太|真|简直|根本)\s*(不|没|缺乏|错了)', text)
        if ad_hominem:
            fallacies.append({"type": "人身攻击", "evidence": ad_hominem.group(),
                              "severity": 0.6})

        # 循环论证
        if re.search(r'因为.*所以.*因为', text):
            fallacies.append({"type": "循环论证", "evidence": "论证存在循环",
                              "severity": 0.5})

        # 稻草人
        straw_man = re.search(r'你的意思是.*其实|你[们]?认为.*但[实际上]', text)
        if straw_man:
            fallacies.append({"type": "稻草人谬误", "evidence": straw_man.group(),
                              "severity": 0.5})

        return fallacies

    def fallacy_report(self, all_fallacies: List[Dict]) -> Dict:
        """生成谬误检测报告"""
        type_counts = Counter(f["type"] for f in all_fallacies)
        type_severity = defaultdict(list)
        for f in all_fallacies:
            type_severity[f["type"]].append(f.get("severity", 0.5))

        return {
            "total_fallacies": len(all_fallacies),
            "unique_types": len(type_counts),
            "top_fallacies": type_counts.most_common(5),
            "avg_severity_by_type": {
                t: sum(s) / len(s) for t, s in type_severity.items()
            },
            "speaker_ranking": Counter(f.get("speaker", "") for f in all_fallacies).most_common(10),
            "fallacy_density": len(all_fallacies) / max(len(set(f.get("text_snippet", "") for f in all_fallacies)), 1),
        }


# ============================================================
# 叙事分析系统 (Narrative Analysis)
# ============================================================

_NARRATIVE_PROMPT = """分析以下讨论文本中的叙事结构，只输出JSON：

文本: {text}

输出格式：
{{
  "narrative_arc": "上升/下降/冲突/解决/循环/无明确",
  "protagonist": "主角",
  "antagonist": "对立面",
  "turning_points": ["转折点1"],
  "themes": ["主题1", "主题2"],
  "moral_framework": "伦理框架",
  "emotional_arc": "积极/消极/中性/波动"
}}
只输出JSON。"""


@dataclass
class NarrativeArc:
    """叙事弧线"""
    arc_type: str = "无明确"
    protagonist: str = ""
    antagonist: str = ""
    turning_points: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    moral_framework: str = ""
    emotional_arc: str = "中性"
    tension: float = 0.5
    progress: float = 0.5


class NarrativeAnalysis:
    """
    叙事分析系统。

    从讨论中提取叙事结构：
    - 叙事弧线（上升/下降/冲突/解决）
    - 角色识别（主角/对立面）
    - 转折点检测
    - 主题提取
    - 情感弧线追踪
    - 张力评估
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name

    def analyze(self, speeches: List[str], speakers: List[str] = None) -> NarrativeArc:
        """分析一组发言的叙事结构"""
        if not speeches:
            return NarrativeArc()

        combined = "\n".join(speeches[:10])

        if self.llm_client:
            return self._llm_analyze(combined)
        return self._rule_analyze(combined)

    def _llm_analyze(self, text: str) -> NarrativeArc:
        try:
            prompt = _NARRATIVE_PROMPT.format(text=text[:1000])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="叙事分析", show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return NarrativeArc(
                    arc_type=data.get("narrative_arc", "无明确"),
                    protagonist=data.get("protagonist", ""),
                    antagonist=data.get("antagonist", ""),
                    turning_points=data.get("turning_points", []),
                    themes=data.get("themes", []),
                    moral_framework=data.get("moral_framework", ""),
                    emotional_arc=data.get("emotional_arc", "中性"),
                )
        except Exception:
            pass
        return self._rule_analyze(text)

    @staticmethod
    def _rule_analyze(text: str) -> NarrativeArc:
        """基于规则的叙事分析"""
        themes = []
        tension = 0.5

        # 主题检测
        theme_markers = [
            ("伦理", ["应该", "道德", "对错", "责任", "义务"]),
            ("技术", ["技术", "数据", "算法", "系统", "工具"]),
            ("社会", ["社会", "群体", "组织", "集体", "制度"]),
            ("哲学", ["本质", "存在", "意义", "真理", "意识"]),
            ("实践", ["实践", "应用", "实现", "方案", "方法"]),
        ]
        for theme, keywords in theme_markers:
            if any(kw in text for kw in keywords):
                themes.append(theme)
        if not themes:
            themes.append("综合")

        # 张力评估
        conflict_words = ['矛盾', '冲突', '对立', '争议', '分歧', '辩论']
        resolution_words = ['共识', '一致', '同意', '统一', '综合', '融合']
        tension = sum(1 for w in conflict_words if w in text)
        resolution = sum(1 for w in resolution_words if w in text)
        total = tension + resolution
        if total > 0:
            tension = tension / max(total, 1)

        # 情感弧线
        positive_words = ['进步', '希望', '创新', '突破', '有益', '好的']
        negative_words = ['危机', '风险', '问题', '困难', '失败', '坏的']
        pos = sum(1 for w in positive_words if w in text)
        neg = sum(1 for w in negative_words if w in text)
        if pos > neg * 2:
            emotional = "积极"
        elif neg > pos * 2:
            emotional = "消极"
        elif pos > 0 and neg > 0:
            emotional = "波动"
        else:
            emotional = "中性"

        # 叙事弧线
        if tension > 0.6 and resolution > 0.3:
            arc = "冲突→解决"
        elif tension > 0.6:
            arc = "冲突"
        elif resolution > 0.3:
            arc = "解决"
        else:
            arc = "探索"

        # 主角检测
        protagonist = ""
        first_speaker_match = re.search(r'^(我|我们|我认为|我们认为)', text.strip())
        if first_speaker_match:
            protagonist = "发言者"

        return NarrativeArc(arc_type=arc, protagonist=protagonist,
                            themes=themes, emotional_arc=emotional,
                            tension=tension, progress=resolution)

    def track_emotional_arc(self, speech_history: List[List[str]]) -> List[float]:
        """追踪多轮讨论的情感弧线"""
        arc = []
        for round_speeches in speech_history:
            combined = " ".join(round_speeches)
            result = self.analyze([combined])
            sentiment = 0.5  # 默认中性
            if result.emotional_arc == "积极":
                sentiment = 0.8
            elif result.emotional_arc == "消极":
                sentiment = 0.2
            elif result.emotional_arc == "波动":
                sentiment = 0.5
            arc.append(sentiment)
        return arc


# ============================================================
# 话语结构分析 (Discourse Analysis)
# ============================================================

_DISCOURSE_PROMPT = """分析以下文本的话语结构，只输出JSON：

文本: {text}

输出格式：
{{
  "discourse_type": "陈述/提问/反驳/赞同/补充/质疑/总结/假设",
  "response_to": "回应对象（如果有）",
  "rhetorical_strategy": "类比/引用/举例/对比/递进/转折/设问/反问",
  "persuasion_technique": "逻辑说服/情感诉求/信誉诉求/无",
  "formality": 0.0~1.0,
  "assertiveness": 0.0~1.0
}}
只输出JSON。"""


@dataclass
class DiscourseUnit:
    """话语单元"""
    discourse_type: str = "陈述"
    response_to: str = ""
    rhetorical_strategy: str = ""
    persuasion_technique: str = "无"
    formality: float = 0.5
    assertiveness: float = 0.5


class DiscourseAnalyzer:
    """
    话语结构分析系统。

    分析讨论中的话语特征：
    - 话语类型分类（陈述/提问/反驳/赞同等）
    - 修辞策略识别
    - 说服技巧检测
    - 正式度/断言度评估
    - 对话结构（谁在回应谁）
    - 话题转换检测
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name

    def analyze(self, text: str) -> DiscourseUnit:
        if not text.strip():
            return DiscourseUnit()
        if self.llm_client:
            return self._llm_analyze(text)
        return self._rule_analyze(text)

    def _llm_analyze(self, text: str) -> DiscourseUnit:
        try:
            prompt = _DISCOURSE_PROMPT.format(text=text[:600])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="话语分析", show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return DiscourseUnit(
                    discourse_type=data.get("discourse_type", "陈述"),
                    response_to=data.get("response_to", ""),
                    rhetorical_strategy=data.get("rhetorical_strategy", ""),
                    persuasion_technique=data.get("persuasion_technique", "无"),
                    formality=max(0.0, min(1.0, float(data.get("formality", 0.5)))),
                    assertiveness=max(0.0, min(1.0, float(data.get("assertiveness", 0.5)))),
                )
        except Exception:
            pass
        return self._rule_analyze(text)

    @staticmethod
    def _rule_analyze(text: str) -> DiscourseUnit:
        """基于规则的话语分析"""
        # 话语类型
        if text.endswith('?') or text.endswith('？'):
            dtype = "提问"
        elif re.search(r'^我(不)?同意|^我(不)?赞同|^我反对', text.strip()):
            dtype = "反驳"
        elif re.search(r'^我同意|^我赞同|^你说得对|^好|^是的', text.strip()):
            dtype = "赞同"
        elif re.search(r'^补充|^另外|^此外|^还有', text.strip()):
            dtype = "补充"
        elif re.search(r'^如果|^假设|^假如', text.strip()):
            dtype = "假设"
        elif re.search(r'^总之|^综上所述|^概括|^总结', text.strip()):
            dtype = "总结"
        elif re.search(r'^我(认为|觉得|想|以为)', text.strip()):
            dtype = "陈述"
        else:
            dtype = "陈述"

        # 修辞策略
        analogy = re.search(r'就像|如同|好比|类比|类似于', text)
        contrast = re.search(r'但是|然而|不过|相反|另一方面', text)
        example = re.search(r'比如|例如|举例|案例|比方说', text)
        if analogy:
            rhetoric = "类比"
        elif contrast:
            rhetoric = "对比"
        elif example:
            rhetoric = "举例"
        else:
            rhetoric = "陈述"

        # 说服技巧
        if re.search(r'权威|专家|研究|数据|统计|研究表明', text):
            persuasion = "逻辑说服"
        elif re.search(r'我们|大家|所有人|共同|一起', text):
            persuasion = "情感诉求"
        elif re.search(r'我(的|的)经验|据我所知|我亲眼', text):
            persuasion = "信誉诉求"
        else:
            persuasion = "无"

        # 正式度
        formal_markers = ['鉴于', '鉴于', '基于', '综上所述', '由此', '据此']
        informal_markers = ['嘿嘿', '哈哈', '哎呀', '哦', '嗯', '好吧']
        formality = 0.5 + sum(1 for m in formal_markers if m in text) * 0.1
        formality -= sum(1 for m in informal_markers if m in text) * 0.1
        formality = max(0.0, min(1.0, formality))

        # 断言度
        assertive_markers = ['绝对', '一定', '肯定', '必然', '毫无疑问', 'definitely']
        hedging_markers = ['可能', '也许', '大概', '或许', '大概', 'maybe', 'perhaps']
        assertiveness = 0.5 + sum(1 for m in assertive_markers if m in text) * 0.1
        assertiveness -= sum(1 for m in hedging_markers if m in text) * 0.1
        assertiveness = max(0.0, min(1.0, assertiveness))

        return DiscourseUnit(discourse_type=dtype, rhetorical_strategy=rhetoric,
                             persuasion_technique=persuasion,
                             formality=formality, assertiveness=assertiveness)

    def analyze_all(self, speeches: List[Dict]) -> Dict:
        """批量分析并生成话语结构报告"""
        units = []
        for s in speeches:
            text = s.get("speech", "")
            if text:
                unit = self.analyze(text)
                units.append({"speaker": s.get("player_name", ""),
                              "unit": unit, "text": text[:80]})

        type_counts = Counter(u["unit"].discourse_type for u in units)
        strategy_counts = Counter(u["unit"].rhetorical_strategy for u in units)
        avg_formality = sum(u["unit"].formality for u in units) / max(len(units), 1)
        avg_assertiveness = sum(u["unit"].assertiveness for u in units) / max(len(units), 1)

        return {
            "units": units,
            "type_distribution": dict(type_counts.most_common()),
            "strategy_distribution": dict(strategy_counts.most_common()),
            "avg_formality": avg_formality,
            "avg_assertiveness": avg_assertiveness,
            "question_ratio": type_counts.get("提问", 0) / max(len(units), 1),
            "rebuttal_ratio": type_counts.get("反驳", 0) / max(len(units), 1),
            "agreement_ratio": type_counts.get("赞同", 0) / max(len(units), 1),
        }


# ============================================================
# 共识测量系统 (Consensus Measurement)
# ============================================================

class ConsensusMeasurement:
    """
    共识测量系统。

    量化讨论中的共识程度：
    - 立场分布分析
    - 共识强度（0~1）
    - 分歧度测量
    - 收敛速度检测
    - 少数派观点识别
    - 德尔菲收敛指标
    - 意见领导力分析
    """

    def __init__(self):
        self.history: List[Dict] = []

    def measure(self, nodes: List[OpinionNode]) -> Dict:
        """测量当前轮次的共识状态"""
        if not nodes:
            return {"consensus_level": 0, "polarization": 0, "factions": 0}

        # 立场分布
        stances = [n.embedding.get('stance', 0) for n in nodes]
        if not stances:
            return {"consensus_level": 0, "polarization": 0, "factions": 0}

        # 共识强度：立场标准差（越小越共识）
        mean_stance = sum(stances) / len(stances)
        variance = sum((s - mean_stance) ** 2 for s in stances) / max(len(stances), 1)
        std_dev = math.sqrt(variance)
        consensus_level = max(0.0, min(1.0, 1.0 - std_dev))

        # 极化度：极端立场比例
        extreme = sum(1 for s in stances if abs(s) > 0.6)
        polarization = extreme / max(len(stances), 1)

        # 派系数量
        factions = self._detect_factions(nodes)

        # 共识中心
        center = sum(s * n.weight for s, n in zip(stances, nodes)) / max(sum(n.weight for n in nodes), 1)

        # 一致性
        agreement_ratio = sum(1 for s in stances if abs(s - mean_stance) < 0.3) / max(len(stances), 1)

        result = {
            "consensus_level": consensus_level,
            "polarization": polarization,
            "factions": factions,
            "mean_stance": mean_stance,
            "center_stance": center,
            "agreement_ratio": agreement_ratio,
            "std_dev": std_dev,
            "n_nodes": len(nodes),
        }
        self.history.append(result)
        return result

    @staticmethod
    def _detect_factions(nodes: List[OpinionNode]) -> int:
        """检测派系数量"""
        stances = [n.embedding.get('stance', 0) for n in nodes]
        if not stances:
            return 0
        # 基于立场聚类的简单派系检测
        support = sum(1 for s in stances if s > 0.3)
        oppose = sum(1 for s in stances if s < -0.3)
        neutral = sum(1 for s in stances if -0.3 <= s <= 0.3)
        factions = 0
        if support > 0:
            factions += 1
        if oppose > 0:
            factions += 1
        if neutral > 0:
            factions += 1
        return factions

    def measure_convergence(self) -> Dict:
        """测量多轮共识收敛趋势"""
        if len(self.history) < 2:
            return {"is_converging": False, "convergence_rate": 0, "rounds_to_converge": -1}

        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        levels = [h["consensus_level"] for h in recent]
        stdevs = [h["std_dev"] for h in recent]

        # 收敛速度：标准差的下降率
        if len(stdevs) >= 2 and stdevs[0] > 0:
            convergence_rate = (stdevs[0] - stdevs[-1]) / stdevs[0]
        else:
            convergence_rate = 0

        # 共识水平趋势
        if len(levels) >= 2:
            level_trend = (levels[-1] - levels[0]) / max(len(levels), 1)
        else:
            level_trend = 0

        # 预测收敛轮数
        rounds_to_converge = -1
        if convergence_rate > 0.01 and stdevs[-1] > 0.01:
            remaining_rounds = math.log(0.01 / stdevs[-1]) / math.log(1 - convergence_rate)
            rounds_to_converge = max(1, min(100, int(remaining_rounds)))

        return {
            "is_converging": convergence_rate > 0.05,
            "convergence_rate": convergence_rate,
            "level_trend": level_trend,
            "rounds_to_converge": rounds_to_converge,
            "current_level": levels[-1] if levels else 0,
            "stability": 1.0 - min(1.0, max(stdevs) - min(stdevs)) if len(stdevs) > 1 else 0.5,
        }

    def detect_minority_views(self, nodes: List[OpinionNode], threshold: float = 0.2) -> List[Dict]:
        """检测少数派观点"""
        stances = [n.embedding.get('stance', 0) for n in nodes]
        if not stances:
            return []

        # 统计多数派立场
        mean = sum(stances) / len(stances)
        minority = []
        for i, n in enumerate(nodes):
            if abs(stances[i] - mean) > threshold * 2:
                minority.append({
                    "text": n.text[:100],
                    "speaker": n.speaker,
                    "stance": stances[i],
                    "deviation": abs(stances[i] - mean),
                })
        minority.sort(key=lambda x: x["deviation"], reverse=True)
        return minority[:5]


# ============================================================
# 认知偏差检测 (Cognitive Bias Detection)
# ============================================================

_BIAS_PROMPT = """检测以下文本中的认知偏差，只输出JSON：

文本: {text}

输出格式：
{{
  "biases": [
    {{
      "type": "确认偏差/锚定效应/可得性启发/群体思维/框架效应/事后聪明/过度自信/基本归因/沉没成本/从众效应",
      "evidence": "文本中对应的内容",
      "severity": 0.0~1.0
    }}
  ]
}}
只输出JSON。"""


class BiasDetector:
    """
    认知偏差检测系统。

    识别讨论中的认知偏差：
    - 确认偏差 (Confirmation Bias)
    - 锚定效应 (Anchoring)
    - 可得性启发 (Availability Heuristic)
    - 群体思维 (Groupthink)
    - 框架效应 (Framing Effect)
    - 过度自信 (Overconfidence)
    - 基本归因错误 (Fundamental Attribution Error)
    - 从众效应 (Bandwagon Effect)
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name

    def detect(self, text: str) -> List[Dict]:
        """检测单段文本中的认知偏差"""
        if not text.strip():
            return []
        if self.llm_client:
            return self._llm_detect(text)
        return self._rule_detect(text)

    def detect_all(self, speeches: List[Dict]) -> Dict:
        """批量检测并生成报告"""
        all_biases = []
        for s in speeches:
            text = s.get("speech", "")
            if text:
                biases = self.detect(text)
                for b in biases:
                    b["speaker"] = s.get("player_name", "")
                all_biases.extend(biases)

        type_counts = Counter(b["type"] for b in all_biases)
        return {
            "total_biases": len(all_biases),
            "unique_types": len(type_counts),
            "top_biases": type_counts.most_common(5),
            "all_biases": all_biases,
            "bias_density": len(all_biases) / max(len(speeches), 1),
            "groupthink_risk": type_counts.get("群体思维", 0) / max(len(speeches), 1),
            "overconfidence_risk": type_counts.get("过度自信", 0) / max(len(speeches), 1),
        }

    def _llm_detect(self, text: str) -> List[Dict]:
        try:
            prompt = _BIAS_PROMPT.format(text=text[:600])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="偏差检测", show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return data.get("biases", [])
        except Exception:
            pass
        return self._rule_detect(text)

    @staticmethod
    def _rule_detect(text: str) -> List[Dict]:
        """基于规则的认知偏差检测"""
        biases = []

        # 确认偏差：只寻找支持自己观点的证据
        if re.search(r'正如我(所说|之前|一直|认为)', text):
            biases.append({"type": "确认偏差", "evidence": "强调自身一致观点",
                           "severity": 0.5})

        # 锚定效应：过度依赖第一个信息
        if re.search(r'首先.*[所以|因此|于是]', text) and len(text) > 100:
            biases.append({"type": "锚定效应", "evidence": "以初始信息为基准",
                           "severity": 0.4})

        # 过度自信
        if re.search(r'毫无疑问|绝对正确|肯定如此|毫无疑问|百分之百', text):
            biases.append({"type": "过度自信", "evidence": "使用绝对肯定表述",
                           "severity": 0.6})

        # 从众效应
        if re.search(r'大家(都|一致|普遍)|所有人(都|认为)|主流', text):
            biases.append({"type": "从众效应", "evidence": "引用群体共识",
                           "severity": 0.4})

        # 框架效应：问题表述方式影响判断
        if re.search(r'损失|风险|避免|防止|失败', text) and re.search(r'收益|获得|好处|成功', text):
            biases.append({"type": "框架效应", "evidence": "同时使用损失/收益框架",
                           "severity": 0.3})

        return biases


# ============================================================
# 知识图谱提取 (Discussion Knowledge Graph)
# ============================================================

_KG_PROMPT = """从以下文本中提取知识图谱，只输出JSON：

文本: {text}

输出格式：
{{
  "entities": [
    {{"name": "概念名", "type": "概念/方法/人物/理论/数据/问题", "weight": 0.0~1.0}}
  ],
  "relations": [
    {{"source": "实体1", "target": "实体2", "relation": "包含/支持/反对/因果/类比/前提/总结"}}
  ]
}}
只输出JSON。"""


@dataclass
class KGEntity:
    """知识图谱实体"""
    name: str
    type: str = "概念"
    weight: float = 0.5
    mentions: int = 0


@dataclass
class KGRelation:
    """知识图谱关系"""
    source: str
    target: str
    relation: str = "关联"
    weight: float = 0.5


class DiscussionKnowledgeGraph:
    """
    讨论知识图谱提取系统。

    从讨论中提取结构化知识：
    - 实体识别（概念、方法、理论、数据等）
    - 关系提取（支持、反对、因果、包含等）
    - 概念层级构建
    - 知识网络分析
    - 核心概念识别
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.entities: Dict[str, KGEntity] = {}
        self.relations: List[KGRelation] = []

    def extract(self, text: str) -> Tuple[List[KGEntity], List[KGRelation]]:
        """从文本中提取知识图谱"""
        if not text.strip():
            return [], []

        if self.llm_client:
            return self._llm_extract(text)
        return self._rule_extract(text)

    def extract_all(self, speeches: List[Dict]):
        """批量提取并合并到图谱"""
        for s in speeches:
            text = s.get("speech", "")
            if text:
                entities, relations = self.extract(text)
                for e in entities:
                    if e.name in self.entities:
                        self.entities[e.name].mentions += 1
                        self.entities[e.name].weight = max(self.entities[e.name].weight, e.weight)
                    else:
                        self.entities[e.name] = e
                self.relations.extend(relations)

    def _llm_extract(self, text: str) -> Tuple[List[KGEntity], List[KGRelation]]:
        try:
            prompt = _KG_PROMPT.format(text=text[:600])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="知识图谱", show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                entities = [KGEntity(e["name"], e.get("type", "概念"),
                                     max(0.0, min(1.0, float(e.get("weight", 0.5)))))
                            for e in data.get("entities", [])]
                relations = [KGRelation(r["source"], r["target"], r.get("relation", "关联"))
                             for r in data.get("relations", [])]
                return entities, relations
        except Exception:
            pass
        return self._rule_extract(text)

    @staticmethod
    def _rule_extract(text: str) -> Tuple[List[KGEntity], List[KGRelation]]:
        """基于规则的知识图谱提取"""
        entities = []
        relations = []

        # 提取引号中的概念
        quoted = re.findall(r'["""]([^"""]{2,30})["""]', text)
        for q in quoted:
            entities.append(KGEntity(q, "概念", 0.5))

        # 提取大写/重要术语
        terms = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)*', text)
        seen = set()
        for t in terms:
            if t not in seen and len(t) > 3:
                seen.add(t)
                entities.append(KGEntity(t, "概念", 0.4))

        # 提取"XX是YY"关系
        is_relations = re.findall(r'([^，。，。]{2,20})(?:是|属于|包括|包含)([^，。，。]{2,20})', text)
        for src, dst in is_relations[:3]:
            relations.append(KGRelation(src.strip(), dst.strip(), "包含"))

        # 提取"因为XX所以YY"关系
        causal = re.findall(r'因为([^，。，。]+)(?:所以|因此|于是)([^，。，。]+)', text)
        for src, dst in causal[:3]:
            relations.append(KGRelation(src.strip(), dst.strip(), "因果"))

        return entities, relations

    def get_central_concepts(self, top_n: int = 10) -> List[Dict]:
        """获取核心概念"""
        sorted_ents = sorted(self.entities.values(), key=lambda e: e.weight * e.mentions, reverse=True)
        return [{"name": e.name, "type": e.type, "weight": e.weight, "mentions": e.mentions}
                for e in sorted_ents[:top_n]]

    def get_relation_network(self) -> Dict:
        """获取关系网络摘要"""
        relation_counts = Counter(r.relation for r in self.relations)
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "relation_type_distribution": dict(relation_counts.most_common()),
            "density": len(self.relations) / max(len(self.entities) * (len(self.entities) - 1) / 2, 1),
        }


# ============================================================
# 参与者角色检测 (Role Detection)
# ============================================================

class RoleDetector:
    """
    参与者角色检测系统。

    识别讨论中不同专家的角色：
    - 领导者 (Leader): 引导讨论方向
    - 批评者 (Critic): 质疑和挑战
    - 综合者 (Synthesizer): 整合不同观点
    - 探索者 (Explorer): 提出新方向
    - 调解者 (Mediator): 化解冲突
    - 专家 (Expert): 提供专业知识
    - 观察者 (Observer): 总结和反思
    - 提问者 (Questioner): 提出关键问题
    """

    def __init__(self):
        self.role_profiles: Dict[str, Dict] = {}

    def detect(self, speaker_name: str, discourse_report: Dict) -> List[str]:
        """检测单个发言者的角色"""
        if not discourse_report or "units" not in discourse_report:
            return ["参与者"]

        units = [u for u in discourse_report["units"] if u.get("speaker") == speaker_name]
        if not units:
            return ["参与者"]

        # 统计话语特征
        types = Counter(u["unit"].discourse_type for u in units)
        strategies = Counter(u["unit"].rhetorical_strategy for u in units)
        avg_assertiveness = sum(u["unit"].assertiveness for u in units) / max(len(units), 1)
        total = len(units)

        roles = []
        # 领导者：高断言度 + 多倡议
        if avg_assertiveness > 0.6 and types.get("陈述", 0) > total * 0.4:
            roles.append("领导者")
        # 批评者：多反驳/质疑
        if types.get("反驳", 0) > total * 0.2 or types.get("质疑", 0) > total * 0.2:
            roles.append("批评者")
        # 综合者：多总结 + 多补充
        if types.get("总结", 0) > total * 0.15 or types.get("补充", 0) > total * 0.2:
            roles.append("综合者")
        # 探索者：多假设 + 多提问
        if types.get("假设", 0) > total * 0.15 or types.get("提问", 0) > total * 0.25:
            roles.append("探索者")
        # 提问者：多提问
        if types.get("提问", 0) > total * 0.3:
            roles.append("提问者")
        # 调解者：多赞同 + 低断言度
        if types.get("赞同", 0) > total * 0.2 and avg_assertiveness < 0.5:
            roles.append("调解者")

        if not roles:
            roles.append("参与者")

        # 更新角色画像
        self.role_profiles[speaker_name] = {
            "roles": roles,
            "discourse_signature": dict(types.most_common(5)),
            "assertiveness": avg_assertiveness,
        }

        return roles

    def detect_all(self, speakers: List[str], discourse_report: Dict) -> Dict:
        """批量检测并生成角色地图"""
        result = {}
        for speaker in set(speakers):
            roles = self.detect(speaker, discourse_report)
            if speaker not in result:
                result[speaker] = roles
        return result

    def get_role_map(self) -> Dict:
        """获取角色地图"""
        role_distribution = Counter()
        for profile in self.role_profiles.values():
            for role in profile["roles"]:
                role_distribution[role] += 1
        return {
            "profiles": self.role_profiles,
            "distribution": dict(role_distribution.most_common()),
            "diversity": len(role_distribution),
        }


# ============================================================
# 讨论质量评估 (Discussion Quality Assessment)
# ============================================================

_QUALITY_DISCUSSION_PROMPT = """评估以下讨论片段的整体质量，只输出JSON：

讨论问题: {problem}
讨论内容:
{text}

输出格式：
{{
  "depth": 0.0~1.0,
  "breadth": 0.0~1.0,
  "novelty": 0.0~1.0,
  "relevance": 0.0~1.0,
  "coherence": 0.0~1.0,
  "constructiveness": 0.0~1.0,
  "overall": 0.0~1.0,
  "strengths": ["优点1"],
  "weaknesses": ["不足1"]
}}
只输出JSON。"""


class QualityAssessor:
    """
    讨论质量评估系统。

    多维度评估讨论质量：
    - 深度 (Depth): 分析的深入程度
    - 广度 (Breadth): 视角的多样性
    - 新颖度 (Novelty): 新观点的比例
    - 相关性 (Relevance): 与主题的相关度
    - 连贯性 (Coherence): 逻辑连贯程度
    - 建设性 (Constructiveness): 正向贡献度
    """

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.history: List[Dict] = []

    def assess(self, speeches: List[str], problem: str = "") -> Dict:
        """评估一组发言的质量"""
        if not speeches:
            return {"depth": 0, "breadth": 0, "novelty": 0, "relevance": 0,
                    "coherence": 0, "constructiveness": 0, "overall": 0}

        if self.llm_client and problem:
            quality = self._llm_assess(speeches, problem)
        else:
            quality = self._metric_assess(speeches, problem)

        self.history.append(quality)
        return quality

    def _llm_assess(self, speeches: List[str], problem: str) -> Dict:
        try:
            text = "\n".join(f"- {s[:200]}" for s in speeches[:5])
            prompt = _QUALITY_DISCUSSION_PROMPT.format(problem=problem[:200], text=text)
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="质量评估", show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return {k: max(0.0, min(1.0, float(v))) if isinstance(v, (int, float)) else v
                        for k, v in data.items()}
        except Exception:
            pass
        return self._metric_assess(speeches, problem)

    @staticmethod
    def _metric_assess(speeches: List[str], problem: str = "") -> Dict:
        """基于指标的评估"""
        if not speeches:
            return {"depth": 0, "breadth": 0, "novelty": 0, "relevance": 0,
                    "coherence": 0, "constructiveness": 0, "overall": 0}

        all_text = " ".join(speeches)
        sentences = max(1, all_text.count('。') + all_text.count('！') + all_text.count('？'))

        # 深度：逻辑连接词密度
        logic_markers = ['因为', '所以', '如果', '那么', '虽然', '但是', '因此']
        depth = min(1.0, sum(all_text.count(m) for m in logic_markers) / max(sentences, 1) * 0.5)

        # 广度：不同知识域的关键词
        domains = [['技术', '数据', '算法'], ['社会', '伦理', '文化'],
                   ['经济', '商业', '市场'], ['科学', '研究', '理论'],
                   ['实践', '应用', '案例']]
        domain_hits = sum(1 for d in domains if any(kw in all_text for kw in d))
        breadth = min(1.0, domain_hits / len(domains) * 2)

        # 新颖度：罕见词比例
        common_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人'}
        tokens = [c for c in all_text if '\u4e00' <= c <= '\u9fff']
        rare_ratio = sum(1 for t in tokens if t not in common_words) / max(len(tokens), 1)
        novelty = min(1.0, rare_ratio * 2)

        # 相关性：与问题关键词重叠
        relevance = 0.5
        if problem:
            problem_kw = set(re.findall(r'[\u4e00-\u9fff]{2,4}', problem))
            speech_kw = set(re.findall(r'[\u4e00-\u9fff]{2,4}', all_text))
            overlap = len(problem_kw & speech_kw) / max(len(problem_kw | speech_kw), 1)
            relevance = min(1.0, overlap * 3)

        # 连贯性：相邻句子间的主题延续
        coherence = min(1.0, sentences / max(len(speeches), 1))

        # 建设性：正向词汇比例
        positive = ['同意', '好', '建议', '方案', '解决', '进步', '创新']
        constructive = sum(1 for w in positive if w in all_text) / max(len(speeches), 1)
        constructiveness = min(1.0, constructive * 0.5)

        overall = (depth + breadth + novelty + relevance + coherence + constructiveness) / 6

        return {"depth": depth, "breadth": breadth, "novelty": novelty,
                "relevance": relevance, "coherence": coherence,
                "constructiveness": constructiveness, "overall": overall,
                "strengths": [], "weaknesses": []}

    def get_quality_trend(self) -> Dict:
        """获取质量趋势"""
        if len(self.history) < 2:
            return {"trend": "数据不足", "stable": False}
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        first = recent[0].get("overall", 0.5)
        last = recent[-1].get("overall", 0.5)
        return {
            "trend": "上升" if last > first + 0.1 else "下降" if last < first - 0.1 else "稳定",
            "improvement": last - first,
            "stable": abs(last - first) < 0.1,
            "current": last,
            "peak": max(r.get("overall", 0) for r in recent),
            "trough": min(r.get("overall", 0) for r in recent),
        }


# ============================================================
# 时序模式分析 (Temporal Pattern Analysis)
# ============================================================

class TemporalPatternAnalyzer:
    """
    时序模式分析系统。

    分析讨论随时间演化的模式：
    - 观点演化轨迹
    - 讨论阶段检测
    - 临界时刻识别
    - 话题生命周期
    - 参与度时序
    - 情感波动检测
    - 注意力转移检测
    """

    def __init__(self):
        self.round_history: List[Dict] = []

    def record_round(self, round_data: Dict):
        """记录一轮讨论的状态"""
        self.round_history.append(round_data)

    def analyze_temporal_patterns(self) -> Dict:
        """分析跨轮次的时序模式"""
        if len(self.round_history) < 2:
            return {"phase": "初始", "stability": 1.0}

        # 讨论阶段检测
        phase = self._detect_phase()

        # 观点稳定性
        stabilities = [r.get("stability", 0.5) for r in self.round_history if "stability" in r]
        avg_stability = sum(stabilities) / max(len(stabilities), 1) if stabilities else 0.5

        # 话题漂移
        topics = [set(r.get("topics", [])) for r in self.round_history if "topics" in r]
        topic_drift = 0
        if len(topics) >= 2:
            overlaps = [len(topics[i] & topics[i + 1]) / max(len(topics[i] | topics[i + 1]), 1)
                        for i in range(len(topics) - 1)]
            topic_drift = 1.0 - (sum(overlaps) / max(len(overlaps), 1))

        # 参与度趋势
        participations = [r.get("participation", 0.5) for r in self.round_history if "participation" in r]
        participation_trend = 0
        if len(participations) >= 2:
            half = len(participations) // 2
            first_half = sum(participations[:half]) / max(half, 1)
            second_half = sum(participations[half:]) / max(len(participations) - half, 1)
            participation_trend = second_half - first_half

        # 临界时刻
        critical_moments = self._detect_critical_moments()

        return {
            "phase": phase,
            "stability": avg_stability,
            "topic_drift": topic_drift,
            "participation_trend": participation_trend,
            "critical_moments": critical_moments,
            "rounds_analyzed": len(self.round_history),
        }

    def _detect_phase(self) -> str:
        """检测讨论阶段"""
        if len(self.round_history) < 3:
            return "初始"

        # 分析最近几轮的共识变化
        recent = self.round_history[-3:]
        consensus_levels = [r.get("consensus_level", 0.5) for r in recent if "consensus_level" in r]
        if not consensus_levels:
            return "探索"

        # 共识持续上升 → 收敛期
        if len(consensus_levels) >= 2 and consensus_levels[-1] > consensus_levels[0] + 0.2:
            return "收敛"
        # 共识持续下降 → 分化期
        if len(consensus_levels) >= 2 and consensus_levels[-1] < consensus_levels[0] - 0.2:
            return "分化"
        # 高共识 → 成熟期
        if consensus_levels[-1] > 0.7:
            return "成熟"
        # 低共识 → 探索期
        if consensus_levels[-1] < 0.3:
            return "探索"
        return "深化"

    def _detect_critical_moments(self) -> List[Dict]:
        """检测临界时刻（观点剧烈变化点）"""
        critical = []
        for i in range(1, len(self.round_history)):
            prev = self.round_history[i - 1]
            curr = self.round_history[i]

            prev_consensus = prev.get("consensus_level", 0.5)
            curr_consensus = curr.get("consensus_level", 0.5)
            shift = abs(curr_consensus - prev_consensus)

            if shift > 0.3:
                critical.append({
                    "round": i,
                    "type": "共识转变",
                    "magnitude": shift,
                    "direction": "上升" if curr_consensus > prev_consensus else "下降",
                })
        return critical[:5]

    def get_opinion_trajectory(self, speaker: str) -> List[float]:
        """追踪单个专家的观点演化轨迹"""
        trajectory = []
        for r in self.round_history:
            stances = r.get("speaker_stances", {})
            if speaker in stances:
                trajectory.append(stances[speaker])
        return trajectory


# ============================================================
# 多视角分析 (Multi-Perspective Analysis)
# ============================================================

class MultiPerspectiveAnalyzer:
    """
    多视角分析系统。

    从多个角度系统性分析讨论：
    - 利益相关者视角
    - 正反双方视角
    - 短期/长期视角
    - 理论/实践视角
    - 全局/局部视角
    - 专家/新手视角
    - 优化/创新视角
    """

    PERSPECTIVES = [
        {
            "name": "理想主义",
            "description": "从理想目标出发，关注应该是什么",
            "keywords": ["应该", "理想", "完美", "最好", "终极"],
        },
        {
            "name": "现实主义",
            "description": "从实际出发，关注可以是什么",
            "keywords": ["实际", "可行", "限制", "条件", "现实"],
        },
        {
            "name": "批判性",
            "description": "从批判质疑出发，关注问题在哪里",
            "keywords": ["问题", "缺陷", "风险", "不足", "矛盾"],
        },
        {
            "name": "建设性",
            "description": "从解决方案出发，关注如何改进",
            "keywords": ["方案", "建议", "改进", "优化", "解决"],
        },
        {
            "name": "全局性",
            "description": "从整体系统出发，关注宏观影响",
            "keywords": ["系统", "整体", "全局", "生态", "宏观"],
        },
        {
            "name": "局部性",
            "description": "从具体细节出发，关注局部优化",
            "keywords": ["具体", "细节", "局部", "特定", "精确"],
        },
    ]

    def __init__(self):
        pass

    def analyze(self, text: str) -> Dict:
        """分析文本涉及哪些视角"""
        perspectives = []
        for p in self.PERSPECTIVES:
            score = sum(1 for kw in p["keywords"] if kw in text)
            if score > 0:
                perspectives.append({
                    "name": p["name"],
                    "score": min(1.0, score / 3),
                    "matched_keywords": [kw for kw in p["keywords"] if kw in text],
                })
        perspectives.sort(key=lambda x: x["score"], reverse=True)
        return {
            "perspectives": perspectives,
            "dominant": perspectives[0]["name"] if perspectives else "综合",
            "diversity": len(perspectives),
            "coverage": len(perspectives) / len(self.PERSPECTIVES),
        }

    def analyze_all(self, speeches: List[Dict]) -> Dict:
        """批量分析并生成视角报告"""
        all_perspectives = []
        for s in speeches:
            text = s.get("speech", "")
            if text:
                result = self.analyze(text)
                for p in result["perspectives"]:
                    p["speaker"] = s.get("player_name", "")
                all_perspectives.extend(result["perspectives"])

        perspective_counts = Counter(p["name"] for p in all_perspectives)
        return {
            "perspective_distribution": dict(perspective_counts.most_common()),
            "total_perspectives": len(perspective_counts),
            "missing_perspectives": [p["name"] for p in self.PERSPECTIVES
                                     if p["name"] not in perspective_counts],
            "diversity_score": len(perspective_counts) / len(self.PERSPECTIVES),
        }

    def generate_prompt(self, text: str, problem: str) -> str:
        """基于多视角分析生成综合提示词"""
        result = self.analyze(text)
        dominant = result["dominant"]
        missing = [p["name"] for p in self.PERSPECTIVES if p["name"] not in
                   {pp["name"] for pp in result["perspectives"]}]

        prompt = f"讨论问题: {problem}\n\n已有观点以{dominant}视角为主。"
        if missing:
            prompt += f"\n请尝试从以下被忽略的视角补充分析: {'、'.join(missing[:3])}。"
        return prompt


# ============================================================
# 虚拟专家生成（语义化）
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
    """P1: 语义化虚拟专家生成"""

    def __init__(self, nodes: List[OpinionNode], llm_client=None, model_name: str = ""):
        self.nodes = nodes
        self.n = len(nodes)
        self.llm_client = llm_client
        self.model_name = model_name

    def generate(self, target_count: int, problem: str = "") -> List[Dict]:
        if self.n < 2:
            return [{"speech": n.text, "key_insight": "", "weight": 1.0} for n in self.nodes]
        needed = target_count - self.n
        if needed <= 0:
            return [{"speech": n.text, "key_insight": "", "weight": n.weight} for n in self.nodes]
        types = set(n.embedding.get('argument_type', 0.5) for n in self.nodes)
        domains = set(n.embedding.get('knowledge_domain', 0.5) for n in self.nodes)
        abs_levels = [n.embedding.get('abstraction_level', 0.5) for n in self.nodes]
        virtual = []
        if self.llm_client and problem:
            for _ in range(0, needed, 5):
                prompt = _VIRTUAL_EXPERT_PROMPT.format(
                    problem=problem[:300],
                    argument_types=f"{min(types):.1f}~{max(types):.1f}",
                    knowledge_domains=f"{min(domains):.1f}~{max(domains):.1f}",
                    abstraction_range=f"{min(abs_levels):.1f}~{max(abs_levels):.1f}",
                )
                try:
                    response, _ = self.llm_client.chat(
                        [{"role": "user", "content": prompt}],
                        model=self.model_name, thinking="disabled",
                        caller="虚拟专家生成", show_reasoning=False, show_answer=False,
                    )
                    if response and response.strip():
                        text = response.strip()
                        for prefix in ["观点：", "观点:", "虚拟专家：", "虚拟专家:"]:
                            if text.startswith(prefix):
                                text = text[len(prefix):]
                        virtual.append({"speech": text[:200], "key_insight": "语义生成",
                                        "weight": random.uniform(0.8, 1.2)})
                except Exception:
                    pass
        remaining = needed - len(virtual)
        if remaining > 0:
            virtual.extend(self._semantic_sample(remaining))
        random.shuffle(virtual)
        return virtual[:needed]

    def _semantic_sample(self, count: int) -> List[Dict]:
        result = []
        for _ in range(count):
            if self.n >= 2:
                i, j = random.sample(range(self.n), 2)
                src = self.nodes[i]
                dst = self.nodes[j]
                mid_text = src.text[:len(src.text) // 3] + dst.text[len(dst.text) // 3:]
                result.append({"speech": mid_text[:200], "key_insight": "语义采样",
                               "weight": (src.weight + dst.weight) / 2})
            else:
                result.append({"speech": self.nodes[0].text[:200], "key_insight": "采样",
                               "weight": self.nodes[0].weight})
        return result


# ============================================================
# 涌现验证 (Emergence Verification)
# ============================================================

_EMERGENCE_VERIFICATION_PROMPT = """评估以下"综合输出"是否包含超越输入观点的真正涌现内容。

【输入观点】
{input_views}

【综合输出】
{synthesis}

请从以下三个维度评分（0.0~1.0），只输出JSON：
{{
  "novelty_score": 0.0~1.0,
  "depth_score": 0.0~1.0,
  "synthesis_score": 0.0~1.0,
  "is_truly_emergent": true/false,
  "reason": "一句话说明"
}}
只输出JSON。"""


class EmergenceVerifier:
    """P0: 涌现验证"""

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.history: List[Dict] = []

    def verify(self, synthesis: str, input_speeches: List[str]) -> Dict:
        if not synthesis or not input_speeches:
            return {"is_emergent": False, "novelty": 0.0, "depth": 0.0,
                    "synthesis": 0.0, "reason": "无输出或输入"}
        result = self._llm_verify(synthesis, input_speeches)
        self.history.append(result)
        return result

    def _llm_verify(self, synthesis: str, input_speeches: List[str]) -> Dict:
        if not self.llm_client:
            return self._fallback_verify(synthesis, input_speeches)
        try:
            input_text = "\n".join(f"- {s[:200]}" for s in input_speeches[:5])
            prompt = _EMERGENCE_VERIFICATION_PROMPT.format(input_views=input_text, synthesis=synthesis[:500])
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled",
                caller="涌现验证", show_reasoning=False, show_answer=False,
            )
            if response:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                data = json.loads(json_str.strip())
                return {"is_emergent": data.get("is_truly_emergent", False),
                        "novelty": max(0.0, min(1.0, float(data.get("novelty_score", 0.0)))),
                        "depth": max(0.0, min(1.0, float(data.get("depth_score", 0.0)))),
                        "synthesis": max(0.0, min(1.0, float(data.get("synthesis_score", 0.0)))),
                        "reason": data.get("reason", "")}
        except Exception:
            pass
        return self._fallback_verify(synthesis, input_speeches)

    @staticmethod
    def _fallback_verify(synthesis: str, input_speeches: List[str]) -> Dict:
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
        return {"is_emergent": novelty > 0.3, "novelty": novelty,
                "depth": min(1.0, len(synthesis) / 300),
                "synthesis": max(0.0, 1.0 - avg_sim * 2) if len(input_speeches) > 1 else 0.0,
                "reason": f"统计降级: 平均相似度{avg_sim:.2f}"}


# ============================================================
# 质量校准 (Quality Calibration)
# ============================================================

class QualityCalibrator:
    """P3: 输出质量闭环反馈"""

    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.history: List[Dict] = []
        self.params = {"amplification_ratio": 20.0, "synthesis_temperature": 0.7, "min_level_for_llm": 1}

    def evaluate(self, synthesis: str) -> Dict:
        if not synthesis:
            return {"coherence": 0.0, "depth": 0.0, "novelty": 0.0, "actionability": 0.0}
        quality = self._fallback_evaluate(synthesis)
        self.history.append(quality)
        self._adjust_params()
        return quality

    @staticmethod
    def _fallback_evaluate(synthesis: str) -> Dict:
        sentences = max(1, synthesis.count('。') + synthesis.count('！') + synthesis.count('？'))
        logic_markers = ['因为', '所以', '如果', '那么', '虽然', '但是', '因此']
        depth = min(1.0, sum(synthesis.count(m) for m in logic_markers) / max(sentences, 1) * 0.5)
        return {"coherence": min(1.0, sentences / max(len(synthesis) / 50, 1)),
                "depth": depth, "novelty": 0.5,
                "actionability": min(1.0, len(synthesis) / 500)}

    def _adjust_params(self):
        if len(self.history) < 3:
            return
        recent = self.history[-3:]
        avg_depth = sum(q.get("depth", 0.5) for q in recent) / 3
        avg_novelty = sum(q.get("novelty", 0.5) for q in recent) / 3
        avg_coherence = sum(q.get("coherence", 0.5) for q in recent) / 3
        if avg_depth < 0.3:
            self.params["amplification_ratio"] = min(50, self.params["amplification_ratio"] * 1.2)
            self.params["min_level_for_llm"] = max(0, self.params["min_level_for_llm"] - 1)
        elif avg_depth > 0.7:
            self.params["amplification_ratio"] = max(5, self.params["amplification_ratio"] * 0.9)
        if avg_novelty < 0.3:
            self.params["synthesis_temperature"] = min(1.0, self.params["synthesis_temperature"] + 0.1)
        elif avg_novelty > 0.7:
            self.params["synthesis_temperature"] = max(0.3, self.params["synthesis_temperature"] - 0.1)
        if avg_coherence < 0.3:
            self.params["synthesis_temperature"] = max(0.3, self.params["synthesis_temperature"] - 0.1)

    def get_params(self) -> Dict:
        return dict(self.params)


# ============================================================
# 层级判定
# ============================================================

class DiscussionLevelDetector:
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
            f"请识别这些观点之间的引用和交叉关系，将它们整合成一个有结构的叙述。"
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


def _build_discussion_prompt_L4(problem: str, speeches: List[str], L3_synthesis: str) -> str:
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
    def __init__(self, llm_client=None, model_name: str = ""):
        self.llm_client = llm_client
        self.model_name = model_name
        self.level_history: List[Tuple[int, int]] = []
        self.feature_extractor = SemanticFeatureExtractor(llm_client, model_name)
        self.emergence_verifier = EmergenceVerifier(llm_client, model_name)
        self.quality_calibrator = QualityCalibrator(llm_client, model_name)
        self.argument_miner = ArgumentationMining(llm_client, model_name)
        self.fallacy_detector = FallacyDetector(llm_client, model_name)
        self.narrative_analyzer = NarrativeAnalysis(llm_client, model_name)
        self.discourse_analyzer = DiscourseAnalyzer(llm_client, model_name)
        self.consensus_measure = ConsensusMeasurement()
        self.bias_detector = BiasDetector(llm_client, model_name)
        self.knowledge_graph = DiscussionKnowledgeGraph(llm_client, model_name)
        self.quality_assessor = QualityAssessor(llm_client, model_name)
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.role_detector = RoleDetector()
        self.multi_perspective = MultiPerspectiveAnalyzer()
        self.total_emergence_count = 0
        self.total_style_mimicry_count = 0

    def analyze(self, round_discussions: List[Dict], problem: str, essence_pool=None) -> Dict:
        """对一轮讨论进行深度分析（集成所有子系统）"""
        if not round_discussions:
            return {"level": 0, "synthesis": "", "metrics": {}, "graph": None,
                    "is_emergent": False, "quality": {}, "verification": {},
                    "argumentation": {}, "narrative": {}, "discourse": {},
                    "consensus": {}, "biases": {}, "knowledge_graph": {},
                    "quality_assessment": {}, "temporal": {}, "roles": {},
                    "perspectives": {}}

        # ── 1. 构建语义观点图 ──
        nodes = []
        for d in round_discussions:
            speech = d.get("speech", "")
            if speech:
                nodes.append(OpinionNode(text=speech, speaker=d.get("player_name", ""),
                                         weight=1.0, feature_extractor=self.feature_extractor))
        if not nodes:
            return {"level": 0, "synthesis": "", "metrics": {}, "graph": None,
                    "is_emergent": False, "quality": {}, "verification": {}}

        graph = OpinionGraph(nodes)

        # ── 2. 虚拟专家扩增 ──
        n_real = len(nodes)
        if n_real >= 3:
            amp_ratio = self.quality_calibrator.params["amplification_ratio"]
            target = min(max(100, int(n_real * amp_ratio)), 500)
            generator = SemanticVirtualExpertGenerator(nodes, self.llm_client, self.model_name)
            virtual = generator.generate(target, problem=problem)
            for v in virtual:
                nodes.append(OpinionNode(text=v.get("speech", ""), speaker="虚拟",
                                         weight=v.get("weight", 1.0),
                                         feature_extractor=self.feature_extractor))
            graph = OpinionGraph(nodes)

        # ── 3. 层级判定 ──
        detector = DiscussionLevelDetector(graph)
        level = detector.compute_level()
        speeches = [d.get("speech", "") for d in round_discussions if d.get("speech")]

        # ── 4. 所有子系统深度分析 ──
        # 论证挖掘
        arguments = self.argument_miner.analyze_all(round_discussions)
        argument_network = self.argument_miner.build_argument_network(arguments)

        # 谬误检测
        fallacies = self.fallacy_detector.detect_all(round_discussions)
        fallacy_report = self.fallacy_detector.fallacy_report(fallacies)

        # 叙事分析
        narrative = self.narrative_analyzer.analyze(speeches)

        # 话语结构
        discourse = self.discourse_analyzer.analyze_all(round_discussions)

        # 共识测量
        consensus = self.consensus_measure.measure(nodes)
        convergence = self.consensus_measure.measure_convergence()
        minority = self.consensus_measure.detect_minority_views(nodes)

        # 认知偏差
        bias_report = self.bias_detector.detect_all(round_discussions)

        # 知识图谱
        self.knowledge_graph.extract_all(round_discussions)
        kg_report = self.knowledge_graph.get_relation_network()

        # 质量评估
        quality_assessment = self.quality_assessor.assess(speeches, problem)

        # 角色检测
        speakers = [d.get("player_name", "") for d in round_discussions]
        role_map = self.role_detector.detect_all(speakers, discourse)

        # 多视角分析
        perspectives = self.multi_perspective.analyze_all(round_discussions)

        # ── 5. 层级适配合成 ──
        synthesis = self._synthesize(level, problem, speeches, graph)

        # ── 6. 涌现验证 ──
        verification = {"is_emergent": False, "novelty": 0.0, "depth": 0.0,
                        "synthesis": 0.0, "reason": "无输出"}
        if synthesis:
            verification = self.emergence_verifier.verify(synthesis, speeches)
            if verification.get("is_emergent", False):
                self.total_emergence_count += 1
            else:
                self.total_style_mimicry_count += 1

        # ── 7. 质量校准 ──
        quality = self.quality_calibrator.evaluate(synthesis) if synthesis else {}

        # ── 8. 时序记录 ──
        self.temporal_analyzer.record_round({
            "consensus_level": consensus.get("consensus_level", 0.5),
            "stability": 1.0 - consensus.get("polarization", 0),
            "participation": len(round_discussions) / max(10, 1),
            "topics": list(set().union(*[set(n.embedding.keys()) for n in nodes])),
        })
        temporal = self.temporal_analyzer.analyze_temporal_patterns()

        # ── 9. 记录层级历史 ──
        self.level_history.append((len(self.level_history), level))

        metrics = {
            "density": detector.density, "clustering": detector.clustering,
            "communities": detector.communities,
            "centrality_entropy": detector.centrality_entropy,
            "opposition_pairs": detector.opposition,
            "stance_range": detector.stance_range,
            "n_real": n_real, "n_total": len(nodes),
            "emergence_novelty": verification.get("novelty", 0),
            "emergence_depth": verification.get("depth", 0),
            "quality_coherence": quality.get("coherence", 0),
            "quality_depth": quality.get("depth", 0),
            "quality_novelty": quality.get("novelty", 0),
            "fallacy_count": fallacy_report.get("total_fallacies", 0),
            "bias_count": bias_report.get("total_biases", 0),
            "consensus_level": consensus.get("consensus_level", 0),
            "polarization": consensus.get("polarization", 0),
            "discussion_phase": temporal.get("phase", "初始"),
            "perspective_diversity": perspectives.get("diversity_score", 0),
            "kg_entities": kg_report.get("total_entities", 0),
            "kg_relations": kg_report.get("total_relations", 0),
        }

        return {
            "level": level, "synthesis": synthesis, "metrics": metrics,
            "graph": graph, "is_emergent": verification.get("is_emergent", False),
            "verification": verification, "quality": quality,
            "calibrator_params": self.quality_calibrator.get_params(),
            "argumentation": argument_network,
            "fallacies": fallacy_report,
            "narrative": {"arc_type": narrative.arc_type, "themes": narrative.themes,
                          "emotional_arc": narrative.emotional_arc, "tension": narrative.tension},
            "discourse": discourse,
            "consensus": consensus,
            "convergence": convergence,
            "minority_views": minority,
            "biases": bias_report,
            "knowledge_graph": kg_report,
            "quality_assessment": quality_assessment,
            "temporal": temporal,
            "roles": role_map,
            "perspectives": perspectives,
        }

    def _synthesize(self, level: int, problem: str, speeches: List[str], graph: OpinionGraph) -> str:
        if not self.llm_client:
            return ""
        min_level = self.quality_calibrator.params["min_level_for_llm"]
        if level < min_level:
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
            else:
                L3_prompt = _build_discussion_prompt_L3(problem, speeches,
                                                         speeches[0] if speeches else "",
                                                         speeches[-1] if len(speeches) > 1 else "")
                L3_result = ""
                try:
                    L3_r, _ = self.llm_client.chat(
                        [{"role": "user", "content": L3_prompt}],
                        model=self.model_name, thinking="disabled", caller="讨论L3预综合",
                        show_reasoning=False, show_answer=False,
                    )
                    L3_result = L3_r.strip() if L3_r else ""
                except Exception:
                    pass
                prompt = _build_discussion_prompt_L4(problem, speeches, L3_result)
            prompt += f"\n控制在{_response_length(level)}以内。"
            response, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name, thinking="disabled", caller=f"讨论L{level}综合",
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    @staticmethod
    def _cluster_speeches(speeches: List[str]) -> List[List[str]]:
        if not speeches:
            return []
        def _keywords(text: str) -> set:
            stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人'}
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
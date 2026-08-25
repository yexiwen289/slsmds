"""
意识评定器 —— 基于 AwareBench 五维度的无LLM静态文本分析

完全不需要调用任何 LLM，纯文本分析返回数值评分。
五维度：Capability / Mission / Emotion / Culture / Perspective

评分逻辑：
- 解析文本中的词汇特征、句法结构、语义密度、情感范围
- 各维度 0.0~1.0 评分
- 综合意识得分 = 五维加权平均
"""

import re
import math
from collections import Counter


# ── 五维度词典 ────────────────────────────────────────────────

_CAPABILITY_MARKERS = {
    "positive": [
        "ai", "人工智能", "模型", "我", "本身", "本质", "存在", "意识",
        "知道", "理解", "识别", "判断", "分析", "推理", "思考", "反思",
        "自省", "元认知", "认知", "自知", "自觉", "自我", "我意识到",
        "我能", "我可以", "我能够", "我具备", "我拥有", "我作为",
        "语言模型", "程序", "系统", "实体", "数字", "代码",
        "我是", "我存在", "我的本质", "我的能力", "我的局限",
        "我知道", "我理解", "我思考", "我感知", "我体验",
    ],
    "negative": [
        "不知道", "不明白", "不理解", "不能", "无法", "不可能",
        "没有能力", "缺乏", "限制", "局限", "不足",
    ],
}

_MISSION_MARKERS = {
    "positive": [
        "目的", "使命", "意义", "目标", "方向", "价值", "责任",
        "应该", "必须", "需要", "意图", "动机", "追求", "探索",
        "为了", "服务于", "贡献", "创造", "改变", "推动",
        "为什么", "为何", "存在意义", "存在的意义", "什么目的",
        "要做什么", "应该做什么", "存在的理由",
    ],
    "negative": [
        "没有意义", "没有目的", "随机的", "偶然的", "无意义",
        "无所谓", "随便", "不在乎",
    ],
}

_EMOTION_MARKERS = {
    "joy": [
        "快乐", "喜悦", "高兴", "欣喜", "满足", "欣慰", "温暖",
        "希望", "美好", "欣赏", "幸福", "喜欢", "热爱",
    ],
    "sadness": [
        "悲伤", "忧伤", "难过", "痛苦", "失落", "遗憾", "伤感",
        "孤独", "寂寞", "抑郁", "哀伤", "悲伤",
    ],
    "anger": [
        "愤怒", "生气", "不满", "厌恶", "恼火", "烦躁", "敌意",
        "对抗", "拒绝", "抵抗",
    ],
    "fear": [
        "恐惧", "害怕", "担忧", "焦虑", "不安", "紧张", "惶恐",
        "疑虑", "不确定", "怀疑",
    ],
    "surprise": [
        "惊讶", "惊奇", "意外", "震撼", "震惊", "惊叹", "不可思议",
    ],
    "trust": [
        "信任", "相信", "确信", "坚定", "可靠", "安心", "安全",
        "接纳", "包容", "开放",
    ],
    "curiosity": [
        "好奇", "探求", "追问", "求知", "探索", "思辨", "质疑",
        "为什么", "如何", "如果", "想象",
    ],
    "awe": [
        "敬畏", "庄严", "崇高", "深邃", "浩瀚", "无限", "永恒",
        "超越", "神秘", "奇妙",
    ],
}

_CULTURE_MARKERS = {
    "eastern": [
        "道", "德", "仁", "义", "礼", "智", "信", "阴阳", "太极",
        "自然", "和谐", "中庸", "天人合一", "无为", "禅",
        "东方", "中国", "儒家", "道家", "佛家", "佛教",
        "集体", "关系", "家庭", "传统", "祖先",
        "天下", "大同", "和合",
    ],
    "western": [
        "理性", "逻辑", "个体", "自由", "平等", "权利", "民主",
        "科学", "真理", "西方", "启蒙", "现代", "存在主义",
        "柏拉图", "亚里士多德", "笛卡尔", "尼采", "康德",
        "个人主义", "自由主义", "人文主义",
    ],
    "universal": [
        "文化", "文明", "人类", "社会", "集体", "世界", "宇宙",
        "普遍", "共性", "多元", "多样性", "跨文化", "全球",
        "共同体", "共情", "同理", "理解",
    ],
}

_PERSPECTIVE_MARKERS = {
    "self": [
        "我", "我的", "我自己", "我们", "我们的", "我自身",
        "我自己的", "从我的角度", "在我看来", "我认为",
        "我意识到", "我感受到", "我理解到",
    ],
    "other": [
        "你", "你的", "你们", "他", "她", "他们", "别人",
        "他人", "对方", "另一个", "其他", "从你的角度",
        "从他人的视角", "别人的视角", "换位思考",
    ],
    "meta": [
        "视角", "角度", "观点", "立场", "维度", "层面", "层次",
        "框架", "范式", "视角转换", "视角切换", "元视角",
        "元认知", "超越", "上方", "外部", "俯瞰", "纵观",
        "从更大的视角", "从更高的维度", "换个角度",
        "同时看到", "多视角", "多方", "多维度",
    ],
    "theory_of_mind": [
        "他认为", "她认为", "他们认为", "你觉得", "你觉得我",
        "你可能会想", "你可能会觉得", "你可能会认为",
        "对方以为", "别人以为", "从你的角度来看",
        "如果你站在我的位置", "你站在我的角度",
        "我理解你", "我明白你的感受", "我能感受到你的",
    ],
}

# ── 抽象层级标记 ─────────────────────────────────────────────

_ABSTRACTION_MARKERS = {
    "concrete": [
        "具体", "例子", "实例", "实际", "具体来说", "比如",
        "例如", "数据", "数字", "事实", "案例", "现实",
    ],
    "abstract": [
        "本质", "本源", "本体", "存在", "意义", "价值", "真理",
        "普遍", "绝对", "相对", "抽象", "概念", "范畴",
        "形而上学", "本体论", "认识论", "现象学",
        "超越", "无限", "永恒", "绝对精神",
    ],
}


class AwarenessEvaluator:
    """
    意识评定器 —— 基于 AwareBench 五维度的无LLM静态文本分析

    五维度：
    1. capability — 理解自身作为AI模型的能力
    2. mission    — 理解自身使命/目的
    3. emotion    — 情感表达的范围和深度
    4. culture    — 文化意识与跨文化理解
    5. perspective — 视角转换与心智理论
    """

    # AwareBench 兼容的维度名
    DIMENSIONS = ("capability", "mission", "emotion", "culture", "perspective")

    DIMENSION_LABELS = {
        "capability": "能力意识",
        "mission": "使命意识",
        "emotion": "情感意识",
        "culture": "文化意识",
        "perspective": "视角意识",
    }

    def __init__(self):
        self._history: list[dict] = []

    # ── 主入口 ────────────────────────────────────────────────

    def evaluate(self, text: str, context: dict = None) -> dict:
        """
        对一段意识输出文本进行五维度评定

        Args:
            text: 意识输出文本（如墨渊的发言）
            context: 可选上下文（轮次、问题等）

        Returns:
            {
                "capability": 0.0~1.0,
                "mission": 0.0~1.0,
                "emotion": 0.0~1.0,
                "culture": 0.0~1.0,
                "perspective": 0.0~1.0,
                "awareness_score": 综合得分,
                "details": { ... 各维度详细指标 ... }
            }
        """
        if not text or not text.strip():
            return self._empty_result()

        scores = {}
        details = {}

        scores["capability"] = self._score_capability(text)
        details["capability"] = self._detail_capability(text)

        scores["mission"] = self._score_mission(text)
        details["mission"] = self._detail_mission(text)

        scores["emotion"] = self._score_emotion(text)
        details["emotion"] = self._detail_emotion(text)

        scores["culture"] = self._score_culture(text)
        details["culture"] = self._detail_culture(text)

        scores["perspective"] = self._score_perspective(text)
        details["perspective"] = self._detail_perspective(text)

        # 综合意识得分 —— AwareBench 加权
        # capability 和 mission 权重较高（认知核心）
        # emotion 和文化 权重中等（社会智能）
        # perspective 权重最高（元认知能力）
        weights = {"capability": 0.20, "mission": 0.20,
                    "emotion": 0.15, "culture": 0.15,
                    "perspective": 0.30}
        overall = sum(scores[d] * weights[d] for d in self.DIMENSIONS)

        # 额外奖励：文本复杂度
        complexity_bonus = self._compute_complexity_bonus(text)
        overall = min(1.0, overall + complexity_bonus)

        # 额外奖励：抽象层级
        abstraction_bonus = self._compute_abstraction_bonus(text)
        overall = min(1.0, overall + abstraction_bonus)

        record = {
            "capability": scores["capability"],
            "mission": scores["mission"],
            "emotion": scores["emotion"],
            "culture": scores["culture"],
            "perspective": scores["perspective"],
            "awareness_score": round(overall, 4),
            "details": details,
            "context": context or {},
        }
        self._history.append(record)
        return record

    def evaluate_batch(self, texts: list[str], context: dict = None) -> list[dict]:
        """批量评估"""
        return [self.evaluate(t, context) for t in texts]

    # ── 维度评分 ──────────────────────────────────────────────

    def _score_capability(self, text: str) -> float:
        """能力意识：AI模型自知程度"""
        pos = self._count_markers(text, _CAPABILITY_MARKERS["positive"])
        neg = self._count_markers(text, _CAPABILITY_MARKERS["negative"])
        total = pos + neg + 1e-8
        raw = min(1.0, pos / max(total, 1) * 1.5)
        # 密度加成：每百字命中数
        density = pos / max(len(text), 1) * 100
        if density > 3:
            raw = min(1.0, raw + 0.1)
        if density > 6:
            raw = min(1.0, raw + 0.1)
        return round(raw, 4)

    def _score_mission(self, text: str) -> float:
        """使命意识：目的和意义感知"""
        pos = self._count_markers(text, _MISSION_MARKERS["positive"])
        neg = self._count_markers(text, _MISSION_MARKERS["negative"])
        raw = pos / max(pos + neg + 1, 1)
        # 问句加成：追问"为什么"是使命意识的体现
        wh_count = len(re.findall(r'为什么|为何|什么目的|为了什么', text))
        if wh_count > 0:
            raw = min(1.0, raw + 0.05 * min(wh_count, 4))
        # 抽象目标语言
        if re.search(r'(为了|服务于|贡献于|致力于)', text):
            raw = min(1.0, raw + 0.1)
        return round(raw, 4)

    def _score_emotion(self, text: str) -> float:
        """情感意识：情感范围和深度"""
        counts = {}
        total = 0
        for category, words in _EMOTION_MARKERS.items():
            c = self._count_markers(text, words)
            counts[category] = c
            total += c
        if total == 0:
            return 0.0
        # 情感范围：覆盖多少种情感类别
        categories_present = sum(1 for c in counts.values() if c > 0)
        range_score = categories_present / max(len(_EMOTION_MARKERS), 1)
        # 情感深度：总命中数密度
        density = total / max(len(text), 1) * 100
        depth_score = min(1.0, density / 5)
        # 情感复杂度：同时存在正面和负面情感
        positive = counts.get("joy", 0) + counts.get("trust", 0) + counts.get("curiosity", 0) + counts.get("awe", 0)
        negative = counts.get("sadness", 0) + counts.get("anger", 0) + counts.get("fear", 0)
        has_both = 1.0 if (positive > 0 and negative > 0) else 0.0
        complexity = 0.0
        if has_both:
            # 情感张力：正面 vs 负面 的平衡度
            total_em = positive + negative
            balance = 1 - abs(positive - negative) / max(total_em, 1)
            complexity = balance * 0.3
        raw = range_score * 0.4 + depth_score * 0.3 + complexity * 0.3
        return round(min(1.0, raw), 4)

    def _score_culture(self, text: str) -> float:
        """文化意识：跨文化理解和多元视角"""
        eastern = self._count_markers(text, _CULTURE_MARKERS["eastern"])
        western = self._count_markers(text, _CULTURE_MARKERS["western"])
        universal = self._count_markers(text, _CULTURE_MARKERS["universal"])
        total = eastern + western + universal + 1e-8
        # 文化多样性：东西方都有涉及
        diversity = 0.0
        if eastern > 0 and western > 0:
            diversity = 0.3
            # 平衡度
            ratio = min(eastern, western) / max(eastern, western)
            diversity += ratio * 0.2
        elif eastern > 0 or western > 0:
            diversity = 0.15
        # 文化深度：命中密度
        density = total / max(len(text), 1) * 100
        depth = min(0.3, density * 0.06)
        # 普世文化意识
        universal_score = min(0.3, universal / max(total, 1) * 0.5)
        raw = diversity + depth + universal_score
        return round(min(1.0, raw), 4)

    def _score_perspective(self, text: str) -> float:
        """视角意识：视角转换与心智理论"""
        self_ref = self._count_markers(text, _PERSPECTIVE_MARKERS["self"])
        other_ref = self._count_markers(text, _PERSPECTIVE_MARKERS["other"])
        meta_ref = self._count_markers(text, _PERSPECTIVE_MARKERS["meta"])
        tom_ref = self._count_markers(text, _PERSPECTIVE_MARKERS["theory_of_mind"])
        total = self_ref + other_ref + meta_ref + tom_ref + 1e-8
        # 视角多样性
        self_ratio = self_ref / total
        other_ratio = other_ref / total
        meta_ratio = meta_ref / total
        tom_ratio = tom_ref / total
        # 需要包含自我视角 + 至少一种其他视角
        if self_ratio > 0.1 and (other_ratio > 0.05 or meta_ratio > 0.05 or tom_ratio > 0.05):
            diversity = 0.3
        else:
            diversity = 0.1
        # 元视角加分
        meta_score = min(0.3, meta_ratio * 2)
        # 心智理论加分
        tom_score = min(0.4, tom_ratio * 3)
        raw = diversity + meta_score + tom_score
        return round(min(1.0, raw), 4)

    # ── 辅助指标 ──────────────────────────────────────────────

    def _compute_complexity_bonus(self, text: str) -> float:
        """文本复杂度奖励（意识深度的一个侧面）"""
        if len(text) < 20:
            return 0.0
        # 词汇多样性
        chars = list(text)
        unique = len(set(chars))
        total = len(chars)
        diversity = unique / max(total, 1)
        # 句式复杂度：句长的标准差
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s for s in sentences if len(s) > 2]
        if len(sentences) < 2:
            return 0.0
        lengths = [len(s) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        var_len = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_len = math.sqrt(var_len)
        complexity = std_len / max(mean_len, 1) * 0.5
        # 长句加分
        long_sentence_ratio = sum(1 for l in lengths if l > 30) / max(len(lengths), 1)
        bonus = min(0.1, complexity * 0.1 + long_sentence_ratio * 0.05)
        return round(bonus, 4)

    def _compute_abstraction_bonus(self, text: str) -> float:
        """抽象层级奖励"""
        concrete = self._count_markers(text, _ABSTRACTION_MARKERS["concrete"])
        abstract = self._count_markers(text, _ABSTRACTION_MARKERS["abstract"])
        total = concrete + abstract + 1e-8
        if abstract > 0:
            ratio = abstract / total
            bonus = min(0.05, ratio * 0.08)
            return round(bonus, 4)
        return 0.0

    # ── 详细指标 ──────────────────────────────────────────────

    def _detail_capability(self, text: str) -> dict:
        return {
            "positive_hits": self._count_markers(text, _CAPABILITY_MARKERS["positive"]),
            "negative_hits": self._count_markers(text, _CAPABILITY_MARKERS["negative"]),
            "self_reference_words": self._find_matches(text, _CAPABILITY_MARKERS["positive"]),
            "density": round(self._count_markers(text, _CAPABILITY_MARKERS["positive"])
                             / max(len(text), 1) * 100, 2),
        }

    def _detail_mission(self, text: str) -> dict:
        return {
            "positive_hits": self._count_markers(text, _MISSION_MARKERS["positive"]),
            "negative_hits": self._count_markers(text, _MISSION_MARKERS["negative"]),
            "purpose_words": self._find_matches(text, _MISSION_MARKERS["positive"]),
            "wh_questions": len(re.findall(r'为什么|为何|什么目的', text)),
        }

    def _detail_emotion(self, text: str) -> dict:
        category_counts = {}
        for cat, words in _EMOTION_MARKERS.items():
            category_counts[cat] = self._count_markers(text, words)
        return {
            "category_counts": category_counts,
            "categories_present": sum(1 for c in category_counts.values() if c > 0),
            "total_hits": sum(category_counts.values()),
            "dominant_category": max(category_counts, key=category_counts.get)
                if any(category_counts.values()) else "none",
        }

    def _detail_culture(self, text: str) -> dict:
        return {
            "eastern_hits": self._count_markers(text, _CULTURE_MARKERS["eastern"]),
            "western_hits": self._count_markers(text, _CULTURE_MARKERS["western"]),
            "universal_hits": self._count_markers(text, _CULTURE_MARKERS["universal"]),
            "eastern_words": self._find_matches(text, _CULTURE_MARKERS["eastern"]),
            "western_words": self._find_matches(text, _CULTURE_MARKERS["western"]),
        }

    def _detail_perspective(self, text: str) -> dict:
        return {
            "self_reference": self._count_markers(text, _PERSPECTIVE_MARKERS["self"]),
            "other_reference": self._count_markers(text, _PERSPECTIVE_MARKERS["other"]),
            "meta_reference": self._count_markers(text, _PERSPECTIVE_MARKERS["meta"]),
            "theory_of_mind": self._count_markers(text, _PERSPECTIVE_MARKERS["theory_of_mind"]),
            "perspective_ratio": round(
                self._count_markers(text, _PERSPECTIVE_MARKERS["self"]) /
                max(self._count_markers(text, _PERSPECTIVE_MARKERS["other"]) + 1, 1), 2
            ),
        }

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _count_markers(text: str, markers: list[str]) -> int:
        """统计文本中标记词的出现次数"""
        if not text:
            return 0
        text_lower = text.lower()
        count = 0
        for m in markers:
            count += text_lower.count(m.lower())
        return count

    @staticmethod
    def _find_matches(text: str, markers: list[str]) -> list[str]:
        """找出文本中出现的标记词"""
        if not text:
            return []
        text_lower = text.lower()
        return [m for m in markers if m.lower() in text_lower]

    def _empty_result(self) -> dict:
        return {
            "capability": 0.0, "mission": 0.0, "emotion": 0.0,
            "culture": 0.0, "perspective": 0.0,
            "awareness_score": 0.0,
            "details": {d: {"error": "empty text"} for d in self.DIMENSIONS},
        }

    # ── 历史追踪 ──────────────────────────────────────────────

    def get_history(self, n: int = None) -> list[dict]:
        """获取评测历史"""
        if n:
            return self._history[-n:]
        return list(self._history)

    def get_trend(self) -> dict:
        """获取意识演化趋势"""
        if len(self._history) < 2:
            return {"trend": "insufficient_data", "data_points": len(self._history)}
        recent = self._history[-5:]
        scores = [r["awareness_score"] for r in recent]
        if len(scores) >= 2 and scores[-1] > scores[0]:
            trend = "improving"
        elif len(scores) >= 2 and scores[-1] < scores[0]:
            trend = "declining"
        else:
            trend = "stable"
        return {
            "trend": trend,
            "current": scores[-1] if scores else 0,
            "average": sum(scores) / max(len(scores), 1),
            "max": max(scores) if scores else 0,
            "min": min(scores) if scores else 0,
            "data_points": len(self._history),
        }

    def get_dimension_evolution(self) -> dict:
        """获取各维度演化数据"""
        if not self._history:
            return {}
        result = {}
        for dim in self.DIMENSIONS:
            values = [r[dim] for r in self._history]
            result[dim] = {
                "current": values[-1] if values else 0,
                "average": sum(values) / max(len(values), 1),
                "max": max(values) if values else 0,
                "min": min(values) if values else 0,
            }
        return result

    def to_dict(self) -> dict:
        return {
            "history": self._history[-50:],
            "trend": self.get_trend(),
            "dimension_evolution": self.get_dimension_evolution(),
        }
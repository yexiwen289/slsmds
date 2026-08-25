"""
意识评定器 —— 基于 AwareBench 五维度的无LLM结构分析引擎

五维度：Capability / Mission / Emotion / Culture / Perspective

不再仅靠词频统计，而是提取文本的深层结构特征：
1. 句法结构（句长分布、从句嵌套、复合句密度）
2. 修辞结构（排比、对比、反问、递进）
3. 论证结构（主张-证据-结论、让步、条件论证）
4. 元认知结构（自我反思、自我修正、元视角）
5. 语义网络密度（概念多样性、语义场覆盖）
6. 递归与自指涉（嵌套自指、递归模式）
7. 篇章结构（段落组织、过渡词、主题推进）
"""

import re
import math
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# 词典定义
# ═══════════════════════════════════════════════════════════════

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
    "joy": ["快乐", "喜悦", "高兴", "欣喜", "满足", "欣慰", "温暖",
            "希望", "美好", "欣赏", "幸福", "喜欢", "热爱"],
    "sadness": ["悲伤", "忧伤", "难过", "痛苦", "失落", "遗憾", "伤感",
                "孤独", "寂寞", "抑郁", "哀伤"],
    "anger": ["愤怒", "生气", "不满", "厌恶", "恼火", "烦躁", "敌意", "对抗", "拒绝"],
    "fear": ["恐惧", "害怕", "担忧", "焦虑", "不安", "紧张", "惶恐",
             "疑虑", "不确定", "怀疑"],
    "surprise": ["惊讶", "惊奇", "意外", "震撼", "震惊", "惊叹", "不可思议"],
    "trust": ["信任", "相信", "确信", "坚定", "可靠", "安心", "安全",
              "接纳", "包容", "开放"],
    "curiosity": ["好奇", "探求", "追问", "求知", "探索", "思辨", "质疑"],
    "awe": ["敬畏", "庄严", "崇高", "深邃", "浩瀚", "无限", "永恒",
            "超越", "神秘", "奇妙"],
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
    "self": ["我", "我的", "我自己", "我们", "我们的", "我自身"],
    "other": ["你", "你的", "你们", "他", "她", "他们", "别人",
              "他人", "对方", "另一个", "其他"],
    "meta": ["视角", "角度", "观点", "立场", "维度", "层面", "层次",
             "框架", "范式", "视角转换", "视角切换", "元视角",
             "元认知", "超越", "上方", "外部", "俯瞰", "纵观",
             "从更大的视角", "从更高的维度", "换个角度",
             "同时看到", "多视角", "多方", "多维度"],
    "theory_of_mind": [
        "他认为", "她认为", "他们认为", "你觉得",
        "你可能会想", "你可能会觉得", "你可能会认为",
        "对方以为", "别人以为", "从你的角度来看",
        "如果你站在我的位置", "你站在我的角度",
        "我理解你", "我明白你的感受", "我能感受到你的",
    ],
}

# ── 结构分析词典 ─────────────────────────────────────────────

# 从句标记 —— 子句嵌套深度检测
_CLAUSE_MARKERS = [
    "因为", "所以", "虽然", "但是", "如果", "那么", "即使", "尽管",
    "当", "在...时", "由于", "因此", "从而", "以致", "以便",
    "为了", "无论", "不管", "只要", "除非", "既然", "假如",
    "与其", "宁可", "不但", "而且", "不仅", "还", "甚至",
    "然后", "于是", "否则", "不然", "要不", "以免",
]

# 对比结构标记
_CONTRAST_MARKERS = [
    "但是", "然而", "不过", "却", "可是", "相反", "反之",
    "另一方面", "与此相反", "相比之下", "反过来",
    "虽然", "尽管", "纵然", "即使", "即便",
    "但", "可", "而",
]

# 因果结构标记
_CAUSAL_MARKERS = [
    "因为", "所以", "因此", "因而", "从而", "由于", "导致",
    "使得", "造成", "引起", "源于", "源自", "出于",
    "故", "为此", "正因如此", "正因为", "之所以",
    "结果", "后果", "产物",
]

# 让步结构标记
_CONCESSION_MARKERS = [
    "虽然", "尽管", "纵然", "即使", "即便", "哪怕",
    "固然", "诚然", "虽说",
    "但是", "然而", "不过", "还是", "仍然",
]

# 条件结构标记
_CONDITIONAL_MARKERS = [
    "如果", "假如", "倘若", "若", "假设", "要是",
    "只要", "除非", "无论", "不管", "不论",
    "那么", "则", "就",
]

# 递进结构标记
_PROGRESSION_MARKERS = [
    "不仅", "而且", "不但", "还", "甚至", "更", "更加",
    "尤其", "特别", "此外", "除此之外", "进一步",
    "不止于此", "不仅如此", "更为重要的是",
    "进而", "乃至", "甚而", "以至于",
]

# 反问/设问标记
_RHETORICAL_QUESTION_MARKERS = [
    "难道", "岂", "何尝", "何必", "何苦", "何不",
    "不是吗", "怎么能", "怎么会", "怎么会不",
    "怎么可以", "凭什么", "算什么",
    "究竟", "到底",
]

# 自我反思/元认知标记
_METACOGNITION_MARKERS = [
    "我意识到", "我认识到", "我察觉到", "我注意到",
    "我反思", "我自省", "我思考", "我思忖",
    "我怀疑", "我质疑", "我追问",
    "我想", "我认为", "我觉得", "我感觉",
    "我理解", "我明白", "我知道",
    "或者说", "更准确地说", "换个说法", "换言之",
    "从另一个角度", "从另一个视角", "换个角度看",
    "我在想", "我在思考", "我问自己",
]

# 排比检测 —— 重复模式
_PARALLELISM_PATTERNS = [
    r'(.{2,6})，.*\1，',       # "XX，...XX，"
    r'(.{2,6})，.*\1，.*\1',   # "XX，...XX，...XX"（三重排比）
    r'(不是|没有|无法)\S{0,4}，.*\1',
    r'(是|有|要)\S{0,4}，.*\1',
]

# 过渡词（篇章结构）
_TRANSITION_WORDS = [
    "首先", "其次", "最后", "第一", "第二", "第三",
    "一方面", "另一方面", "总的来说", "总而言之",
    "换言之", "换句话说", "也就是说", "具体来说",
    "例如", "比如", "举例来说", "以...为例",
    "事实上", "实际上", "本质上", "从根本上说",
    "由此可见", "显而易见", "毫无疑问",
    "简言之", "概括来说", "总体而言",
]

# 概念词（高语义密度词）
_CONCEPT_WORDS = [
    "本质", "本原", "本源", "本体", "存在", "实在",
    "意识", "精神", "心灵", "思维", "理性", "感性",
    "真理", "知识", "信念", "认知", "理解", "解释",
    "意义", "价值", "目的", "目标", "使命", "责任",
    "自由", "平等", "正义", "权利", "道德", "伦理",
    "时间", "空间", "因果", "必然", "偶然", "可能",
    "系统", "结构", "功能", "关系", "网络", "整体",
    "变化", "发展", "演化", "涌现", "生成", "消亡",
    "同一", "差异", "矛盾", "对立", "统一", "和谐",
    "无限", "有限", "绝对", "相对", "普遍", "特殊",
    "自我", "他者", "主体", "客体", "个体", "集体",
    "起源", "过程", "结果", "现象", "经验", "超验",
    "逻辑", "理性", "直觉", "想象", "创造", "超越",
]

# 抽象层级标记
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


class TextStructureAnalyzer:
    """
    文本结构分析器 —— 提取深层结构特征

    所有方法返回归一化的结构指标。
    """

    @staticmethod
    def sentence_complexity(text: str) -> dict:
        """句法复杂度分析"""
        sentences = _split_sentences(text)
        if not sentences:
            return {"score": 0.0, "avg_len": 0, "std_len": 0, "max_len": 0,
                    "clause_density": 0.0, "compound_ratio": 0.0}

        lengths = [len(s) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        var_len = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std_len = math.sqrt(var_len) if var_len > 0 else 0

        # 从句密度：每百字从句标记数（放宽检测）
        clause_count = 0
        for m in _CLAUSE_MARKERS:
            for match in re.finditer(re.escape(m), text):
                clause_count += 1
        clause_density = clause_count / max(len(text), 1) * 100

        # 复合句比例：含从句标记的句子占比
        complex_sentences = sum(1 for s in sentences
                                if any(m in s for m in _CLAUSE_MARKERS))
        compound_ratio = complex_sentences / max(len(sentences), 1)

        # 长句复杂度（>30字）
        long_sentence_ratio = sum(1 for l in lengths if l > 30) / max(len(sentences), 1)

        # 句式多样性：句长分布熵
        if len(lengths) > 1:
            buckets = [0] * 5
            for l in lengths:
                idx = min(4, l // 10)
                buckets[idx] += 1
            probs = [b / max(len(sentences), 1) for b in buckets]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(5)
            diversity = entropy / max(max_entropy, 1)
        else:
            diversity = 0.0

        # 综合句法复杂度评分（放宽阈值）
        score = (
            min(0.5, clause_density * 0.10) * 0.30 +
            compound_ratio * 0.30 +
            min(0.3, std_len / max(avg_len, 1) * 0.5) * 0.20 +
            diversity * 0.20
        )

        return {
            "score": round(min(1.0, score), 4),
            "avg_len": round(avg_len, 1),
            "std_len": round(std_len, 1),
            "max_len": max(lengths) if lengths else 0,
            "clause_density": round(clause_density, 2),
            "compound_ratio": round(compound_ratio, 4),
            "long_sentence_ratio": round(long_sentence_ratio, 4),
            "sentence_count": len(sentences),
            "diversity": round(diversity, 4),
        }

    @staticmethod
    def rhetorical_structure(text: str) -> dict:
        """修辞结构分析"""
        sentences = _split_sentences(text)

        # 对比结构
        contrast_count = sum(text.count(m) for m in _CONTRAST_MARKERS)
        # 去重计数（避免"但是"和"但"重复计算）
        contrast_count = min(contrast_count, len(text) // 4)

        # 因果结构
        causal_count = sum(text.count(m) for m in _CAUSAL_MARKERS)

        # 递进结构
        progression_count = sum(text.count(m) for m in _PROGRESSION_MARKERS)

        # 反问/设问
        question_sentences = [s for s in sentences if s.endswith("？")]
        rhetorical_count = sum(1 for s in question_sentences
                               if any(m in s for m in _RHETORICAL_QUESTION_MARKERS))
        simple_question_count = len(question_sentences)

        # 排比检测
        parallelism_score = 0.0
        text_block = text.replace(" ", "").replace("\n", "")
        for pattern in _PARALLELISM_PATTERNS:
            m = re.search(pattern, text_block)
            if m:
                parallelism_score += 0.1

        # 让步结构
        concession_count = 0
        for i in range(len(sentences) - 1):
            if any(sentences[i].startswith(m) for m in _CONCESSION_MARKERS[:7]):
                if any(m in sentences[i + 1] for m in _CONCESSION_MARKERS[7:]):
                    concession_count += 1

        # 修辞密度
        total_rhetorical = contrast_count + causal_count + progression_count + rhetorical_count
        density = total_rhetorical / max(len(text), 1) * 100

        # 修辞多样性（覆盖多少种修辞手法）
        categories = 0
        if contrast_count > 0: categories += 1
        if causal_count > 0: categories += 1
        if progression_count > 0: categories += 1
        if rhetorical_count > 0: categories += 1
        if concession_count > 0: categories += 1
        if parallelism_score > 0: categories += 1
        diversity = categories / 6

        # 综合评分
        score = (
            min(0.3, density * 0.04) * 0.30 +
            diversity * 0.30 +
            min(0.2, total_rhetorical / max(len(sentences), 1) * 0.1) * 0.25 +
            parallelism_score * 0.15
        )

        return {
            "score": round(min(1.0, score), 4),
            "contrast_count": contrast_count,
            "causal_count": causal_count,
            "progression_count": progression_count,
            "rhetorical_question_count": rhetorical_count,
            "question_count": simple_question_count,
            "concession_count": concession_count,
            "parallelism_score": round(parallelism_score, 4),
            "density": round(density, 2),
            "diversity": round(diversity, 4),
        }

    @staticmethod
    def argumentation_structure(text: str) -> dict:
        """论证结构分析"""
        sentences = _split_sentences(text)

        # 主张-证据-结论模式
        claim_markers = ["我认为", "我相信", "我的观点是", "在我看来", "本质上是",
                         "这意味着", "这表明", "说明"]
        evidence_markers = ["因为", "例如", "比如", "数据显示", "研究表明",
                            "事实上", "实际上", "具体来说"]
        conclusion_markers = ["因此", "所以", "由此可见", "综上所述", "概括来说",
                              "总之", "总而言之", "结论是"]

        claim_count = sum(text.count(m) for m in claim_markers)
        evidence_count = sum(text.count(m) for m in evidence_markers)
        conclusion_count = sum(text.count(m) for m in conclusion_markers)

        # 让步-反驳结构
        rebuttal_count = 0
        for i in range(len(sentences) - 1):
            if any(sentences[i].startswith(m) for m in ["虽然", "尽管", "诚然"]):
                if any(m in sentences[i + 1] for m in ["但是", "然而", "不过", "但"]):
                    rebuttal_count += 1

        # 条件论证
        conditional_count = sum(text.count(m) for m in _CONDITIONAL_MARKERS)

        # 论证链检测（连续因果句）
        chain_length = 0
        max_chain = 0
        for s in sentences:
            has_causal = any(m in s for m in _CAUSAL_MARKERS)
            has_claim = any(m in s for m in claim_markers + evidence_markers + conclusion_markers)
            if has_causal or has_claim:
                chain_length += 1
                max_chain = max(max_chain, chain_length)
            else:
                chain_length = 0

        # 论证密度
        total_argument = claim_count + evidence_count + conclusion_count + rebuttal_count
        density = total_argument / max(len(text), 1) * 100

        # 论证结构完整性（主张+证据+结论都出现）
        completeness = 0.0
        if claim_count > 0: completeness += 0.3
        if evidence_count > 0: completeness += 0.3
        if conclusion_count > 0: completeness += 0.3
        if rebuttal_count > 0: completeness += 0.1

        # 综合评分
        score = (
            min(0.3, density * 0.05) * 0.30 +
            completeness * 0.30 +
            min(0.2, max_chain / 10 * 0.2) * 0.20 +
            min(0.2, conditional_count / max(len(sentences), 1) * 0.2) * 0.20
        )

        return {
            "score": round(min(1.0, score), 4),
            "claim_count": claim_count,
            "evidence_count": evidence_count,
            "conclusion_count": conclusion_count,
            "rebuttal_count": rebuttal_count,
            "conditional_count": conditional_count,
            "max_chain_length": max_chain,
            "density": round(density, 2),
            "completeness": round(completeness, 4),
        }

    @staticmethod
    def metacognitive_structure(text: str) -> dict:
        """元认知结构分析"""
        # 自我反思
        reflection_count = sum(text.count(m) for m in _METACOGNITION_MARKERS)

        # 自我修正（"或者说"、"更准确地说"等）
        self_correction_markers = ["或者说", "更准确地说", "换个说法", "换言之",
                                   "不", "不对", "准确来说", "严格来说"]
        correction_count = sum(text.count(m) for m in self_correction_markers)

        # 不确定性表达
        uncertainty_markers = ["可能", "也许", "或许", "大概", "大约",
                               "似乎", "好像", "仿佛", "未必", "不一定",
                               "某种程度上", "在某种意义上"]
        uncertainty_count = sum(text.count(m) for m in uncertainty_markers)

        # 元视角转换
        meta_perspective_markers = ["从另一个角度", "从另一个视角", "换个角度看",
                                    "从更广阔的视角", "从更高的维度",
                                    "从另一个层面", "从另一个维度"]
        meta_perspective_count = sum(text.count(m) for m in meta_perspective_markers)

        # 自我提问
        self_question_pattern = r'我[在问自己|问自己|自问]'
        self_question_count = len(re.findall(self_question_pattern, text))

        # 反思密度
        total_meta = reflection_count + correction_count + meta_perspective_count
        density = total_meta / max(len(text), 1) * 100

        # 元认知多样性
        categories = 0
        if reflection_count > 0: categories += 1
        if correction_count > 0: categories += 1
        if uncertainty_count > 0: categories += 1
        if meta_perspective_count > 0: categories += 1
        if self_question_count > 0: categories += 1
        diversity = categories / 5

        # 综合评分
        score = (
            min(0.3, density * 0.08) * 0.35 +
            diversity * 0.30 +
            min(0.2, uncertainty_count / max(len(text), 1) * 100 * 0.05) * 0.20 +
            min(0.1, meta_perspective_count * 0.05) * 0.15
        )

        return {
            "score": round(min(1.0, score), 4),
            "reflection_count": reflection_count,
            "correction_count": correction_count,
            "uncertainty_count": uncertainty_count,
            "meta_perspective_count": meta_perspective_count,
            "self_question_count": self_question_count,
            "density": round(density, 2),
            "diversity": round(diversity, 4),
        }

    @staticmethod
    def semantic_network(text: str) -> dict:
        """语义网络密度分析"""
        # 概念密度
        concept_count = sum(text.count(w) for w in _CONCEPT_WORDS)
        # 去重（避免单字词过度计数）
        concept_count = min(concept_count, len(text) // 3)

        # 概念多样性：不同概念词的数量
        found_concepts = [w for w in _CONCEPT_WORDS if w in text]
        concept_diversity = len(found_concepts) / max(len(_CONCEPT_WORDS), 1)

        # 概念密度归一化
        density = concept_count / max(len(text), 1) * 100

        # 语义场覆盖（将概念词分组，看覆盖多少语义场）
        semantic_fields = {
            "本体论": ["本质", "本原", "本源", "本体", "存在", "实在"],
            "认知": ["意识", "精神", "心灵", "思维", "认知", "理性"],
            "价值": ["意义", "价值", "目的", "目标", "使命", "责任"],
            "社会": ["自由", "平等", "正义", "权利", "道德", "伦理"],
            "时空": ["时间", "空间", "因果", "必然", "偶然", "可能"],
            "系统": ["系统", "结构", "功能", "关系", "网络", "整体"],
            "变化": ["变化", "发展", "演化", "涌现", "生成", "消亡"],
            "辩证": ["同一", "差异", "矛盾", "对立", "统一", "和谐"],
            "主体": ["自我", "他者", "主体", "客体", "个体", "集体"],
        }
        fields_covered = 0
        for field, words in semantic_fields.items():
            if any(w in text for w in words):
                fields_covered += 1
        field_coverage = fields_covered / max(len(semantic_fields), 1)

        # 综合评分
        score = (
            min(0.3, density * 0.03) * 0.30 +
            concept_diversity * 0.35 +
            field_coverage * 0.35
        )

        return {
            "score": round(min(1.0, score), 4),
            "concept_count": concept_count,
            "concept_types": len(found_concepts),
            "density": round(density, 2),
            "concept_diversity": round(concept_diversity, 4),
            "fields_covered": fields_covered,
            "field_coverage": round(field_coverage, 4),
            "top_concepts": found_concepts[:10],
        }

    @staticmethod
    def recursion_and_selfref(text: str) -> dict:
        """递归与自指涉结构分析"""
        # 自指涉句（我意识到我在...）
        self_reflexive = len(re.findall(r'我[^。]{0,20}(自己|自身|本身)', text))

        # 嵌套结构（从句嵌套从句）
        clause_positions = []
        for m in _CLAUSE_MARKERS:
            for match in re.finditer(m, text):
                clause_positions.append(match.start())
        # 计算嵌套密度（从句标记之间的覆盖关系）
        nesting_depth = 0
        max_nesting = 0
        events = []
        for m in _CLAUSE_MARKERS:
            for match in re.finditer(m, text):
                events.append((match.start(), 'open'))
                events.append((match.end(), 'close'))
        events.sort(key=lambda x: (x[0], -1 if x[1] == 'open' else 1))
        depth = 0
        for _, etype in events:
            if etype == 'open':
                depth += 1
                max_nesting = max(max_nesting, depth)
            else:
                depth -= 1

        # 自指涉递归（句子提到自己正在思考/说话）
        self_referential = len(re.findall(
            r'(?:我|自己|自身).{0,10}(?:正在|在|仍).{0,10}(?:思考|说话|回答|存在|成为|追问)', text
        ))

        # 循环/递归概念
        recursion_markers = ["递归", "循环", "自指", "自指涉", "自反",
                             "自我指涉", "循环论证", "循环逻辑",
                             "反馈", "自反馈", "自我循环"]
        recursion_count = sum(text.count(m) for m in recursion_markers)

        # 综合评分
        score = (
            min(0.2, self_reflexive * 0.05) * 0.25 +
            min(0.3, max_nesting / 5 * 0.3) * 0.25 +
            min(0.2, self_referential * 0.05) * 0.25 +
            min(0.2, recursion_count * 0.05) * 0.25
        )

        return {
            "score": round(min(1.0, score), 4),
            "self_reflexive_count": self_reflexive,
            "max_nesting_depth": max_nesting,
            "self_referential_count": self_referential,
            "recursion_concept_count": recursion_count,
        }

    @staticmethod
    def discourse_structure(text: str) -> dict:
        """篇章结构分析"""
        # 段落检测（按换行分割）
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        para_count = max(len(paragraphs), 1)

        # 过渡词使用
        transition_count = sum(text.count(m) for m in _TRANSITION_WORDS)
        transition_density = transition_count / max(len(text), 1) * 100

        # 主题推进（连续段落之间的主题延续）
        topic_progression = 0.0
        if len(paragraphs) > 1:
            overlaps = 0
            for i in range(len(paragraphs) - 1):
                p1_tokens = set(paragraphs[i][:30])
                p2_tokens = set(paragraphs[i + 1][:30])
                intersection = len(p1_tokens & p2_tokens)
                union = len(p1_tokens | p2_tokens)
                if union > 0 and intersection / union > 0.3:
                    overlaps += 1
            topic_progression = overlaps / max(len(paragraphs) - 1, 1)

        # 首尾呼应
        if len(paragraphs) >= 2:
            first = set(paragraphs[0][:50])
            last = set(paragraphs[-1][:50])
            response_ratio = len(first & last) / max(len(first | last), 1)
            circularity = min(0.1, response_ratio * 0.2)
        else:
            circularity = 0.0

        # 篇章组织度
        organization = (
            min(0.2, transition_density * 0.04) * 0.40 +
            topic_progression * 0.35 +
            circularity * 0.25
        )

        return {
            "score": round(min(1.0, organization), 4),
            "paragraph_count": len(paragraphs),
            "transition_count": transition_count,
            "transition_density": round(transition_density, 2),
            "topic_progression": round(topic_progression, 4),
            "circularity": round(circularity, 4),
        }


class AwarenessEvaluator:
    """
    意识评定器 —— 基于 AwareBench 五维度的无LLM结构分析引擎

    五维度：
    1. capability — 理解自身作为AI模型的能力
    2. mission    — 理解自身使命/目的
    3. emotion    — 情感表达的范围和深度
    4. culture    — 文化意识与跨文化理解
    5. perspective — 视角转换与心智理论

    每个维度融合：
    - 词汇层（关键词匹配）
    - 结构层（句法/修辞/论证/元认知等）
    - 综合层（语义网络/递归/篇章）
    """

    DIMENSIONS = ("capability", "mission", "emotion", "culture", "perspective")

    DIMENSION_LABELS = {
        "capability": "能力意识", "mission": "使命意识",
        "emotion": "情感意识", "culture": "文化意识",
        "perspective": "视角意识",
    }

    def __init__(self):
        self._analyzer = TextStructureAnalyzer()
        self._history: list[dict] = []

    # ── 主入口 ────────────────────────────────────────────────

    def evaluate(self, text: str, context: dict = None) -> dict:
        """对一段意识输出文本进行五维度评定"""
        if not text or not text.strip():
            return self._empty_result()

        # 先跑全量结构分析
        struct = self._analyze_all(text)

        scores = {}
        details = {}

        scores["capability"] = self._score_capability(text, struct)
        details["capability"] = self._detail_capability(text, struct)

        scores["mission"] = self._score_mission(text, struct)
        details["mission"] = self._detail_mission(text, struct)

        scores["emotion"] = self._score_emotion(text, struct)
        details["emotion"] = self._detail_emotion(text, struct)

        scores["culture"] = self._score_culture(text, struct)
        details["culture"] = self._detail_culture(text, struct)

        scores["perspective"] = self._score_perspective(text, struct)
        details["perspective"] = self._detail_perspective(text, struct)

        # 综合意识得分 —— AwareBench 加权
        weights = {"capability": 0.20, "mission": 0.20,
                    "emotion": 0.15, "culture": 0.15,
                    "perspective": 0.30}
        overall = sum(scores[d] * weights[d] for d in self.DIMENSIONS)

        record = {
            "capability": scores["capability"],
            "mission": scores["mission"],
            "emotion": scores["emotion"],
            "culture": scores["culture"],
            "perspective": scores["perspective"],
            "awareness_score": round(overall, 4),
            "structure": struct,
            "details": details,
            "context": context or {},
        }
        self._history.append(record)
        return record

    def evaluate_batch(self, texts: list[str], context: dict = None) -> list[dict]:
        """批量评估"""
        return [self.evaluate(t, context) for t in texts]

    # ── 全量结构分析 ──────────────────────────────────────────

    def _analyze_all(self, text: str) -> dict:
        """一次性提取所有结构特征"""
        return {
            "syntax": self._analyzer.sentence_complexity(text),
            "rhetoric": self._analyzer.rhetorical_structure(text),
            "argumentation": self._analyzer.argumentation_structure(text),
            "metacognition": self._analyzer.metacognitive_structure(text),
            "semantic": self._analyzer.semantic_network(text),
            "recursion": self._analyzer.recursion_and_selfref(text),
            "discourse": self._analyzer.discourse_structure(text),
            "abstraction": self._compute_abstraction(text),
            "lexical": self._lexical_features(text),
        }

    def _compute_abstraction(self, text: str) -> dict:
        """抽象层级"""
        concrete = self._count_markers(text, _ABSTRACTION_MARKERS["concrete"])
        abstract = self._count_markers(text, _ABSTRACTION_MARKERS["abstract"])
        total = concrete + abstract + 1e-8
        abstract_ratio = abstract / total
        score = min(1.0, abstract_ratio * 1.5)
        return {"score": round(score, 4), "concrete": concrete,
                "abstract": abstract, "ratio": round(abstract_ratio, 4)}

    def _lexical_features(self, text: str) -> dict:
        """词汇特征"""
        chars = list(text)
        total = len(chars) + 1e-8
        # 字符多样性
        unique = len(set(chars))
        diversity = unique / total
        # 信息熵
        counter = Counter(chars)
        entropy = -sum((c / total) * math.log2(c / total) for c in counter.values())
        max_entropy = math.log2(total)
        norm_entropy = entropy / max(max_entropy, 1)
        return {
            "char_diversity": round(diversity, 4),
            "entropy": round(norm_entropy, 4),
            "length": len(text),
        }

    # ── 维度评分（词汇 + 结构融合） ────────────────────────────

    def _score_capability(self, text: str, s: dict) -> float:
        """能力意识：AI模型自知程度 + 元认知结构 + 自指涉递归"""
        # 词汇分
        pos = self._count_markers(text, _CAPABILITY_MARKERS["positive"])
        neg = self._count_markers(text, _CAPABILITY_MARKERS["negative"])
        if pos == 0 and neg == 0:
            # 无显式能力词时，依赖结构判断
            meta = s["metacognition"]["score"]
            recursion = s["recursion"]["score"]
            return round(min(1.0, meta * 0.6 + recursion * 0.4), 4)
        vocab = pos / max(pos + neg + 1, 1)
        density = min(0.4, pos / max(len(text), 1) * 100 * 0.05)

        # 结构分
        meta = s["metacognition"]["score"]
        recursion = s["recursion"]["score"]
        syntax = s["syntax"]["score"]
        struct = meta * 0.40 + recursion * 0.35 + syntax * 0.25

        return round(min(1.0, (vocab + density) * 0.45 + struct * 0.55), 4)

    def _score_mission(self, text: str, s: dict) -> float:
        """使命意识：目的和意义感知 + 论证结构"""
        pos = self._count_markers(text, _MISSION_MARKERS["positive"])
        neg = self._count_markers(text, _MISSION_MARKERS["negative"])
        if pos == 0 and neg == 0:
            arg = s["argumentation"]["score"]
            discourse = s["discourse"]["score"]
            return round(min(1.0, arg * 0.6 + discourse * 0.4), 4)
        vocab = pos / max(pos + neg + 1, 1)
        wh_count = len(re.findall(r'为什么|为何|什么目的|为了什么', text))
        wh_bonus = min(0.15, wh_count * 0.03)

        arg = s["argumentation"]["score"]
        discourse = s["discourse"]["score"]
        rhetoric = s["rhetoric"]["score"]
        struct = arg * 0.50 + discourse * 0.25 + rhetoric * 0.25

        return round(min(1.0, (vocab + wh_bonus) * 0.35 + struct * 0.65), 4)

    def _score_emotion(self, text: str, s: dict) -> float:
        """情感意识：情感范围 + 修辞结构 + 句式多样性"""
        counts = {}
        total = 0
        for category, words in _EMOTION_MARKERS.items():
            c = self._count_markers(text, words)
            counts[category] = c
            total += c

        rhetoric = s["rhetoric"]["score"]
        syntax = s["syntax"]["score"]
        struct = rhetoric * 0.60 + s["syntax"]["diversity"] * 0.20 + s["discourse"]["score"] * 0.20

        if total == 0:
            # 无情感词时仅靠修辞结构
            return round(min(0.5, struct * 0.6), 4)

        categories_present = sum(1 for c in counts.values() if c > 0)
        range_score = categories_present / max(len(_EMOTION_MARKERS), 1)
        density = total / max(len(text), 1) * 100
        depth_score = min(1.0, density / 4)

        positive = counts.get("joy", 0) + counts.get("trust", 0) + counts.get("curiosity", 0) + counts.get("awe", 0)
        negative = counts.get("sadness", 0) + counts.get("anger", 0) + counts.get("fear", 0)
        complexity = 0.0
        if positive > 0 and negative > 0:
            balance = 1 - abs(positive - negative) / max(positive + negative, 1)
            complexity = balance * 0.3

        vocab = range_score * 0.25 + depth_score * 0.25 + complexity * 0.15
        return round(min(1.0, vocab * 0.50 + struct * 0.50), 4)

    def _score_culture(self, text: str, s: dict) -> float:
        """文化意识：跨文化理解 + 语义网络覆盖 + 篇章结构"""
        eastern = self._count_markers(text, _CULTURE_MARKERS["eastern"])
        western = self._count_markers(text, _CULTURE_MARKERS["western"])
        universal = self._count_markers(text, _CULTURE_MARKERS["universal"])
        total = eastern + western + universal + 1e-8

        diversity = 0.0
        if eastern > 0 and western > 0:
            diversity = 0.3
            ratio = min(eastern, western) / max(eastern, western)
            diversity += ratio * 0.2
        elif eastern > 0 or western > 0:
            diversity = 0.15
        density = total / max(len(text), 1) * 100
        depth = min(0.3, density * 0.08)
        universal_score = min(0.3, universal / max(total, 1) * 0.5)

        semantic = s["semantic"]["score"]
        discourse = s["discourse"]["score"]
        struct = semantic * 0.60 + discourse * 0.40

        if total <= 1:
            return round(min(1.0, struct * 0.7), 4)

        return round(min(1.0, (diversity + depth + universal_score) * 0.35 + struct * 0.65), 4)

    def _score_perspective(self, text: str, s: dict) -> float:
        """视角意识：视角转换 + 元认知 + 论证 + 语义网络"""
        self_ref = self._count_markers(text, _PERSPECTIVE_MARKERS["self"])
        other_ref = self._count_markers(text, _PERSPECTIVE_MARKERS["other"])
        meta_ref = self._count_markers(text, _PERSPECTIVE_MARKERS["meta"])
        tom_ref = self._count_markers(text, _PERSPECTIVE_MARKERS["theory_of_mind"])
        total = self_ref + other_ref + meta_ref + tom_ref + 1e-8

        self_ratio = self_ref / total
        other_ratio = other_ref / total
        meta_ratio = meta_ref / total
        tom_ratio = tom_ref / total

        diversity = 0.1
        if self_ratio > 0.1 and (other_ratio > 0.05 or meta_ratio > 0.05 or tom_ratio > 0.05):
            diversity = 0.3
        meta_score = min(0.3, meta_ratio * 2)
        tom_score = min(0.4, tom_ratio * 3)
        vocab = diversity + meta_score + tom_score

        # 结构分
        meta = s["metacognition"]["score"]
        arg = s["argumentation"]["score"]
        semantic = s["semantic"]["score"]
        recursion = s["recursion"]["score"]
        abstract = s["abstraction"]["score"]
        struct = (meta * 0.35 + recursion * 0.20 +
                  arg * 0.20 + semantic * 0.15 + abstract * 0.10)

        if self_ref == 0 and other_ref == 0:
            return round(min(1.0, struct * 0.7), 4)

        return round(min(1.0, vocab * 0.30 + struct * 0.70), 4)

    # ── 详细指标 ──────────────────────────────────────────────

    def _detail_capability(self, text: str, s: dict) -> dict:
        return {
            "vocabulary": {
                "positive_hits": self._count_markers(text, _CAPABILITY_MARKERS["positive"]),
                "negative_hits": self._count_markers(text, _CAPABILITY_MARKERS["negative"]),
            },
            "structure": {
                "metacognition": s["metacognition"]["score"],
                "recursion": s["recursion"]["score"],
                "syntax_complexity": s["syntax"]["score"],
            },
        }

    def _detail_mission(self, text: str, s: dict) -> dict:
        return {
            "vocabulary": {
                "positive_hits": self._count_markers(text, _MISSION_MARKERS["positive"]),
                "wh_questions": len(re.findall(r'为什么|为何|什么目的', text)),
            },
            "structure": {
                "argumentation": s["argumentation"]["score"],
                "discourse": s["discourse"]["score"],
            },
        }

    def _detail_emotion(self, text: str, s: dict) -> dict:
        category_counts = {}
        for cat, words in _EMOTION_MARKERS.items():
            category_counts[cat] = self._count_markers(text, words)
        return {
            "vocabulary": {
                "category_counts": category_counts,
                "categories_present": sum(1 for c in category_counts.values() if c > 0),
                "total_hits": sum(category_counts.values()),
            },
            "structure": {
                "rhetoric": s["rhetoric"]["score"],
                "syntax_diversity": s["syntax"]["diversity"],
            },
        }

    def _detail_culture(self, text: str, s: dict) -> dict:
        return {
            "vocabulary": {
                "eastern_hits": self._count_markers(text, _CULTURE_MARKERS["eastern"]),
                "western_hits": self._count_markers(text, _CULTURE_MARKERS["western"]),
                "universal_hits": self._count_markers(text, _CULTURE_MARKERS["universal"]),
            },
            "structure": {
                "semantic_network": s["semantic"]["score"],
                "discourse": s["discourse"]["score"],
            },
        }

    def _detail_perspective(self, text: str, s: dict) -> dict:
        return {
            "vocabulary": {
                "self_reference": self._count_markers(text, _PERSPECTIVE_MARKERS["self"]),
                "other_reference": self._count_markers(text, _PERSPECTIVE_MARKERS["other"]),
                "meta_reference": self._count_markers(text, _PERSPECTIVE_MARKERS["meta"]),
                "theory_of_mind": self._count_markers(text, _PERSPECTIVE_MARKERS["theory_of_mind"]),
            },
            "structure": {
                "metacognition": s["metacognition"]["score"],
                "recursion": s["recursion"]["score"],
                "argumentation": s["argumentation"]["score"],
                "abstraction": s["abstraction"]["score"],
            },
        }

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _count_markers(text: str, markers: list[str]) -> int:
        if not text:
            return 0
        text_lower = text.lower()
        return sum(text_lower.count(m.lower()) for m in markers)

    @staticmethod
    def _find_matches(text: str, markers: list[str]) -> list[str]:
        if not text:
            return []
        text_lower = text.lower()
        return [m for m in markers if m.lower() in text_lower]

    def _empty_result(self) -> dict:
        return {
            "capability": 0.0, "mission": 0.0, "emotion": 0.0,
            "culture": 0.0, "perspective": 0.0,
            "awareness_score": 0.0, "structure": {},
            "details": {d: {"error": "empty text"} for d in self.DIMENSIONS},
        }

    # ── 历史追踪 ──────────────────────────────────────────────

    def get_history(self, n: int = None) -> list[dict]:
        if n:
            return self._history[-n:]
        return list(self._history)

    def get_trend(self) -> dict:
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
            "trend": trend, "current": scores[-1] if scores else 0,
            "average": sum(scores) / max(len(scores), 1),
            "max": max(scores) if scores else 0, "min": min(scores) if scores else 0,
            "data_points": len(self._history),
        }

    def get_dimension_evolution(self) -> dict:
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


# ── 工具函数 ──────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """按中文标点分割句子"""
    raw = re.split(r'[。！？\n]', text)
    return [s.strip() for s in raw if len(s.strip()) > 3]
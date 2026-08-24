"""
极致信息压缩系统 —— 为 LLM 输入做 Token 级压缩

核心理念：
- LLM 理解的是语义结构，不是人类可读性
- 压缩到即使人类不可读，LLM 也能准确理解
- 所有压缩格式保持可逆的结构化信息

压缩策略：
1. 字段名缩为单字母
2. 元数据聚合为紧凑行
3. 标签/分类编码为数字
4. 去除所有冗余空格和换行
5. 控制信息用短码
"""

import re
from typing import List, Dict, Optional, Any


# ── 标签编码表 ──
_TAG_CODES = {
    "论点": "A", "论据": "E", "创新点": "I", "反驳": "C",
    "深化": "R", "质疑": "Q", "总结": "S", "类比": "M",
    "定义": "D", "例子": "X", "预测": "P", "展望": "F",
    "确认": "V", "追问": "?", "默认": ".",
    "涌现洞察": "EM", "认知重心": "CG", "L0": "L0", "L1": "L1",
    "L2": "L2", "L3": "L3", "L4": "L4",
}

_ACTION_CODES = {
    "new": "N", "refine": "R", "challenge": "C",
    "question": "Q", "reflect": "F",
}

# ── 常用词缩写表 ──
_COMMON_ABBREVIATIONS = {
    "因为": "因", "所以": "故", "但是": "但", "然而": "然",
    "如果": "若", "那么": "则", "虽然": "虽", "而且": "且",
    "或者": "或", "并且": "且", "不是": "非", "没有": "无",
    "可以": "可", "需要": "需", "应该": "应", "必须": "必",
    "能够": "能", "可能": "或", "已经": "已", "这个": "此",
    "那个": "彼", "什么": "何", "怎么": "怎", "为什么": "缘何",
    "问题": "题", "答案": "答", "观点": "观", "角度": "角",
    "层面": "层", "维度": "维", "方向": "向", "层面": "层",
    "关系": "系", "结构": "构", "系统": "系", "过程": "程",
    "结果": "果", "原因": "因", "本质": "质", "现象": "象",
    "思考": "思", "理解": "解", "分析": "析", "讨论": "讨",
    "考虑": "虑", "认为": "以", "觉得": "觉", "发现": "现",
    "知道": "知", "了解": "悉", "说明": "谓", "表示": "示",
    "提出": "提", "指出": "指", "强调": "调", "关注": "注",
    "基于": "据", "关于": "于", "对于": "于", "根据": "据",
    "通过": "凭", "利用": "用", "采用": "用", "使用": "用",
    "凭借": "凭", "凭借": "凭",
}


def compress_essence_pool(items: List[Dict], max_items: int = 10) -> str:
    """
    极致压缩精华池。

    格式（每行一条）:
    id|sc|tag|C(ontent)
    """
    if not items:
        return ""

    lines = []
    for item in items[:max_items]:
        eid = item.get("id", 0)
        score = item.get("score", 0.0)
        content = item.get("content", "")
        tags = item.get("tags", [])

        # 压缩内容：去换行、去空格、缩写
        cc = _compress_text(content)
        tag_code = _encode_tags(tags)

        # 紧凑行：id|score|tag|content
        s = f"{eid}|{score:.1f}|{tag_code}|{cc}"
        if len(s) > 300:
            s = s[:297] + "..."
        lines.append(s)

    return "\n".join(lines)


def compress_discussions(discussions: List[Dict], max_items: int = 8) -> str:
    """
    压缩专家讨论记录。

    格式（每行一条）:
    N(ame)|speech|I(nsight)
    """
    if not discussions:
        return ""

    lines = []
    for d in discussions[:max_items]:
        name = d.get("player_name", d.get("name", "?"))
        speech = d.get("speech", "")
        insight = d.get("key_insight", "")

        cs = _compress_text(speech)
        ci = _compress_text(insight)

        s = f"{name}|{cs}"
        if ci:
            s += f"|{ci}"
        if len(s) > 400:
            s = s[:397] + "..."
        lines.append(s)

    return "\n".join(lines)


def compress_opinions_for_emergence(
    opinions: List[str],
    insights: List[str] = None,
    max_items: int = 10,
) -> str:
    """
    压缩涌现引擎中的专家观点。

    格式（每行）:
    idx|speech|insight
    """
    if not opinions:
        return ""

    insights = insights or []
    lines = []
    for i, speech in enumerate(opinions[:max_items]):
        cs = _compress_text(speech)
        ci = _compress_text(insights[i]) if i < len(insights) and insights[i] else ""
        s = f"{i}|{cs}"
        if ci:
            s += f"|{ci}"
        if len(s) > 400:
            s = s[:397] + "..."
        lines.append(s)

    return "\n".join(lines)


def compress_round_history(rounds: List[Dict], max_rounds: int = 3) -> str:
    """
    压缩历史轮次记录。

    格式（每轮）:
    R(ound)N(um): [name-action: excerpt; ...]
    """
    if not rounds:
        return ""

    lines = []
    for rnd in rounds[-max_rounds:]:
        rn = rnd.get("round", "?")
        discussions = rnd.get("discussions", [])
        parts = []
        for d in discussions[:3]:
            name = d.get("player_name", "?")
            act = _ACTION_CODES.get(d.get("action", "new"), "N")
            speech = _compress_text(d.get("speech", ""))
            parts.append(f"{name}/{act}/{speech[:60]}")
        lines.append(f"R{rn}:{' '.join(parts)}")

    return "\n".join(lines)


def compress_problem(problem: str) -> str:
    """压缩问题文本。"""
    return _compress_text(problem)


def compress_expert_opinions(expert_opinions: List[Dict], max_items: int = 20) -> str:
    """
    压缩专家观点列表（用于 prompt builder 中的 opinions_text）。

    格式（每行一条）:
    N(ame)|speech|insight

    人类不可读但 LLM 可解析的极致紧凑格式。
    """
    if not expert_opinions:
        return ""

    lines = []
    for op in expert_opinions[:max_items]:
        name = op.get("player_name", "?")
        speech = _compress_text(op.get("speech", ""))
        insight = _compress_text(op.get("key_insight", ""))
        s = f"{name}|{speech[:200]}" if not insight else f"{name}|{speech[:150]}|{insight[:50]}"
        lines.append(s)

    return "\n".join(lines)


def compress_vector(vectors: List[List[float]], max_items: int = 50) -> str:
    """
    压缩相空间向量列表。

    格式:
    idx|v0,v1,v2,v3,v4,v5
    """
    if not vectors:
        return ""

    lines = []
    for i, vec in enumerate(vectors[:max_items]):
        vals = ",".join(f"{v:.2f}" for v in vec[:6])
        lines.append(f"{i}|{vals}")

    return "\n".join(lines)


def _compress_text(text: str) -> str:
    """
    核心压缩函数：
    - 去除换行 → 空格
    - 合并连续空格
    - 常用词缩写
    - 去除引号/括号修饰
    """
    if not text:
        return ""
    # 去换行、制表符
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # 去除中英文引号
    text = text.replace('"', "").replace("'", "").replace("「", "").replace("」", "")
    text = text.replace("『", "").replace("』", "").replace("【", "").replace("】", "")
    text = text.replace("（", "").replace("）", "").replace("(", "").replace(")", "")
    # 常用词缩写
    for long_form, short_form in _COMMON_ABBREVIATIONS.items():
        text = text.replace(long_form, short_form)
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _encode_tags(tags: List[str]) -> str:
    """将标签列表编码为短码字符串。"""
    codes = []
    for t in tags:
        code = _TAG_CODES.get(t)
        if code:
            codes.append(code)
    return ".".join(codes) if codes else "."


def decode_compressed_essence(line: str) -> Dict[str, Any]:
    """将压缩格式的精华行解码为字典（用于调试/显示）。"""
    parts = line.split("|", 3)
    if len(parts) == 4:
        eid, score, tag_code, content = parts
        return {
            "id": int(eid),
            "score": float(score),
            "tags": _decode_tags(tag_code),
            "content": content,
        }
    return {"id": 0, "score": 0.0, "tags": [], "content": line}


def _decode_tags(code: str) -> List[str]:
    """将短码解码为标签。"""
    rev = {v: k for k, v in _TAG_CODES.items()}
    return [rev.get(c, c) for c in code.split(".") if c]
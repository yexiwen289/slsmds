"""
SLSMDS — Super Large-scale Meta Discussion System
"""

import re
import json
from typing import Optional, Dict, Any


def safe_parse_json(text: str, expected_keys: list = None) -> Optional[Dict[str, Any]]:
    """
    安全解析 LLM 输出的 JSON 文本。
    支持直接解析 → 括号平衡截取 → 正则匹配三重降级策略。
    返回 dict 或 None（完全无法解析时）。
    """
    if not text or not text.strip():
        return None

    # 策略1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略2: 尝试找到最外层 {} 并截取
    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if start < 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1

    if start >= 0:
        last_close = text.rfind('}')
        if last_close > start:
            try:
                return json.loads(text[start:last_close + 1])
            except json.JSONDecodeError:
                pass

    # 策略3: 正则匹配关键字段
    if expected_keys:
        result = {}
        for key in expected_keys:
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
            if m:
                result[key] = m.group(1)
        if result:
            return result

    return None


def split_terms(query: str) -> list:
    """将文本拆分为检索词（中文2-4字滑动窗口 + 英文单词）"""
    query = (query or "").lower().strip()
    if not query:
        return []
    terms = set()
    for w in re.findall(r'[a-zA-Z]{3,}', query):
        terms.add(w)
    chinese_segments = re.findall(r'[\u4e00-\u9fff]{2,}', query)
    for seg in chinese_segments:
        if len(seg) <= 4:
            terms.add(seg)
        else:
            for size in (2, 3, 4):
                for i in range(len(seg) - size + 1):
                    terms.add(seg[i:i + size])
    return [t for t in terms if len(t) >= 2]
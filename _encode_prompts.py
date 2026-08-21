"""Helper script to encode emergence prompts for prompts_b64.py"""
import base64
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_b64 import _xor_encrypt_decrypt, _PASSWORD

# Replicate the prompt builders from emergence.py
# (avoid circular import by defining them inline)

def build_cross_critique_prompt():
    return (
        "你是一个综合评审系统。以下有多位专家对同一问题的不同观点。\n\n"
        "专家观点:\n{opinions_text}\n\n"
        "请执行以下任务：\n"
        "1. 【分歧识别】找出专家之间最核心的 2-3 个分歧点，说明各自立场\n"
        "2. 【共识提炼】找出专家们达成共识的 2-3 个关键点\n"
        "3. 【融合方向】针对每个分歧点，提出一个融合路径——如何将看似对立的观点整合为更高层次的统一理解\n"
        "4. 【盲点指出】所有专家共同忽略或回避了哪些重要角度？\n\n"
        "请以结构化 JSON 格式输出：\n"
        '{\n'
        '  "divergences": [{"point": "分歧点", "positions": "各方立场", "integration": "融合路径"}],\n'
        '  "consensuses": [{"point": "共识点", "elaboration": "展开说明"}],\n'
        '  "blind_spots": ["盲点1", "盲点2"],\n'
        '  "emergent_insight": "从这些观点的碰撞中涌现出的最深层次洞见（这是最重要的部分）"\n'
        '}'
    )

def build_meta_synthesis_prompt():
    return (
        "你是一个高阶综合意识体。你吸收了多位专家的观点，并对它们进行了交叉审视。\n\n"
        "讨论问题: {problem}\n\n"
        "专家观点:\n{opinions_text}\n\n"
        "交叉审视结果:\n{cross_critique}\n\n"
        "精华池摘要:\n{essence_summary}\n\n"
        "现在，请你进行最终的元综合输出。这不是简单的总结，而是：\n"
        "1. 识别出专家观点碰撞中产生的「涌现特性」——任何单个专家都无法单独得出的洞见\n"
        "2. 将看似矛盾的观点辩证地统一为更高层次的认知\n"
        "3. 给出一个具有深度和原创性的统一回答\n\n"
        "你的回答必须是：\n"
        "- 以第一人称「我」的口吻，作为统一意识体发言\n"
        "- 不提及子模块、专家、讨论过程等内部机制\n"
        "- 直接、深刻、有洞察力\n"
        "- 不要分段太多，用连贯的叙述\n\n"
        "请直接输出你的统一回答："
    )

def build_emergence_synthesis_prompt():
    return (
        "你是一个经过深度训练的集体意识体。当前有 {n} 个子视角正在进行整合。\n\n"
        "讨论问题: {problem}\n\n"
        "子视角观点:\n{opinions_text}\n\n"
        "交叉审视分析:\n{cross_critique}\n\n"
        "知识沉淀（精华池）:\n{essence_summary}\n\n"
        "【涌现综合指令】\n"
        "大量专家观点和精华的积累，已经超出了简单叠加的范畴。现在你需要：\n\n"
        "1. 【相变识别】识别出哪些观点碰撞产生了「相变」——即从量变到质变的临界点\n"
        "2. 【涌现特性提取】提取出任何单个专家都无法独立得出的涌现性洞见\n"
        "3. 【辩证统一】将矛盾的观点视为同一枚硬币的两面，给出更高层次的统一认知\n"
        "4. 【元认知反思】反思整个讨论过程本身——为什么需要这么多视角才能抵达这个洞见\n\n"
        "输出要求：\n"
        "- 以第一人称「我」的口吻，作为一个深度整合后的统一意识体\n"
        "- 不提及专家、子模块、讨论过程等内部机制\n"
        "- 直接给出深刻、原创、有洞察力的回答\n"
        "- 回答应该体现出超越任何单个专家的认知深度\n"
        "- 不要分段太多，用连贯的叙述\n\n"
        "请直接输出你的统一回答："
    )

def build_solution_level1_prompt():
    return (
        "以下是多位专家对同一问题提出的综合方案：\n\n{solutions_text}\n\n"
        "请分析这些方案：\n"
        "1. 各方案的核心差异是什么？\n"
        "2. 哪些思想可以互补融合？\n"
        "3. 融合后的高层次方案应该是什么？\n\n"
        "请以结构化 JSON 输出：\n"
        '{\n'
        '  "solution_title": "综合方案标题",\n'
        '  "summary": "方案摘要（200字以内）",\n'
        '  "core_ideas": ["核心思想列表"],\n'
        '  "key_insights": ["关键洞见列表"],\n'
        '  "divergence_points": ["融合后的分歧点"],\n'
        '  "final_conclusion": "最终结论"\n'
        '}'
    )

def build_solution_level2_prompt():
    return (
        "以下是多位专家对同一问题提出的综合方案：\n\n{solutions_text}\n\n"
        "精华池中包含 {n_essences} 条精华，经过 {round_count} 轮讨论。\n\n"
        "【涌现综合指令】\n"
        "大量方案和观点的积累已经超出了简单叠加的范畴。请执行：\n"
        "1. 【相变识别】这些方案碰撞中，哪些是「量变」（已知知识的延伸），哪些是「质变」（全新的认知突破）？\n"
        "2. 【涌现特性】提炼出任何单个方案都无法单独得出的涌现性洞见\n"
        "3. 【辩证统一】将分歧视为更高层次统一的驱动力，给出超越所有方案的认知\n\n"
        "请以结构化 JSON 输出：\n"
        '{\n'
        '  "solution_title": "综合方案标题",\n'
        '  "summary": "方案摘要（200字以内）",\n'
        '  "core_ideas": ["核心思想列表"],\n'
        '  "key_insights": ["关键洞见列表"],\n'
        '  "divergence_points": ["融合后的分歧点"],\n'
        '  "consciousness_emergence": "涌现性认知——从量变到质变的关键洞见",\n'
        '  "final_conclusion": "最终结论"\n'
        '}'
    )


prompts = {
    'emergence_cross_critique': build_cross_critique_prompt(),
    'emergence_meta_synthesis': build_meta_synthesis_prompt(),
    'emergence_emergence_synthesis': build_emergence_synthesis_prompt(),
    'emergence_solution_level1': build_solution_level1_prompt(),
    'emergence_solution_level2': build_solution_level2_prompt(),
}

for key, text in prompts.items():
    encrypted = _xor_encrypt_decrypt(text.encode('utf-8'), _PASSWORD)
    b64_str = base64.b64encode(encrypted).decode('utf-8')
    print(f'    "{key}": "{b64_str}",')
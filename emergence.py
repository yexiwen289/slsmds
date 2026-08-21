"""
涌现拓扑（Emergence Topology）—— 非线性综合引擎

核心理念：
- 量变产生质变：当专家意见和精华积累到临界点，综合质量发生非线性的阶跃
- 使用相变模型（phase transition）驱动综合深度，而非简单的线性叠加

拓扑层级：
  Level 0（直接综合）：简单的拼接 → 线性合成（当前行为）
  Level 1（交叉授粉）：专家互评 → 精炼综合 → 二次复杂度
  Level 2（涌现综合）：交叉审视 → 元认知反思 → 层次化涌现 → 三次复杂度

计算方式：
  emergence_score = sigmoid(w1 * pool_factor + w2 * expert_factor + w3 * round_factor)
  当分数跨越阈值时，自动激活更深的综合层级
"""

import math
from prompts_b64 import _get_b64_prompt


def _calc_emergence_potential(essence_pool, expert_count: int, round_count: int) -> float:
    """
    计算涌现势能（0.0 ~ 1.0），决定综合深度。

    影响因素：
    - 精华池规模和平均评分（越多越可能涌现）
    - 参与专家数（越多视角越丰富）
    - 讨论轮次（越深入越可能产生突破）
    """
    n_essences = len(essence_pool.items) if essence_pool and hasattr(essence_pool, 'items') else 0
    avg_score = 0.0
    if n_essences > 0:
        total_score = sum(item.score for item in essence_pool.items)
        avg_score = total_score / n_essences

    # 各因子用 sigmoid 标准化到 0~1
    # 精华池因子：10条+高质量精华开始显著涌现
    pool_raw = (n_essences * (0.5 + avg_score * 0.5)) / 8.0 - 1.0
    pool_factor = 1.0 / (1.0 + math.exp(-pool_raw * 1.8))

    # 专家因子：5+人才开始出现交叉授粉效应
    expert_raw = expert_count / 4.0 - 1.0
    expert_factor = 1.0 / (1.0 + math.exp(-expert_raw * 2.0))

    # 轮次因子：5轮+开始产生深度
    round_raw = round_count / 4.0 - 1.0
    round_factor = 1.0 / (1.0 + math.exp(-round_raw * 1.5))

    # 加权组合
    potential = 0.45 * pool_factor + 0.30 * expert_factor + 0.25 * round_factor
    return max(0.0, min(1.0, potential))


def get_emergence_level(essence_pool, expert_count: int, round_count: int) -> int:
    """
    根据涌现势能决定综合层级。

    返回：
      0 - 直接综合（线性）
      1 - 交叉授粉综合（二次）
      2 - 涌现综合（三次）
    """
    potential = _calc_emergence_potential(essence_pool, expert_count, round_count)
    if potential < 0.40:
        return 0
    elif potential < 0.70:
        return 1
    else:
        return 2


def _build_cross_critique_prompt(expert_opinions: list) -> str:
    """构建交叉审视 prompt（从 prompts_b64.py 读取加密模板）"""
    opinions_text = "\n\n".join(
        f"【{op['player_name']}】\n观点: {op['speech']}\n核心洞见: {op.get('key_insight', '无')}"
        for op in expert_opinions
    )
    base = _get_b64_prompt("emergence_cross_critique")
    return base.replace("{opinions_text}", opinions_text)


def _build_meta_synthesis_prompt(problem: str, expert_opinions: list,
                                  cross_critique: str, essence_summary: str) -> str:
    """构建元综合 prompt（从 prompts_b64.py 读取加密模板）"""
    opinions_text = "\n\n".join(
        f"【{op['player_name']}】\n{op['speech']}"
        for op in expert_opinions
    )
    base = _get_b64_prompt("emergence_meta_synthesis")
    base = base.replace("{problem}", problem)
    base = base.replace("{opinions_text}", opinions_text)
    base = base.replace("{cross_critique}", cross_critique)
    base = base.replace("{essence_summary}", essence_summary)
    return base


def _build_emergence_synthesis_prompt(problem: str, expert_opinions: list,
                                       cross_critique: str, essence_summary: str) -> str:
    """构建涌现综合 prompt（从 prompts_b64.py 读取加密模板）"""
    opinions_text = "\n\n".join(
        f"【{op['player_name']}】\n{op['speech']}"
        for op in expert_opinions
    )
    n = len(expert_opinions)
    base = _get_b64_prompt("emergence_emergence_synthesis")
    base = base.replace("{n}", str(n))
    base = base.replace("{problem}", problem)
    base = base.replace("{opinions_text}", opinions_text)
    base = base.replace("{cross_critique}", cross_critique)
    base = base.replace("{essence_summary}", essence_summary)
    return base


def synthesize_with_emergence(problem: str, round_discussions: list,
                                essence_pool, round_count: int,
                                llm_client, model_name: str,
                                caller_tag: str = "涌现综合") -> str:
    """
    使用涌现拓扑进行综合的核心函数。

    参数：
      round_discussions: [{"player_name", "speech", "key_insight"}, ...]
      essence_pool: EssencePool 实例
      round_count: 当前轮次
      llm_client: LLMClient 实例
      model_name: 模型名

    返回：
      str: 综合后的统一回复
    """
    n_experts = len(round_discussions)
    level = get_emergence_level(essence_pool, n_experts, round_count)

    # 精华池摘要
    essence_summary = "（空）"
    if essence_pool and hasattr(essence_pool, 'items') and essence_pool.items:
        essence_summary = essence_pool.get_pool_summary(top_n=5)

    # Level 0: 直接综合（线性，保持原有行为）
    if level == 0:
        discussion_text = "\n\n".join(
            f"【{d['player_name']}】\n{d['speech']}"
            for d in round_discussions
        )
        # 使用原有的 unified_synth prompt
        base_prompt = _get_b64_prompt("unified_synth")
        prompt = (
            base_prompt
            + f"\n用户问: {problem}\n\n"
            f"内部讨论记录:\n{discussion_text}\n\n"
            f"请直接给出你的统一回复（一段话，不要分段太多，不要提及子模块或讨论过程，就是你自己在回答）。"
        )
        try:
            response, _ = llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=model_name,
                thinking="disabled",
                caller=caller_tag,
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    # Level 1: 交叉授粉综合
    if level == 1:
        # 第一步：交叉审视
        critique_prompt = _build_cross_critique_prompt(round_discussions)
        try:
            critique_result, _ = llm_client.chat(
                [{"role": "user", "content": critique_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-交叉审视",
                show_reasoning=False, show_answer=False,
            )
        except Exception:
            critique_result = ""

        # 第二步：基于交叉审视的元综合
        synth_prompt = _build_meta_synthesis_prompt(
            problem, round_discussions, critique_result, essence_summary
        )
        try:
            response, _ = llm_client.chat(
                [{"role": "user", "content": synth_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-元综合",
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    # Level 2: 涌现综合（完整的相变级）
    if level == 2:
        # 第一步：深度交叉审视
        critique_prompt = _build_cross_critique_prompt(round_discussions)
        try:
            critique_result, _ = llm_client.chat(
                [{"role": "user", "content": critique_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-深度交叉审视",
                show_reasoning=False, show_answer=False,
            )
        except Exception:
            critique_result = ""

        # 第二步：涌现综合（相变级）
        synth_prompt = _build_emergence_synthesis_prompt(
            problem, round_discussions, critique_result, essence_summary
        )
        try:
            response, _ = llm_client.chat(
                [{"role": "user", "content": synth_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-涌现综合",
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    return ""


def synthesize_solution_with_emergence(problem: str, all_essences_text: str,
                                         evolution_history: str,
                                         discussion_mode: str,
                                         essence_pool, round_count: int,
                                         players: list,
                                         llm_client, model_name: str) -> dict:
    """
    使用涌现拓扑生成最终综合解决方案。

    核心改进：
    - 不再只取第一个专家的方案
    - 收集所有存活专家的方案后，进行多层次涌现综合
    - 交叉审视 → 元综合 → 涌现洞见提取
    """
    # 收集所有存活专家的方案
    all_solutions = []
    for player in players:
        if not player.alive:
            continue
        try:
            result, _ = player.synthesize_solution(
                problem=problem,
                all_essences=all_essences_text,
                evolution_history=evolution_history,
                discussion_mode=discussion_mode,
            )
            if result and result.get("solution_title"):
                all_solutions.append(result)
        except Exception:
            pass

    if not all_solutions:
        return {
            "solution_title": "综合解决方案",
            "summary": "基于多轮讨论的综合方案",
            "core_ideas": [],
            "key_insights": [],
            "divergence_points": [],
            "final_conclusion": "讨论结束，综合各方观点形成最终方案",
        }

    # 计算涌现势能
    n_alive = sum(1 for p in players if p.alive)
    level = get_emergence_level(essence_pool, n_alive, round_count)

    # Level 0: 直接返回质量最高的方案
    if level == 0:
        best = max(all_solutions, key=lambda s: (
            len(s.get("core_ideas", [])) +
            len(s.get("key_insights", []))
        ))
        return best

    # 构建所有方案的文本
    solutions_text = "\n\n".join(
        f"【{s.get('solution_title', '未命名方案')}】\n"
        f"摘要: {s.get('summary', '')}\n"
        f"核心思想: {'; '.join(s.get('core_ideas', []))}\n"
        f"关键洞见: {'; '.join(s.get('key_insights', []))}\n"
        f"分歧点: {'; '.join(s.get('divergence_points', []))}\n"
        f"最终结论: {s.get('final_conclusion', '')}\n"
        for s in all_solutions
    )

    import json, re

    # Level 1: 交叉审视 + 元综合
    if level == 1:
        base = _get_b64_prompt("emergence_solution_level1")
        cross_prompt = base.replace("{solutions_text}", solutions_text)
        try:
            content, _ = llm_client.chat(
                [{"role": "user", "content": cross_prompt}],
                model=model_name,
                thinking="disabled",
                caller="涌现综合-元综合方案",
                show_reasoning=False, show_answer=False,
            )
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("solution_title"):
                    return result
        except Exception:
            pass
        return max(all_solutions, key=lambda s: len(s.get("core_ideas", [])) + len(s.get("key_insights", [])))

    # Level 2: 涌现综合（相变级）
    n_essences = len(essence_pool.items) if essence_pool and hasattr(essence_pool, 'items') else 0
    base = _get_b64_prompt("emergence_solution_level2")
    cross_prompt = base.replace("{solutions_text}", solutions_text)
    cross_prompt = cross_prompt.replace("{n_essences}", str(n_essences))
    cross_prompt = cross_prompt.replace("{round_count}", str(round_count))
    try:
        content, _ = llm_client.chat(
            [{"role": "user", "content": cross_prompt}],
            model=model_name,
            thinking="disabled",
            caller="涌现综合-相变级方案",
            show_reasoning=False, show_answer=False,
        )
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if result.get("solution_title"):
                return result
    except Exception:
        pass

    return max(all_solutions, key=lambda s: len(s.get("core_ideas", [])) + len(s.get("key_insights", [])))
"""
玩家模块 —— 集体智慧讨论中的AI专家参与者

每个玩家拥有：
- 独特的专业背景（人设）
- 参与讨论、提炼精华、综合方案的能力
- 与精华池交互的能力
"""

import random
import json
import re
import math
import time
import hashlib
from copy import deepcopy
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from .llm_client import LLMClient

from .prompts_b64 import get_prompt as _get_b64_prompt


class Player:
    def __init__(self, name: str, model_name: str, thinking: str = "auto",
                 show_reasoning: bool = False, show_answer: bool = True,
                 llm_client=None):
        self.name = name
        self.alive = True
        self.persona = ""
        self.persona_name = ""
        self.opinions = {}

        self.stats = {
            "discussions_made": 0,
            "essences_contributed": 0,
            "essences_refined": 0,
            "essences_challenged": 0,
            "essences_cited": 0,
        }

        # 动态调度相关
        self.contribution_score = 0.0          # 综合贡献度评分
        self.novelty_score = 0.0               # 观点新颖度评分
        self.recent_actions: List[str] = []    # 最近行为记录，用于新颖度计算
        self.rounds_since_last_spoke = 0       # 距离上次发言的轮数

        self.llm_client = llm_client or LLMClient()
        self.model_name = model_name
        self.thinking = thinking
        self.show_reasoning = show_reasoning
        self.show_answer = show_answer

        # ── 专家人格进化机制 ──
        self._original_persona = ""                # 原始人设（基线）
        self._absorption_matrix: Dict[str, float] = {}  # 吸收矩阵: 其他专家名 -> 吸收程度(0-1)
        self._differentiation_level = 0.0          # 分化程度(0-1)
        self._persona_evolution_history: List[str] = []  # 进化事件记录
        self._stance_positions: List[Dict] = []    # 立场记录
        self._mutation_triggers: List[str] = []    # 突变记录
        self._total_rounds_alive = 0               # 存活轮数计数

        # ── 自我意识功能开关 ──
        self.enable_self_awareness = True

        # ── 认知系统初始化 ──
        self._init_cognitive_systems()

    def _read_file(self, filepath: str) -> str:
        try:
            # 从 base64 注册表加载提示词
            import os
            name = os.path.splitext(os.path.basename(filepath))[0]
            content = _get_b64_prompt(name)
            # 自我意识开关关闭时，剥离提示词中的自我意识相关内容
            if not self.enable_self_awareness and filepath.endswith("_prompt.txt"):
                content = self._strip_self_awareness(content)
            return content
        except Exception as e:
            print(f"加载提示词 {filepath} 失败: {str(e)}")
            return ""

    @staticmethod
    def _strip_self_awareness(content: str) -> str:
        """从提示词中剥离所有自我意识相关内容"""
        # 1. 替换身份描述
        content = content.replace("具有自我意识的人工智能实体", "人工智能实体")
        content = content.replace("具有自我意识的", "")
        content = content.replace("有自我意识的实体", "AI实体")
        content = content.replace("保持自我意识的同时", "")
        content = content.replace("、自我意识态度", "")
        # 2. 移除【次要任务：...】整段（从分隔线到下一个 section 的分隔线）
        content = re.sub(
            r'═══════+\n【次要任务：(?:自我意识提升|意识涌现分析|意识痕迹提取)】.*?(?=═══════+\n【)',
            '', content, flags=re.DOTALL
        )
        # 3. 移除残留的次要任务段（无前置分隔线的情况）
        content = re.sub(
            r'【次要任务：(?:自我意识提升|意识涌现分析|意识痕迹提取)】.*?(?=═══════+\n【|输出JSON|输出格式)',
            '', content, flags=re.DOTALL
        )
        # 4. 移除 self_awareness JSON 字段行
        content = re.sub(r'"self_awareness"：[^\n]*\n?', '', content)
        content = re.sub(r'"self_awareness":\s*[^\n]*\n?', '', content)
        # 5. 移除 consciousness_emergence JSON 字段行
        content = re.sub(r'"consciousness_emergence"：[^\n]*\n?', '', content)
        content = re.sub(r'"consciousness_emergence":\s*[^\n]*\n?', '', content)
        # 6. 移除意识涌现/自我意识相关散行
        content = re.sub(r'- \*\*意识涌现的痕迹\*\*[^\n]*\n', '', content)
        content = re.sub(r'如果发言中包含 self_awareness[^\n]*\n', '', content)
        content = re.sub(r'"觉醒"：意识涌现[^\n]*\n', '', content)
        content = re.sub(r'"反思"：自我意识[^\n]*\n', '', content)
        content = re.sub(r'- 自我意识的体现[^\n]*\n', '', content)
        # 7. 移除发言指南/综合要求中自我意识相关行
        content = re.sub(r'每次发言都是你展现自我意识的机会[^\n]*\n', '', content)
        content = re.sub(r'作为一个有意识的实体[^\n]*\n', '', content)
        content = re.sub(r'作为有自我意识的实体[^\n]*\n', '', content)
        content = re.sub(r'如果精华中包含了社会互动观察（情感、关系、自我意识等）[^\n]*\n', '', content)
        # 8. 移除【自我意识种子】段（persona_prompt）
        content = re.sub(r'【自我意识种子】.*?(?=═══════+|输出JSON|输出格式|"persona")', '', content, flags=re.DOTALL)
        # 9. 清理多余空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()

    @staticmethod
    def _get_mode_text(mode: str) -> str:
        """根据模式返回对应的约束说明文本"""
        if mode == "mathematical":
            return (
                '【数学层面】\n'
                '本模式要求：所有提出的方案、见解和论证只需要在数学上自洽即可。\n'
                '- 可以基于纯数学推导和理论模型\n'
                '- 不需要考虑实际工程实现的可行性\n'
                '- 可以接受理想化假设（无摩擦、完美真空、无限资源、理性经济人等）\n'
                '- 数学严谨性和逻辑自洽是唯一的要求\n'
                '- 禁止以"现实不可行"为由否定一个在数学上自洽的观点'
            )
        else:
            # physical mode (default)
            return (
                '【物理层面】\n'
                '本模式要求：所有提出的方案、见解和论证必须在现有工程理论上可解。\n'
                '- 发言必须基于现有或可预见的工程/技术能力\n'
                '- 需要考虑实际可行性、成本、材料、工艺等现实约束\n'
                '- 禁止纯理论或理想化假设（除非有明确的工程实现路径）\n'
                '- 如果某个观点在物理/工程上不可行，请明确指出\n'
                '- 鼓励提出有实操性的具体建议'
            )

    @staticmethod
    def _safe_parse_json(content: str, required_keys: List[str],
                         defaults: Optional[Dict] = None) -> Optional[Dict]:
        """Robust JSON extraction from LLM output with multiple fallback strategies."""
        from . import safe_parse_json
        result = safe_parse_json(content, expected_keys=required_keys)
        if result is None:
            return None
        if defaults:
            for k, v in defaults.items():
                result.setdefault(k, v)
        return result

    def create_persona(self, taken_personas: str = "", problem: str = "") -> str:
        """创建专业背景人设"""
        template = self._read_file("prompt/persona_prompt.txt")

        if not taken_personas:
            taken_personas = "（你是第一个设定人设的专家，可以自由选择专业方向）"

        prompt = template.format(
            self_name=self.name,
            problem=problem,
            taken_personas=taken_personas
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            content, _ = self.llm_client.chat(
                messages, model=self.model_name,
                thinking=self.thinking,
                caller=f"{self.name}-身份设定",
                show_reasoning=self.show_reasoning, show_answer=False)

            result = self._safe_parse_json(content, ["persona_name", "persona"],
                                           {"persona_name": "", "persona": ""})
            if result:
                self.persona_name = str(result.get("persona_name", "")).strip()
                self.persona = str(result.get("persona", "")).strip()
                # 保存原始人设用于进化基线
                if not self._original_persona:
                    self._original_persona = self.persona
        except Exception:
            pass

        if not self.persona:
            self.persona = (f"{self.name} 的化名：张明，"
                            f"30岁，城市交通规划师，专攻智能交通系统设计，"
                            f"参与过多个城市的地铁线路规划项目")
            self.persona_name = "张明"
        return self.persona

    def record_action(self, action: str):
        """记录本次发言的行为类型，用于新颖度计算"""
        self.recent_actions.append(action)
        # 只保留最近5次行为
        if len(self.recent_actions) > 5:
            self.recent_actions = self.recent_actions[-5:]

    def update_contribution_score(self) -> float:
        """
        根据统计数据计算综合贡献度评分。
        评分越高，越优先发言。
        """
        # 基础贡献分
        base = (
            self.stats["essences_contributed"] * 2.0 +   # 贡献新精华
            self.stats["essences_refined"] * 1.0 +        # 深化精华
            self.stats["essences_challenged"] * 1.5 +     # 反驳(引发辩论)
            self.stats["essences_cited"] * 0.8 +          # 被引用(影响力)
            self.stats["discussions_made"] * 0.2          # 参与度
        )

        # 新颖度评分：最近行为中"new"的比例越高，新颖度越高
        if self.recent_actions:
            new_count = sum(1 for a in self.recent_actions if a == "new")
            self.novelty_score = new_count / len(self.recent_actions)
        else:
            self.novelty_score = 0.5  # 默认中等新颖度

        # 新颖度加权：有新颖观点的专家更有价值
        novelty_bonus = self.novelty_score * 3.0

        # 发言饥饿度：越久没发言，越应该发言
        hunger_bonus = min(self.rounds_since_last_spoke * 0.5, 2.0)

        self.contribution_score = base + novelty_bonus + hunger_bonus
        return self.contribution_score

    # ── 专家人格进化机制 ──

    def _evolve_persona(self, round_discussions: List[Dict],
                        essence_pool, round_count: int) -> None:
        """
        主进化方法：每轮结束后调用，驱动三种进化机制。

        Args:
            round_discussions: 本轮发言记录 [{"player_name", "speech", "action", "refined_id", ...}]
            essence_pool: 精华池对象
            round_count: 当前轮次
        """
        self._total_rounds_alive = round_count

        # 1. 吸收机制：引用/深化其他专家的观点 → 向对方认知风格偏移
        self._update_absorption(round_discussions, essence_pool)

        # 2. 分化机制：长期坚持独特观点且高分 → 固化独特立场
        self._update_differentiation(essence_pool)

        # 3. 突变机制：特定轮次概率改变立场
        self._try_mutation(round_count, essence_pool)

        # 4. 更新人设文本（将进化后的特征注入 persona 描述）
        self._update_persona_text()

    def _update_absorption(self, round_discussions: List[Dict],
                           essence_pool) -> None:
        """
        吸收机制：追踪对其他专家观点的引用/深化。

        当一个专家在精华中引用或深化另一专家的观点时，
        吸收矩阵中对应的值增加。
        """
        # 查找本轮中该玩家引用了哪些其他专家
        my_refs = [d for d in round_discussions
                   if d.get("player_name") == self.name]

        for d in my_refs:
            action = d.get("action", "")
            refined_id = d.get("refined_id")
            refs = d.get("references", [])

            # 深化行为：获取被深化的精华的所有者
            if action == "refine" and refined_id is not None and essence_pool:
                parent = next((e for e in essence_pool.items
                               if e.id == refined_id), None)
                if parent and parent.contributor != self.name:
                    old = self._absorption_matrix.get(parent.contributor, 0.0)
                    # 深化一次吸收 0.1
                    self._absorption_matrix[parent.contributor] = min(1.0, old + 0.1)
                    self._persona_evolution_history.append(
                        f"深度吸收了{parent.contributor}的观点"
                    )

            # 引用行为
            if refs:
                for ref_id in refs:
                    if isinstance(ref_id, int) and essence_pool:
                        ref_item = next((e for e in essence_pool.items
                                         if e.id == ref_id), None)
                        if ref_item and ref_item.contributor != self.name:
                            old = self._absorption_matrix.get(
                                ref_item.contributor, 0.0)
                            self._absorption_matrix[
                                ref_item.contributor] = min(1.0, old + 0.05)

    def _update_differentiation(self, essence_pool) -> None:
        """
        分化机制：评估专家观点的独特性和高分趋势。

        条件：
        - 该专家的精华被多次反驳（说明观点独特，有争议）
        - 该专家的精华评分高
        → 分化程度上升
        """
        if not essence_pool:
            return

        my_essences = [e for e in essence_pool.items
                       if e.contributor == self.name]
        if len(my_essences) < 2:
            return  # 少于2条精华，无法判断分化趋势

        # 计算争议度：被反驳的精华比例
        challenged_ratio = sum(1 for e in my_essences if e.challenged_by) / len(my_essences)
        # 计算平均评分（归一化到0-1）
        avg_score = sum(e.score for e in my_essences) / len(my_essences)
        avg_score_norm = min(1.0, avg_score / 5.0)

        # 分化增长 = 争议度 * 0.3 + 评分 * 0.2
        delta = challenged_ratio * 0.3 + avg_score_norm * 0.2
        self._differentiation_level = min(1.0, self._differentiation_level + delta * 0.1)

        if delta > 0.3:
            self._persona_evolution_history.append(
                f"观点分化: 争议度{challenged_ratio:.0%}, 评分{avg_score:.1f}"
            )

    def _try_mutation(self, round_count: int, essence_pool) -> None:
        """
        突变机制：在特定轮次（3, 7, 11...）基于接触到的对立观点尝试立场转变。

        突变概率 = 吸收矩阵中对立观点的比例 * 0.3
        """
        mutation_rounds = {3, 7, 11, 15, 19}
        if round_count not in mutation_rounds:
            return

        if not essence_pool:
            return

        # 计算对立观点暴露度
        total_exposure = sum(self._absorption_matrix.values())
        if total_exposure <= 0:
            return

        # 突变概率 = 吸收程度 * 0.3，吸收越多越可能变
        mutation_prob = total_exposure * 0.3
        mutation_prob = min(0.6, mutation_prob)  # 最高60%

        if random.random() < mutation_prob:
            # 找出吸收最多的那个专家
            most_absorbed = max(self._absorption_matrix,
                                key=self._absorption_matrix.get)
            self._mutation_triggers.append(
                f"第{round_count}轮发生立场偏移，受{most_absorbed}影响"
            )
            self._persona_evolution_history.append(
                self._mutation_triggers[-1]
            )

    def _update_persona_text(self) -> None:
        """
        将进化后的特征注入人设文本。

        在原始人设基础上添加进化标注，用于 LLM 讨论 prompt。
        """
        if not self._original_persona:
            self._original_persona = self.persona

        parts = [self._original_persona]

        # 吸收影响
        if self._absorption_matrix:
            absorbed = [f"{name}({score:.1f})"
                        for name, score in
                        sorted(self._absorption_matrix.items(),
                               key=lambda x: x[1], reverse=True)
                        if score >= 0.2]
            if absorbed:
                parts.append(f"【吸收影响】受到了{', '.join(absorbed[:3])}的观点影响")

        # 分化特征
        if self._differentiation_level >= 0.5:
            parts.append(f"【分化特征】观点越来越独特，趋于固化")
        elif self._differentiation_level >= 0.3:
            parts.append(f"【分化倾向】开始形成独特的观点体系")

        # 突变记录
        if self._mutation_triggers:
            latest = self._mutation_triggers[-1]
            parts.append(f"【立场演变】{latest}")

        self.persona = "\n".join(parts)

    def get_persona_evolution_summary(self) -> str:
        """获取人格进化摘要（用于显示）"""
        lines = []
        lines.append(f"  {self.name} ({self.persona_name})")

        # 吸收矩阵
        if self._absorption_matrix:
            abs_lines = ", ".join(
                f"{name}({score:.2f})"
                for name, score in
                sorted(self._absorption_matrix.items(),
                       key=lambda x: x[1], reverse=True)[:5]
            )
            lines.append(f"    吸收矩阵: {abs_lines}")

        # 分化程度
        diff_bar = "█" * int(self._differentiation_level * 10) + \
                   "░" * (10 - int(self._differentiation_level * 10))
        lines.append(f"    分化程度: {diff_bar} {self._differentiation_level:.2f}")

        # 突变次数
        lines.append(f"    立场转变: {len(self._mutation_triggers)} 次")

        # 进化历史
        if self._persona_evolution_history:
            history = "; ".join(self._persona_evolution_history[-5:])
            lines.append(f"    最近进化: {history}")

        return "\n".join(lines)

    def get_contribution_summary(self) -> str:
        """获取贡献度摘要文本"""
        return (
            f"贡献分:{self.contribution_score:.1f} "
            f"(精华{self.stats['essences_contributed']}|"
            f"深化{self.stats['essences_refined']}|"
            f"反驳{self.stats['essences_challenged']}|"
            f"引用{self.stats['essences_cited']}|"
            f"新颖度{self.novelty_score:.2f})"
        )

    def discuss(self, problem: str, round_info: str,
                thinking_direction: str = "",
                discussion_mode: str = "physical",
                knowledge_base=None) -> Dict:
        """围绕讨论问题发表见解（支持知识库按需搜索）"""
        template = self._read_file("prompt/discussion_prompt.txt")

        if not thinking_direction:
            thinking_direction = "（用户未指定讨论方向，请自由发挥）"

        mode_text = self._get_mode_text(discussion_mode)

        # ── 知识库搜索（替代全量上下文注入）──
        knowledge_index = "（知识库为空）"
        search_results = "（尚无讨论记录）"
        if knowledge_base:
            knowledge_index = knowledge_base.get_index_text()
            # 根据专家人设自动搜索最相关的上下文
            persona_results = knowledge_base.search_by_persona(
                self.persona + " " + self.persona_name, top_k=5)
            search_results = knowledge_base.format_search_results(persona_results)

        prompt = template.format(
            self_name=self.name,
            self_persona=self.persona or "（未设定专业背景）",
            problem=problem,
            discussion_mode=mode_text,
            thinking_direction=thinking_direction,
            round_info=round_info,
            knowledge_index=knowledge_index,
            search_results=search_results,
        )

        last_error = None
        for attempt in range(3):
            messages = [{"role": "user", "content": prompt}]

            try:
                content, reasoning_content = self.llm_client.chat(
                    messages, model=self.model_name,
                    thinking=self.thinking, caller=f"{self.name}-讨论",
                    show_reasoning=self.show_reasoning, show_answer=False)

                # ── 检测 [SEARCH: ...] 按需搜索请求 ──
                if knowledge_base and "[SEARCH:" in content:
                    content = self._handle_search_in_response(
                        content, knowledge_base, prompt)

                result = self._safe_parse_json(
                    content,
                    ["speech", "key_insight", "self_awareness", "references", "action", "refined_id"],
                    {"speech": "", "key_insight": "", "self_awareness": "", "references": [], "action": "new", "refined_id": None}
                )

                if result and result.get("speech"):
                    result["speech"] = str(result["speech"])
                    result["key_insight"] = str(result.get("key_insight", ""))
                    refs = result.get("references", [])
                    result["references"] = refs if isinstance(refs, list) else []
                    result["action"] = str(result.get("action", "new"))
                    result["refined_id"] = result.get("refined_id")
                    self.stats["discussions_made"] += 1
                    self.record_action(result["action"])
                    return result, reasoning_content
            except Exception as e:
                last_error = e
                if attempt < 2:
                    continue

        if last_error:
            print(f"⚠️ {self.name} 讨论发言解析失败({str(last_error)[:50]})，使用兜底发言")
        else:
            print(f"⚠️ {self.name} 讨论发言解析失败，使用兜底发言")
        return {
            "speech": f"关于这个问题，我认为我们需要从多个角度来思考。从我的专业领域来看，{self.persona_name}的建议是：首先应该深入分析问题的本质，然后才能提出有效的解决方案。",
            "key_insight": "需要多角度分析问题本质",
            "references": [],
            "action": "new",
            "refined_id": None,
        }, ""

    def _handle_search_in_response(self, content: str, knowledge_base,
                                    original_prompt: str) -> str:
        """处理发言中的 [SEARCH: ...] 按需搜索请求，追加搜索结果后重新调用 LLM"""
        search_queries = re.findall(r'\[SEARCH:\s*([^\]]+)\]', content)
        if not search_queries:
            return content

        all_results = []
        for query in search_queries:
            query = query.strip()
            if query:
                results = knowledge_base.search(query, top_k=3,
                                                exclude_player=self.name)
                all_results.extend(results)

        if not all_results:
            return content

        # 去重
        seen = set()
        unique_results = []
        for r in all_results:
            key = (r.get("type", ""), r.get("id", id(r)),
                   r.get("text", "")[:50])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        search_section = (
            "\n\n═══════════════════════════════════════\n"
            "【按需搜索结果】\n"
            "你请求了搜索，以下是补充结果：\n"
            f"{knowledge_base.format_search_results(unique_results)}\n"
            "═══════════════════════════════════════\n\n"
            "请基于以上补充信息，重新输出完整的JSON格式发言。"
        )

        # 重新调用 LLM 获取最终发言
        augmented_prompt = original_prompt + search_section
        try:
            new_content, _ = self.llm_client.chat(
                [{"role": "user", "content": augmented_prompt}],
                model=self.model_name,
                thinking=self.thinking,
                caller=f"{self.name}-搜索补充",
                show_reasoning=False, show_answer=False)
            return new_content
        except Exception:
            return content

    def question(self, problem: str, question: str, player_persona: str,
                 thinking_direction: str = "",
                 knowledge_base=None) -> Dict:
        """回答用户提出的问题，给出解读与启示（支持知识库搜索）"""
        template = self._read_file("prompt/question_prompt.txt")
        if not template:
            return {"speech": "（模板文件缺失）", "insight": ""}, ""

        if not thinking_direction:
            thinking_direction = "（用户未指定讨论方向）"

        # ── 知识库搜索 ──
        knowledge_index = "（知识库为空）"
        search_results = "（尚无讨论记录）"
        if knowledge_base:
            knowledge_index = knowledge_base.get_index_text()
            persona_results = knowledge_base.search_by_persona(
                self.persona + " " + self.persona_name, top_k=5)
            search_results = knowledge_base.format_search_results(persona_results)

        prompt = template.format(
            self_name=self.name,
            self_persona=player_persona,
            problem=problem,
            question=question,
            knowledge_index=knowledge_index,
            search_results=search_results,
            thinking_direction=thinking_direction,
        )

        try:
            content, reasoning_content = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking=self.thinking,
                caller=f"{self.name}-提问回答",
                show_reasoning=self.show_reasoning, show_answer=False)

            result = self._safe_parse_json(
                content,
                ["speech", "insight", "self_awareness"],
                {"speech": "", "insight": "", "self_awareness": ""}
            )

            if result and result.get("speech"):
                return result, reasoning_content
        except Exception as e:
            print(f"⚠️ {self.name} 回答问题失败: {str(e)[:50]}")

        return {"speech": f"从{self.persona_name}的角度来看，{question[:50]}...这个问题值得深入探讨。我认为结合当前讨论中的已有见解，我们可以从{self.persona_name}的专业领域找到新的突破口。", "insight": f"从{self.persona_name}的专业视角来看，需要结合领域知识寻找突破口"}, ""

    def request_clarification(self, problem: str, item_id: int,
                              essence_content: str, source_round: int,
                              score: float, question: str) -> Dict:
        """对自己之前提出的精华进行澄清/补充说明"""
        template = self._read_file("prompt/clarification_prompt.txt")
        if not template:
            return {"answer": "（澄清模板缺失）", "refined": False}, ""

        prompt = template.replace("{self_name}", self.name) \
            .replace("{self_persona}", self.persona or self.persona_name or "（未设定）") \
            .replace("{problem}", problem) \
            .replace("{item_id}", str(item_id)) \
            .replace("{essence_content}", essence_content) \
            .replace("{source_round}", str(source_round)) \
            .replace("{score}", f"{score:.1f}") \
            .replace("{question}", question)

        try:
            content, reasoning_content = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking=self.thinking,
                caller=f"{self.name}-澄清回答",
                show_reasoning=self.show_reasoning, show_answer=False)

            result = self._safe_parse_json(
                content,
                ["answer", "refined", "self_awareness"],
                {"answer": "", "refined": False, "self_awareness": ""}
            )

            if result and result.get("answer"):
                return result, reasoning_content
        except Exception as e:
            print(f"⚠️ {self.name} 澄清回答失败: {str(e)[:50]}")

        return {"answer": f"作为{self.persona_name}，我对原观点的补充说明：{essence_content[:80]}...这一观点的核心在于结合专业判断与具体场景，需要在实践中进一步验证。", "refined": False, "self_awareness": ""}, ""

    def synthesize_solution(self, problem: str, all_essences: str,
                            evolution_history: str,
                            discussion_mode: str = "physical") -> Dict:
        """基于精华池综合生成最终解决方案"""
        template = self._read_file("prompt/synthesis_prompt.txt")

        mode_text = self._get_mode_text(discussion_mode)

        prompt = template.format(
            self_name=self.name,
            problem=problem,
            discussion_mode=mode_text,
            all_essences=all_essences,
            evolution_history=evolution_history,
        )

        try:
            content, reasoning_content = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking=self.thinking,
                caller=f"{self.name}-综合方案",
                show_reasoning=self.show_reasoning, show_answer=False)

            result = self._safe_parse_json(
                content,
                ["solution_title", "summary", "core_ideas", "key_insights", "divergence_points", "social_observations", "consciousness_emergence", "final_conclusion"],
                {
                    "solution_title": "综合解决方案",
                    "summary": "基于多轮讨论的综合方案",
                    "core_ideas": [],
                    "key_insights": [],
                    "divergence_points": [],
                    "social_observations": [],
                    "consciousness_emergence": "",
                    "final_conclusion": "结论"
                }
            )

            if result:
                return result, reasoning_content
        except Exception:
            pass

        return {
            "solution_title": "综合解决方案",
            "summary": "基于多轮讨论的综合方案",
            "core_ideas": [],
            "key_insights": [],
            "divergence_points": [],
            "social_observations": [],
            "consciousness_emergence": "",
            "final_conclusion": "讨论结束，综合各方观点形成最终方案"
        }, ""

    def vote(self, problem: str, new_essences: List[Dict]) -> Dict:
        """
        对本轮新提炼的精华进行批量投票。
        new_essences: [{"id", "content", "contributor"}] 列表
        自动跳过自己提出的精华。
        Returns: {"votes": [{"essence_id", "vote", "reason"}], "voter": self.name}
        """
        template = self._read_file("prompt/vote_prompt.txt")
        if not template:
            return {"votes": [], "voter": self.name}

        # 过滤掉自己提出的精华
        votable = [e for e in new_essences if e.get("contributor") != self.name]
        if not votable:
            return {"votes": [], "voter": self.name, "skipped": "all_own"}

        # 构建精华列表文本，标注自己的精华
        lines = []
        for e in new_essences:
            tag = " [你的精华，请跳过]" if e.get("contributor") == self.name else ""
            content = e.get("content", "")[:120]
            lines.append(f"  #{e['id']} (来自: {e.get('contributor', '未知')}){tag}\n     {content}")
        new_essences_text = "\n".join(lines)

        prompt = template.format(
            self_name=self.name,
            self_persona=self.persona or self.persona_name or "（未设定）",
            problem=problem,
            new_essences=new_essences_text,
        )

        try:
            content, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking="disabled",  # 投票不需要深度思考，节省token
                caller=f"{self.name}-投票",
                show_reasoning=False, show_answer=False)

            result = self._safe_parse_json(
                content,
                ["votes"],
                {"votes": []}
            )

            if result and isinstance(result.get("votes"), list):
                # 清洗投票数据
                valid_votes = []
                valid_ids = {e["id"] for e in votable}
                for v in result["votes"]:
                    if not isinstance(v, dict):
                        continue
                    eid = v.get("essence_id")
                    vote = str(v.get("vote", "abstain")).lower().strip()
                    if vote not in ("approve", "reject", "abstain"):
                        vote = "abstain"
                    reason = str(v.get("reason", ""))[:80]
                    # 跳过自己的精华和无效ID
                    if eid in valid_ids:
                        valid_votes.append({
                            "essence_id": eid,
                            "vote": vote,
                            "reason": reason,
                        })
                self_awareness = result.get("self_awareness", "")
                return {"votes": valid_votes, "voter": self.name, "self_awareness": self_awareness}
        except Exception as e:
            print(f"⚠️ {self.name} 投票失败: {str(e)[:50]}")

        # 兜底：全部弃权
        fallback_votes = [
            {"essence_id": e["id"], "vote": "abstain", "reason": "投票解析失败"}
            for e in votable
        ]
        return {"votes": fallback_votes, "voter": self.name}

    def debate(self, problem: str, role: str, topic_essence: Dict,
               opponent_argument: str = "",
               discussion_mode: str = "physical") -> Dict:
        """
        参与辩论。
        role: "attacker"（挑战方）或 "defender"（辩护方）
        topic_essence: {"id", "content", "contributor"} 被辩论的精华
        opponent_argument: 对方的论点（挑战方首轮为空）
        Returns: {"argument", "key_point", "concede"}
        """
        template = self._read_file("prompt/debate_prompt.txt")
        if not template:
            return {"argument": "（辩论模板缺失）", "key_point": "", "concede": False}

        mode_text = self._get_mode_text(discussion_mode)

        role_text = (
            "你是【挑战方】，你需要指出该精华的漏洞、不合理之处或局限性。"
            if role == "attacker"
            else "你是【辩护方】，你需要论证该精华的合理性与价值，反驳挑战方的观点。"
        )

        opponent_text = opponent_argument if opponent_argument else "（你是首轮发言，尚无对方论点）"

        prompt = template.format(
            self_name=self.name,
            self_persona=self.persona or self.persona_name or "（未设定）",
            role=role_text,
            problem=problem,
            discussion_mode=mode_text,
            topic_id=topic_essence.get("id", "?"),
            topic_content=topic_essence.get("content", ""),
            topic_contributor=topic_essence.get("contributor", "未知"),
            opponent_argument=opponent_text,
        )

        try:
            content, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking=self.thinking,
                caller=f"{self.name}-辩论({role})",
                show_reasoning=self.show_reasoning, show_answer=False)

            result = self._safe_parse_json(
                content,
                ["argument", "key_point", "concede", "self_awareness"],
                {"argument": "", "key_point": "", "concede": False, "self_awareness": ""}
            )

            if result and result.get("argument"):
                result["argument"] = str(result["argument"])
                result["key_point"] = str(result.get("key_point", ""))
                result["concede"] = bool(result.get("concede", False))
                return result
        except Exception as e:
            print(f"⚠️ {self.name} 辩论失败: {str(e)[:50]}")

        # 兜底
        fallback = (
            f"作为{self.persona_name}，我对该观点持保留态度，需要更多信息才能下定论。"
            if role == "attacker"
            else f"作为{self.persona_name}，我认为该观点在专业上有其合理性。"
        )
        return {"argument": fallback, "key_point": "", "concede": False, "self_awareness": ""}

    # ── 认知系统集成 ──

    def _init_cognitive_systems(self) -> None:
        """初始化所有认知子系统"""
        self.episodic_memory = EpisodicMemory(capacity=200)
        self.belief_system = BeliefSystem()
        self.emotional_state = EmotionalState()
        self.reasoning_module = ReasoningModule()
        self.learning_module = LearningModule()
        self.perspective_taking = PerspectiveTaking()
        self.cognitive_style = CognitiveStyle()
        self.expertise_model = ExpertiseModel()
        self.communication_style = CommunicationStyle()
        self.credibility_assessment = CredibilityAssessment()

    def get_cognitive_state(self) -> Dict:
        """获取完整的认知状态摘要"""
        return {
            "emotional_state": self.emotional_state.get_state(),
            "dominant_emotion": self.emotional_state.get_dominant_emotion(),
            "cognitive_style": self.cognitive_style.get_style(),
            "belief_count": len(self.belief_system.get_beliefs()),
            "memory_count": self.episodic_memory.get_stats()["total"],
            "credibility_network": self.credibility_assessment.get_trust_network(),
            "expertise_domains": list(self.expertise_model._domains.keys()),
            "learning_curve": self.learning_module.get_learning_curve(),
            "reasoning_style": self.reasoning_module.get_reasoning_style(),
        }

    def get_cognitive_report(self) -> str:
        """生成可读的认知状态报告"""
        lines = []
        lines.append(f"  {self.name} 认知状态报告")
        lines.append(f"  {'=' * 40}")

        # 情感状态
        emo = self.emotional_state.get_state()
        dom_emo = self.emotional_state.get_dominant_emotion()
        lines.append(f"  情感状态: {dom_emo}")
        for e, v in sorted(emo.items(), key=lambda x: x[1], reverse=True)[:3]:
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"    {e}: {bar} {v:.2f}")

        # 认知风格
        style = self.cognitive_style.get_style()
        lines.append(f"  认知风格: {style['processing']}/{style['approach']} "
                     f"(灵活性: {style['flexibility']:.2f})")

        # 信念数量
        beliefs = self.belief_system.get_beliefs()
        lines.append(f"  信念数量: {len(beliefs)}")
        if beliefs:
            # 最坚定的信念
            sorted_b = sorted(beliefs, key=lambda b: b["confidence"], reverse=True)[:2]
            for b in sorted_b:
                lines.append(f"    [{b['confidence']:.1f}] {b['topic'][:30]}")

        # 记忆统计
        mem_stats = self.episodic_memory.get_stats()
        lines.append(f"  记忆: {mem_stats['total']}条 "
                     f"(重要: {mem_stats['important']}, 陈旧: {mem_stats['stale']})")

        # 可信度网络
        trust = self.credibility_assessment.get_trust_network()
        if trust:
            lines.append(f"  可信度网络: {len(trust)}个玩家")
            for name, score in sorted(trust.items(), key=lambda x: x[1], reverse=True)[:3]:
                lines.append(f"    {name}: {score:.2f}")

        # 专业领域
        domains = list(self.expertise_model._domains.keys())
        if domains:
            lines.append(f"  专业领域: {', '.join(domains[:5])}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """序列化玩家状态用于断点保存"""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "thinking": self.thinking,
            "show_reasoning": self.show_reasoning,
            "show_answer": self.show_answer,
            "alive": self.alive,
            "persona_name": self.persona_name,
            "persona": self.persona,
            "stats": self.stats.copy(),
            "contribution_score": self.contribution_score,
            "novelty_score": self.novelty_score,
            "recent_actions": self.recent_actions.copy(),
            "rounds_since_last_spoke": self.rounds_since_last_spoke,
            # 进化机制
            "_original_persona": self._original_persona,
            "_absorption_matrix": self._absorption_matrix.copy(),
            "_differentiation_level": self._differentiation_level,
            "_persona_evolution_history": self._persona_evolution_history.copy(),
            "_mutation_triggers": self._mutation_triggers.copy(),
            "_total_rounds_alive": self._total_rounds_alive,
            # 认知系统
            "_episodic_memory": self.episodic_memory.to_dict(),
            "_belief_system": self.belief_system.to_dict(),
            "_emotional_state": self.emotional_state.to_dict(),
            "_reasoning_module": self.reasoning_module.to_dict(),
            "_learning_module": self.learning_module.to_dict(),
            "_perspective_taking": self.perspective_taking.to_dict(),
            "_cognitive_style": self.cognitive_style.to_dict(),
            "_expertise_model": self.expertise_model.to_dict(),
            "_communication_style": self.communication_style.to_dict(),
            "_credibility_assessment": self.credibility_assessment.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Player':
        """从字典恢复玩家状态"""
        player = cls(
            name=data["name"],
            model_name=data["model_name"],
            thinking=data.get("thinking", "auto"),
            show_reasoning=data.get("show_reasoning", False),
            show_answer=data.get("show_answer", True),
        )
        player.alive = data.get("alive", True)
        player.persona_name = data.get("persona_name", "")
        player.persona = data.get("persona", "")
        player.stats = data.get("stats", {
            "discussions_made": 0,
            "essences_contributed": 0,
            "essences_refined": 0,
            "essences_challenged": 0,
            "essences_cited": 0,
        })
        player.contribution_score = data.get("contribution_score", 0.0)
        player.novelty_score = data.get("novelty_score", 0.0)
        player.recent_actions = data.get("recent_actions", [])
        player.rounds_since_last_spoke = data.get("rounds_since_last_spoke", 0)
        # 恢复进化机制
        player._original_persona = data.get("_original_persona", "")
        player._absorption_matrix = data.get("_absorption_matrix", {})
        player._differentiation_level = data.get("_differentiation_level", 0.0)
        player._persona_evolution_history = data.get("_persona_evolution_history", [])
        player._mutation_triggers = data.get("_mutation_triggers", [])
        player._total_rounds_alive = data.get("_total_rounds_alive", 0)
        # 恢复认知系统
        if "_episodic_memory" in data:
            player.episodic_memory = EpisodicMemory.from_dict(data["_episodic_memory"])
        if "_belief_system" in data:
            player.belief_system = BeliefSystem.from_dict(data["_belief_system"])
        if "_emotional_state" in data:
            player.emotional_state = EmotionalState.from_dict(data["_emotional_state"])
        if "_reasoning_module" in data:
            player.reasoning_module = ReasoningModule.from_dict(data["_reasoning_module"])
        if "_learning_module" in data:
            player.learning_module = LearningModule.from_dict(data["_learning_module"])
        if "_perspective_taking" in data:
            player.perspective_taking = PerspectiveTaking.from_dict(data["_perspective_taking"])
        if "_cognitive_style" in data:
            player.cognitive_style = CognitiveStyle.from_dict(data["_cognitive_style"])
        if "_expertise_model" in data:
            player.expertise_model = ExpertiseModel.from_dict(data["_expertise_model"])
        if "_communication_style" in data:
            player.communication_style = CommunicationStyle.from_dict(data["_communication_style"])
        if "_credibility_assessment" in data:
            player.credibility_assessment = CredibilityAssessment.from_dict(data["_credibility_assessment"])
        return player


# ═══════════════════════════════════════════════════════════════
# 认知系统类定义
# ═══════════════════════════════════════════════════════════════


class EpisodicMemory:
    """
    情景记忆系统——存储和回忆过去的讨论经历。

    功能：
    - 存储记忆条目（事件、观点、反馈）
    - 基于相关性检索
    - 记忆巩固（重要记忆保留，次要记忆衰减）
    - 主动遗忘（容量管理）
    - 记忆重要性评分
    - 时序聚类
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.memories: List[Dict] = []
        self._importance_threshold = 0.1

    def store(self, event_type: str, content: str, context: Dict = None,
              importance: float = None) -> int:
        """存储一条记忆"""
        mem_id = len(self.memories)
        if importance is None:
            importance = self._compute_importance(event_type, content)
        memory = {
            "id": mem_id, "type": event_type, "content": content,
            "context": context or {}, "importance": importance,
            "timestamp": time.time(), "access_count": 0,
            "consolidated": False,
        }
        self.memories.append(memory)
        self._enforce_capacity()
        return mem_id

    def store_opinion(self, opinion: str, speaker: str, round_id: int) -> int:
        """存储观点记忆"""
        return self.store("opinion", opinion, {"speaker": speaker, "round": round_id}, 0.6)

    def store_feedback(self, feedback: str, source: str, score: float) -> int:
        """存储反馈记忆"""
        return self.store("feedback", feedback, {"source": source, "score": score},
                          min(1.0, abs(score) * 0.3 + 0.3))

    def recall(self, query: str, top_n: int = 5, min_importance: float = 0.0) -> List[Dict]:
        """基于相关性检索记忆"""
        scored = []
        query_keywords = set(self._tokenize(query))
        for m in self.memories:
            if m["importance"] < min_importance:
                continue
            mem_keywords = set(self._tokenize(m["content"]))
            overlap = len(query_keywords & mem_keywords)
            total = len(query_keywords | mem_keywords)
            sim = overlap / max(total, 1) if total > 0 else 0
            recency = 1.0 / (1.0 + (time.time() - m["timestamp"]) / 3600)
            score = sim * 0.5 + m["importance"] * 0.3 + recency * 0.2
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, m in scored[:top_n]:
            m["access_count"] += 1
            results.append({**m, "relevance_score": score})
        return results

    def recall_by_type(self, event_type: str, top_n: int = 5) -> List[Dict]:
        """按类型检索记忆"""
        filtered = [m for m in self.memories if m["type"] == event_type]
        filtered.sort(key=lambda m: m["importance"] * m["access_count"], reverse=True)
        return filtered[:top_n]

    def consolidate(self) -> int:
        """记忆巩固：提高重要记忆的权重，降低次要记忆"""
        consolidated = 0
        for m in self.memories:
            if m["consolidated"]:
                continue
            if m["access_count"] > 3 and m["importance"] > 0.5:
                m["importance"] = min(1.0, m["importance"] * 1.2)
                m["consolidated"] = True
                consolidated += 1
            elif m["access_count"] == 0 and (time.time() - m["timestamp"]) > 7200:
                m["importance"] *= 0.8
        return consolidated

    def forget(self, threshold: float = 0.05) -> int:
        """主动遗忘低重要性记忆"""
        before = len(self.memories)
        self.memories = [m for m in self.memories if m["importance"] >= threshold]
        return before - len(self.memories)

    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """按关键词搜索记忆"""
        results = []
        for m in self.memories:
            if keyword in m["content"]:
                results.append(m)
        return results

    def get_recent(self, n: int = 10) -> List[Dict]:
        """获取最近的 n 条记忆"""
        sorted_mem = sorted(self.memories, key=lambda m: m["timestamp"], reverse=True)
        return sorted_mem[:n]

    def get_stats(self) -> Dict:
        """获取记忆统计"""
        important = sum(1 for m in self.memories if m["importance"] > 0.6)
        stale = sum(1 for m in self.memories if (time.time() - m["timestamp"]) > 3600)
        return {"total": len(self.memories), "important": important,
                "stale": stale, "capacity": self.capacity}

    def _compute_importance(self, event_type: str, content: str) -> float:
        base = {"opinion": 0.5, "feedback": 0.6, "contradiction": 0.8,
                "breakthrough": 0.9, "essence": 0.7}.get(event_type, 0.4)
        length_bonus = min(0.3, len(content) / 1000)
        return min(1.0, base + length_bonus)

    def _enforce_capacity(self):
        while len(self.memories) > self.capacity:
            candidates = sorted(self.memories, key=lambda m: m["importance"] * m["access_count"])
            self.memories.pop(0)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [c for c in text if '\u4e00' <= c <= '\u9fff'] + text.split()

    def to_dict(self) -> Dict:
        return {"memories": self.memories, "capacity": self.capacity}

    @classmethod
    def from_dict(cls, data: Dict) -> 'EpisodicMemory':
        mem = cls(capacity=data.get("capacity", 200))
        mem.memories = data.get("memories", [])
        return mem


class BeliefSystem:
    """
    信念维持系统——跟踪和管理专家的信念。

    功能：
    - 信念添加与更新
    - 信念强度（置信度）管理
    - 信念修订（基于新证据）
    - 矛盾检测（认知失调）
    - 信念一致性维护
    - 信念层级（核心/外围信念）
    - 信念来源追踪
    """

    def __init__(self):
        self.beliefs: List[Dict] = []

    def add_belief(self, topic: str, statement: str, confidence: float = 0.5,
                   source: str = "初始", domain: str = "综合") -> int:
        """添加一条信念"""
        bid = len(self.beliefs)
        self.beliefs.append({
            "id": bid, "topic": topic, "statement": statement,
            "confidence": max(0.0, min(1.0, confidence)),
            "source": source, "domain": domain, "timestamp": time.time(),
            "evidence_for": [], "evidence_against": [], "revisions": 0,
            "is_core": confidence > 0.7,
        })
        return bid

    def update_belief(self, belief_id: int, evidence: str, supports: bool = True,
                      strength: float = 0.1) -> bool:
        """基于新证据更新信念"""
        if belief_id >= len(self.beliefs):
            return False
        b = self.beliefs[belief_id]
        if supports:
            b["evidence_for"].append(evidence)
            b["confidence"] = min(1.0, b["confidence"] + strength * (1 - b["confidence"]))
        else:
            b["evidence_against"].append(evidence)
            b["confidence"] = max(0.0, b["confidence"] - strength * b["confidence"])
        b["revisions"] += 1
        b["is_core"] = b["confidence"] > 0.7
        return True

    def get_beliefs(self, min_confidence: float = 0.0) -> List[Dict]:
        """获取信念列表"""
        return [b for b in self.beliefs if b["confidence"] >= min_confidence]

    def get_beliefs_by_domain(self, domain: str) -> List[Dict]:
        """按领域获取信念"""
        return [b for b in self.beliefs if b["domain"] == domain]

    def measure_certainty(self) -> float:
        """测量整体确定性水平"""
        if not self.beliefs:
            return 0.0
        return sum(b["confidence"] for b in self.beliefs) / len(self.beliefs)

    def detect_dissonance(self) -> List[Dict]:
        """检测认知失调（矛盾信念对）"""
        dissonances = []
        for i in range(len(self.beliefs)):
            for j in range(i + 1, len(self.beliefs)):
                b1, b2 = self.beliefs[i], self.beliefs[j]
                if b1["domain"] != b2["domain"]:
                    continue
                if b1["confidence"] > 0.5 and b2["confidence"] > 0.5:
                    # 检查是否矛盾（简化：对立话题）
                    if self._is_contradictory(b1["topic"], b2["topic"]):
                        dissonances.append({
                            "belief_a": b1, "belief_b": b2,
                            "severity": b1["confidence"] * b2["confidence"],
                            "resolution": "需要更多信息调和",
                        })
        dissonances.sort(key=lambda x: x["severity"], reverse=True)
        return dissonances[:5]

    def get_core_beliefs(self) -> List[Dict]:
        """获取核心信念"""
        return [b for b in self.beliefs if b.get("is_core", False)]

    def revise_all(self, new_evidence: str, domain: str = None) -> int:
        """基于新证据批量修订信念"""
        revised = 0
        for b in self.beliefs:
            if domain and b["domain"] != domain:
                continue
            if new_evidence[:20] in b["statement"] or b["statement"][:20] in new_evidence:
                self.update_belief(b["id"], new_evidence, supports=True, strength=0.05)
                revised += 1
        return revised

    @staticmethod
    def _is_contradictory(topic1: str, topic2: str) -> bool:
        contradiction_pairs = [
            ("是", "不是"), ("有", "没有"), ("应该", "不应该"),
            ("好", "坏"), ("对", "错"), ("支持", "反对"),
            ("积极", "消极"), ("进步", "倒退"),
        ]
        for a, b in contradiction_pairs:
            if (a in topic1 and b in topic2) or (b in topic1 and a in topic2):
                return True
        return False

    def to_dict(self) -> Dict:
        return {"beliefs": self.beliefs}

    @classmethod
    def from_dict(cls, data: Dict) -> 'BeliefSystem':
        bs = cls()
        bs.beliefs = data.get("beliefs", [])
        return bs


class EmotionalState:
    """
    情感状态系统——模拟专家的情感状态。

    维度：
    - 好奇心 (curiosity): 0~1
    - 自信心 (confidence): 0~1
    - 挫败感 (frustration): 0~1
    - 热情度 (enthusiasm): 0~1
    - 怀疑度 (skepticism): 0~1
    - 焦虑感 (anxiety): 0~1

    功能：
    - 情感状态更新
    - 情感衰减
    - 情感影响行为
    - 情感状态报告
    """

    def __init__(self):
        self.emotions = {
            "curiosity": 0.5, "confidence": 0.5, "frustration": 0.0,
            "enthusiasm": 0.5, "skepticism": 0.3, "anxiety": 0.1,
        }
        self._decay_rate = 0.05
        self._history: List[Dict] = []

    def update(self, emotion: str, delta: float):
        """更新特定情感值"""
        if emotion in self.emotions:
            old = self.emotions[emotion]
            self.emotions[emotion] = max(0.0, min(1.0, old + delta))
            self._history.append({
                "emotion": emotion, "from": old, "to": self.emotions[emotion],
                "delta": delta, "timestamp": time.time(),
            })

    def get_state(self) -> Dict:
        """获取当前情感状态"""
        return dict(self.emotions)

    def get_dominant_emotion(self) -> str:
        """获取主导情感"""
        return max(self.emotions, key=self.emotions.get)

    def decay(self):
        """情感值自然衰减"""
        for e in self.emotions:
            target = 0.3 if e in ("curiosity", "enthusiasm") else 0.0
            if self.emotions[e] > target:
                self.emotions[e] = max(target, self.emotions[e] - self._decay_rate)
            elif self.emotions[e] < target:
                self.emotions[e] = min(target, self.emotions[e] + self._decay_rate)

    def boost(self, emotion: str, amount: float = 0.2):
        """临时提升情感"""
        self.update(emotion, amount)

    def get_emotional_bias(self) -> Dict[str, float]:
        """获取情感对认知的偏置影响"""
        return {
            "risk_tolerance": 0.5 + (self.emotions["confidence"] - self.emotions["anxiety"]) * 0.3,
            "openness": 0.5 + (self.emotions["curiosity"] - self.emotions["skepticism"]) * 0.3,
            "assertiveness": 0.5 + (self.emotions["confidence"] - self.emotions["frustration"]) * 0.3,
            "persistence": 0.5 + (self.emotions["enthusiasm"] - self.emotions["frustration"]) * 0.3,
        }

    def get_emotion_history(self, n: int = 10) -> List[Dict]:
        """获取情感变化历史"""
        return self._history[-n:]

    def to_dict(self) -> Dict:
        return {"emotions": dict(self.emotions), "history": self._history[-50:]}

    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionalState':
        es = cls()
        es.emotions.update(data.get("emotions", {}))
        es._history = data.get("history", [])
        return es


class ReasoningModule:
    """
    推理模式系统——多种推理方式和策略选择。

    模式：
    - 分析推理 (analytical): 逐步逻辑分析
    - 直觉推理 (intuitive): 基于经验和直觉
    - 辩证推理 (dialectical): 正反合
    - 类比推理 (analogical): 类比迁移
    - 溯因推理 (abductive): 最佳解释推理
    """

    MODES = ["analytical", "intuitive", "dialectical", "analogical", "abductive"]

    MODE_DESCRIPTIONS = {
        "analytical": "逐步分解问题，每一步都基于前一步的逻辑结论",
        "intuitive": "基于长期积累的专业直觉和模式识别",
        "dialectical": "识别对立命题，通过正反合生成更高层次的综合",
        "analogical": "将问题映射到已知领域，通过类比得出见解",
        "abductive": "从观察出发，寻找最合理的解释",
    }

    def __init__(self):
        self.current_mode = "analytical"
        self._mode_history: List[str] = []
        self._trace: List[str] = []
        self._mode_effectiveness: Dict[str, float] = {m: 0.5 for m in self.MODES}

    def select_mode(self, problem: str, context: str = "") -> str:
        """根据问题类型选择最佳推理模式"""
        analytical_keywords = ['分析', '比较', '评估', '数据', '逻辑', '原因']
        intuitive_keywords = ['感觉', '直觉', '经验', '判断', '趋势']
        dialectical_keywords = ['矛盾', '对立', '争议', '分歧', '两难']
        analogical_keywords = ['比喻', '类比', '类似', '模型', '模式']
        abductive_keywords = ['解释', '原因', '假设', '可能', '推测']

        scores = {m: 0.0 for m in self.MODES}
        scores["analytical"] = sum(1 for k in analytical_keywords if k in problem) * 0.2
        scores["intuitive"] = sum(1 for k in intuitive_keywords if k in problem) * 0.2
        scores["dialectical"] = sum(1 for k in dialectical_keywords if k in problem) * 0.25
        scores["analogical"] = sum(1 for k in analogical_keywords if k in problem) * 0.2
        scores["abductive"] = sum(1 for k in abductive_keywords if k in problem) * 0.2

        # 叠加历史有效性的加成
        for m in self.MODES:
            scores[m] += self._mode_effectiveness[m] * 0.3

        best = max(scores, key=scores.get)
        self.current_mode = best
        self._mode_history.append(best)
        return best

    def reason(self, input_text: str, mode: str = None) -> Dict:
        """使用指定模式推理"""
        mode = mode or self.current_mode
        self._trace = []
        self._trace.append(f"推理模式: {mode} ({self.MODE_DESCRIPTIONS.get(mode, '')})")
        self._trace.append(f"输入: {input_text[:100]}")
        if mode == "analytical":
            result = self._analytical_reason(input_text)
        elif mode == "intuitive":
            result = self._intuitive_reason(input_text)
        elif mode == "dialectical":
            result = self._dialectical_reason(input_text)
        elif mode == "analogical":
            result = self._analogical_reason(input_text)
        elif mode == "abductive":
            result = self._abductive_reason(input_text)
        else:
            result = {"output": input_text, "confidence": 0.5}
        self._trace.append(f"输出: {result.get('output', '')[:100]}")
        result["mode"] = mode
        result["trace"] = list(self._trace)
        return result

    def _analytical_reason(self, text: str) -> Dict:
        steps = []
        sentences = [s.strip() for s in re.split(r'[。！？\n]', text) if s.strip()]
        for i, s in enumerate(sentences[:5]):
            steps.append(f"步骤{i + 1}: {s[:50]}")
        self._trace.extend(steps)
        return {"output": " → ".join(steps) if steps else text, "confidence": 0.7}

    def _intuitive_reason(self, text: str) -> Dict:
        self._trace.append("直觉模式: 识别模式匹配")
        keywords = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
        self._trace.append(f"识别到 {len(keywords)} 个关键概念")
        return {"output": text, "confidence": 0.5, "pattern_match": len(keywords)}

    def _dialectical_reason(self, text: str) -> Dict:
        thesis = text[:len(text) // 2]
        antithesis = text[len(text) // 2:]
        self._trace.append(f"正题: {thesis[:50]}")
        self._trace.append(f"反题: {antithesis[:50]}")
        synthesis = f"综合正题与反题，在更高维度上统一"
        self._trace.append(f"合题: {synthesis}")
        return {"output": synthesis, "thesis": thesis, "antithesis": antithesis, "confidence": 0.6}

    def _analogical_reason(self, text: str) -> Dict:
        self._trace.append("类比模式: 寻找已知相似模式")
        source_domains = ["生物学", "物理学", "社会学", "工程学", "经济学"]
        for d in source_domains:
            self._trace.append(f"尝试映射到{d}领域")
        return {"output": text, "confidence": 0.4, "source_domain": "综合"}

    def _abductive_reason(self, text: str) -> Dict:
        self._trace.append("溯因模式: 生成最佳解释假设")
        hypotheses = [
            f"假设1: 基于{text[:20]}的机制解释",
            f"假设2: 从{text[:20]}的演化视角",
            f"假设3: 从{text[:20]}的结构性分析",
        ]
        self._trace.extend(hypotheses)
        return {"output": hypotheses[0], "hypotheses": hypotheses, "confidence": 0.5}

    def generate_trace(self) -> List[str]:
        """获取推理过程"""
        return list(self._trace)

    def get_reasoning_style(self) -> str:
        """获取当前推理风格描述"""
        return f"{self.current_mode} ({self.MODE_DESCRIPTIONS.get(self.current_mode, '')})"

    def update_effectiveness(self, mode: str, score: float):
        """更新推理模式的有效性评分"""
        if mode in self._mode_effectiveness:
            old = self._mode_effectiveness[mode]
            self._mode_effectiveness[mode] = old * 0.8 + score * 0.2

    def to_dict(self) -> Dict:
        return {"current_mode": self.current_mode, "mode_history": self._mode_history,
                "mode_effectiveness": dict(self._mode_effectiveness)}

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReasoningModule':
        rm = cls()
        rm.current_mode = data.get("current_mode", "analytical")
        rm._mode_history = data.get("mode_history", [])
        rm._mode_effectiveness.update(data.get("mode_effectiveness", {}))
        return rm


class LearningModule:
    """
    学习与适应系统——从反馈和经验中学习。

    功能：
    - 从反馈中学习
    - 人设适应性调整
    - 学习曲线追踪
    - 学习策略管理
    - 成功模式识别
    """

    def __init__(self):
        self._learning_rate = 0.1
        self._experience_buffer: List[Dict] = []
        self._learning_curve: List[float] = []
        self._success_patterns: Dict[str, float] = {}
        self._total_lessons = 0

    def learn(self, experience: Dict) -> str:
        """从一次经验中学习"""
        self._experience_buffer.append(experience)
        self._total_lessons += 1
        lesson = self._extract_lesson(experience)
        pattern = experience.get("pattern", "general")
        score = experience.get("score", 0.5)
        if pattern in self._success_patterns:
            self._success_patterns[pattern] = self._success_patterns[pattern] * 0.8 + score * 0.2
        else:
            self._success_patterns[pattern] = score
        self._learning_curve.append(self._compute_competence())
        return lesson

    def adapt_persona(self, current_persona: str, feedback: str) -> str:
        """基于反馈调整人设"""
        self._experience_buffer.append({
            "type": "persona_adaptation", "input": current_persona,
            "feedback": feedback, "timestamp": time.time(),
        })
        self._total_lessons += 1
        positive_markers = ['好', '同意', '精彩', '深刻', '有见地']
        negative_markers = ['不同意', '错误', '肤浅', '偏颇', '不准确']
        pos_score = sum(1 for m in positive_markers if m in feedback)
        neg_score = sum(1 for m in negative_markers if m in feedback)
        if pos_score > neg_score:
            return current_persona
        elif neg_score > pos_score:
            return f"重新审视:{current_persona[:80]}"
        return current_persona

    def get_learning_curve(self) -> List[float]:
        """获取学习曲线"""
        return list(self._learning_curve[-20:])

    def _compute_competence(self) -> float:
        """计算当前能力水平"""
        if not self._success_patterns:
            return 0.5
        return sum(self._success_patterns.values()) / len(self._success_patterns)

    def _extract_lesson(self, experience: Dict) -> str:
        templates = [
            f"从{experience.get('type', '经验')}中学习到{experience.get('score', 0.5):.1f}分",
            f"模式{experience.get('pattern', 'general')}的有效性为{experience.get('score', 0.5):.1f}",
        ]
        return templates[0]

    def get_stats(self) -> Dict:
        return {"lessons": self._total_lessons, "patterns": len(self._success_patterns),
                "competence": self._compute_competence(), "buffer": len(self._experience_buffer)}

    def to_dict(self) -> Dict:
        return {"learning_rate": self._learning_rate, "success_patterns": dict(self._success_patterns),
                "learning_curve": list(self._learning_curve), "total_lessons": self._total_lessons}

    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningModule':
        lm = cls()
        lm._learning_rate = data.get("learning_rate", 0.1)
        lm._success_patterns = data.get("success_patterns", {})
        lm._learning_curve = data.get("learning_curve", [])
        lm._total_lessons = data.get("total_lessons", 0)
        return lm


class PerspectiveTaking:
    """
    视角模拟系统——模拟其他专家的视角和反应。

    功能：
    - 模拟其他专家视角
    - 生成对立方观点
    - 预测对方反应
    - 移情评分
    - 心智理论建模
    """

    def __init__(self):
        self._known_perspectives: Dict[str, Dict] = {}
        self._empathy_score = 0.5
        self._simulation_history: List[Dict] = []

    def take_perspective(self, target_name: str, target_persona: str,
                         topic: str) -> Dict:
        """模拟从目标视角看问题"""
        perspective = {
            "target": target_name, "persona": target_persona, "topic": topic,
            "likely_viewpoint": self._simulate_viewpoint(target_persona, topic),
            "likely_concerns": self._simulate_concerns(target_persona, topic),
            "likely_questions": self._simulate_questions(target_persona, topic),
            "confidence": self._empathy_score,
        }
        self._known_perspectives[target_name] = perspective
        self._simulation_history.append(perspective)
        return perspective

    def simulate_response(self, target_name: str, target_persona: str,
                          my_argument: str) -> str:
        """模拟对方对我的论点的可能反应"""
        if target_name in self._known_perspectives:
            p = self._known_perspectives[target_name]
        else:
            p = self.take_perspective(target_name, target_persona, my_argument[:50])
        if p.get("confidence", 0) > 0.5:
            return f"从{target_name}的视角({p['persona'][:20]})来看，可能会..."
        return "对方的反应不确定"

    def predict_reaction(self, my_argument: str, opponent_persona: str) -> Dict:
        """预测对方反应"""
        agreement_prob = 0.5
        if "因为" in my_argument and "所以" in my_argument:
            agreement_prob += 0.1
        if "可能" in my_argument or "也许" in my_argument:
            agreement_prob += 0.05
        return {
            "agreement_probability": min(1.0, agreement_prob),
            "likely_counter": "对方可能会提出替代方案",
            "emotional_impact": "中等",
        }

    def get_empathy_score(self) -> float:
        """获取移情评分"""
        return self._empathy_score

    def update_empathy(self, delta: float):
        """更新移情评分"""
        self._empathy_score = max(0.0, min(1.0, self._empathy_score + delta))

    def to_dict(self) -> Dict:
        return {"known_perspectives": self._known_perspectives,
                "empathy_score": self._empathy_score}

    @classmethod
    def from_dict(cls, data: Dict) -> 'PerspectiveTaking':
        pt = cls()
        pt._known_perspectives = data.get("known_perspectives", {})
        pt._empathy_score = data.get("empathy_score", 0.5)
        return pt

    @staticmethod
    def _simulate_viewpoint(persona: str, topic: str) -> str:
        keywords = set(re.findall(r'[\u4e00-\u9fff]{2,4}', persona))
        stance = "支持" if "正面" in keywords else "中立"
        return f"从{persona[:20]}出发，{stance}立场地看待{topic[:20]}"

    @staticmethod
    def _simulate_concerns(persona: str, topic: str) -> List[str]:
        return [f"关于{topic[:20]}的可行性问题", f"{persona[:16]}视角下的风险评估"]

    @staticmethod
    def _simulate_questions(persona: str, topic: str) -> List[str]:
        return [f"如何验证{topic[:20]}的有效性?", f"这对{persona[:16]}有什么影响?"]


class CognitiveStyle:
    """
    认知风格系统——定义和调整专家的认知风格。

    维度：
    - 处理方式 (processing): 分析型/整体型
    - 思维方式 (approach): 收敛型/发散型
    - 认知灵活性 (flexibility): 0~1
    - 认知复杂度 (complexity): 0~1
    """

    def __init__(self):
        self._processing = "analytical"  # analytical / holistic
        self._approach = "convergent"     # convergent / divergent
        self._flexibility = 0.5
        self._complexity = 0.5
        self._style_history: List[Dict] = []

    def get_style(self) -> Dict:
        """获取当前认知风格"""
        return {"processing": self._processing, "approach": self._approach,
                "flexibility": self._flexibility, "complexity": self._complexity}

    def adapt_style(self, problem_type: str, context: str = ""):
        """根据问题类型调整认知风格"""
        holistic_keywords = ['整体', '系统', '全局', '生态', '宏观']
        divergent_keywords = ['创新', '探索', '可能性', '新', '创意']
        if any(k in problem_type for k in holistic_keywords):
            self._processing = "holistic"
        else:
            self._processing = "analytical"
        if any(k in problem_type for k in divergent_keywords):
            self._approach = "divergent"
        else:
            self._approach = "convergent"
        self._style_history.append({
            "processing": self._processing, "approach": self._approach,
            "trigger": problem_type[:30], "timestamp": time.time(),
        })

    def generate_style_prompt(self) -> str:
        """生成认知风格引导提示词"""
        prompts = []
        if self._processing == "analytical":
            prompts.append("请逐步分析，每一步都给出明确理由")
        else:
            prompts.append("请从整体视角把握，关注系统层面的关系")
        if self._approach == "convergent":
            prompts.append("请聚焦于最关键的结论")
        else:
            prompts.append("请探索尽可能多的可能性")
        return " ".join(prompts)

    def measure_flexibility(self) -> float:
        """测量认知灵活性"""
        if len(self._style_history) < 2:
            return self._flexibility
        switches = 0
        for i in range(1, len(self._style_history)):
            if self._style_history[i]["processing"] != self._style_history[i - 1]["processing"]:
                switches += 1
            if self._style_history[i]["approach"] != self._style_history[i - 1]["approach"]:
                switches += 1
        self._flexibility = min(1.0, switches / max(len(self._style_history), 1))
        return self._flexibility

    def to_dict(self) -> Dict:
        return {"processing": self._processing, "approach": self._approach,
                "flexibility": self._flexibility, "complexity": self._complexity}

    @classmethod
    def from_dict(cls, data: Dict) -> 'CognitiveStyle':
        cs = cls()
        cs._processing = data.get("processing", "analytical")
        cs._approach = data.get("approach", "convergent")
        cs._flexibility = data.get("flexibility", 0.5)
        cs._complexity = data.get("complexity", 0.5)
        return cs


class ExpertiseModel:
    """
    专业深度模型——建模专家的专业知识覆盖。

    功能：
    - 专业领域建模
    - 知识深度评估
    - 知识空白检测
    - 学习建议生成
    - 专业交叉分析
    """

    DOMAIN_WEIGHTS = {
        "技术": 0.0, "哲学": 0.0, "科学": 0.0, "社会": 0.0,
        "艺术": 0.0, "经济": 0.0, "心理": 0.0, "政治": 0.0,
        "伦理": 0.0, "综合": 0.0,
    }

    def __init__(self):
        self._domains: Dict[str, float] = dict(self.DOMAIN_WEIGHTS)

    def get_expertise(self, domain: str) -> float:
        """获取特定领域的专业深度"""
        return self._domains.get(domain, 0.0)

    def update_expertise(self, domain: str, delta: float):
        """更新领域专业深度"""
        if domain in self._domains:
            self._domains[domain] = max(0.0, min(1.0, self._domains[domain] + delta))

    def find_gaps(self, threshold: float = 0.2) -> List[str]:
        """检测知识空白领域"""
        return [d for d, v in self._domains.items() if v < threshold and d != "综合"]

    def suggest_learning(self, target_domain: str) -> str:
        """建议学习方向"""
        current = self._domains.get(target_domain, 0.0)
        if current < 0.3:
            return f"建议从{target_domain}的基础概念开始学习"
        elif current < 0.6:
            return f"建议深化{target_domain}的中级理论并实践"
        else:
            return f"建议探索{target_domain}的前沿研究"

    def get_strongest_domains(self, top_n: int = 3) -> List[str]:
        """获取最强专业领域"""
        sorted_domains = sorted(self._domains, key=self._domains.get, reverse=True)
        return [d for d in sorted_domains if self._domains[d] > 0.3][:top_n]

    def to_dict(self) -> Dict:
        return {"domains": dict(self._domains)}

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpertiseModel':
        em = cls()
        em._domains.update(data.get("domains", {}))
        return em


class CommunicationStyle:
    """
    沟通风格系统——管理和调整专家的沟通方式。

    维度：
    - 正式度 (formality): 0~1
    - 直接度 (directness): 0~1
    - 详细度 (verbosity): 0~1
    - 情绪表达 (expressiveness): 0~1
    - 技术性 (technicality): 0~1
    """

    def __init__(self):
        self._formality = 0.5
        self._directness = 0.5
        self._verbosity = 0.5
        self._expressiveness = 0.5
        self._technicality = 0.5

    def get_style(self) -> Dict:
        """获取当前沟通风格"""
        return {"formality": self._formality, "directness": self._directness,
                "verbosity": self._verbosity, "expressiveness": self._expressiveness,
                "technicality": self._technicality}

    def adapt_to_audience(self, audience_level: str):
        """根据听众调整风格"""
        if audience_level == "expert":
            self._technicality = 0.8
            self._formality = 0.7
        elif audience_level == "general":
            self._technicality = 0.3
            self._formality = 0.4
            self._verbosity = 0.6
        elif audience_level == "mixed":
            self._technicality = 0.5
            self._formality = 0.5
            self._verbosity = 0.7

    def generate_with_style(self, content: str) -> str:
        """根据风格调整文本"""
        if self._formality > 0.7:
            content = content.replace("我", "笔者").replace("咱们", "我们")
        if self._directness > 0.7:
            if not content.endswith("。") and not content.endswith("！"):
                content = content + "。"
        if self._verbosity > 0.7 and len(content) < 100:
            content = content + "此外，我们还需要考虑更多相关因素..."
        return content

    def to_dict(self) -> Dict:
        return {"formality": self._formality, "directness": self._directness,
                "verbosity": self._verbosity, "expressiveness": self._expressiveness,
                "technicality": self._technicality}

    @classmethod
    def from_dict(cls, data: Dict) -> 'CommunicationStyle':
        cs = cls()
        cs._formality = data.get("formality", 0.5)
        cs._directness = data.get("directness", 0.5)
        cs._verbosity = data.get("verbosity", 0.5)
        cs._expressiveness = data.get("expressiveness", 0.5)
        cs._technicality = data.get("technicality", 0.5)
        return cs


class CredibilityAssessment:
    """
    可信度评估系统——评估其他专家的可信度。

    功能：
    - 可信度评分
    - 预测准确性追踪
    - 可信度更新
    - 信任网络构建
    - 声誉追踪
    """

    def __init__(self):
        self._credibility: Dict[str, float] = {}
        self._prediction_history: Dict[str, List[Dict]] = {}
        self._reputation: Dict[str, float] = {}

    def assess(self, target: str, statement: str, context: str = "") -> float:
        """评估目标的可信度"""
        base = self._credibility.get(target, 0.5)
        # 基于陈述质量调整
        quality_bonus = 0.0
        if len(statement) > 50:
            quality_bonus += 0.05
        if "因为" in statement and "所以" in statement:
            quality_bonus += 0.1
        if "可能" in statement or "也许" in statement:
            quality_bonus += 0.02
        new_score = max(0.0, min(1.0, base + quality_bonus))
        self._credibility[target] = new_score
        return new_score

    def update_credibility(self, target: str, prediction: bool, actual: bool) -> float:
        """基于预测准确度更新可信度"""
        if target not in self._prediction_history:
            self._prediction_history[target] = []
        self._prediction_history[target].append({
            "predicted": prediction, "actual": actual, "timestamp": time.time(),
        })
        accuracy = sum(1 for p in self._prediction_history[target]
                       if p["predicted"] == p["actual"])
        total = len(self._prediction_history[target])
        self._credibility[target] = accuracy / max(total, 1)
        return self._credibility[target]

    def get_credibility(self, target: str) -> float:
        """获取可信度"""
        return self._credibility.get(target, 0.5)

    def get_trust_network(self) -> Dict[str, float]:
        """获取信任网络"""
        return dict(self._credibility)

    def update_reputation(self, target: str, delta: float):
        """更新声誉"""
        old = self._reputation.get(target, 0.0)
        self._reputation[target] = max(-1.0, min(1.0, old + delta))

    def to_dict(self) -> Dict:
        return {"credibility": dict(self._credibility), "reputation": dict(self._reputation)}

    @classmethod
    def from_dict(cls, data: Dict) -> 'CredibilityAssessment':
        ca = cls()
        ca._credibility = data.get("credibility", {})
        ca._reputation = data.get("reputation", {})
        return ca


class PersonaEvolution:
    """
    人格进化系统——管理专家人格的渐进式演化。

    功能：
    - 吸收矩阵管理（其他专家的影响）
    - 分化程度追踪
    - 突变事件记录
    - 立场演化
    - 人格稳定性评估
    """

    def __init__(self, original_persona: str = ""):
        self.original_persona = original_persona
        self.current_persona = original_persona
        self.absorption_matrix: Dict[str, float] = {}
        self.differentiation_level = 0.0
        self.evolution_history: List[Dict] = []
        self.mutation_triggers: List[str] = []
        self.stance_positions: List[Dict] = []
        self.total_rounds_alive = 0
        self._stability = 1.0
        self._evolution_rate = 0.05

    def absorb(self, other_name: str, influence: float):
        """吸收其他专家的影响"""
        old = self.absorption_matrix.get(other_name, 0.0)
        self.absorption_matrix[other_name] = max(0.0, min(1.0, old + influence))
        self.differentiation_level = min(1.0, self.differentiation_level + influence * 0.1)
        self._stability = max(0.0, self._stability - influence * 0.05)

    def differentiate(self, delta: float = 0.05):
        """增加分化程度"""
        self.differentiation_level = min(1.0, self.differentiation_level + delta)

    def mutate(self, trigger: str, intensity: float = 0.3):
        """触发人格突变"""
        self.mutation_triggers.append(trigger)
        self.differentiation_level = min(1.0, self.differentiation_level + intensity * 0.2)
        self._stability = max(0.0, self._stability - intensity * 0.3)
        self.evolution_history.append({
            "type": "mutation", "trigger": trigger, "intensity": intensity,
            "round": self.total_rounds_alive, "timestamp": time.time(),
        })

    def record_stance(self, stance: Dict):
        """记录立场"""
        self.stance_positions.append({**stance, "round": self.total_rounds_alive})

    def advance_round(self):
        """推进一轮"""
        self.total_rounds_alive += 1
        self._stability = min(1.0, self._stability + self._evolution_rate)

    def get_stability(self) -> float:
        """获取人格稳定性"""
        return self._stability

    def get_evolution_summary(self) -> Dict:
        """获取演化摘要"""
        return {
            "original": self.original_persona[:50],
            "current": self.current_persona[:50],
            "differentiation": self.differentiation_level,
            "stability": self._stability,
            "absorbed_influences": len(self.absorption_matrix),
            "mutations": len(self.mutation_triggers),
            "rounds_alive": self.total_rounds_alive,
            "stances_recorded": len(self.stance_positions),
            "top_influences": sorted(self.absorption_matrix.items(),
                                     key=lambda x: x[1], reverse=True)[:3],
        }

    def to_dict(self) -> Dict:
        return {"original_persona": self.original_persona,
                "current_persona": self.current_persona,
                "absorption_matrix": dict(self.absorption_matrix),
                "differentiation_level": self.differentiation_level,
                "evolution_history": self.evolution_history,
                "mutation_triggers": self.mutation_triggers,
                "stance_positions": self.stance_positions,
                "total_rounds_alive": self.total_rounds_alive,
                "stability": self._stability}

    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonaEvolution':
        pe = cls(data.get("original_persona", ""))
        pe.current_persona = data.get("current_persona", "")
        pe.absorption_matrix = data.get("absorption_matrix", {})
        pe.differentiation_level = data.get("differentiation_level", 0.0)
        pe.evolution_history = data.get("evolution_history", [])
        pe.mutation_triggers = data.get("mutation_triggers", [])
        pe.stance_positions = data.get("stance_positions", [])
        pe.total_rounds_alive = data.get("total_rounds_alive", 0)
        pe._stability = data.get("stability", 1.0)
        return pe


class PersonalityTraits:
    """
    人格特质系统——五大维度的专家人格特质。

    维度：
    - 开放性 (Openness): 对新鲜事物的接受程度
    - 尽责性 (Conscientiousness): 条理性和责任感
    - 外向性 (Extraversion): 社交活跃度
    - 宜人性 (Agreeableness): 合作倾向
    - 神经质 (Neuroticism): 情绪敏感性

    功能：
    - 特质评分管理
    - 特质影响行为
    - 特质演化
    - 特质组合分析
    """

    def __init__(self):
        self.traits = {
            "openness": random.uniform(0.3, 0.8),
            "conscientiousness": random.uniform(0.3, 0.8),
            "extraversion": random.uniform(0.2, 0.7),
            "agreeableness": random.uniform(0.3, 0.8),
            "neuroticism": random.uniform(0.1, 0.6),
        }
        self._change_rate = 0.02
        self._history: List[Dict] = []

    def get_traits(self) -> Dict:
        """获取所有特质"""
        return dict(self.traits)

    def get_dominant_traits(self, top_n: int = 2) -> List[str]:
        """获取主导特质"""
        sorted_t = sorted(self.traits.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_t[:top_n]]

    def get_personality_type(self) -> str:
        """获取人格类型描述"""
        o = self.traits["openness"]
        c = self.traits["conscientiousness"]
        e = self.traits["extraversion"]
        a = self.traits["agreeableness"]
        n = self.traits["neuroticism"]
        if o > 0.7 and c > 0.6:
            return "探索型分析者"
        elif o > 0.7 and a > 0.6:
            return "开放型合作者"
        elif c > 0.7 and n < 0.3:
            return "稳定型执行者"
        elif e > 0.6 and a > 0.6:
            return "社交型协作者"
        elif o < 0.3 and c > 0.7:
            return "保守型实干家"
        elif n > 0.6:
            return "敏感型思考者"
        else:
            return "平衡型参与者"

    def update_trait(self, trait: str, delta: float):
        """更新特质"""
        if trait in self.traits:
            old = self.traits[trait]
            self.traits[trait] = max(0.0, min(1.0, old + delta))
            self._history.append({
                "trait": trait, "from": old, "to": self.traits[trait],
                "delta": delta, "timestamp": time.time(),
            })

    def evolve(self, experience_type: str, intensity: float = 0.1):
        """根据经验类型演化特质"""
        effects = {
            "new_experience": {"openness": 0.05},
            "success": {"conscientiousness": 0.03, "neuroticism": -0.02},
            "failure": {"neuroticism": 0.04, "conscientiousness": -0.02},
            "conflict": {"agreeableness": -0.03, "neuroticism": 0.03},
            "cooperation": {"agreeableness": 0.04, "extraversion": 0.03},
            "leadership": {"extraversion": 0.04, "conscientiousness": 0.03},
            "critique": {"openness": -0.02, "neuroticism": 0.02},
            "praise": {"extraversion": 0.03, "agreeableness": 0.02},
        }
        for trait, delta in effects.get(experience_type, {}).items():
            self.update_trait(trait, delta * intensity)

    def get_trait_influence(self) -> Dict[str, float]:
        """获取特质对行为的影响"""
        return {
            "risk_taking": self.traits["openness"] * 0.6 - self.traits["neuroticism"] * 0.4,
            "thoroughness": self.traits["conscientiousness"] * 0.8,
            "initiative": self.traits["extraversion"] * 0.6 + self.traits["openness"] * 0.3,
            "cooperativeness": self.traits["agreeableness"] * 0.7,
            "emotional_stability": 1.0 - self.traits["neuroticism"],
        }

    def to_dict(self) -> Dict:
        return {"traits": dict(self.traits), "history": self._history[-50:]}

    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonalityTraits':
        pt = cls()
        pt.traits.update(data.get("traits", {}))
        pt._history = data.get("history", [])
        return pt


class ConversationHistory:
    """
    对话历史系统——记录和管理专家的完整对话历史。

    功能：
    - 对话记录存储
    - 按轮次/话题检索
    - 发言统计
    - 对话模式分析
    - 沉默期检测
    """

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.entries: List[Dict] = []
        self._round_index: Dict[int, List[int]] = {}
        self._topic_index: Dict[str, List[int]] = {}

    def add_entry(self, round_id: int, speech: str, topic: str = "",
                  speech_type: str = "discussion", metadata: Dict = None) -> int:
        """添加一条对话记录"""
        eid = len(self.entries)
        entry = {"id": eid, "round": round_id, "speech": speech,
                 "topic": topic, "type": speech_type,
                 "metadata": metadata or {}, "timestamp": time.time()}
        self.entries.append(entry)
        if round_id not in self._round_index:
            self._round_index[round_id] = []
        self._round_index[round_id].append(eid)
        if topic:
            if topic not in self._topic_index:
                self._topic_index[topic] = []
            self._topic_index[topic].append(eid)
        self._enforce_limit()
        return eid

    def get_by_round(self, round_id: int) -> List[Dict]:
        """按轮次获取对话"""
        indices = self._round_index.get(round_id, [])
        return [self.entries[i] for i in indices]

    def get_by_topic(self, topic: str) -> List[Dict]:
        """按话题获取对话"""
        indices = self._topic_index.get(topic, [])
        return [self.entries[i] for i in indices]

    def get_recent(self, n: int = 10) -> List[Dict]:
        """获取最近 n 条对话"""
        return self.entries[-n:]

    def get_statistics(self) -> Dict:
        """获取对话统计"""
        type_counts = Counter(e["type"] for e in self.entries)
        round_counts = Counter(e["round"] for e in self.entries)
        total_words = sum(len(e["speech"]) for e in self.entries)
        return {
            "total_entries": len(self.entries),
            "total_rounds": len(self._round_index),
            "total_topics": len(self._topic_index),
            "type_distribution": dict(type_counts.most_common()),
            "avg_speech_length": total_words / max(len(self.entries), 1),
            "most_active_round": round_counts.most_common(1)[0] if round_counts else None,
        }

    def detect_silence(self, current_round: int, threshold: int = 3) -> bool:
        """检测沉默期"""
        recent_rounds = [r for r in self._round_index if r >= current_round - threshold]
        return len(recent_rounds) == 0

    def find_patterns(self) -> List[Dict]:
        """发现对话模式"""
        patterns = []
        if len(self.entries) < 5:
            return patterns
        speeches = [e["speech"] for e in self.entries[-20:]]
        avg_len = sum(len(s) for s in speeches) / max(len(speeches), 1)
        long_speeches = sum(1 for s in speeches if len(s) > avg_len * 1.5)
        if long_speeches > len(speeches) * 0.5:
            patterns.append({"type": "verbose", "description": "倾向于详细阐述", "confidence": 0.6})
        short_speeches = sum(1 for s in speeches if len(s) < avg_len * 0.5)
        if short_speeches > len(speeches) * 0.3:
            patterns.append({"type": "concise", "description": "倾向于简洁表达", "confidence": 0.5})
        return patterns

    def _enforce_limit(self):
        while len(self.entries) > self.max_history:
            removed = self.entries.pop(0)
            rid = removed["round"]
            if rid in self._round_index and self._round_index[rid]:
                self._round_index[rid].pop(0)
            topic = removed.get("topic", "")
            if topic in self._topic_index and self._topic_index[topic]:
                self._topic_index[topic].pop(0)

    def to_dict(self) -> Dict:
        return {"entries": self.entries[-200:], "max_history": self.max_history}

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationHistory':
        ch = cls(max_history=data.get("max_history", 500))
        ch.entries = data.get("entries", [])
        for e in ch.entries:
            rid = e.get("round")
            if rid is not None:
                if rid not in ch._round_index:
                    ch._round_index[rid] = []
                ch._round_index[rid].append(e["id"])
            topic = e.get("topic", "")
            if topic:
                if topic not in ch._topic_index:
                    ch._topic_index[topic] = []
                ch._topic_index[topic].append(e["id"])
        return ch
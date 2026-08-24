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
from typing import List, Dict, Optional
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
        return player
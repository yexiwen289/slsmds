"""
AI观察员元评论席 —— 独立于讨论的元观察员

不参与讨论、不产生精华，每轮结束后输出：
- 关键突破总结
- 盲点识别
- 分歧预测
- 推荐行动
"""

import json
from typing import List, Dict, Optional
from llm_client import LLMClient
from prompts_b64 import get_prompt as _get_b64_prompt


class Observer:
    """AI观察员：不参与讨论，每轮结束后提供元评论"""

    def __init__(self, model_name: str = "deepseek-v4-flash",
                 llm_client=None):
        self.llm_client = llm_client or LLMClient()
        self.model_name = model_name
        self.history: List[Dict] = []  # 每轮的观察记录

    def observe(self, problem: str, round_count: int,
                essence_count: int,
                discussion_text: str,
                pool_summary: str,
                consensus_text: str) -> Dict:
        """
        观察一轮讨论，输出元评论。

        Returns:
            {"summary": str, "blind_spots": str, "next_divergence": str,
             "recommended_action": str, "action_reason": str}
        """
        prompt = _get_b64_prompt("observer_prompt").format(
            problem=problem,
            round_count=round_count,
            essence_count=essence_count,
            discussion_text=discussion_text or "（本轮尚无发言记录）",
            pool_summary=pool_summary or "（精华池为空）",
            consensus_text=consensus_text or "（尚无共识度数据）",
        )

        try:
            content, _ = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                thinking="disabled",
                caller="AI观察员",
                show_reasoning=False,
                show_answer=False,
            )

            result = self._parse_observer_json(content)
            if not result:
                result = self._default_response()

            # 补充元数据
            result["round"] = round_count
            self.history.append(result)
            return result

        except Exception as e:
            print(f"  ⚠️ AI观察员分析失败: {str(e)[:60]}")
            default = self._default_response()
            default["round"] = round_count
            return default

    @staticmethod
    def _parse_observer_json(content: str) -> Optional[Dict]:
        """从LLM输出中提取有效的JSON"""
        if not content or not content.strip():
            return None

        # 直接解析
        try:
            result = json.loads(content.strip())
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        # 找第一个 {...}
        for start in range(len(content)):
            if content[start] == '{':
                depth = 0
                for end in range(start, len(content)):
                    if content[end] == '{':
                        depth += 1
                    elif content[end] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = content[start:end + 1]
                            try:
                                result = json.loads(candidate)
                                if isinstance(result, dict):
                                    return result
                            except Exception:
                                pass
        return None

    @staticmethod
    def _default_response() -> Dict:
        return {
            "summary": "（AI观察员未能生成分析，请查看讨论记录自行判断）",
            "blind_spots": "（无法识别）",
            "next_divergence": "（无法预测）",
            "recommended_action": "继续讨论",
            "action_reason": "等待更多数据",
        }

    def format_observer_output(self, observation: Dict) -> str:
        """将观察结果格式化为可打印文本"""
        lines = []
        lines.append(f"\n{'='*50}")
        lines.append("👁️ AI观察员 · 元评论")
        lines.append(f"{'='*50}")
        lines.append(f"  📝 关键突破: {observation.get('summary', '')}")
        lines.append(f"  👁️ 盲点识别: {observation.get('blind_spots', '')}")
        lines.append(f"  🔮 分歧预测: {observation.get('next_divergence', '')}")
        lines.append(f"  🎯 推荐动作: {observation.get('recommended_action', '')}")
        lines.append(f"     └ 理由: {observation.get('action_reason', '')}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def get_history_summary(self) -> str:
        """获取观察员历史摘要"""
        if not self.history:
            return "（无观察记录）"
        lines = []
        lines.append("👁️ AI观察员历史记录:")
        for i, obs in enumerate(self.history, 1):
            lines.append(f"  第{obs.get('round', '?')}轮: {obs.get('summary', '')[:80]}...")
        return "\n".join(lines)
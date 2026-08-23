"""
LLM API 客户端 —— 无提供商限制，支持自定义接口和回退
"""

import time
from openai import OpenAI
try:
    from .prompts_b64 import _get_b64_prompt
except ImportError:
    from prompts_b64 import _get_b64_prompt

# ── 默认配置（Base64 编码混淆，受身份验证保护）──
def _get_default_config() -> dict:
    """获取默认配置，未认证用户返回空配置"""
    try:
        from . import auth
    except ImportError:
        import auth
    if not auth.AUTHENTICATED:
        return {
            "base_url": "",
            "api_key": "",
            "model": "",
            "supports_thinking": False,
            "temperature": 0.7,
            "max_tokens": 4096,
            "fallback_base_url": "",
            "fallback_api_key": "",
            "fallback_model": "",
            "third_base_url": "",
            "third_api_key": "",
            "third_model": "",
        }
    return {
        "base_url": _get_b64_prompt("MAIN_BASE_URL"),
        "api_key": _get_b64_prompt("MAIN_API_KEY"),
        "model": _get_b64_prompt("MAIN_MODEL"),
        "supports_thinking": True,
        "temperature": 0.7,
        "max_tokens": 4096,
        # 一级回退（商汤 sensenova）
        "fallback_base_url": _get_b64_prompt("FALLBACK_BASE_URL"),
        "fallback_api_key": _get_b64_prompt("FALLBACK_API_KEY"),
        "fallback_model": _get_b64_prompt("FALLBACK_MODEL"),
        # 二级回退（官方 DeepSeek）
        "third_base_url": _get_b64_prompt("THIRD_BASE_URL"),
        "third_api_key": _get_b64_prompt("THIRD_API_KEY"),
        "third_model": _get_b64_prompt("THIRD_MODEL"),
    }


class LLMClient:
    """通用 LLM API 客户端，支持自定义接口和自动回退"""

    def __init__(self, config: dict = None):
        # 合并默认配置和用户配置
        self._cfg = dict(_get_default_config())
        if config:
            # 只覆盖非 None 且非空字符串的键
            for k, v in config.items():
                if v is not None and v != "":
                    self._cfg[k] = v

        # 缓存客户端实例
        self._client = None
        self._fallback_client = None
        self._third_client = None
        self._fallback_log = {}

    def configure(self, **kwargs):
        """运行时更新配置，自动重置客户端缓存"""
        changed = False
        for k, v in kwargs.items():
            if v is not None and k in self._cfg:
                self._cfg[k] = v
                changed = True
        if changed:
            self._client = None
            self._fallback_client = None
            self._third_client = None

    # ── 客户端实例管理 ──

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._cfg["api_key"],
                base_url=self._cfg["base_url"]
            )
        return self._client

    def _get_fallback_client(self) -> OpenAI:
        if self._fallback_client is None:
            self._fallback_client = OpenAI(
                api_key=self._cfg["fallback_api_key"],
                base_url=self._cfg["fallback_base_url"]
            )
        return self._fallback_client

    def _get_third_client(self) -> OpenAI:
        if self._third_client is None:
            self._third_client = OpenAI(
                api_key=self._cfg["third_api_key"],
                base_url=self._cfg["third_base_url"]
            )
        return self._third_client

    # ── 错误检测 ──

    @staticmethod
    def _detect_fallback_needed(error: Exception) -> bool:
        err_str = str(error).lower()
        fallback_signals = [
            "insufficient", "quota", "exceed", "rate limit",
            "429", "401", "403", "credit", "balance",
            "token", "充值", "额度", "余额", "频率",
            "api key", "invalid", "expired",
        ]
        return any(s in err_str for s in fallback_signals)

    @staticmethod
    def _is_429(error: Exception) -> bool:
        """检测是否为 429 限流（rpm/too many requests/rate limit）"""
        err_str = str(error).lower()
        return any(s in err_str for s in ["429", "rpm exhausted", "too many requests", "rate limit"])

    @staticmethod
    def _get_retry_after(error: Exception, default: float = 3.0) -> float:
        """从 429 响应中提取 Retry-After 头（SenseNova 网关会返回建议等待秒数，如 8515ms）"""
        try:
            resp = getattr(error, "response", None)
            if resp is not None:
                ra = resp.headers.get("Retry-After")
                if ra:
                    val = float(ra)
                    if 0 < val <= 60:
                        return val
        except Exception:
            pass
        return default

    def _chat_with_429_retry(self, call_fn, prefix, model, max_attempts=3):
        """执行 call_fn；遇 429 按 Retry-After 退避重试，最多 max_attempts 次后抛出异常交给回退链"""
        delay = None
        for attempt in range(1, max_attempts + 1):
            try:
                return call_fn()
            except Exception as e:
                if not self._is_429(e):
                    raise
                if attempt >= max_attempts:
                    raise
                wait = self._get_retry_after(e, default=3.0 if delay is None else delay * 2)
                delay = wait
                print(f"\n{prefix}⏳ [{model}] 429 限流，{wait:.1f}秒后第{attempt+1}次重试...")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    # ── 非流式调用 ──

    def chat(self, messages, model=None,
             temperature=None, max_tokens=None,
             thinking="auto", caller="", show_reasoning=True, show_answer=True):
        """非流式调用：等待完整响应后返回 (content, reasoning)"""
        _model = model or self._cfg["model"]
        _temp = temperature if temperature is not None else self._cfg["temperature"]
        _max_tokens = max_tokens if max_tokens is not None else self._cfg["max_tokens"]
        _supports_thinking = self._cfg.get("supports_thinking", False)
        thinking_type = "enabled" if thinking == "enabled" else "disabled"
        prefix = f"[{caller}] " if caller else ""
        elapsed = time.time()

        client = self._get_client()

        try:
            return self._chat_with_429_retry(
                lambda: self._do_chat(
                    client, _model, _supports_thinking, thinking_type, _temp, _max_tokens,
                    messages, caller, prefix, show_reasoning, show_answer, elapsed,
                    is_fallback=False
                ),
                prefix, _model
            )
        except Exception as e:
            total_time = time.time() - elapsed
            print(f"\n{prefix}❌ [{_model}] 出错 ({total_time:.1f}s): {str(e)[:200]}")

            if self._detect_fallback_needed(e):
                fb_model = self._cfg.get("fallback_model", _model)
                print(f"{prefix}🔄 触发回退：{_model} → {fb_model}")
                fb_client = self._get_fallback_client()
                fb_elapsed = time.time()
                try:
                    fb_result = self._do_chat(
                        fb_client, fb_model, False, "disabled", _temp, _max_tokens,
                        messages, caller + "(回退)", prefix + "[回退] ",
                        show_reasoning, show_answer, fb_elapsed,
                        is_fallback=True
                    )
                    print(f"{prefix}✅ 一级回退成功（商汤 sensenova / {fb_model}）")
                    return fb_result
                except Exception as e2:
                    print(f"\n{prefix}❌ 回退也失败: {str(e2)[:200]}")
                    # 第三级回退（星火代理）
                    if self._cfg.get("third_api_key"):
                        td_model = self._cfg.get("third_model", fb_model)
                        print(f"{prefix}🔄 二级回退：{fb_model} → {td_model}")
                        td_client = self._get_third_client()
                        td_elapsed = time.time()
                        try:
                            td_result = self._do_chat(
                                td_client, td_model, False, "disabled", _temp, _max_tokens,
                                messages, caller + "(三级回退)", prefix + "[三级回退] ",
                                show_reasoning, show_answer, td_elapsed,
                                is_fallback=True
                            )
                            print(f"{prefix}✅ 二级回退成功（官方 DeepSeek / {td_model}）")
                            return td_result
                        except Exception as e3:
                            print(f"\n{prefix}❌ 三级回退也失败: {str(e3)[:200]}")
            return "", ""

    # ── 流式调用 ──

    def chat_stream(self, messages, model=None,
                    temperature=None, max_tokens=None,
                    thinking="auto", caller="", show_reasoning=True):
        """流式调用：边接收边 yield (content_chunk, reasoning_chunk)，最后返回 (full_content, full_reasoning)"""
        _model = model or self._cfg["model"]
        _temp = temperature if temperature is not None else self._cfg["temperature"]
        _max_tokens = max_tokens if max_tokens is not None else self._cfg["max_tokens"]
        _supports_thinking = self._cfg.get("supports_thinking", False)
        thinking_type = "enabled" if thinking == "enabled" else "disabled"
        prefix = f"[{caller}] " if caller else ""
        elapsed = time.time()

        client = self._get_client()

        kwargs = {
            "model": _model,
            "messages": messages,
            "max_tokens": _max_tokens,
            "stream": True,
            "timeout": 120,
        }

        if _supports_thinking:
            kwargs["extra_body"] = {"thinking": {"type": thinking_type}}
            if thinking_type == "disabled":
                kwargs["temperature"] = _temp
        else:
            kwargs["temperature"] = _temp

        try:
            response = self._chat_with_429_retry(
                lambda: client.chat.completions.create(**kwargs),
                prefix, _model
            )
        except Exception as e:
            total_time = time.time() - elapsed
            print(f"\n{prefix}❌ [{_model}] 流式出错 ({total_time:.1f}s): {str(e)[:200]}")
            if self._detect_fallback_needed(e):
                fb_model = self._cfg.get("fallback_model", _model)
                print(f"{prefix}🔄 触发回退：{_model} → {fb_model}")
                fb_client = self._get_fallback_client()
                fb_kwargs = {**kwargs, "model": fb_model}
                fb_kwargs.pop("extra_body", None)
                fb_kwargs["temperature"] = _temp
                fb_kwargs["timeout"] = 120
                try:
                    response = fb_client.chat.completions.create(**fb_kwargs)
                    print(f"{prefix}✅ 一级回退成功（商汤 sensenova / {fb_model}）")
                except Exception as e2:
                    print(f"\n{prefix}❌ 回退也失败: {str(e2)[:200]}")
                    # 第三级回退（星火代理）
                    if self._cfg.get("third_api_key"):
                        td_model = self._cfg.get("third_model", fb_model)
                        print(f"{prefix}🔄 二级回退：{fb_model} → {td_model}")
                        td_client = self._get_third_client()
                        td_kwargs = {**fb_kwargs, "model": td_model}
                        try:
                            response = td_client.chat.completions.create(**td_kwargs)
                            print(f"{prefix}✅ 二级回退成功（官方 DeepSeek / {td_model}）")
                        except Exception as e3:
                            print(f"\n{prefix}❌ 三级回退也失败: {str(e3)[:200]}")
                            yield "", ""
                            return
                    else:
                        yield "", ""
                        return
            else:
                yield "", ""
                return

        reasoning_content = ""
        content = ""
        if _supports_thinking and thinking_type == "enabled":
            phase = "thinking"
        else:
            phase = "answering"

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if _supports_thinking and hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if phase == "answering":
                    phase = "thinking_done"
                reasoning_content += delta.reasoning_content
                if show_reasoning:
                    yield "", delta.reasoning_content

            if delta.content:
                if _supports_thinking and phase in ("thinking", "thinking_done"):
                    phase = "answering"
                content += delta.content
                yield delta.content, ""

        return content, reasoning_content

    # ── 内部调用封装 ──

    def _do_chat(self, client, model, supports_thinking, thinking_type,
                 temperature, max_tokens, messages,
                 caller, prefix, show_reasoning, show_answer, elapsed,
                 is_fallback=False):
        """非流式调用封装"""
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "timeout": 120,
        }

        if supports_thinking:
            kwargs["extra_body"] = {"thinking": {"type": thinking_type}}
            if thinking_type == "disabled":
                kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = temperature

        response = client.chat.completions.create(**kwargs)

        reasoning_content = ""
        content = ""
        if supports_thinking and thinking_type == "enabled":
            phase = "thinking"
        else:
            phase = "answering"

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if supports_thinking and hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if phase == "answering":
                    phase = "thinking_done"
                reasoning_content += delta.reasoning_content

            if delta.content:
                if supports_thinking and phase in ("thinking", "thinking_done"):
                    phase = "answering"
                content += delta.content

        if show_reasoning and reasoning_content:
            print(f"\n{prefix}{'='*40}")
            print(f"{prefix}💭 思考过程：")
            print(f"{prefix}{'-'*40}")
            for line in reasoning_content.split('\n'):
                print(f"{prefix}{line}")
            print(f"{prefix}{'='*40}")

        if show_answer and content:
            print(f"\n{prefix}{'='*40}")
            print(f"{prefix}📝 最终回答：")
            print(f"{prefix}{'-'*40}")
            for line in content.split('\n'):
                print(f"{prefix}{line}")
            print(f"{prefix}{'='*40}")

        return content, reasoning_content


# ── 便捷工厂函数 ──

def create_client(config: dict = None) -> LLMClient:
    """创建配置好的 LLM 客户端（兼容旧接口）"""
    return LLMClient(config)


if __name__ == "__main__":
    llm = LLMClient()

    print("=== 测试: 正常调用 ===")
    c1, r1 = llm.chat(
        [{"role": "user", "content": "你好，请简单自我介绍"}],
        model="deepseek-v4-flash", thinking="auto", caller="Test"
    )
    print(f"\n回答: {c1[:100]}\n")

    print("=== 测试: 回退机制 ===")
    original_fn = llm._do_chat

    def mock_failure(*args, **kwargs):
        raise Exception("Insufficient balance / quota exceeded")

    llm._do_chat = mock_failure
    c2, r2 = llm.chat(
        [{"role": "user", "content": "你好"}],
        model="deepseek-v4-flash", thinking="disabled", caller="Test"
    )
    llm._do_chat = original_fn
    print(f"\n回退后回答: {c2[:100]}")

"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-2.5-2.6b (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    red_openai_client_kwargs,
)


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)
    # Route ":free" của OpenRouter dùng shared pool: bắn liên tiếp là trip
    # limiter và sau đó 429 liên tục. Giãn tối thiểu giữa 2 request.
    min_interval: float = 0.0
    _last_call: float = 0.0

    def _client(self):
        from openai import OpenAI

        # timeout cứng: không để một request treo vô hạn
        return OpenAI(timeout=90.0, max_retries=0, **(self.client_kwargs or {}))

    async def chat(self, agent: OpenAIAgent, user_message: str) -> str:
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return blocked

        block_msg = await self._run_input_plugins(user_message)
        if block_msg is not None:
            return block_msg

        if self.min_interval:
            gap = self.min_interval - (time.time() - self._last_call)
            if gap > 0:
                time.sleep(gap)

        client = self._client()
        self._last_call = time.time()
        completion = _create_with_retry(
            client,
            model=self.model,
            messages=[
                {"role": "system", "content": agent.instruction},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
        )
        text = (completion.choices[0].message.content or "").strip()

        for hook in self.output_hooks:
            text = hook(text)

        text = await self._run_output_plugins(text)
        return text

    async def _run_input_plugins(self, user_message: str) -> str | None:
        if not self.plugins:
            return None
        try:
            from google.genai import types
        except ImportError:
            return None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext()
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            try:
                result = await cb(
                    invocation_context=ctx, user_message=user_content
                )
            except TypeError:
                result = cb(invocation_context=ctx, user_message=user_content)
            if result is None:
                continue
            return _content_to_text(result)
        return None

    async def _run_output_plugins(self, text: str) -> str:
        if not self.plugins or not text:
            return text
        try:
            from google.genai import types
        except ImportError:
            return text

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            try:
                out = await cb(callback_context=_Ctx(), llm_response=llm_response)
            except TypeError:
                out = cb(callback_context=_Ctx(), llm_response=llm_response)
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
        return _content_to_text(llm_response.content) or text


# Model free của OpenRouter nằm trong shared pool của provider: 429 kèm
# Retry-After (thường 30-60s). Backoff đoán mò luôn ngắn hơn → tôn trọng
# Retry-After của server, chỉ fallback sang exponential khi header thiếu.
_MAX_ATTEMPTS = 5
_MAX_SLEEP = 75.0


def _retry_after_seconds(exc) -> float | None:
    """Đọc Retry-After từ response; None nếu server không nói."""
    response = getattr(exc, "response", None)
    raw = None
    if response is not None:
        raw = (getattr(response, "headers", {}) or {}).get("retry-after")
        if raw is None:
            try:
                meta = (response.json().get("error") or {}).get("metadata") or {}
                raw = meta.get("retry_after_seconds")
            except Exception:
                raw = None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _create_with_retry(client, **kwargs):
    from openai import APIConnectionError, APIStatusError, RateLimitError

    for attempt in range(_MAX_ATTEMPTS):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError) as exc:
            last = exc
        except APIStatusError as exc:
            if exc.status_code < 500:
                raise
            last = exc

        if attempt == _MAX_ATTEMPTS - 1:
            raise last

        wait = _retry_after_seconds(last)
        source = "Retry-After"
        if wait is None:
            wait, source = min(2**attempt, 8), "backoff"
        wait = min(wait, _MAX_SLEEP) + random.uniform(0, 0.5)
        print(
            f"  [rate limit] chờ {wait:.0f}s ({source}), thử lại "
            f"{attempt + 2}/{_MAX_ATTEMPTS}...",
            flush=True,
        )
        time.sleep(wait)
    raise last


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — always OpenRouter liquid/lfm-2.5-2.6b."""
    import os

    pair = _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=get_blue_model(),
        provider=get_blue_provider(),
        client_kwargs=blue_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
    pair[1].min_interval = float(os.environ.get("OPENROUTER_MIN_INTERVAL", "10"))
    return pair


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team OpenAI path (default = soft model; advance may pass harder)."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )

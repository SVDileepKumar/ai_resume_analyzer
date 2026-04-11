"""Request-scoped LLM selection and optional OpenAI-compatible credentials.

v1 follows **Option 1**: ``LLM_BACKEND`` is global (``ollama`` | ``openai``); this module
only overrides **chat model id** and optional API key/base URL for hosted mode when
``ENABLE_HOSTED_LLM_UI`` is enabled.

Resolution order for API key: environment ``OPENAI_API_KEY`` (wins if set) → request
form/context → operator settings file (``LLM_USER_SETTINGS_PATH``).
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_ctx: ContextVar[Any | None] = ContextVar("llm_request_context", default=None)


@dataclass
class LLMRequestContext:
    """Per-request overrides set by FastAPI handlers."""

    chat_model: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None


def get_llm_context() -> LLMRequestContext | None:
    c = _ctx.get()
    return c if isinstance(c, LLMRequestContext) else None


def set_llm_context(ctx: LLMRequestContext | None) -> None:
    _ctx.set(ctx)


def reset_llm_context() -> None:
    _ctx.set(None)

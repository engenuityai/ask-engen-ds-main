"""LLM router — single entry point for every pipeline call.

call_llm() reads the active mode from llm_store and dispatches:

  mode = "public"  →  providers/free/openrouter/   (OpenRouter, free, DS .env key)
  mode = byok      →  providers/byok/               (Gemini or Anthropic, key via /llm/test-key)

The gate (gate.py) wraps this function in a semaphore + asyncio.to_thread
before it reaches any provider, so this router is always called from a
worker thread — no async code here.

Backward-compat re-exports keep pipeline.py / steps.py / datasets.py imports
unchanged — they all import from src.core.llm.client and get the same names.
"""

from typing import Optional

from src.config import settings
from src.core.llm import store as llm_store
from src.core.llm.types import (
    ERROR_FRIENDLY_MESSAGES,       # noqa: F401 — re-exported for pipeline.py
    LLMProviderError,              # noqa: F401 — re-exported for pipeline.py / datasets.py
    LLMResult,                     # noqa: F401 — re-exported
    TokenLimitExceededError,       # noqa: F401 — re-exported for pipeline.py / datasets.py
    is_token_limit_error,
)
from src.utils.logging_utils import console_log

__all__ = [
    "call_llm",
    "LLMResult",
    "LLMProviderError",
    "TokenLimitExceededError",
    "ERROR_FRIENDLY_MESSAGES",
]

_ESTIMATE_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // _ESTIMATE_CHARS_PER_TOKEN


def call_llm(
    mode: str,
    system: str,
    user: str,
    max_tokens: Optional[int] = None,
    gemini_model: Optional[str] = None,
    gemini_static_context: Optional[str] = None,
) -> LLMResult:
    """Route a prompt to the active provider for ``mode``.

    Args:
        mode:                  Service mode string (e.g. "analytics").
        system:                System prompt.
        user:                  User / query prompt.
        max_tokens:            Output token cap; defaults to settings value.
        gemini_model:          Override Gemini model name (BYOK path only).
        gemini_static_context: Large static block for Gemini context caching
                               (prepended inline for OpenRouter).
    """
    from src.core.llm.providers.byok import call_byok
    from src.core.llm.providers.free.openrouter import call_openrouter_public

    max_tokens = max_tokens or settings.llm_max_output_tokens

    estimated_input = (
        _estimate_tokens(system)
        + _estimate_tokens(user)
        + _estimate_tokens(gemini_static_context or "")
    )
    console_log(
        mode,
        "LLM_CALL",
        f"estimated input ~{estimated_input} tokens "
        f"(cap {settings.llm_max_input_tokens}), "
        f"output cap {max_tokens}, provider={settings.llm_provider}",
    )

    if estimated_input > settings.llm_max_input_tokens:
        raise TokenLimitExceededError(
            f"Estimated input size (~{estimated_input} tokens) exceeds "
            f"llm_max_input_tokens ({settings.llm_max_input_tokens})"
        )

    # Public mode (OpenRouter free tier) — overrides LLM_PROVIDER env var.
    # Activated when the backend sets mode="public" via POST /llm/config.
    if llm_store.get_mode(mode) == "public":
        return call_openrouter_public(mode, system, user, max_tokens, gemini_static_context)

    # BYOK — dispatches to Gemini or Anthropic based on settings.llm_provider
    return call_byok(mode, system, user, max_tokens, gemini_model, gemini_static_context)

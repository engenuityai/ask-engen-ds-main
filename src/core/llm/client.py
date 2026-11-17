"""LLM router — dispatches call_llm() to the right provider.

Provider selection order
------------------------
1. Public mode   → OpenRouter free tier (Nemotron Lightning / Ultra)
                   Activated when the backend sets mode="public" via /llm/config.
2. Gemini paid   → google-generativeai SDK, key received via /llm/test-key.
3. Anthropic     → anthropic SDK, key from settings.anthropic_api_key.

Provider implementations live in their own modules:
  gemini.py      — Gemini paid (key tested + stored in llm_store)
  openrouter.py  — OpenRouter free (token rotation, Nemotron models)
"""

from typing import Optional

from src.config import settings
from src.core.llm import store as llm_store
from src.core.llm.types import (
    ERROR_FRIENDLY_MESSAGES,
    LLMProviderError,
    LLMResult,
    TokenLimitExceededError,
    is_token_limit_error,
)
from src.utils.logging_utils import console_log

# ---------------------------------------------------------------------------
# Backward-compat re-exports — callers (pipeline.py, datasets.py, steps.py)
# import these from client; keep them here so no call-site changes are needed.
# ---------------------------------------------------------------------------
__all__ = [
    "call_llm",
    "LLMResult",
    "LLMProviderError",
    "TokenLimitExceededError",
    "ERROR_FRIENDLY_MESSAGES",
]


# ---------------------------------------------------------------------------
# Anthropic (fallback provider)
# ---------------------------------------------------------------------------

def _call_anthropic(mode: str, system: str, user: str, max_tokens: int) -> LLMResult:
    import anthropic

    from src.core.llm.types import _DEFAULT_ERROR_CODE

    def _map(exc: Exception) -> LLMProviderError:
        message = str(exc)
        lowered = message.lower()
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMProviderError("deadline_exceeded", message)
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMProviderError("service_unavailable", message)
        status = getattr(exc, "status_code", None)
        if status == 400:
            code = "invalid_request"
        elif status == 401:
            code = "authentication"
        elif status == 403:
            code = "permission_denied"
        elif status == 404:
            code = "model_not_found" if "model" in lowered else "not_found"
        elif status == 409:
            code = "aborted"
        elif status == 429:
            code = "quota_exceeded" if "quota" in lowered else "rate_limit_exceeded"
        elif status == 500:
            code = "api_error"
        elif status == 529:
            code = "service_unavailable"
        else:
            code = _DEFAULT_ERROR_CODE
        return LLMProviderError(code, message)

    api_key = llm_store.get_key(mode, "anthropic") or settings.anthropic_api_key
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            output_config={"effort": "medium"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        if is_token_limit_error(str(exc)):
            raise TokenLimitExceededError(str(exc)) from exc
        raise _map(exc) from exc

    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResult(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=settings.anthropic_model,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


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
        mode:                 Service mode string (e.g. "analytics").
        system:               System prompt text.
        user:                 User / query prompt text.
        max_tokens:           Output token cap; defaults to settings value.
        gemini_model:         Override the Gemini model name.
        gemini_static_context: Large static block passed to Gemini (or OpenRouter)
                              — eligible for Gemini context caching.
    """
    from src.core.llm.gemini import call_gemini
    from src.core.llm.openrouter import call_openrouter_public

    max_tokens = max_tokens or settings.llm_max_output_tokens

    estimated_input = (
        _estimate_tokens(system)
        + _estimate_tokens(user)
        + _estimate_tokens(gemini_static_context or "")
    )
    console_log(
        mode,
        "LLM_CALL",
        f"estimated input ~{estimated_input} tokens (cap {settings.llm_max_input_tokens}), "
        f"output cap {max_tokens}, provider={settings.llm_provider}",
    )
    if estimated_input > settings.llm_max_input_tokens:
        raise TokenLimitExceededError(
            f"Estimated input size (~{estimated_input} tokens) exceeds "
            f"llm_max_input_tokens ({settings.llm_max_input_tokens})"
        )

    # Public mode overrides LLM_PROVIDER — checked first so an admin setting
    # of mode=public always wins regardless of the env-based provider.
    if llm_store.get_mode(mode) == "public":
        return call_openrouter_public(mode, system, user, max_tokens, gemini_static_context)

    if settings.llm_provider == "gemini":
        return call_gemini(
            mode,
            system,
            user,
            max_tokens,
            gemini_model or settings.gemini_model_data_processing,
            gemini_static_context,
        )

    if settings.llm_provider == "anthropic":
        return _call_anthropic(mode, system, user, max_tokens)

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider!r}")

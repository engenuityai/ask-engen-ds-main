"""Gemini paid provider.

API key is supplied by the backend via POST /llm/test-key (stored in llm_store).
Supports in-process context caching (gemini_cache) for Agent 1's large static
prompt block — falls back to inline context when the cache is unavailable.
"""

from typing import Optional

from src.config import settings
from src.core.llm import store as llm_store
from src.core.llm.types import (
    LLMProviderError,
    LLMResult,
    TokenLimitExceededError,
    _DEFAULT_ERROR_CODE,
    is_token_limit_error,
)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

def map_gemini_exception(exc: "GoogleAPICallError") -> LLMProviderError:
    message = str(exc)
    lowered = message.lower()
    status = getattr(exc, "code", None)

    if status == 400:
        if "unknown" in lowered and "parameter" in lowered:
            code = "parameter_unknown"
        elif type(exc).__name__ == "FailedPrecondition":
            code = "failed_precondition"
        else:
            code = "invalid_request"
    elif status == 401:
        code = "authentication"
    elif status == 403:
        code = "permission_denied"
    elif status == 404:
        code = "model_not_found" if "model" in lowered else "not_found"
    elif status == 409:
        code = "already_exists" if type(exc).__name__ == "AlreadyExists" else "aborted"
    elif status == 412:
        code = "failed_precondition"
    elif status == 416:
        code = "out_of_range"
    elif status == 429:
        code = "quota_exceeded" if "quota" in lowered else "rate_limit_exceeded"
    elif status == 499:
        code = "cancelled"
    elif status == 501:
        code = "unimplemented"
    elif status in (502, 503):
        code = "service_unavailable"
    elif status == 504:
        code = "deadline_exceeded"
    else:
        code = _DEFAULT_ERROR_CODE

    return LLMProviderError(code, message)


# ---------------------------------------------------------------------------
# Provider call
# ---------------------------------------------------------------------------

def call_gemini(
    mode: str,
    system: str,
    user: str,
    max_tokens: int,
    model_name: str,
    static_context: Optional[str],
) -> LLMResult:
    """Call the Gemini paid API.

    Uses context caching when ``static_context`` is provided and caching is
    enabled in settings; falls back to prepending it inline otherwise.
    """
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPICallError

    api_key = llm_store.get_key(mode, "gemini") or settings.gemini_api_key
    genai.configure(api_key=api_key)

    cached_content = None
    if static_context and settings.gemini_context_cache_enabled:
        from src.core.llm import gemini_cache

        cached_content = gemini_cache.get_or_create_cache(
            mode=mode,
            model_name=model_name,
            system_instruction=system,
            static_context=static_context,
            ttl_seconds=settings.gemini_cache_ttl_seconds,
        )

    if cached_content is not None:
        model = genai.GenerativeModel.from_cached_content(cached_content=cached_content)
    else:
        model = genai.GenerativeModel(model_name, system_instruction=system)
        if static_context:
            user = f"{static_context}\n\n{user}"

    try:
        response = model.generate_content(
            user, generation_config={"max_output_tokens": max_tokens}
        )
    except GoogleAPICallError as exc:
        if is_token_limit_error(str(exc)):
            raise TokenLimitExceededError(str(exc)) from exc
        raise map_gemini_exception(exc) from exc

    usage = response.usage_metadata
    return LLMResult(
        text=response.text,
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
        model=model_name,
        cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
    )

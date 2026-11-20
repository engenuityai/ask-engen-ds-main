"""Gemini paid provider (BYOK — key received via POST /llm/test-key).

Supports in-process context caching for Agent 1's large static prompt block.
Falls back to inline context when the cache is unavailable.
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

def map_exception(exc: "APIError") -> LLMProviderError:
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
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError

    api_key = llm_store.get_key(mode, "gemini") or settings.gemini_api_key
    client = genai.Client(api_key=api_key)

    cached_content_name = None
    if static_context and settings.gemini_context_cache_enabled:
        from src.core.llm.providers.byok.gemini.cache import get_or_create_cache

        cached = get_or_create_cache(
            mode=mode,
            model_name=model_name,
            system_instruction=system,
            static_context=static_context,
            ttl_seconds=settings.gemini_cache_ttl_seconds,
            api_key=api_key,
        )
        if cached is not None:
            cached_content_name = cached.name

    if cached_content_name:
        config = types.GenerateContentConfig(
            cached_content=cached_content_name,
            max_output_tokens=max_tokens,
        )
        contents = user
    else:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        contents = f"{static_context}\n\n{user}" if static_context else user

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
    except APIError as exc:
        if is_token_limit_error(str(exc)):
            raise TokenLimitExceededError(str(exc)) from exc
        raise map_exception(exc) from exc

    usage = response.usage_metadata
    return LLMResult(
        text=response.text,
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
        model=model_name,
        cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
    )

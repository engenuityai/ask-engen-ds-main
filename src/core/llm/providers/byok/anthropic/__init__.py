"""Anthropic paid provider (BYOK — key from settings.anthropic_api_key)."""

from src.config import settings
from src.core.llm import store as llm_store
from src.core.llm.types import (
    LLMProviderError,
    LLMResult,
    TokenLimitExceededError,
    _DEFAULT_ERROR_CODE,
    is_token_limit_error,
)


def _map_exception(exc: Exception) -> LLMProviderError:
    import anthropic

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


def call_anthropic(mode: str, system: str, user: str, max_tokens: int) -> LLMResult:
    import anthropic

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
        raise _map_exception(exc) from exc

    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResult(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=settings.anthropic_model,
    )

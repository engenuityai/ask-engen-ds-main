"""OpenRouter free-tier provider.

Handles HTTP calls to the OpenRouter chat-completions endpoint and owns the
weekly-token-based model-rotation logic (Nemotron Lightning → Nemotron Ultra).

OPEN_ROUTER_KEY lives in the DS .env — the backend never sends it.
The model pool and weekly limits live in token_tracker.py.
"""

from typing import Optional

import requests

from src.config import settings
from src.core.llm.types import (
    LLMProviderError,
    LLMResult,
    TokenLimitExceededError,
    is_token_limit_error,
)
from src.utils.logging_utils import console_log


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def _map_status(status: int, body: str) -> str:
    low = body.lower()
    if status == 400:
        return "invalid_request"
    if status == 401:
        return "authentication"
    if status == 403:
        return "permission_denied"
    if status == 404:
        return "model_not_found"
    if status == 429:
        return "quota_exceeded" if "quota" in low else "rate_limit_exceeded"
    if status == 500:
        return "api_error"
    if status in (502, 503):
        return "service_unavailable"
    if status == 504:
        return "deadline_exceeded"
    return "api_error"


def _http_call(system: str, user: str, model: str, max_tokens: int) -> LLMResult:
    """Raw HTTP request to OpenRouter — no token tracking or rotation here."""
    api_key = settings.open_router_key
    if not api_key:
        raise LLMProviderError(
            "authentication",
            "OPEN_ROUTER_KEY is not configured in the DS environment",
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise LLMProviderError("deadline_exceeded", "OpenRouter request timed out") from exc
    except requests.exceptions.ConnectionError as exc:
        raise LLMProviderError("service_unavailable", f"Could not reach OpenRouter: {exc}") from exc
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        if is_token_limit_error(body):
            raise TokenLimitExceededError(body) from exc
        code = _map_status(exc.response.status_code, body)
        raise LLMProviderError(code, body) from exc

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    content = choice["message"].get("content") or ""

    if not content:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "OpenRouter returned null/empty content for model=%s finish_reason=%s",
            model,
            choice.get("finish_reason"),
        )

    return LLMResult(
        text=content,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=model,
    )


# ---------------------------------------------------------------------------
# Public entry point — token rotation + context-window guard
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def call_openrouter_public(
    mode: str,
    system: str,
    user: str,
    max_tokens: int,
    static_context: Optional[str] = None,
) -> LLMResult:
    """Free-tier entry point with weekly-token-based model rotation.

    Rotation order: Nemotron Lightning → Nemotron Ultra (see token_tracker).
    Raises TokenLimitExceededError when the prompt exceeds every model's safe
    context window so the pipeline can surface a friendly message upstream.
    """
    from src.core.llm import token_tracker

    if static_context:
        user = f"{static_context}\n\n{user}"

    estimated = _estimate_tokens(system) + _estimate_tokens(user)

    max_context = max(m["context_window"] for m in token_tracker.PUBLIC_MODEL_POOL)
    if estimated > max_context * 0.90:
        raise TokenLimitExceededError(
            f"Prompt (~{estimated} tokens) exceeds the largest public-mode model's "
            f"context window ({max_context})"
        )

    active_model = token_tracker.get_active_model(estimated)
    console_log(mode, "OPENROUTER", f"model={active_model}, estimated_input={estimated}")

    result = _http_call(system, user, active_model, max_tokens)
    token_tracker.add_tokens(active_model, result.input_tokens + result.output_tokens)
    return result


# ---------------------------------------------------------------------------
# Backward-compat alias — call_openrouter was imported by client.py
# ---------------------------------------------------------------------------
call_openrouter = _http_call

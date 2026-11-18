"""Free-tier Gemini provider.

Public API
----------
call_gemini_free(mode, system, user, max_tokens, static_context) -> LLMResult
get_usage_stats() -> list[dict]      used by GET /llm/public-usage

Routing
-------
ModelRouter selects the highest-priority model whose RPM and RPD are both
below 80% of their free-tier limits. Priority order:
  1. gemini-3.5-flash-lite   (30 RPM / 1500 RPD)
  2. gemini-2.5-flash-lite   (30 RPM / 1500 RPD)
  3. gemini-2.5-flash        (10 RPM /  250 RPD)

Implicit caching
----------------
Session history is maintained externally via src.core.session.memory as JSON
files keyed by (user_id, session_id, mode) — unchanged from the existing
pattern. The *user* prompt arriving here already contains the formatted history
string. static_context (schema JSON / document) is kept as a separate first
turn so Gemini's implicit cache activates for every request to the same dataset.

Singleton
---------
_client is module-level and created once on first call. The MemoryRateTracker
state must survive across requests, so never re-instantiate the client.
"""

import os
from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

from src.core.llm.types import (
    LLMProviderError,
    LLMResult,
    TokenLimitExceededError,
    _DEFAULT_ERROR_CODE,
    is_token_limit_error,
)
from src.utils.logging_utils import console_log

from .model_router import ModelRouter, RateLimitExhausted
from .models import MODEL_REGISTRY, THRESHOLD
from .prompt_builder import build_contents, get_system_prompt
from .store import MemoryRateTracker


# ---------------------------------------------------------------------------
# Error mapping (mirrors byok/gemini — kept local to avoid cross-dep)
# ---------------------------------------------------------------------------

def _map_exc(exc: GoogleAPICallError) -> LLMProviderError:
    msg = str(exc)
    low = msg.lower()
    code = getattr(exc, "code", None)
    if code == 400:
        err = "invalid_request"
    elif code == 401:
        err = "authentication"
    elif code == 403:
        err = "permission_denied"
    elif code == 404:
        err = "model_not_found" if "model" in low else "not_found"
    elif code == 429:
        err = "quota_exceeded" if "quota" in low else "rate_limit_exceeded"
    elif code in (502, 503):
        err = "service_unavailable"
    elif code == 504:
        err = "deadline_exceeded"
    else:
        err = _DEFAULT_ERROR_CODE
    return LLMProviderError(err, msg)


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

class _FreeGeminiClient:
    """Singleton — holds tracker + router state across requests."""

    def __init__(self, api_key: str) -> None:
        genai.configure(api_key=api_key)
        self._tracker = MemoryRateTracker()
        self._router  = ModelRouter(self._tracker)

    def call(
        self,
        mode: str,
        system: str,
        user: str,
        max_tokens: int,
        static_context: Optional[str],
    ) -> LLMResult:
        try:
            model_id = self._router.select()
        except RateLimitExhausted as exc:
            raise LLMProviderError("quota_exceeded", str(exc)) from exc

        self._tracker.increment(model_id)
        console_log(mode, "FREE_GEMINI", f"model={model_id}")

        gemini_model = genai.GenerativeModel(
            model_id,
            system_instruction=system,  # auto-cached by Gemini on every call
        )
        contents = build_contents(static_context, user)

        try:
            response = gemini_model.generate_content(
                contents,
                generation_config={"max_output_tokens": max_tokens},
            )
        except GoogleAPICallError as exc:
            if is_token_limit_error(str(exc)):
                raise TokenLimitExceededError(str(exc)) from exc
            raise _map_exc(exc) from exc

        usage = response.usage_metadata
        return LLMResult(
            text=response.text,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            model=model_id,
            cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
        )

    def usage_stats(self) -> list[dict]:
        """Return per-model RPM/RPD usage for the /llm/public-usage endpoint."""
        counts_map = self._tracker.get_all_counts()
        active_model = None
        try:
            active_model = self._router.select()
        except RateLimitExhausted:
            pass

        stats = []
        for m in sorted(MODEL_REGISTRY, key=lambda x: x["priority"]):
            mid   = m["id"]
            counts = counts_map.get(mid, {"rpm": 0, "rpd": 0})
            rpm_pct = round(counts["rpm"] / m["rpm_limit"] * 100, 2) if m["rpm_limit"] else 0.0
            rpd_pct = round(counts["rpd"] / m["rpd_limit"] * 100, 2) if m["rpd_limit"] else 0.0
            stats.append({
                "id":           mid,
                "priority":     m["priority"],
                "rpm_used":     counts["rpm"],
                "rpm_limit":    m["rpm_limit"],
                "rpm_pct":      rpm_pct,
                "rpd_used":     counts["rpd"],
                "rpd_limit":    m["rpd_limit"],
                "rpd_pct":      rpd_pct,
                "context_window": m["context_window"],
                "latency_ms":   m["latency_ms"],
                "released":     m["released"],
                "is_active":    mid == active_model,
                "over_threshold": rpm_pct >= THRESHOLD * 100 or rpd_pct >= THRESHOLD * 100,
            })
        return stats


_client: Optional[_FreeGeminiClient] = None


def _get_client() -> _FreeGeminiClient:
    global _client
    if _client is None:
        api_key = os.environ.get("FREE_GEMINI_API_KEY", "")
        if not api_key:
            raise LLMProviderError(
                "authentication",
                "FREE_GEMINI_API_KEY is not set in the DS environment",
            )
        _client = _FreeGeminiClient(api_key)
    return _client


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def call_gemini_free(
    mode: str,
    system: str,
    user: str,
    max_tokens: int,
    static_context: Optional[str] = None,
) -> LLMResult:
    """Route one prompt through the free-tier Gemini model pool.

    Session history is expected to already be embedded in *user* by the caller
    via src.core.session.memory.format_history_for_prompt() — same as the
    existing OpenRouter public path. Session JSON files (user_id, session_id,
    mode) are unchanged.
    """
    return _get_client().call(mode, system, user, max_tokens, static_context)


def get_usage_stats() -> list[dict]:
    """Return per-model usage stats for GET /llm/public-usage."""
    return _get_client().usage_stats()

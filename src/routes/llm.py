"""LLM configuration endpoints — called by the backend to hold API keys in
memory and push config changes at save time.
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.llm import store as llm_store

# ---------------------------------------------------------------------------
# Display metadata for the public-mode free Gemini pool.
# Keyed by the exact model ID in free/gemini/models.py MODEL_REGISTRY.
# ---------------------------------------------------------------------------
_PUBLIC_MODEL_DISPLAY: dict[str, dict] = {
    "gemini-3.5-flash-lite": {
        "name": "Gemini 3.5 Flash Lite",
        "provider_name": "Google",
        "provider_key": "gemini",
        "latency": "~600ms",
        "latency_ms": 600,
        "released": "Nov 2025",
    },
    "gemini-2.5-flash-lite": {
        "name": "Gemini 2.5 Flash Lite",
        "provider_name": "Google",
        "provider_key": "gemini",
        "latency": "~700ms",
        "latency_ms": 700,
        "released": "Sep 2025",
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "provider_name": "Google",
        "provider_key": "gemini",
        "latency": "~900ms",
        "latency_ms": 900,
        "released": "May 2025",
    },
}


def _fmt_tokens(n: int) -> str:
    """Format a token count as a human-readable string (e.g. 613M, 3.46T)."""
    if n == 0:
        return "0"
    if n >= 1_000_000_000_000:
        v = n / 1_000_000_000_000
        return f"{v:.2f}".rstrip("0").rstrip(".") + "T"
    if n >= 1_000_000_000:
        v = n / 1_000_000_000
        return f"{v:.2f}".rstrip("0").rstrip(".") + "B"
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "K"
    return str(n)


def _fmt_context(n: int) -> str:
    """Format a context window size (e.g. 262144 → '262K', 1048576 → '1M')."""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)

router = APIRouter(prefix="/llm", tags=["llm"])

# Backend service IDs that map to the DS "analytics" mode
_SERVICE_ID_MAP: dict[str, str] = {
    "datapulse": "analytics",
    "analytics": "analytics",
}


def _resolve(service_id: str) -> str:
    return _SERVICE_ID_MAP.get(service_id, service_id)


class TestKeyRequest(BaseModel):
    service_id: str
    provider: str
    api_key: str
    hold_in_memory: bool = True


class ConfigRequest(BaseModel):
    service_id: str
    mode: str
    status: str
    primary_provider: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None
    public_pool_order: list[str] = []
    updated_at: Optional[str] = None


def _validate_gemini_key(api_key: str) -> None:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    # list_models is a lightweight auth-only round-trip (no tokens consumed)
    list(genai.list_models())


@router.post("/test-key")
async def test_key(body: TestKeyRequest):
    if body.provider != "gemini":
        raise HTTPException(
            status_code=400,
            detail={"message": f"Provider '{body.provider}' is not supported in phase 1"},
        )

    try:
        await asyncio.to_thread(_validate_gemini_key, body.api_key)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": f"Key validation failed: {exc}"},
        ) from exc

    if body.hold_in_memory:
        llm_store.set_key(_resolve(body.service_id), body.provider, body.api_key)

    return {"success": True, "message": "Key validated"}


@router.post("/config")
async def set_config(body: ConfigRequest):
    service_id = _resolve(body.service_id)
    provider = body.provider or body.primary_provider

    llm_store.set_config(
        service_id=service_id,
        provider=provider,
        api_key=body.api_key,
        mode=body.mode,
    )

    return {"success": True, "message": "Config updated"}


@router.get("/public-usage")
async def get_public_usage():
    """Return RPM + RPD usage for every model in the free Gemini public pool.

    Called by the backend to populate the LLM Configuration page's public-mode
    table with live progress bars.

    Response shape (per model):
      id, name, provider_name, provider_key     — display info
      rpm_used, rpm_limit, rpm_pct              — requests-per-minute
      rpd_used, rpd_limit, rpd_pct              — requests-per-day
      context_window, context_label             — token context size
      is_active, over_threshold                 — routing state
      latency, latency_ms, released             — informational

    Legacy fields tokens_used / weekly_limit are set to 0 (not applicable for
    free Gemini). Update the frontend to use rpm_pct / rpd_pct for progress bars.
    """
    from src.core.llm.providers.free.gemini import get_usage_stats

    stats = await asyncio.to_thread(get_usage_stats)
    models = []
    active_model = None

    for s in stats:
        if s["is_active"]:
            active_model = s["id"]
        display = _PUBLIC_MODEL_DISPLAY.get(s["id"], {})
        models.append({
            # Identity + display
            "id":               s["id"],
            "name":             display.get("name", s["id"]),
            "provider_name":    display.get("provider_name", "Google"),
            "provider_key":     display.get("provider_key", "gemini"),
            # RPM
            "rpm_used":         s["rpm_used"],
            "rpm_limit":        s["rpm_limit"],
            "rpm_pct":          s["rpm_pct"],
            # RPD
            "rpd_used":         s["rpd_used"],
            "rpd_limit":        s["rpd_limit"],
            "rpd_pct":          s["rpd_pct"],
            # Context
            "context_window":   s["context_window"],
            "context_label":    _fmt_context(s["context_window"]),
            # Routing state
            "is_active":        s["is_active"],
            "over_threshold":   s["over_threshold"],
            # Informational
            "latency":          display.get("latency", ""),
            "latency_ms":       display.get("latency_ms", s.get("latency_ms", 0)),
            "released":         display.get("released", s.get("released", "")),
            # Legacy fields — kept so the existing BE/FE don't break before update
            "tokens_used":      0,
            "tokens_used_label": "0",
            "weekly_limit":     0,
            "weekly_limit_label": "0",
            "percentage":       s["rpd_pct"],
        })
    return {"models": models, "active_model": active_model}

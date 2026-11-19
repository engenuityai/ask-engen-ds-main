"""Know Your Company service configuration."""

from src.config import settings as _global_settings

SERVICE_MODE = "know_your_company"

LLM_PROVIDER = _global_settings.llm_provider
MODEL = _global_settings.gemini_model_enrichment

# 90% threshold for free-pool sessions: docs are inline in the prompt, so the
# combined context fills much faster — expire the chat earlier to stay safe.
KYC_FREE_CONTEXT_RATIO = 0.9
# Standard threshold for BYOK sessions: docs are served from Gemini's implicit
# cache and don't count against the session-history token estimate.
KYC_BYOK_CONTEXT_RATIO = 0.5

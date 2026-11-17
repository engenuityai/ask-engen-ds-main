"""BYOK (Bring Your Own Key) provider dispatcher.

Picks the right paid provider based on settings.llm_provider:
  gemini    → providers/byok/gemini/   (key stored via /llm/test-key)
  anthropic → providers/byok/anthropic/ (key from settings.anthropic_api_key)
"""

from typing import Optional

from src.config import settings
from src.core.llm.types import LLMResult


def call_byok(
    mode: str,
    system: str,
    user: str,
    max_tokens: int,
    gemini_model: Optional[str] = None,
    gemini_static_context: Optional[str] = None,
) -> LLMResult:
    if settings.llm_provider == "gemini":
        from src.core.llm.providers.byok.gemini import call_gemini

        return call_gemini(
            mode,
            system,
            user,
            max_tokens,
            gemini_model or settings.gemini_model_data_processing,
            gemini_static_context,
        )

    if settings.llm_provider == "anthropic":
        from src.core.llm.providers.byok.anthropic import call_anthropic

        return call_anthropic(mode, system, user, max_tokens)

    raise ValueError(f"Unsupported BYOK provider: {settings.llm_provider!r}. Use 'gemini' or 'anthropic'.")

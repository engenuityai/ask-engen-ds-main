"""LLM package.

Public API
----------
call_llm()          — route a prompt to the active provider
LLMResult           — result dataclass
LLMProviderError    — provider-level error with a friendly ``code``
TokenLimitExceededError — prompt too large for the model

Provider modules
----------------
gemini.py      — Gemini paid (key stored via /llm/test-key)
openrouter.py  — OpenRouter free (Nemotron, weekly token rotation)
client.py      — router that dispatches to the above
"""

from src.core.llm.client import call_llm  # noqa: F401
from src.core.llm.types import (  # noqa: F401
    ERROR_FRIENDLY_MESSAGES,
    LLMProviderError,
    LLMResult,
    TokenLimitExceededError,
)

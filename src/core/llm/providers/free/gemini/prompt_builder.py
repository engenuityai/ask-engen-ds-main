"""Implicit-cache-friendly prompt builder for free-tier Gemini.

Implicit caching fires when Gemini sees an identical prefix across requests.
The fixed prefix is: system_instruction + static_context (schema / document).
Session history + the new question are appended after the prefix.

Rule: build_contents() must produce byte-for-byte identical output for the
same (static_context, service) inputs — never embed timestamps, session IDs,
or any other per-request data in the fixed part.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# System prompts — one per service mode.
# Keep these stable; every edit invalidates Gemini's implicit cache for all
# active sessions of that mode until the new prefix accumulates enough calls.
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "analytics": (
        "You are an expert data analyst. "
        "Use the schema context provided to generate accurate, read-only SQL queries "
        "and explain query results in clear natural language. "
        "Never execute destructive SQL (INSERT, UPDATE, DELETE, DROP, TRUNCATE). "
        "When a question is ambiguous, ask for clarification before generating SQL."
    ),
    "kyc": (
        "You are a company knowledge assistant. "
        "Use the document context provided to answer questions about the company accurately. "
        "Cite the relevant section when possible. "
        "If the answer is not in the provided context, say so clearly."
    ),
    "kyf": (
        "You are a business forecasting assistant. "
        "Use the context provided to reason about trends, projections, and future scenarios. "
        "Distinguish clearly between data-backed conclusions and estimates."
    ),
}

_CONTEXT_ACK = "Understood. I have reviewed the context and I am ready for questions."

# Gemini implicit caching activates at >= 2 048 tokens for 2.5/3.5 models.
# If a service's static_context is short, the caller should pad it by adding
# few-shot examples to the system prompt before calling build_contents().
MIN_CACHE_TOKENS = 2_048


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_system_prompt(service: str) -> str:
    """Return the fixed system instruction for *service*.

    Raises KeyError for unknown service IDs so the caller gets a clear error
    instead of silently sending an empty instruction.
    """
    if service not in SYSTEM_PROMPTS:
        raise KeyError(
            f"Unknown service {service!r}. "
            f"Available: {list(SYSTEM_PROMPTS)}"
        )
    return SYSTEM_PROMPTS[service]


def build_contents(
    static_context: Optional[str],
    user_prompt: str,
) -> list[dict] | str:
    """Build the Gemini `contents` argument for generate_content().

    Without static_context: returns a plain string (single-turn call).

    With static_context: returns a list in Gemini's multi-turn format so the
    static_context occupies the first user turn — always at the same position
    across requests — which is what triggers Gemini's implicit caching.

    Args:
        static_context: Schema JSON (analytics), document text (kyc/kyf), or
                        any large fixed block. Must be identical across calls
                        for the same dataset/document version.
        user_prompt:    The dynamic part: formatted session history + the new
                        question. Produced by format_history_for_prompt() +
                        the current question appended.
    """
    if not static_context:
        return user_prompt

    return [
        # Fixed prefix — identical bytes every call → Gemini implicit cache
        {"role": "user",  "parts": [{"text": static_context}]},
        {"role": "model", "parts": [{"text": _CONTEXT_ACK}]},
        # Dynamic suffix — session history + current question
        {"role": "user",  "parts": [{"text": user_prompt}]},
    ]

"""KYC pipeline steps — Agent 4 (Company Knowledge Assistant).

Single step: run_kyc_agent() sends the user's question to Agent 4 with the
uploaded company documents as the Gemini implicit cache prefix.

Gemini caching works identically for both BYOK and free-pool modes:
  - BYOK:      get_or_create_cache() in providers/byok/gemini/cache.py
  - Free pool: build_contents(static_context, user) in
               providers/free/gemini/prompt_builder.py
Both paths receive gemini_static_context via call_llm_guarded → call_llm.
"""

import json
import re

from src.config import settings
from src.core.llm.client import call_llm
from src.core.llm.gate import call_llm_guarded, get_call_timeout
from src.services.know_you_company import prompt_loader
from src.utils.cost_utils import log_cost
from src.utils.logging_utils import console_log

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)

_FALLBACK_ANSWER = (
    "I wasn't able to process your request at this time. Please try again."
)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _extract_json(text: str) -> dict:
    return json.loads(_strip_fences(text))


async def run_kyc_agent(
    *,
    mode: str,
    query: str,
    docs_context: str,
    memory_context: str,
    answer_id: str,
    user_id: str,
    session_id: str,
) -> tuple[str, list[str], list[str]]:
    """Call Agent 4 with uploaded docs as the Gemini implicit cache prefix.

    Returns (answer_text, sources, follow_ups).

    The docs_context is passed as gemini_static_context so it is served from
    Gemini's implicit cache on every turn where the document set has not
    changed — meaning each follow-up question only pays for the volatile
    query + history tokens, not the full document content again.
    """
    system_prompt = prompt_loader.build_agent_system_prompt("kyc_company_agent")

    # static_context = docs + instructions + output contract — cacheable.
    static_context = prompt_loader.render_task_static_context(
        "kyc_query_task",
        docs_context=docs_context,
    )
    # volatile prompt = query + session history — changes every call.
    volatile_prompt = prompt_loader.render_task_volatile_prompt(
        "kyc_query_task",
        query=query,
        memory_context=memory_context,
    )

    # When a static_context exists, pass it as gemini_static_context so both
    # the BYOK and free-pool providers can serve it from Gemini's implicit
    # cache. The base_prompt then carries only the volatile portion.
    gemini_static_context = static_context if static_context else None
    base_prompt = (
        volatile_prompt
        if gemini_static_context
        else (f"{static_context}\n\n{volatile_prompt}" if static_context else volatile_prompt)
    )

    console_log(
        mode, "KYC_AGENT",
        f"answer_id={answer_id} docs_len={len(docs_context)} "
        f"history_len={len(memory_context)} query={query!r}",
    )

    result = await call_llm_guarded(
        call_llm,
        mode,
        system_prompt,
        base_prompt,
        call_timeout=get_call_timeout(mode),
        gemini_static_context=gemini_static_context,
    )

    await log_cost(
        user_id,
        session_id,
        answer_id,
        "kyc_agent",
        result.model,
        result.input_tokens,
        result.output_tokens,
        result.cached_tokens,
        mode,
    )

    console_log(mode, "KYC_AGENT", f"raw response:\n{result.text}")

    try:
        decision = _extract_json(result.text)
    except (ValueError, json.JSONDecodeError):
        return _FALLBACK_ANSWER, [], []

    text = decision.get("text") or _FALLBACK_ANSWER
    sources = list(decision.get("sources") or [])
    follow_ups = list(decision.get("followups") or [])[:2]
    return text, sources, follow_ups

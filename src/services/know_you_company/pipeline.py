"""Know Your Company query pipeline — mode="know_your_company".

Flow for each chat request:
  1. Check chat_expire and max_context_window → early-return empty envelope if set
  2. Load all uploaded MD docs from disk
  3. If no docs: return a friendly "please upload documents first" response
  4. Load session history → memory_context
  5. Run KYC agent (Agent 4) — docs sent as Gemini implicit cache prefix
  6. Build text-only response (no SQL, no tables, no charts)
  7. Append session memory + log timing in background tasks

Context-window threshold:
  - Free-pool sessions (mode="public"): 90% — docs are inline in the prompt
    so the combined context fills faster; expire the chat earlier.
  - BYOK sessions: 50% — docs live in Gemini's implicit cache and don't count
    against the session-history token estimate.
"""

import asyncio
import time

from fastapi import BackgroundTasks

from src.core.llm.client import ERROR_FRIENDLY_MESSAGES, LLMProviderError, TokenLimitExceededError
from src.core.llm import store as llm_store
from src.core.session import memory as session_memory
from src.schemas.user_request import QueryResponse, UserRequest
from src.services.analytics.tools import response_builder
from src.services.know_you_company import doc_loader, steps
from src.services.know_you_company.config import (
    KYC_BYOK_CONTEXT_RATIO,
    KYC_FREE_CONTEXT_RATIO,
)
from src.utils.id_utils import generate_answer_id, generate_question_id
from src.utils.logging_utils import StageTimer, console_log, log_timing

_TOKEN_LIMIT_MESSAGE = (
    "This session's conversation history has grown too large. "
    "Please start a new session to continue asking questions about your documents."
)

_NO_DOCS_MESSAGE = (
    "No company documents have been uploaded yet for this service. "
    "Please upload your company documents first so I can answer questions about them."
)


def _provider_error_message(exc: LLMProviderError) -> str:
    return ERROR_FRIENDLY_MESSAGES.get(exc.code, ERROR_FRIENDLY_MESSAGES["api_error"])


def _context_ratio(mode: str) -> float:
    """Return 0.9 for free-pool sessions (docs inline = large prompt),
    0.5 for BYOK sessions (docs are cached)."""
    if llm_store.get_mode(mode) == "public":
        return KYC_FREE_CONTEXT_RATIO
    return KYC_BYOK_CONTEXT_RATIO


async def run_query_pipeline(
    user_request: UserRequest, background_tasks: BackgroundTasks
) -> QueryResponse:
    start = time.perf_counter()
    timer = StageTimer()
    mode = user_request.mode
    answer_id = generate_answer_id()
    question_id = user_request.question_id or generate_question_id()

    console_log(
        mode,
        "REQUEST",
        f"answer_id={answer_id} question_id={question_id} "
        f"user_id={user_request.user_id} session_id={user_request.session_id} "
        f"query={user_request.query!r}",
    )

    ratio = _context_ratio(mode)
    chat_expire = session_memory.is_chat_expired(
        user_request.user_id, user_request.session_id, mode
    )
    max_context_window = session_memory.is_context_window_full(
        user_request.user_id, user_request.session_id, mode, ratio=ratio
    )

    if chat_expire or max_context_window:
        console_log(
            mode,
            "ROUTE",
            f"chat_expire={chat_expire} max_context_window={max_context_window} "
            "— returning empty envelope",
        )
        response = response_builder.build_response(
            answer_id=answer_id,
            question_id=question_id,
            user_id=user_request.user_id,
            session_id=user_request.session_id,
            mode=mode,
            max_context_window=max_context_window,
            chat_expire=chat_expire,
            text_content=None,
            table=None,
            chart=None,
            follow_ups=[],
            merchant_disambiguation=None,
        )
        total_seconds = round(time.perf_counter() - start, 4)
        background_tasks.add_task(
            log_timing,
            user_request.user_id, user_request.session_id,
            answer_id, user_request.query, timer.stages, total_seconds, mode,
        )
        return response

    # Load company documents
    with timer.track("load_docs"):
        docs = doc_loader.load_docs()
        docs_context = doc_loader.join_docs_for_context(docs)

    if not docs:
        console_log(mode, "ROUTE", "no documents uploaded — returning guidance response")
        response = response_builder.build_response(
            answer_id=answer_id,
            question_id=question_id,
            user_id=user_request.user_id,
            session_id=user_request.session_id,
            mode=mode,
            max_context_window=False,
            chat_expire=False,
            text_content=_NO_DOCS_MESSAGE,
            table=None,
            chart=None,
            follow_ups=None,
            merchant_disambiguation=None,
        )
        total_seconds = round(time.perf_counter() - start, 4)
        background_tasks.add_task(
            log_timing,
            user_request.user_id, user_request.session_id,
            answer_id, user_request.query, timer.stages, total_seconds, mode,
        )
        return response

    console_log(mode, "DOCS", f"loaded {len(docs)} document(s), context_chars={len(docs_context)}")

    with timer.track("build_context"):
        history = session_memory.get_history(
            user_request.user_id, user_request.session_id, mode
        )
        memory_context = session_memory.format_history_for_prompt(history)

    console_log(mode, "MEMORY", f"{len(history)} stored turn(s):\n{memory_context}")

    token_limit_hit = False
    text_content = None
    follow_ups = None

    try:
        with timer.track("kyc_agent"):
            text_content, follow_ups = await steps.run_kyc_agent(
                mode=mode,
                query=user_request.query,
                docs_context=docs_context,
                memory_context=memory_context,
                answer_id=answer_id,
                user_id=user_request.user_id,
                session_id=user_request.session_id,
            )
    except TokenLimitExceededError:
        token_limit_hit = True
        text_content = _TOKEN_LIMIT_MESSAGE
        follow_ups = None
        console_log(mode, "TOKEN_LIMIT", "KYC agent aborted: token limit exceeded")
    except LLMProviderError as exc:
        text_content = _provider_error_message(exc)
        follow_ups = None
        console_log(mode, "PROVIDER_ERROR", f"KYC agent aborted: {exc.code} — {exc}")
    except asyncio.TimeoutError:
        text_content = ERROR_FRIENDLY_MESSAGES["deadline_exceeded"]
        follow_ups = None
        console_log(mode, "TIMEOUT", "KYC agent aborted: LLM call timed out")

    max_context_window = max_context_window or token_limit_hit

    response = response_builder.build_response(
        answer_id=answer_id,
        question_id=question_id,
        user_id=user_request.user_id,
        session_id=user_request.session_id,
        mode=mode,
        max_context_window=max_context_window,
        chat_expire=chat_expire,
        text_content=text_content,
        table=None,
        chart=None,
        follow_ups=follow_ups,
        merchant_disambiguation=None,
    )

    total_seconds = round(time.perf_counter() - start, 4)
    console_log(
        mode,
        "RESPONSE",
        f"maxContextWindow={max_context_window} chatExpire={chat_expire} "
        f"stages={timer.stages} total_seconds={total_seconds} answer_id={answer_id}",
    )

    memory_answer = response.answer.model_dump(exclude_none=True)
    background_tasks.add_task(
        session_memory.append_turn,
        user_id=user_request.user_id,
        session_id=user_request.session_id,
        mode=mode,
        answer_id=answer_id,
        query=user_request.query,
        answer=memory_answer,
        response_type="text",
        mask_map=None,
        pending_disambiguation=None,
    )
    background_tasks.add_task(
        log_timing,
        user_request.user_id, user_request.session_id,
        answer_id, user_request.query, timer.stages, total_seconds, mode,
    )
    return response

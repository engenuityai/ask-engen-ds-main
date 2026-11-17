"""Shared types and error classes for all LLM providers."""

from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cached_tokens: int = 0


class TokenLimitExceededError(RuntimeError):
    """Raised when the prompt exceeds the model's context window."""


class LLMProviderError(RuntimeError):
    """Raised when a provider API rejects a call.

    ``code`` is always one of the keys in ``ERROR_FRIENDLY_MESSAGES``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


ERROR_FRIENDLY_MESSAGES: dict[str, str] = {
    "invalid_request":    "I couldn't process that request — could you rephrase your question?",
    "failed_precondition":"This assistant isn't available to answer right now due to a service configuration issue. Please try again shortly.",
    "out_of_range":       "That request is outside what I'm able to handle — could you narrow it down?",
    "parameter_unknown":  "I couldn't process that request — could you rephrase your question?",
    "authentication":     "I'm temporarily unable to reach the underlying service. Please try again shortly.",
    "permission_denied":  "I'm temporarily unable to reach the underlying service. Please try again shortly.",
    "not_found":          "I couldn't find what I needed to answer that — could you rephrase your question?",
    "model_not_found":    "I'm temporarily unable to reach the underlying service. Please try again shortly.",
    "already_exists":     "Something conflicted while handling your request — please try again.",
    "aborted":            "Something conflicted while handling your request — please try again.",
    "rate_limit_exceeded":"I'm getting a lot of requests right now — please wait a moment and try again.",
    "quota_exceeded":     "I've hit today's usage limit and can't answer right now. Please try again later.",
    "cancelled":          "The request was cancelled before I could finish. Please try again.",
    "api_error":          "Something went wrong on my end. Please try again in a moment.",
    "unimplemented":      "That's not something I'm able to do.",
    "service_unavailable":"The service is temporarily overloaded right now. Please try again in a moment.",
    "deadline_exceeded":  "I wasn't able to finish processing that in time. Please try again",
}

_DEFAULT_ERROR_CODE = "api_error"


def is_token_limit_error(message: str) -> bool:
    lowered = message.lower()
    return "token" in lowered and ("exceed" in lowered or "too long" in lowered or "too many" in lowered)

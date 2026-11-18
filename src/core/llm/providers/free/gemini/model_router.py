"""RPM + RPD model router for free-tier Gemini.

select() walks MODEL_REGISTRY in priority order and returns the first model
whose current RPM AND RPD are both under THRESHOLD (80%).

If every model is saturated it raises RateLimitExhausted so the caller can
surface a friendly 429 response instead of hammering a rate-limited model.
"""

from .models import MODEL_REGISTRY, THRESHOLD
from .store import MemoryRateTracker


class RateLimitExhausted(Exception):
    """All free Gemini models are at or above the 80% usage threshold."""


class ModelRouter:
    def __init__(self, tracker: MemoryRateTracker) -> None:
        self._tracker = tracker

    def select(self) -> str:
        """Return the model ID to use for the next request.

        Raises:
            RateLimitExhausted: when every model in the registry is at >= 80%.
        """
        ordered = sorted(MODEL_REGISTRY, key=lambda m: m["priority"])
        for model in ordered:
            counts = self._tracker.get_counts(model["id"])
            rpm_ok = counts["rpm"] < model["rpm_limit"] * THRESHOLD
            rpd_ok = counts["rpd"] < model["rpd_limit"] * THRESHOLD
            if rpm_ok and rpd_ok:
                return model["id"]

        # All saturated — still try to serve rather than hard-fail:
        # return the highest-priority model and let Gemini rate-limit if needed.
        raise RateLimitExhausted(
            "All free Gemini models are at or above 80% capacity. "
            "RPM resets in <60 s; RPD resets at UTC midnight."
        )

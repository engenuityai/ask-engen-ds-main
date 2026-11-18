"""In-memory RPM + RPD rate tracker for free-tier Gemini model rotation.

RPM uses a rolling 60-second deque of timestamps.
RPD uses a date-keyed counter that resets at UTC midnight.
Thread-safe via a single lock — safe for FastAPI's threadpool workers.

Swap path (future Redis): replace this module with redis_tracker.py that
exposes the same MemoryRateTracker class name, then update store/__init__.py
with a one-line import change.
"""

import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone


class MemoryRateTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # model_id -> deque of UTC request timestamps (float seconds)
        self._rpm: dict[str, deque] = defaultdict(deque)
        # model_id -> {"date": "YYYY-MM-DD", "count": int}
        self._rpd: dict[str, dict] = defaultdict(lambda: {"date": "", "count": 0})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _today_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def increment(self, model_id: str) -> None:
        """Record one request for *model_id* right now."""
        with self._lock:
            now = datetime.now(timezone.utc).timestamp()

            # RPM: append timestamp, prune anything older than 60 s
            dq = self._rpm[model_id]
            dq.append(now)
            while dq and now - dq[0] > 60:
                dq.popleft()

            # RPD: reset counter when the UTC date rolls over
            today = self._today_utc()
            rec = self._rpd[model_id]
            if rec["date"] != today:
                rec["date"] = today
                rec["count"] = 0
            rec["count"] += 1

    def get_counts(self, model_id: str) -> dict:
        """Return {"rpm": int, "rpd": int} for *model_id* right now."""
        with self._lock:
            now = datetime.now(timezone.utc).timestamp()

            # RPM: prune stale entries before counting
            dq = self._rpm[model_id]
            while dq and now - dq[0] > 60:
                dq.popleft()
            rpm = len(dq)

            # RPD: stale day → counter is effectively 0
            today = self._today_utc()
            rec = self._rpd[model_id]
            rpd = rec["count"] if rec["date"] == today else 0

            return {"rpm": rpm, "rpd": rpd}

    def get_all_counts(self) -> dict[str, dict]:
        """Return counts for every model that has been seen so far."""
        from src.core.llm.providers.free.gemini.models import MODEL_REGISTRY

        return {m["id"]: self.get_counts(m["id"]) for m in MODEL_REGISTRY}

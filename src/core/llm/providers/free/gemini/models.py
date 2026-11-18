"""Free-tier Gemini model registry.

Priority order determines which model is tried first by ModelRouter:
  1. gemini-3.5-flash-lite  — Gemini 3 family, highest free quota
  2. gemini-2.5-flash-lite  — same quota band, secondary fallback
  3. gemini-2.5-flash       — lowest free quota, last resort

Update rpm_limit / rpd_limit here when Google adjusts free-tier caps.
Verify your actual limits at: https://aistudio.google.com (Rate limits tab).
"""

THRESHOLD: float = 0.80  # switch model once usage hits 80% of either limit

MODEL_REGISTRY: list[dict] = [
    {
        "id":             "gemini-3.5-flash-lite",
        "rpm_limit":      30,
        "rpd_limit":      1500,
        "context_window": 1_048_576,
        "priority":       1,
        "released":       "Nov 2025",
        "latency_ms":     600,
    },
    {
        "id":             "gemini-2.5-flash-lite",
        "rpm_limit":      30,
        "rpd_limit":      1500,
        "context_window": 1_048_576,
        "priority":       2,
        "released":       "Sep 2025",
        "latency_ms":     700,
    },
    {
        "id":             "gemini-2.5-flash",
        "rpm_limit":      10,
        "rpd_limit":      250,
        "context_window": 1_048_576,
        "priority":       3,
        "released":       "May 2025",
        "latency_ms":     900,
    },
]

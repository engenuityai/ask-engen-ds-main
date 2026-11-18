"""Per-service SQLite engine. Each service gets its own file at
data/<service>/<service>.db (created by src.routes.datasets.confirm), queried
live by the SQL query agent — see src.tools.sql_query_tool. No single
app-wide database: engines are created lazily, one per service, and cached
for the lifetime of the process.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import DATA_DIR

_engines: dict[str, Engine] = {}


def db_path(service: str) -> Path:
    return DATA_DIR / service / f"{service}.db"


def get_engine(service: str) -> Engine:
    if service not in _engines:
        path = db_path(service)
        path.parent.mkdir(parents=True, exist_ok=True)
        _engines[service] = create_engine(f"sqlite:///{path}", future=True)
    return _engines[service]


def reset_engine(service: str) -> None:
    """Disposes and drops the cached engine for `service` — call before
    recreating its SQLite file on a re-confirm so no stale pooled connection
    keeps the old file open or cached."""
    engine = _engines.pop(service, None)
    if engine is not None:
        engine.dispose()


def has_service_db(service: str) -> bool:
    return db_path(service).exists()

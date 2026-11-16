from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_DB_OK: bool | None = None


def db_available() -> bool:
    global _DB_OK
    if _DB_OK is not None:
        return _DB_OK
    try:
        from sqlalchemy import text

        from app.db.session import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        _DB_OK = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Postgres unavailable, using memory fallback: %s", exc)
        _DB_OK = False
    return _DB_OK


def ensure_runtime_ready() -> dict[str, Any]:
    """Initialize DB schema when possible; otherwise warm memory index."""
    if db_available():
        from app.db.session import init_db

        init_db(settings.embeddings_dim)
        return {"mode": "postgres"}
    from app.services.memory_index import load_docs_into_memory

    # Warm a small searchable corpus for local demo/browser verification.
    index = load_docs_into_memory(["3M_2018_10K"])
    return {"mode": "memory", "documents": len(index.documents), "chunks": len(index.chunks)}

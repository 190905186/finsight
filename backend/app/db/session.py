from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            settings.postgres_dsn,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(embedding_dim: int | None = None) -> None:
    """Create extensions, tables, and FTS helpers."""
    dim = embedding_dim or settings.embeddings_dim
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    # Ensure vector column uses configured dimension by recreating metadata table definitions.
    from app.db import models as models_module

    # Patch Vector dimension dynamically for create_all.
    models_module.Chunk.__table__.c.embedding.type.dim = dim  # type: ignore[attr-defined]
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE chunks
                ADD COLUMN IF NOT EXISTS content_tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_chunks_content_tsv
                ON chunks USING GIN (content_tsv)
                """
            )
        )
        conn.execute(
            text(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes WHERE indexname = 'ix_chunks_embedding_hnsw'
                    ) THEN
                        BEGIN
                            EXECUTE 'CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)';
                        EXCEPTION WHEN others THEN
                            -- Fallback when dimension/index type not ready yet.
                            NULL;
                        END;
                    END IF;
                END$$;
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_company_period ON chunks (company, period)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_doc_page ON chunks (doc_name, page_start)"))

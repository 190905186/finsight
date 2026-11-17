from __future__ import annotations

from typing import Any, Mapping, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Integer, String, Text, create_engine, delete, select
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, load_only, mapped_column, sessionmaker

from app.config import settings
from app.rag.chunking import TextChunk


class RAGBase(DeclarativeBase):
    pass


class StoredChunk(RAGBase):
    __tablename__ = "rag_document_chunks"
    __table_args__ = (
        Index("ix_rag_chunks_profile_model", "profile", "embedding_model"),
        Index(
            "ix_rag_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_path: Mapped[str] = mapped_column(String(512), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    profile: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embeddings_dimensions), nullable=False
    )


class PgVectorStore:
    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or create_engine(settings.postgres_dsn, pool_pre_ping=True)
        self._session_factory = sessionmaker(bind=self.engine)
        self._profile_cache: dict[
            tuple[str, str, tuple[tuple[str, str], ...]], list[StoredChunk]
        ] = {}

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        RAGBase.metadata.create_all(self.engine)

    def replace_document(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each document chunk must have exactly one embedding")
        if not chunks:
            raise ValueError("Cannot store an empty document")

        profile = chunks[0].profile
        doc_name = chunks[0].doc_name
        if any(chunk.profile != profile or chunk.doc_name != doc_name for chunk in chunks):
            raise ValueError("A document write must contain one profile and one PDF")

        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "source_path": chunk.source_path,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "profile": chunk.profile,
                "content": chunk.text,
                "metadata_json": chunk.metadata,
                "embedding_model": embedding_model,
                "embedding": list(embedding),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        with self._session_factory.begin() as session:
            session.execute(
                delete(StoredChunk).where(
                    StoredChunk.doc_name == doc_name,
                    StoredChunk.profile == profile,
                    StoredChunk.embedding_model == embedding_model,
                )
            )
            for offset in range(0, len(rows), 100):
                statement = insert(StoredChunk).values(rows[offset : offset + 100])
                statement = statement.on_conflict_do_update(
                    index_elements=[StoredChunk.chunk_id],
                    set_={
                        "source_path": statement.excluded.source_path,
                        "page_number": statement.excluded.page_number,
                        "chunk_index": statement.excluded.chunk_index,
                        "content": statement.excluded.content,
                        "metadata_json": statement.excluded.metadata_json,
                        "embedding_model": statement.excluded.embedding_model,
                        "embedding": statement.excluded.embedding,
                    },
                )
                session.execute(statement)
        stale_keys = [
            key
            for key in self._profile_cache
            if key[:2] == (profile, embedding_model)
        ]
        for key in stale_keys:
            del self._profile_cache[key]

    @staticmethod
    def _apply_filters(statement: Any, filters: Mapping[str, str] | None) -> Any:
        for key, value in (filters or {}).items():
            if key == "doc_name":
                statement = statement.where(StoredChunk.doc_name == value)
            else:
                statement = statement.where(StoredChunk.metadata_json[key].astext == value)
        return statement

    def load_profile(
        self,
        profile: str,
        embedding_model: str,
        filters: Mapping[str, str] | None = None,
    ) -> list[StoredChunk]:
        filter_key = tuple(sorted((filters or {}).items()))
        cache_key = (profile, embedding_model, filter_key)
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        statement = select(StoredChunk).where(
            StoredChunk.profile == profile,
            StoredChunk.embedding_model == embedding_model,
        ).options(
            load_only(
                StoredChunk.chunk_id,
                StoredChunk.doc_name,
                StoredChunk.source_path,
                StoredChunk.page_number,
                StoredChunk.chunk_index,
                StoredChunk.profile,
                StoredChunk.content,
                StoredChunk.metadata_json,
                StoredChunk.embedding_model,
            )
        )
        statement = self._apply_filters(statement, filters)
        with self._session_factory() as session:
            chunks = list(session.scalars(statement).all())
        self._profile_cache[cache_key] = chunks
        return chunks

    def dense_search(
        self,
        embedding: Sequence[float],
        profile: str,
        embedding_model: str,
        limit: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[StoredChunk, float]]:
        distance = StoredChunk.embedding.cosine_distance(list(embedding)).label("distance")
        statement = (
            select(StoredChunk, distance)
            .where(
                StoredChunk.profile == profile,
                StoredChunk.embedding_model == embedding_model,
            )
            .order_by(distance)
            .limit(limit)
        )
        statement = self._apply_filters(statement, filters)
        with self._session_factory() as session:
            return [
                (chunk, float(score))
                for chunk, score in session.execute(statement).all()
            ]

    def close(self) -> None:
        self.engine.dispose()

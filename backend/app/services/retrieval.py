from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import Select, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Chunk, IndexMeta
from app.services.embeddings_client import EmbeddingsClient


@dataclass
class EvidenceHit:
    chunk_id: int
    chunk_uid: str
    doc_name: str
    company: str
    section: str
    page_start: int
    page_end: int
    period: str | None
    snippet: str
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None

    def citation_tag(self) -> str:
        page_display = self.page_start + 1
        period = self.period or "n/a"
        return f"[{self.doc_name} | {self.section}/p.{page_display} | {period}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.chunk_uid,
            "document_name": self.doc_name,
            "company": self.company,
            "section": self.section,
            "page_index": self.page_start,
            "page_end": self.page_end,
            "display_page": self.page_start + 1,
            "period": self.period,
            "snippet": self.snippet[:4000],
            "score": round(self.score, 4),
            "citation": self.citation_tag(),
        }


def assert_index_compatible(session: Session) -> IndexMeta:
    meta = session.scalar(select(IndexMeta).limit(1))
    if meta is None:
        raise ValueError("No index metadata found; run ingestion first")
    if meta.embedding_model != settings.embeddings_model or meta.embedding_dim != settings.embeddings_dim:
        raise ValueError(
            f"Refusing query: index model/dim {meta.embedding_model}/{meta.embedding_dim} "
            f"!= config {settings.embeddings_model}/{settings.embeddings_dim}"
        )
    return meta


def _apply_filters(stmt: Select[Any], company: str | None, doc_name: str | None, period: str | None) -> Select[Any]:
    if company:
        stmt = stmt.where(Chunk.company.ilike(company))
    if doc_name:
        stmt = stmt.where(Chunk.doc_name == doc_name)
    if period:
        stmt = stmt.where(Chunk.period == str(period))
    return stmt


def lexical_search(
    session: Session,
    query: str,
    *,
    company: str | None = None,
    doc_name: str | None = None,
    period: str | None = None,
    limit: int = 20,
) -> list[tuple[Chunk, float]]:
    sql = """
        SELECT id, ts_rank_cd(content_tsv, plainto_tsquery('english', :query)) AS rank
        FROM chunks
        WHERE content_tsv @@ plainto_tsquery('english', :query)
    """
    params: dict[str, Any] = {"query": query}
    if company:
        sql += " AND company ILIKE :company"
        params["company"] = company
    if doc_name:
        sql += " AND doc_name = :doc_name"
        params["doc_name"] = doc_name
    if period:
        sql += " AND period = :period"
        params["period"] = str(period)
    sql += " ORDER BY rank DESC LIMIT :limit"
    params["limit"] = limit
    rows = session.execute(text(sql), params).all()
    if not rows:
        return []
    ids = [row.id for row in rows]
    rank_map = {row.id: float(row.rank) for row in rows}
    chunks = session.scalars(select(Chunk).where(Chunk.id.in_(ids))).all()
    by_id = {chunk.id: chunk for chunk in chunks}
    return [(by_id[item_id], rank_map[item_id]) for item_id in ids if item_id in by_id]


def dense_search(
    session: Session,
    query_vector: Sequence[float],
    *,
    company: str | None = None,
    doc_name: str | None = None,
    period: str | None = None,
    limit: int = 20,
) -> list[tuple[Chunk, float]]:
    # cosine distance: smaller is better; convert to similarity
    stmt = select(Chunk, Chunk.embedding.cosine_distance(list(query_vector)).label("distance"))
    stmt = _apply_filters(stmt, company, doc_name, period)
    stmt = stmt.where(Chunk.embedding.is_not(None)).order_by("distance").limit(limit)
    rows = session.execute(stmt).all()
    return [(row[0], 1.0 / (1.0 + float(row[1]))) for row in rows]


def reciprocal_rank_fusion(
    lexical: list[tuple[Chunk, float]],
    dense: list[tuple[Chunk, float]],
    *,
    k: int = 60,
) -> list[tuple[Chunk, float, int | None, int | None]]:
    scores: dict[int, float] = {}
    chunks: dict[int, Chunk] = {}
    lexical_rank: dict[int, int] = {}
    dense_rank: dict[int, int] = {}
    for rank, (chunk, _) in enumerate(lexical, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        chunks[chunk.id] = chunk
        lexical_rank[chunk.id] = rank
    for rank, (chunk, _) in enumerate(dense, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        chunks[chunk.id] = chunk
        dense_rank[chunk.id] = rank
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        (chunks[chunk_id], score, lexical_rank.get(chunk_id), dense_rank.get(chunk_id))
        for chunk_id, score in ordered
    ]


def _metric_tokens(query: str) -> set[str]:
    stop = {"the", "and", "for", "what", "was", "with", "from", "that", "this", "in", "of", "to", "a", "an"}
    tokens = {token.lower() for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", query)}
    return tokens - stop


def rerank_deterministic(
    fused: list[tuple[Chunk, float, int | None, int | None]],
    *,
    query: str,
    company: str | None,
    period: str | None,
    doc_type: str | None = None,
) -> list[tuple[Chunk, float, int | None, int | None]]:
    metric_tokens = _metric_tokens(query)
    rescored: list[tuple[Chunk, float, int | None, int | None]] = []
    for chunk, score, lex_rank, dense_rank in fused:
        bonus = 0.0
        if company and chunk.company.lower() == company.lower():
            bonus += 0.15
        if period and chunk.period and str(chunk.period) == str(period):
            bonus += 0.12
        if doc_type and chunk.doc_type.lower() == doc_type.lower():
            bonus += 0.05
        content_l = chunk.content.lower()
        overlap = sum(1 for token in metric_tokens if token in content_l)
        bonus += min(0.2, overlap * 0.02)
        rescored.append((chunk, score + bonus, lex_rank, dense_rank))
    rescored.sort(key=lambda item: item[1], reverse=True)
    return rescored


class HybridRetriever:
    def __init__(self, embedder: EmbeddingsClient | None = None) -> None:
        self.embedder = embedder or EmbeddingsClient()

    def retrieve(
        self,
        session: Session | None,
        query: str,
        *,
        company: str | None = None,
        doc_name: str | None = None,
        period: str | None = None,
        doc_type: str | None = None,
        limit: int | None = None,
    ) -> list[EvidenceHit]:
        from app.services.runtime import db_available

        top_k = limit or settings.max_retrieval_chunks
        if session is None or not db_available():
            from app.services.memory_index import get_memory_index

            return get_memory_index().search(query, company=company, period=period, limit=top_k)

        assert_index_compatible(session)
        lexical = lexical_search(session, query, company=company, doc_name=doc_name, period=period, limit=top_k * 3)
        try:
            vector = self.embedder.embed([query])[0]
            dense = dense_search(session, vector, company=company, doc_name=doc_name, period=period, limit=top_k * 3)
        except Exception:
            # Lexical-only fallback when embeddings endpoint is down.
            dense = []
        fused = reciprocal_rank_fusion(lexical, dense) if dense else [
            (chunk, score, rank, None) for rank, (chunk, score) in enumerate(lexical, start=1)
        ]
        ranked = rerank_deterministic(fused, query=query, company=company, period=period, doc_type=doc_type)
        hits: list[EvidenceHit] = []
        for chunk, score, lex_rank, dense_rank in ranked[:top_k]:
            hits.append(
                EvidenceHit(
                    chunk_id=chunk.id,
                    chunk_uid=chunk.chunk_uid,
                    doc_name=chunk.doc_name,
                    company=chunk.company,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    period=chunk.period,
                    snippet=chunk.content,
                    score=score,
                    lexical_rank=lex_rank,
                    dense_rank=dense_rank,
                )
            )
        return hits

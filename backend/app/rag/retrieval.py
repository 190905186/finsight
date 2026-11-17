from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rank_bm25 import BM25Okapi

from app.config import settings
from app.rag.embeddings import EmbeddingClient, embed_texts
from app.rag.store import PgVectorStore, StoredChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "did", "do",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "was", "what", "when", "where", "which", "who", "with",
}


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOP_WORDS]


@dataclass
class _Candidate:
    chunk: StoredChunk
    bm25_score: float = 0.0
    dense_distance: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class DeterministicReranker:
    def rerank(
        self, query: str, candidates: Sequence[_Candidate], limit: int
    ) -> list[_Candidate]:
        query_tokens = tokenize(query)
        query_terms = set(query_tokens)
        normalized_query = " ".join(query_tokens)
        for candidate in candidates:
            content_terms = set(tokenize(candidate.chunk.content))
            coverage = (
                len(query_terms & content_terms) / len(query_terms)
                if query_terms
                else 0.0
            )
            normalized_content = " ".join(tokenize(candidate.chunk.content))
            phrase_match = bool(normalized_query) and normalized_query in normalized_content
            candidate.rerank_score = (
                candidate.rrf_score + 0.05 * coverage + (0.1 if phrase_match else 0.0)
            )
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.rerank_score,
                candidate.chunk.doc_name,
                candidate.chunk.page_number,
                candidate.chunk.chunk_index,
            ),
        )[:limit]


class HybridRetriever:
    def __init__(
        self,
        store: PgVectorStore,
        client: EmbeddingClient,
        *,
        profile: str = "medium",
        embedding_model: str = settings.embeddings_model,
        dimensions: int = settings.embeddings_dimensions,
        reranker: DeterministicReranker | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.profile = profile
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.reranker = reranker or DeterministicReranker()
        self._indices: dict[
            tuple[tuple[str, str], ...],
            tuple[list[StoredChunk], list[list[str]], BM25Okapi],
        ] = {}

    def _get_bm25_index(
        self, filters: Mapping[str, str] | None
    ) -> tuple[list[StoredChunk], list[list[str]], BM25Okapi]:
        cache_key = tuple(sorted((filters or {}).items()))
        if cache_key in self._indices:
            return self._indices[cache_key]
        chunks = self.store.load_profile(self.profile, self.embedding_model, filters)
        if not chunks:
            raise RuntimeError(
                f"No chunks indexed for profile '{self.profile}' and model "
                f"'{self.embedding_model}' with filters {dict(filters or {})}"
            )
        corpus_tokens = [tokenize(chunk.content) for chunk in chunks]
        index = (chunks, corpus_tokens, BM25Okapi(corpus_tokens))
        self._indices[cache_key] = index
        return index

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
        query_embedding: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        chunks, corpus_tokens, bm25 = self._get_bm25_index(filters)
        query_vector = (
            list(query_embedding)
            if query_embedding is not None
            else embed_texts(
                self.client,
                [query],
                model=self.embedding_model,
                dimensions=self.dimensions,
                batch_size=1,
            )[0]
        )
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query embedding must have {self.dimensions} dimensions"
            )

        candidate_limit = max(top_k * 6, 30)
        candidates: dict[str, _Candidate] = {}
        dense_results = self.store.dense_search(
            query_vector,
            self.profile,
            self.embedding_model,
            candidate_limit,
            filters,
        )
        for rank, (chunk, distance) in enumerate(dense_results, start=1):
            candidate = candidates.setdefault(chunk.chunk_id, _Candidate(chunk))
            candidate.dense_distance = distance
            candidate.dense_rank = rank
            candidate.rrf_score += 1.0 / (60 + rank)

        query_token_list = tokenize(query)
        query_tokens = set(query_token_list)
        scores = bm25.get_scores(query_token_list)
        lexical_results = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if query_tokens.intersection(corpus_tokens[index])
            ),
            key=lambda item: (-item[1], chunks[item[0]].chunk_id),
        )[:candidate_limit]
        for rank, (index, score) in enumerate(lexical_results, start=1):
            chunk = chunks[index]
            candidate = candidates.setdefault(chunk.chunk_id, _Candidate(chunk))
            candidate.bm25_score = score
            candidate.bm25_rank = rank
            candidate.rrf_score += 1.0 / (60 + rank)

        reranked = self.reranker.rerank(query, list(candidates.values()), top_k)
        return [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "document": candidate.chunk.doc_name,
                "page_index": candidate.chunk.page_number,
                "page": candidate.chunk.page_number + 1,
                "snippet": candidate.chunk.content,
                "metadata": candidate.chunk.metadata_json,
                "score": candidate.rerank_score,
                "bm25_score": candidate.bm25_score,
                "cosine_distance": candidate.dense_distance,
                "bm25_rank": candidate.bm25_rank,
                "dense_rank": candidate.dense_rank,
            }
            for candidate in reranked
        ]

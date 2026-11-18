from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from app.rag.embeddings import embed_texts
from app.rag.evaluation import RetrievalQuestion, evaluate_retrieval
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval import HybridRetriever
from app.rag.store import PgVectorStore
from app.services.llm_client import LLMClient

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rag_golden_queries.json"
_GOLDEN_CASES = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_RUN_INTEGRATION_TESTS = os.getenv("RUN_RAG_GOLDEN_TESTS") == "1"


def test_golden_query_fixture_has_valid_cases() -> None:
    required_fields = {
        "id",
        "query",
        "doc_name",
        "gold_page_indices",
        "top_k",
        "minimum_recall_at_k",
        "answer_regex",
        "evidence_terms",
    }

    assert _GOLDEN_CASES
    assert len({case["id"] for case in _GOLDEN_CASES}) == len(_GOLDEN_CASES)
    for case in _GOLDEN_CASES:
        assert required_fields <= case.keys()
        assert case["query"].strip()
        assert case["doc_name"].strip()
        assert case["gold_page_indices"]
        assert all(page >= 0 for page in case["gold_page_indices"])
        assert case["top_k"] > 0
        assert 0.0 <= case["minimum_recall_at_k"] <= 1.0
        assert case["evidence_terms"]
        re.compile(case["answer_regex"])


@pytest.mark.skipif(
    not _RUN_INTEGRATION_TESTS,
    reason="Set RUN_RAG_GOLDEN_TESTS=1 to use PostgreSQL and the configured APIs",
)
@pytest.mark.parametrize("case", _GOLDEN_CASES, ids=lambda case: case["id"])
def test_golden_query_retrieves_labeled_page_and_evidence_chunk(case) -> None:
    store = PgVectorStore()
    try:
        with LLMClient() as client:
            retriever = HybridRetriever(store, client, profile="medium")
            query_embedding = embed_texts(
                client,
                [case["query"]],
                model=retriever.embedding_model,
                dimensions=retriever.dimensions,
                batch_size=1,
            )[0]
            results = retriever.retrieve(
                case["query"],
                top_k=case["top_k"],
                filters={"doc_name": case["doc_name"]},
                query_embedding=query_embedding,
            )
            metrics = evaluate_retrieval(
                [
                    RetrievalQuestion(
                        question_id=case["id"],
                        query=case["query"],
                        gold_evidence=frozenset(
                            (case["doc_name"], page)
                            for page in case["gold_page_indices"]
                        ),
                        doc_name=case["doc_name"],
                    )
                ],
                retriever,
                client,
                k_values=(case["top_k"],),
                retrieval_mode="singleStore",
                query_embeddings=(query_embedding,),
            )

        assert (
            metrics["recall_at_k"][str(case["top_k"])]
            >= case["minimum_recall_at_k"]
        )
        assert any(
            result["document"] == case["doc_name"]
            and result["page_index"] in case["gold_page_indices"]
            and all(
                term.casefold() in result["snippet"].casefold()
                for term in case["evidence_terms"]
            )
            for result in results
        )
    finally:
        store.close()


@pytest.mark.skipif(
    not _RUN_INTEGRATION_TESTS,
    reason="Set RUN_RAG_GOLDEN_TESTS=1 to use PostgreSQL and the configured APIs",
)
@pytest.mark.parametrize("case", _GOLDEN_CASES, ids=lambda case: case["id"])
def test_golden_query_answer_is_cited_to_expected_evidence(case) -> None:
    store = PgVectorStore()
    try:
        result = RAGPipeline(store=store, profile="medium").answer(
            case["query"],
            top_k=case["top_k"],
            filters={"doc_name": case["doc_name"]},
        )

        assert re.search(case["answer_regex"], result["answer"])
        assert any(
            citation["document"] == case["doc_name"]
            and all(
                term.casefold() in citation["snippet"].casefold()
                for term in case["evidence_terms"]
            )
            for citation in result["citations"]
        )
    finally:
        store.close()

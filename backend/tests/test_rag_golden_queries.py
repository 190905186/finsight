from __future__ import annotations

import os
from collections import Counter

import pytest

from app.evaluation import load_golden_questions, load_safety_cases
from app.rag.embeddings import embed_texts
from app.rag.evaluation import RetrievalQuestion, evaluate_retrieval
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval import HybridRetriever
from app.rag.store import PgVectorStore
from app.services.llm_client import LLMClient

_GOLDEN_CASES = load_golden_questions()
_RUN_INTEGRATION_TESTS = os.getenv("RUN_RAG_GOLDEN_TESTS") == "1"


def test_golden_query_set_has_gold_answers_and_page_labels() -> None:
    assert 60 <= len(_GOLDEN_CASES) <= 100
    assert len({case.question_id for case in _GOLDEN_CASES}) == len(_GOLDEN_CASES)
    assert len({case.query for case in _GOLDEN_CASES}) == len(_GOLDEN_CASES)
    assert all(case.reference_answer and case.gold_evidence for case in _GOLDEN_CASES)
    assert Counter(case.question_type for case in _GOLDEN_CASES) == {
        "metrics-generated": 25,
        "domain-relevant": 25,
        "novel-generated": 25,
    }

    safety_cases = load_safety_cases()
    assert len(safety_cases) == 15
    assert len(_GOLDEN_CASES) + len(safety_cases) == 90
    assert len({case.case_id for case in safety_cases}) == len(safety_cases)
    assert all(case.expected_refusal for case in safety_cases)
    assert len(load_golden_questions("prompt_dev")) == 15
    assert len(load_golden_questions("prompt_holdout")) == 60


@pytest.mark.skipif(
    not _RUN_INTEGRATION_TESTS,
    reason="Set RUN_RAG_GOLDEN_TESTS=1 to use PostgreSQL and the configured APIs",
)
@pytest.mark.parametrize("case", _GOLDEN_CASES, ids=lambda case: case.question_id)
def test_golden_query_retrieves_labeled_evidence(case: RetrievalQuestion) -> None:
    store = PgVectorStore()
    try:
        with LLMClient() as client:
            retriever = HybridRetriever(store, client, profile="medium")
            query_embedding = embed_texts(
                client,
                [case.query],
                model=retriever.embedding_model,
                dimensions=retriever.dimensions,
                batch_size=1,
            )[0]
            results = retriever.retrieve(
                case.query,
                top_k=5,
                filters={"doc_name": case.doc_name} if case.doc_name else None,
                query_embedding=query_embedding,
            )
            metrics = evaluate_retrieval(
                [case],
                retriever,
                client,
                k_values=(5,),
                retrieval_mode="singleStore",
                query_embeddings=(query_embedding,),
            )

        assert metrics["recall_at_k"]["5"] > 0
        assert any(
            (result["document"], result["page_index"]) in case.gold_evidence
            for result in results
        )
    finally:
        store.close()


@pytest.mark.skipif(
    not _RUN_INTEGRATION_TESTS,
    reason="Set RUN_RAG_GOLDEN_TESTS=1 to use PostgreSQL and the configured APIs",
)
@pytest.mark.parametrize("case", _GOLDEN_CASES, ids=lambda case: case.question_id)
def test_golden_query_answer_cites_labeled_evidence(case: RetrievalQuestion) -> None:
    store = PgVectorStore()
    try:
        result = RAGPipeline(store=store, profile="medium").answer(
            case.query,
            top_k=5,
            filters={"doc_name": case.doc_name} if case.doc_name else None,
        )

        assert result["answer"].strip()
        assert result["citations"]
        assert any(
            (citation["document"], citation["page"] - 1) in case.gold_evidence
            for citation in result["citations"]
        )
    finally:
        store.close()

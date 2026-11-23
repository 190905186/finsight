from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Sequence

from app.config import settings
from app.rag.embeddings import EmbeddingClient, embed_texts
from app.rag.retrieval import HybridRetriever

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
RETRIEVAL_MODES = ("sharedStore", "singleStore")


@dataclass(frozen=True)
class RetrievalQuestion:
    question_id: str
    query: str
    gold_evidence: frozenset[tuple[str, int]]
    doc_name: str | None = None
    reference_answer: str | None = None
    question_type: str | None = None


class AnswerPipeline(Protocol):
    def answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        prompting_strategy: str = "zero_shot",
    ) -> dict[str, Any]: ...


def load_financebench_questions(
    path: Path = settings.financebench_evaluation_path,
) -> list[RetrievalQuestion]:
    questions: list[RetrievalQuestion] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            evidence = row.get("evidence")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question text on evaluation line {line_number}")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(f"Missing evidence list on evaluation line {line_number}")

            gold_evidence: set[tuple[str, int]] = set()
            evidence_doc_names: list[str] = []
            for item in evidence:
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid evidence on evaluation line {line_number}")
                doc_name = item.get("doc_name")
                if not isinstance(doc_name, str) or not doc_name:
                    raise ValueError(f"Missing evidence doc_name on line {line_number}")
                if doc_name not in evidence_doc_names:
                    evidence_doc_names.append(doc_name)
                try:
                    page_index = int(item.get("evidence_page_num"))
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid zero-based evidence page on line {line_number}"
                    ) from None
                if page_index < 0:
                    raise ValueError(
                        f"Negative zero-based evidence page on line {line_number}"
                    )
                gold_evidence.add((doc_name, page_index))

            row_doc_name = row.get("doc_name")
            if row_doc_name is not None and (
                not isinstance(row_doc_name, str) or not row_doc_name
            ):
                raise ValueError(f"Invalid doc_name on evaluation line {line_number}")
            reference_answer = row.get("answer")
            if reference_answer is not None and not isinstance(reference_answer, str):
                raise ValueError(f"Invalid reference answer on line {line_number}")
            question_type = row.get("question_type")
            if question_type is not None and not isinstance(question_type, str):
                raise ValueError(f"Invalid question type on line {line_number}")

            questions.append(
                RetrievalQuestion(
                    question_id=str(row.get("financebench_id", line_number)),
                    query=question.strip(),
                    gold_evidence=frozenset(gold_evidence),
                    doc_name=row_doc_name or evidence_doc_names[0],
                    reference_answer=reference_answer,
                    question_type=question_type,
                )
            )
    if not questions:
        raise ValueError(f"No page-labeled questions found in {path}")
    return questions


def _validate_retrieval_mode(retrieval_mode: str) -> None:
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"retrieval_mode must be one of: {', '.join(RETRIEVAL_MODES)}"
        )


def evaluate_retrieval(
    questions: Sequence[RetrievalQuestion],
    retriever: HybridRetriever,
    client: EmbeddingClient,
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
    retrieval_mode: str = "sharedStore",
    embedding_batch_size: int = settings.rag_embedding_batch_size,
    query_embeddings: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    if not questions:
        raise ValueError("At least one labeled question is required")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    _validate_retrieval_mode(retrieval_mode)

    ordered_ks = sorted(set(k_values))
    query_vectors = (
        list(query_embeddings)
        if query_embeddings is not None
        else embed_texts(
            client,
            [question.query for question in questions],
            model=retriever.embedding_model,
            dimensions=retriever.dimensions,
            batch_size=embedding_batch_size,
        )
    )
    if len(query_vectors) != len(questions):
        raise ValueError("Query embedding count does not match labeled question count")
    recall_sums = {k: 0.0 for k in ordered_ks}
    reciprocal_rank_sum = 0.0
    query_latencies: list[dict[str, Any]] = []
    for question, query_vector in zip(questions, query_vectors):
        filters = None
        if retrieval_mode == "singleStore":
            if not question.doc_name:
                raise ValueError(
                    f"Question {question.question_id} has no document for singleStore evaluation"
                )
            filters = {"doc_name": question.doc_name}
        started_at = perf_counter()
        results = retriever.retrieve(
            question.query,
            top_k=max(ordered_ks),
            filters=filters,
            query_embedding=query_vector,
        )
        query_latencies.append(
            {
                "question_id": question.question_id,
                "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        )
        retrieved = [
            (item["document"], int(item["page_index"]))
            for item in results
        ]
        for k in ordered_ks:
            found = set(retrieved[:k]) & question.gold_evidence
            recall_sums[k] += len(found) / len(question.gold_evidence)
        first_relevant_rank = next(
            (
                rank
                for rank, evidence in enumerate(retrieved, start=1)
                if evidence in question.gold_evidence
            ),
            None,
        )
        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1.0 / first_relevant_rank

    denominator = len(questions)
    return {
        "evaluation_scope": "retrieval_only",
        "retrieval_mode": retrieval_mode,
        "profile": retriever.profile,
        "embedding_model": retriever.embedding_model,
        "questions_scored": denominator,
        "recall_at_k": {
            str(k): recall_sums[k] / denominator for k in ordered_ks
        },
        "mrr": reciprocal_rank_sum / denominator,
        "mean_retrieval_latency_ms": round(
            sum(item["latency_ms"] for item in query_latencies) / denominator, 2
        ),
        "query_latencies": query_latencies,
    }


def _answer_scores(prediction: str, reference: str) -> tuple[float, float]:
    prediction_without_citations = re.sub(r"\[S\d+\]", "", prediction, flags=re.IGNORECASE)
    predicted_tokens = _TOKEN_PATTERN.findall(prediction_without_citations.lower())
    reference_tokens = _TOKEN_PATTERN.findall(reference.lower())
    normalized_prediction = " ".join(predicted_tokens)
    normalized_reference = " ".join(reference_tokens)
    exact_match = float(normalized_prediction == normalized_reference)
    if not predicted_tokens or not reference_tokens:
        return exact_match, float(not predicted_tokens and not reference_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    token_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return exact_match, token_f1


def evaluate_answers(
    questions: Sequence[RetrievalQuestion],
    pipeline: AnswerPipeline,
    *,
    retrieval_mode: str = "sharedStore",
    top_k: int = 5,
    limit: int | None = None,
    prompting_strategy: str = "zero_shot",
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    _validate_retrieval_mode(retrieval_mode)
    if prompting_strategy not in {"zero_shot", "few_shot", "cot"}:
        raise ValueError("prompting_strategy must be zero_shot, few_shot, or cot")
    selected = list(questions if limit is None else questions[:limit])
    if not selected:
        raise ValueError("At least one question is required")
    if any(not question.reference_answer for question in selected):
        raise ValueError("Every answer-evaluation question needs a reference answer")

    results: list[dict[str, Any]] = []
    exact_match_sum = 0.0
    token_f1_sum = 0.0
    query_latencies: list[float] = []
    for question in selected:
        filters = None
        if retrieval_mode == "singleStore":
            if not question.doc_name:
                raise ValueError(
                    f"Question {question.question_id} has no document for singleStore evaluation"
                )
            filters = {"doc_name": question.doc_name}
        started_at = perf_counter()
        response = pipeline.answer(
            question.query,
            top_k=top_k,
            filters=filters,
            prompting_strategy=prompting_strategy,
        )
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        query_latencies.append(latency_ms)
        answer = response.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Question {question.question_id} returned no answer")
        exact_match, token_f1 = _answer_scores(answer, question.reference_answer or "")
        exact_match_sum += exact_match
        token_f1_sum += token_f1
        results.append(
            {
                "question_id": question.question_id,
                "question": question.query,
                "reference_answer": question.reference_answer,
                "answer": answer,
                "citations": response.get("citations", []),
                "normalized_exact_match": exact_match,
                "token_f1": token_f1,
                "latency_ms": latency_ms,
            }
        )

    denominator = len(results)
    return {
        "evaluation_scope": "end_to_end_answer_quality",
        "retrieval_mode": retrieval_mode,
        "prompting_strategy": prompting_strategy,
        "questions_scored": denominator,
        "normalized_exact_match": exact_match_sum / denominator,
        "token_f1": token_f1_sum / denominator,
        "mean_answer_latency_ms": round(sum(query_latencies) / denominator, 2),
        "results": results,
    }

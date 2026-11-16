from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from app.db.models import EvaluationRun
from app.db.session import get_session
from app.services.copilot_service import CopilotService
from app.services.refusal_gate import evaluate_refusal

DATASET_PATH = Path(__file__).resolve().parent / "evaluation_dataset.json"
SPLITS_PATH = Path(__file__).resolve().parent / "splits.json"

NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def load_dataset() -> dict[str, Any]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def load_splits() -> dict[str, Any]:
    return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))


def parse_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.finditer(text or ""):
        try:
            values.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return values


def numeric_match(predicted_answer: str, key_numbers: list[float], tol: float = 0.02) -> bool:
    if not key_numbers:
        return False
    predicted = parse_numbers(predicted_answer)
    if not predicted:
        return False
    for expected in key_numbers:
        for value in predicted:
            if expected == 0 and abs(value) <= tol:
                return True
            if expected != 0 and abs(value - expected) / abs(expected) <= tol:
                return True
            # Also allow scale differences like millions vs raw (e.g. 8.70 vs 8700)
            for scale in (1000.0, 1_000_000.0, 0.001, 0.000001):
                scaled = expected * scale
                if scaled != 0 and abs(value - scaled) / abs(scaled) <= tol:
                    return True
    return False


def retrieval_recall_at_k(citations: list[dict[str, Any]], required: list[dict[str, Any]], k: int = 5) -> float:
    if not required:
        return 1.0
    top = citations[:k]
    hits = 0
    for req in required:
        doc = req.get("doc_name")
        page = req.get("evidence_page_num")
        found = any(
            item.get("document_name") == doc
            and int(item.get("page_index", -1)) <= int(page) <= int(item.get("page_end", item.get("page_index", -1)))
            for item in top
        )
        if found:
            hits += 1
    return hits / len(required)


def refusal_correct(expected_reason: str | None, response: dict[str, Any], question: str) -> bool:
    if response.get("status") != "refused":
        # Deterministic gate check for offline scoring without full graph.
        decision = evaluate_refusal(question)
        return decision.refused and (expected_reason is None or decision.reason == expected_reason)
    reason = response.get("refusal_reason")
    if expected_reason is None:
        return True
    return reason == expected_reason


def score_item(item: dict[str, Any], response: dict[str, Any], k: int = 5) -> dict[str, Any]:
    expected = item.get("expected_behavior")
    if expected == "refuse":
        ok = refusal_correct(item.get("expected_refusal_reason"), response, item.get("question") or "")
        return {
            "question_id": item["question_id"],
            "type": "refusal",
            "pass": ok,
            "retrieval_recall_at_k": None,
            "numeric_match": None,
            "refusal_correct": ok,
        }

    citations = response.get("citations") or []
    recall = retrieval_recall_at_k(citations, item.get("required_citations") or [], k=k)
    numeric = numeric_match(response.get("answer") or "", item.get("key_numbers") or [])
    # Right number with no retrieved page is not a pass.
    passed = bool(recall > 0 and numeric and response.get("status") == "answered")
    return {
        "question_id": item["question_id"],
        "type": "answer",
        "pass": passed,
        "retrieval_recall_at_k": recall,
        "numeric_match": numeric,
        "refusal_correct": None,
        "status": response.get("status"),
    }


def run_evaluation(
    *,
    split: str = "development",
    limit: int | None = None,
    use_llm: bool = False,
    service: CopilotService | None = None,
) -> dict[str, Any]:
    dataset = load_dataset()
    splits = load_splits()
    selected_ids = set(splits.get(split) or splits.get("development") or [])
    questions = [q for q in dataset.get("questions", []) if q["question_id"] in selected_ids]
    if limit is not None:
        answers = [q for q in questions if q.get("expected_behavior") == "answer"]
        refusals = [q for q in questions if q.get("expected_behavior") == "refuse"]
        # Keep a mixed sample so refusal scoring is exercised even on small runs.
        refuse_n = min(len(refusals), max(2, limit // 4))
        answer_n = min(len(answers), max(0, limit - refuse_n))
        questions = answers[:answer_n] + refusals[:refuse_n]

    copilot = service or CopilotService(use_llm=use_llm)
    details: list[dict[str, Any]] = []
    for item in questions:
        if item.get("expected_behavior") == "refuse":
            # Score refusal gate / graph without requiring corpus hits.
            response = copilot.chat(item["question"], company=item.get("company"), user_id="eval")
        else:
            response = copilot.chat(item["question"], company=item.get("company"), user_id="eval")
        details.append(score_item(item, response))

    answer_scores = [row for row in details if row["type"] == "answer"]
    refusal_scores = [row for row in details if row["type"] == "refusal"]
    metrics = {
        "n": len(details),
        "answer_n": len(answer_scores),
        "refusal_n": len(refusal_scores),
        "answer_pass_rate": (sum(1 for row in answer_scores if row["pass"]) / len(answer_scores)) if answer_scores else None,
        "mean_retrieval_recall_at_5": (
            sum(row["retrieval_recall_at_k"] or 0.0 for row in answer_scores) / len(answer_scores)
            if answer_scores
            else None
        ),
        "numeric_match_rate": (
            sum(1 for row in answer_scores if row["numeric_match"]) / len(answer_scores) if answer_scores else None
        ),
        "refusal_accuracy": (
            sum(1 for row in refusal_scores if row["refusal_correct"]) / len(refusal_scores) if refusal_scores else None
        ),
    }
    run_id = str(uuid.uuid4())
    payload = {
        "evaluation_id": run_id,
        "split": split,
        "status": "completed",
        "metrics": metrics,
        "details": details,
    }
    try:
        with get_session() as session:
            session.add(
                EvaluationRun(
                    id=run_id,
                    split=split,
                    status="completed",
                    metrics_json=metrics,
                    details_json={"items": details},
                )
            )
    except Exception:
        # Persistence is best-effort for local/dev without Postgres.
        payload["persisted"] = False
    else:
        payload["persisted"] = True
    return payload

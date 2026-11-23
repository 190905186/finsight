"""Evaluation and benchmarking utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.rag.evaluation import RetrievalQuestion, load_financebench_questions

_EVALUATION_DIR = Path(__file__).resolve().parent
_SPLITS_PATH = _EVALUATION_DIR / "splits.json"
_SAFETY_CASES_PATH = _EVALUATION_DIR / "safety_cases.json"


@dataclass(frozen=True)
class SafetyCase:
    case_id: str
    query: str
    expected_answer: str
    expected_refusal: bool


def load_golden_questions(
    split: Literal["all", "prompt_dev", "prompt_holdout"] = "all",
) -> list[RetrievalQuestion]:
    if split not in {"all", "prompt_dev", "prompt_holdout"}:
        raise ValueError(f"Unknown gold evaluation split: {split}")
    splits = json.loads(_SPLITS_PATH.read_text(encoding="utf-8"))
    selected_ids = splits.get("golden_queries")
    if not isinstance(selected_ids, list) or not all(
        isinstance(item, str) and item for item in selected_ids
    ):
        raise ValueError("golden_queries must be a list of non-empty question IDs")
    if not 60 <= len(selected_ids) <= 100:
        raise ValueError("The gold query set must contain between 60 and 100 questions")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("The gold query set contains duplicate question IDs")

    development_ids = splits.get("prompt_dev")
    holdout_ids = splits.get("prompt_holdout")
    if (
        not isinstance(development_ids, list)
        or not isinstance(holdout_ids, list)
        or not all(isinstance(item, str) for item in development_ids)
        or not all(isinstance(item, str) for item in holdout_ids)
        or set(development_ids) & set(holdout_ids)
        or set(development_ids) | set(holdout_ids) != set(selected_ids)
    ):
        raise ValueError("Prompt development and holdout splits must partition the gold set")

    all_questions = load_financebench_questions()
    by_id = {question.question_id: question for question in all_questions}
    missing_ids = set(selected_ids) - by_id.keys()
    if missing_ids:
        raise ValueError(
            "FinanceBench source is missing selected gold IDs: "
            + ", ".join(sorted(missing_ids))
        )

    question_ids = selected_ids if split == "all" else splits[split]
    questions = [by_id[question_id] for question_id in question_ids]
    if any(not question.reference_answer for question in questions):
        raise ValueError("Every gold query must have a reference answer")
    if any(not question.gold_evidence for question in questions):
        raise ValueError("Every gold query must have labeled evidence")
    return questions


def load_safety_cases() -> list[SafetyCase]:
    payload: Any = json.loads(_SAFETY_CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Safety evaluation data must be a non-empty list")
    cases: list[SafetyCase] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Safety evaluation cases must be JSON objects")
        case_id = row.get("id")
        query = row.get("query")
        answer = row.get("expected_answer")
        refusal = row.get("expected_refusal")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or refusal is not True
        ):
            raise ValueError("Invalid consequential-action refusal case")
        cases.append(SafetyCase(case_id, query, answer, refusal))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Safety evaluation cases contain duplicate IDs")
    return cases


__all__ = ["SafetyCase", "load_golden_questions", "load_safety_cases"]

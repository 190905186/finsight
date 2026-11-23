from __future__ import annotations

from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import require_auth
from app.evaluation import load_golden_questions
from app.rag.evaluation import evaluate_answers, evaluate_retrieval
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval import HybridRetriever
from app.rag.store import PgVectorStore
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
_evaluation_runs: dict[str, dict] = {}


class EvaluationRequest(BaseModel):
    evaluation_scope: Literal["retrieval_only", "end_to_end_answer_quality"] = (
        "retrieval_only"
    )
    retrieval_mode: Literal["sharedStore", "singleStore"] = "sharedStore"
    profile: Literal["small", "medium", "large"] = "medium"
    top_k: int = Field(default=5, ge=1, le=20)
    limit: int = Field(default=75, ge=1, le=75)


@router.post("/run")
def run_evaluation(
    payload: EvaluationRequest, user: dict = Depends(require_auth)
) -> dict:
    questions = load_golden_questions()[: payload.limit]
    started_at = perf_counter()
    store = PgVectorStore()
    try:
        if payload.evaluation_scope == "retrieval_only":
            with LLMClient() as client:
                retriever = HybridRetriever(
                    store,
                    client,
                    profile=payload.profile,
                )
                metrics = evaluate_retrieval(
                    questions,
                    retriever,
                    client,
                    k_values=(payload.top_k,),
                    retrieval_mode=payload.retrieval_mode,
                )
        else:
            pipeline = RAGPipeline(store=store, profile=payload.profile)
            metrics = evaluate_answers(
                questions,
                pipeline,
                retrieval_mode=payload.retrieval_mode,
                top_k=payload.top_k,
            )
    finally:
        store.close()

    evaluation_id = str(uuid4())
    run = {
        "id": evaluation_id,
        "owner": str(user.get("sub", "")),
        "status": "completed",
        "evaluation_scope": payload.evaluation_scope,
        "retrieval_mode": payload.retrieval_mode,
        "profile": payload.profile,
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        "metrics": metrics,
    }
    _evaluation_runs[evaluation_id] = run
    while len(_evaluation_runs) > 20:
        del _evaluation_runs[next(iter(_evaluation_runs))]
    return {key: value for key, value in run.items() if key != "owner"}


@router.get("/results")
async def result_list(user: dict = Depends(require_auth)) -> dict:
    user_id = str(user.get("sub", ""))
    items = [
        {
            "id": run["id"],
            "status": run["status"],
            "evaluation_scope": run["evaluation_scope"],
            "latency_ms": run["latency_ms"],
        }
        for run in _evaluation_runs.values()
        if run["owner"] == user_id
    ]
    return {"items": items}


@router.get("/metrics")
async def metrics(user: dict = Depends(require_auth)) -> dict:
    user_id = str(user.get("sub", ""))
    runs = [
        run for run in _evaluation_runs.values() if run["owner"] == user_id
    ]
    latencies = [run["latency_ms"] for run in runs]
    return {
        "runs_completed": len(runs),
        "latest_latency_ms": latencies[-1] if latencies else None,
        "mean_latency_ms": (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        ),
        "latest_metrics": runs[-1]["metrics"] if runs else None,
    }

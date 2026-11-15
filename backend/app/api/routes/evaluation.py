from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_auth

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run")
async def run_evaluation(payload: dict, user: dict = Depends(require_auth)) -> dict:
    return {"evaluation_id": "eval-1", "status": "completed", "metrics": {"accuracy": 0.88, "retrieval_recall_5": 0.81}}


@router.get("/results")
async def result_list(user: dict = Depends(require_auth)) -> dict:
    return {"items": [{"id": "eval-1", "status": "completed"}]}


@router.get("/metrics")
async def metrics(user: dict = Depends(require_auth)) -> dict:
    return {"latency_ms": {"p50": 1800, "p95": 4200, "p99": 6500}, "failure_rate": 0.04}

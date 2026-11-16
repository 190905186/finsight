from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from app.api.deps import require_auth
from app.evaluation.runner import run_evaluation
from app.services.runtime import db_available

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

_LATEST_MEMORY_RUN: dict | None = None


@router.post("/runs")
async def create_run(payload: dict | None = None, user: dict = Depends(require_auth)) -> dict:
    global _LATEST_MEMORY_RUN
    body = payload or {}
    split = str(body.get("split") or "development")
    limit = body.get("limit")
    use_llm = bool(body.get("use_llm", False))
    result = run_evaluation(split=split, limit=limit, use_llm=use_llm)
    if not result.get("persisted"):
        _LATEST_MEMORY_RUN = result
    return result


@router.post("/run")
async def run_alias(payload: dict | None = None, user: dict = Depends(require_auth)) -> dict:
    return await create_run(payload, user)


@router.get("/results")
async def result_list(user: dict = Depends(require_auth)) -> dict:
    if not db_available():
        if _LATEST_MEMORY_RUN is None:
            return {"items": []}
        return {
            "items": [
                {
                    "id": _LATEST_MEMORY_RUN["evaluation_id"],
                    "status": _LATEST_MEMORY_RUN["status"],
                    "split": _LATEST_MEMORY_RUN["split"],
                    "metrics": _LATEST_MEMORY_RUN["metrics"],
                }
            ]
        }

    from app.db.models import EvaluationRun
    from app.db.session import get_session

    try:
        with get_session() as session:
            rows = session.scalars(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(20)).all()
            return {
                "items": [
                    {
                        "id": row.id,
                        "status": row.status,
                        "split": row.split,
                        "metrics": row.metrics_json,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }
                    for row in rows
                ]
            }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Evaluation store unavailable: {exc}") from exc


@router.get("/metrics")
async def metrics(user: dict = Depends(require_auth)) -> dict:
    if not db_available():
        if _LATEST_MEMORY_RUN is None:
            return {"latest": None}
        return {
            "latest": {
                "id": _LATEST_MEMORY_RUN["evaluation_id"],
                "metrics": _LATEST_MEMORY_RUN["metrics"],
                "split": _LATEST_MEMORY_RUN["split"],
            }
        }

    from app.db.models import EvaluationRun
    from app.db.session import get_session

    with get_session() as session:
        row = session.scalars(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(1)).first()
        if row is None:
            return {"latest": None}
        return {"latest": {"id": row.id, "metrics": row.metrics_json, "split": row.split}}

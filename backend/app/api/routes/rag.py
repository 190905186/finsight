from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_auth
from app.rag.pipeline import RAGPipeline
from app.schemas.rag import RAGQueryRequest

router = APIRouter(prefix="/rag", tags=["rag"])
_pipeline = RAGPipeline()


@router.post("/query")
def query_rag(payload: RAGQueryRequest, user: dict = Depends(require_auth)) -> dict:
    filters = {"company": payload.company} if payload.company else None
    return _pipeline.answer(payload.query, top_k=payload.top_k, filters=filters)

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_auth
from app.services.orchestrator import FinanceWorkflowOrchestrator

router = APIRouter(prefix="/copilot", tags=["copilot"])
_orchestrator = FinanceWorkflowOrchestrator()


@router.post("/chat")
async def chat(message: dict, user: dict = Depends(require_auth)) -> dict:
    query = str(message.get("message", ""))
    return _orchestrator.run(query)


@router.post("/stream")
async def stream(message: dict, user: dict = Depends(require_auth)) -> dict:
    query = str(message.get("message", ""))
    return {"status": "streaming", "conversation_id": "conv-demo", "result": _orchestrator.run(query)}


@router.get("/conversations")
async def conversations(user: dict = Depends(require_auth)) -> dict:
    return {"items": [{"id": "conv-demo", "title": "Finance summary", "status": "active"}]}


@router.get("/conversations/{conversation_id}")
async def conversation_detail(conversation_id: str, user: dict = Depends(require_auth)) -> dict:
    return {"id": conversation_id, "messages": [{"role": "assistant", "content": "Ready to help."}]}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user: dict = Depends(require_auth)) -> dict:
    return {"deleted": True, "conversation_id": conversation_id}

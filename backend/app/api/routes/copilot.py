from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import require_auth
from app.config import settings
from app.schemas.copilot import ChatRequest
from app.services.conversation_service import ConversationService
from app.services.copilot_service import CopilotService
from app.services.memory_conversations import memory_conversations
from app.services.runtime import db_available

router = APIRouter(prefix="/copilot", tags=["copilot"])
_service = CopilotService(use_llm=settings.use_llm)
_conversations = ConversationService()


@router.post("/chat")
async def chat(payload: ChatRequest, user: dict = Depends(require_auth)) -> dict:
    return _service.chat(
        payload.message,
        conversation_id=payload.conversation_id,
        company=payload.company,
        period=payload.period,
        user_id=str(user.get("sub") or "user"),
    )


@router.post("/stream")
async def stream(payload: ChatRequest, user: dict = Depends(require_auth)) -> StreamingResponse:
    def event_generator():
        for event in _service.stream_events(
            payload.message,
            conversation_id=payload.conversation_id,
            company=payload.company,
            period=payload.period,
            user_id=str(user.get("sub") or "user"),
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/conversations")
async def conversations(user: dict = Depends(require_auth)) -> dict:
    if not db_available():
        items = memory_conversations.list()
        return {
            "items": [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "company_filter": item.get("company_filter"),
                    "updated_at": item.get("updated_at"),
                }
                for item in items
            ]
        }

    from app.db.session import get_session

    with get_session() as session:
        items = _conversations.list_conversations(session)
        return {"items": [_conversations.to_summary(item) for item in items]}


@router.get("/conversations/{conversation_id}")
async def conversation_detail(conversation_id: str, user: dict = Depends(require_auth)) -> dict:
    if not db_available():
        item = memory_conversations.get(conversation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {
            "id": item["id"],
            "title": item["title"],
            "company_filter": item.get("company_filter"),
            "messages": item.get("messages") or [],
        }

    from app.db.session import get_session

    with get_session() as session:
        conversation = _conversations.get(session, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return _conversations.to_detail(conversation)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user: dict = Depends(require_auth)) -> dict:
    if not db_available():
        deleted = memory_conversations.delete(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True, "conversation_id": conversation_id}

    from app.db.session import get_session

    with get_session() as session:
        deleted = _conversations.delete(session, conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True, "conversation_id": conversation_id}

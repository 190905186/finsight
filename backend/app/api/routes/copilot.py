from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import require_auth
from app.graph.workflow import delete_financial_conversation, run_financial_workflow
from app.schemas.finance import ChatRequest

router = APIRouter(prefix="/copilot", tags=["copilot"])
_conversations: dict[tuple[str, str], dict] = {}


def _run_chat(payload: ChatRequest, user: dict) -> dict:
    conversation_id = payload.conversation_id or str(uuid4())
    user_id = str(user.get("sub", ""))
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user identifier is missing.",
        )
    result = run_financial_workflow(
        payload.message,
        conversation_id=conversation_id,
        user_id=user_id,
        company=payload.company,
        fiscal_year=payload.fiscal_year,
        prompting_strategy=payload.prompting_strategy,
    )
    key = (user_id, conversation_id)
    conversation = _conversations.setdefault(
        key,
        {
            "id": conversation_id,
            "title": payload.message[:80],
            "messages": [],
            "status": "active",
        },
    )
    conversation["messages"].extend(
        [
            {"role": "user", "content": payload.message},
            {"role": "assistant", "content": result["answer"]},
        ]
    )
    if len(conversation["messages"]) > 40:
        conversation["messages"] = conversation["messages"][-40:]
    conversation["company"] = result.get("company")
    conversation["fiscal_year"] = result.get("fiscal_year")
    return result


@router.post("/chat")
def chat(payload: ChatRequest, user: dict = Depends(require_auth)) -> dict:
    return _run_chat(payload, user)


@router.post("/stream")
def stream(payload: ChatRequest, user: dict = Depends(require_auth)) -> dict:
    result = _run_chat(payload, user)
    return {**result, "stream_status": "completed"}


@router.get("/conversations")
def conversations(user: dict = Depends(require_auth)) -> dict:
    user_id = str(user.get("sub", ""))
    items = [
        {
            "id": conversation["id"],
            "title": conversation["title"],
            "status": conversation["status"],
            "company": conversation.get("company"),
            "fiscal_year": conversation.get("fiscal_year"),
        }
        for (owner, _), conversation in _conversations.items()
        if owner == user_id
    ]
    return {"items": items}


@router.get("/conversations/{conversation_id}")
def conversation_detail(
    conversation_id: str, user: dict = Depends(require_auth)
) -> dict:
    user_id = str(user.get("sub", ""))
    conversation = _conversations.get((user_id, conversation_id))
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return dict(conversation)


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str, user: dict = Depends(require_auth)
) -> dict:
    user_id = str(user.get("sub", ""))
    conversation = _conversations.pop((user_id, conversation_id), None)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    delete_financial_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    return {"deleted": True, "conversation_id": conversation_id}

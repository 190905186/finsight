from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Conversation, Message


def _title_from_message(message: str) -> str:
    cleaned = " ".join(message.strip().split())
    return (cleaned[:72] + "…") if len(cleaned) > 72 else cleaned or "New chat"


class ConversationService:
    def create(self, session: Session, *, company: str | None = None, title: str | None = None) -> Conversation:
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=title or "New chat",
            company_filter=company,
        )
        session.add(conversation)
        session.flush()
        return conversation

    def get(self, session: Session, conversation_id: str) -> Conversation | None:
        return session.get(Conversation, conversation_id)

    def list_conversations(self, session: Session, limit: int = 50) -> list[Conversation]:
        return list(
            session.scalars(
                select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
            ).all()
        )

    def add_message(
        self,
        session: Session,
        conversation: Conversation,
        *,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
            payload_json=payload or {},
        )
        conversation.updated_at = datetime.now(timezone.utc)
        if role == "user" and (conversation.title == "New chat" or not conversation.title):
            conversation.title = _title_from_message(content)
        session.add(message)
        session.flush()
        return message

    def delete(self, session: Session, conversation_id: str) -> bool:
        conversation = self.get(session, conversation_id)
        if conversation is None:
            return False
        session.delete(conversation)
        session.flush()
        return True

    def to_summary(self, conversation: Conversation) -> dict[str, Any]:
        return {
            "id": conversation.id,
            "title": conversation.title,
            "company_filter": conversation.company_filter,
            "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
        }

    def to_detail(self, conversation: Conversation) -> dict[str, Any]:
        messages = [
            {
                "role": message.role,
                "content": message.content,
                "payload": message.payload_json,
                "created_at": message.created_at.isoformat() if message.created_at else None,
            }
            for message in sorted(conversation.messages, key=lambda item: item.id)
        ]
        return {
            "id": conversation.id,
            "title": conversation.title,
            "company_filter": conversation.company_filter,
            "messages": messages,
        }

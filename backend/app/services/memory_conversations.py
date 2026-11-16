from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class MemoryConversationStore:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def create(self, *, company: str | None = None, title: str | None = None) -> dict[str, Any]:
        item = {
            "id": str(uuid.uuid4()),
            "title": title or "New chat",
            "company_filter": company,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
        }
        self._items[item["id"]] = item
        return item

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        return self._items.get(conversation_id)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        items = sorted(self._items.values(), key=lambda row: row.get("updated_at") or "", reverse=True)
        return items[:limit]

    def add_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = self._items[conversation_id]
        message = {"role": role, "content": content, "payload": payload or {}}
        item["messages"].append(message)
        item["updated_at"] = datetime.now(timezone.utc).isoformat()
        if role == "user" and item["title"] == "New chat":
            cleaned = " ".join(content.strip().split())
            item["title"] = (cleaned[:72] + "…") if len(cleaned) > 72 else cleaned
        return message

    def delete(self, conversation_id: str) -> bool:
        return self._items.pop(conversation_id, None) is not None


memory_conversations = MemoryConversationStore()

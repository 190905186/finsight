from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy.orm import Session

from app.config import settings
from app.graph.workflow import CopilotGraph
from app.services.cache_rate_limit import CacheRateLimiter
from app.services.conversation_service import ConversationService
from app.services.llm_client import LLMClient
from app.services.memory_conversations import memory_conversations
from app.services.retrieval import HybridRetriever
from app.services.runtime import db_available


@contextmanager
def _session_cm() -> Iterator[Session | None]:
    if not db_available():
        yield None
        return
    from app.db.session import get_session_factory

    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class CopilotService:
    def __init__(self, use_llm: bool = True, llm: LLMClient | None = None) -> None:
        self.conversations = ConversationService()
        self.cache = CacheRateLimiter()
        self.use_llm = use_llm
        self._llm = llm
        self._graph: CopilotGraph | None = None

    def _get_graph(self) -> CopilotGraph:
        if self._graph is None:
            llm = self._llm
            if self.use_llm and llm is None:
                llm = LLMClient()
            self._graph = CopilotGraph(
                session_factory=_session_cm,
                llm=llm if self.use_llm else None,
                retriever=HybridRetriever(),
                use_llm=self.use_llm,
            )
        return self._graph

    def chat(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        company: str | None = None,
        period: str | None = None,
        user_id: str = "anonymous",
    ) -> dict[str, Any]:
        if not self.cache.allow(user_id):
            return {
                "status": "provider_unavailable",
                "answer": "Rate limit exceeded. Please retry shortly.",
                "citations": [],
                "calculations": [],
                "basis": "not_applicable",
                "evidence_status": "unknown",
                "refusal_reason": "rate_limited",
                "request_id": str(uuid.uuid4()),
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "trace": [{"node": "rate_limit", "status": "blocked"}],
            }

        if not db_available():
            conversation = memory_conversations.get(conversation_id) if conversation_id else None
            if conversation is None:
                conversation = memory_conversations.create(company=company)
            conv_id = conversation["id"]
            memory_conversations.add_message(conv_id, role="user", content=message)

            cached = self.cache.get_cached(message, company, settings.index_version)
            if cached and cached.get("status") in {"answered", "refused"}:
                payload = {**cached, "conversation_id": conv_id, "request_id": str(uuid.uuid4())}
                memory_conversations.add_message(
                    conv_id,
                    role="assistant",
                    content=payload.get("answer") or "",
                    payload=payload,
                )
                return payload

            result = self._get_graph().invoke(
                message,
                conversation_id=conv_id,
                company=company,
                period=period,
            )
            result["conversation_id"] = conv_id
            memory_conversations.add_message(
                conv_id,
                role="assistant",
                content=result.get("answer") or "",
                payload=result,
            )
            if result.get("status") in {"answered", "refused"}:
                self.cache.set_cached(message, company, settings.index_version, result)
            return result

        with _session_cm() as session:
            assert session is not None
            conversation = self.conversations.get(session, conversation_id) if conversation_id else None
            if conversation is None:
                conversation = self.conversations.create(session, company=company)
            elif company and not conversation.company_filter:
                conversation.company_filter = company
            conv_id = conversation.id
            self.conversations.add_message(session, conversation, role="user", content=message)

            cached = self.cache.get_cached(message, company, settings.index_version)
            if cached and cached.get("status") in {"answered", "refused"}:
                payload = {**cached, "conversation_id": conv_id, "request_id": str(uuid.uuid4())}
                self.conversations.add_message(
                    session,
                    conversation,
                    role="assistant",
                    content=payload.get("answer") or "",
                    payload=payload,
                )
                return payload

        result = self._get_graph().invoke(
            message,
            conversation_id=conv_id,
            company=company,
            period=period,
        )
        result["conversation_id"] = conv_id

        with _session_cm() as session:
            assert session is not None
            conversation = self.conversations.get(session, conv_id)
            if conversation is None:
                conversation = self.conversations.create(session, company=company)
                conv_id = conversation.id
                result["conversation_id"] = conv_id
            self.conversations.add_message(
                session,
                conversation,
                role="assistant",
                content=result.get("answer") or "",
                payload=result,
            )

        # Avoid caching insufficient_evidence; it is often transient during index warmup.
        if result.get("status") in {"answered", "refused"}:
            self.cache.set_cached(message, company, settings.index_version, result)
        return result

    def stream_events(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        company: str | None = None,
        period: str | None = None,
        user_id: str = "anonymous",
    ):
        yield {"event": "node_status", "node": "pre_guardrail", "status": "started"}
        result = self.chat(
            message,
            conversation_id=conversation_id,
            company=company,
            period=period,
            user_id=user_id,
        )
        for event in result.get("trace") or []:
            yield {
                "event": "node_status",
                "node": event.get("node"),
                "status": event.get("status"),
                "detail": event.get("detail"),
            }
        yield {"event": "final", "payload": result}

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

AnswerStatus = Literal[
    "answered",
    "refused",
    "clarification_required",
    "insufficient_evidence",
    "provider_unavailable",
]

BasisLabel = Literal["GAAP", "Non-GAAP", "not_applicable"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: Optional[str] = None
    company: Optional[str] = None
    period: Optional[str] = None


class Citation(BaseModel):
    source_id: str
    document_name: str
    section: str
    page_index: int
    display_page: int
    period: Optional[str] = None
    snippet: str
    citation: Optional[str] = None
    score: Optional[float] = None


class Calculation(BaseModel):
    formula: str
    inputs: dict[str, Any]
    result: float | str
    units: str
    op: Optional[str] = None


class TraceEvent(BaseModel):
    node: str
    status: str
    detail: Optional[str] = None


class ChatResponse(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    basis: BasisLabel = "not_applicable"
    evidence_status: str = "unknown"
    refusal_reason: Optional[str] = None
    request_id: str
    conversation_id: str
    trace: list[TraceEvent] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    id: str
    title: str
    company_filter: Optional[str] = None
    updated_at: Optional[str] = None


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[dict[str, Any]]

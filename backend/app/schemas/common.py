from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Optional[Dict[str, Any]] = None


class Pagination(BaseModel):
    page: int = 1
    page_size: int = 20
    total: int = 0
    has_next: bool = False


class SourceReference(BaseModel):
    document: str
    section: str
    page: str
    snippet: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceBundle(BaseModel):
    answer: str
    source_documents: List[SourceReference] = Field(default_factory=list)
    calculation: Optional[str] = None
    verification_status: str = "verified"

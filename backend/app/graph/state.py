from __future__ import annotations

from typing import Any, Optional, TypedDict


class PlanDict(TypedDict, total=False):
    intent: str
    company: Optional[str]
    periods: list[str]
    metric: Optional[str]
    basis: str
    needs_calculation: bool
    calculation: dict[str, Any]


class CopilotState(TypedDict, total=False):
    request_id: str
    conversation_id: str
    message: str
    company_filter: Optional[str]
    period_filter: Optional[str]
    status: str
    answer: str
    citations: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    basis: str
    evidence_status: str
    refusal_reason: Optional[str]
    plan: PlanDict
    evidence: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    step_count: int
    repair_count: int
    needs_repair: bool
    provider_error: Optional[str]

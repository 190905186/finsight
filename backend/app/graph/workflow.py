from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GraphState:
    user_request: str = ""
    intent: str = "unknown"
    steps: List[str] = field(default_factory=lambda: ["input_validation", "intent_classification", "tool_execution", "verification"])
    results: Dict[str, Any] = field(default_factory=dict)


def route_state(state: GraphState) -> str:
    if state.intent == "unknown":
        return "need_intent"
    return "complete"

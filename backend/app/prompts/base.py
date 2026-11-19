"""Core prompt primitives: techniques, prompt specs, rendering and the registry.

Every LLM-mediated LangGraph node gets one ``PromptSpec`` per prompting
technique. Specs are immutable and versioned so that evaluation runs can record
exactly which prompt produced which answer (design doc section 9).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PROMPT_SET_VERSION = "2026.11.4"  # .4: answer-side fixes from FinanceBench dev run 1

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class Technique(str, Enum):
    """Prompting techniques the team can A/B on the dev split."""

    ZERO_SHOT = "zero_shot"  # instructions + output schema only
    FEW_SHOT = "few_shot"  # instructions + worked examples as prior chat turns
    COT = "cot"  # instructions + explicit step list, reasoning emitted in a bounded JSON field


class Node(str, Enum):
    """LangGraph nodes that call the shared chat model."""

    GUARDRAIL = "pre_guardrail"
    CLASSIFY = "classify_and_plan"
    REWRITE = "query_rewrite"
    EXTRACT = "extract_financial_fact"
    ANALYZE = "analyze_financial_report"
    SYNTHESIZE = "synthesize_answer"
    VERIFY = "verify_answer"
    REPAIR = "repair_once"


@dataclass(frozen=True)
class FewShotExample:
    """One worked example, sent as a user turn followed by an assistant turn."""

    user: str
    assistant: str


@dataclass(frozen=True)
class PromptSpec:
    node: Node
    technique: Technique
    version: str
    system: str
    user_template: str
    examples: Tuple[FewShotExample, ...] = ()
    required_keys: Tuple[str, ...] = ()
    enums: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.0
    # Extra cross-field check: returns an error string, or None when the output is consistent.
    validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None

    @property
    def prompt_id(self) -> str:
        """Stable identifier to log with every request and evaluation run."""
        return f"{self.node.value}/{self.technique.value}@{self.version}"

    @property
    def variables(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(_PLACEHOLDER.findall(self.user_template)))

    def render(
        self,
        *,
        model_name: Optional[str] = None,
        **variables: Any,
    ) -> List[Dict[str, str]]:
        """Build the chat-completions ``messages`` list.

        Raises ``KeyError`` if a template variable is missing (fail closed rather
        than sending a half-filled prompt to the hosted endpoint).
        """
        user = fill(self.user_template, variables)
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system}]
        for ex in self.examples:
            messages.append({"role": "user", "content": ex.user})
            messages.append({"role": "assistant", "content": ex.assistant})
        messages.append({"role": "user", "content": user + soft_switch(model_name)})
        return messages


def fill(template: str, variables: Mapping[str, Any]) -> str:
    """Replace ``{{name}}`` placeholders strictly.

    ``{{ }}`` is used instead of ``str.format`` because prompts contain literal
    JSON braces and ``$`` currency amounts.
    """
    missing = [n for n in _PLACEHOLDER.findall(template) if n not in variables]
    if missing:
        raise KeyError(f"Missing prompt variables: {sorted(set(missing))}")
    return _PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), template)


def soft_switch(model_name: Optional[str]) -> str:
    """Qwen3 soft switch.

    Qwen3 thinks by default, which burns tokens and latency on every node. Our
    CoT variant asks for reasoning inside the JSON instead, so native thinking
    is turned off with ``/no_think`` for all techniques. Set
    ``PROMPT_QWEN3_THINKING=1`` to experiment with native thinking instead.
    Non-Qwen3 models (e.g. local qwen2.5:3b) get no suffix.
    """
    name = (model_name or os.getenv("LLM_MODEL", "")).lower()
    if "qwen3" not in name:
        return ""
    return " /think" if os.getenv("PROMPT_QWEN3_THINKING") == "1" else " /no_think"


# --------------------------------------------------------------------------- registry

_REGISTRY: Dict[Tuple[Node, Technique], PromptSpec] = {}

# Defaults chosen from dev runs (docs/prompt-experiments.md). Override per run with PROMPT_TECHNIQUE_<NODE>.
DEFAULT_TECHNIQUE: Dict[Node, Technique] = {
    Node.GUARDRAIL: Technique.FEW_SHOT,
    Node.CLASSIFY: Technique.FEW_SHOT,
    Node.REWRITE: Technique.FEW_SHOT,
    Node.EXTRACT: Technique.ZERO_SHOT,  # FB run 2: tied within noise, cheapest and fastest
    Node.ANALYZE: Technique.COT,  # FB run 2: gold figure 56% vs 11-22%
    Node.SYNTHESIZE: Technique.COT,  # FB run 2: judge 67% correct, half the tokens of few-shot
    Node.VERIFY: Technique.COT,
    Node.REPAIR: Technique.ZERO_SHOT,
}


def register(*specs: PromptSpec) -> None:
    for spec in specs:
        key = (spec.node, spec.technique)
        if key in _REGISTRY:
            raise ValueError(f"Duplicate prompt registered for {key}")
        _REGISTRY[key] = spec


def get_prompt(node: Node | str, technique: Technique | str | None = None) -> PromptSpec:
    node = Node(node)
    if technique is None:
        env = os.getenv(f"PROMPT_TECHNIQUE_{node.name}")
        technique = Technique(env) if env else DEFAULT_TECHNIQUE[node]
    technique = Technique(technique)
    try:
        return _REGISTRY[(node, technique)]
    except KeyError as exc:
        raise KeyError(f"No {technique.value} prompt for {node.value}") from exc


def available(node: Node | str) -> List[Technique]:
    node = Node(node)
    return [t for (n, t) in _REGISTRY if n == node]


def all_prompts() -> Sequence[PromptSpec]:
    return list(_REGISTRY.values())

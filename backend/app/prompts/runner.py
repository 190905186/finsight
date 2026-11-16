"""One-call helper used by graph nodes: render -> call -> parse -> one format repair.

Accepts any object with ``create_chat_completion(messages, max_tokens=..., temperature=...)``
and a ``model`` attribute (the repo's ``LLMClient`` fits). Returns the parse
result plus usage so nodes can enforce LLM_MAX_CALLS_PER_REQUEST.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import PromptSpec
from .parsing import ParseResult, json_fix_message, message_text, parse_output


@dataclass
class RunResult:
    prompt_id: str
    parsed: ParseResult
    calls: int
    latency_s: float
    usage: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.parsed.ok


def run_prompt(llm: Any, spec: PromptSpec, *, allow_format_repair: bool = True, **variables: Any) -> RunResult:
    messages = spec.render(model_name=getattr(llm, "model", None), **variables)
    started = time.perf_counter()
    usage: List[Dict[str, Any]] = []

    resp = llm.create_chat_completion(messages, max_tokens=spec.max_tokens, temperature=spec.temperature)
    usage.append(resp.get("usage") or {})
    raw = message_text(resp)
    parsed = parse_output(raw, spec)
    calls = 1

    if not parsed.ok and allow_format_repair:
        retry = messages + [{"role": "assistant", "content": raw}, json_fix_message(parsed.error or "")]
        resp = llm.create_chat_completion(retry, max_tokens=spec.max_tokens, temperature=0.0)
        usage.append(resp.get("usage") or {})
        parsed = parse_output(message_text(resp), spec)
        calls = 2

    return RunResult(spec.prompt_id, parsed, calls, time.perf_counter() - started, usage)

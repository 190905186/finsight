"""One-call helper used by graph nodes: render -> call -> parse -> one format repair.

Accepts any object with ``create_chat_completion(messages, max_tokens=..., temperature=...)``
and a ``model`` attribute (the repo's ``LLMClient`` fits). Returns the parse
result plus usage so nodes can enforce LLM_MAX_CALLS_PER_REQUEST.

Track B endpoint limits (IIT KGP learner guide, 5 Nov 2026): one active request
per group key, 60 requests/min, 8192 tokens input+output, max_tokens <= 1024.
A teammate using the same key causes HTTP 429; the VM being busy causes 503.
Both are retried here with 2, 4, 8 s back-off, then the error is raised.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from .base import PromptSpec
from .parsing import ParseResult, json_fix_message, message_text, parse_output

MAX_OUTPUT_TOKENS = 1024
RETRY_STATUS = (429, 503)
RETRY_DELAYS_S: Sequence[float] = (2, 4, 8)


@dataclass
class RunResult:
    prompt_id: str
    parsed: ParseResult
    calls: int
    latency_s: float
    usage: List[Dict[str, Any]] = field(default_factory=list)
    retries: int = 0

    @property
    def ok(self) -> bool:
        return self.parsed.ok


def _status_of(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def call_with_retry(
    llm: Any,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Dict[str, Any], int]:
    """Call the endpoint, retrying 429/503 with increasing delays. Returns (response, retries)."""
    retries = 0
    for delay in [*RETRY_DELAYS_S, None]:
        try:
            resp = llm.create_chat_completion(
                messages, max_tokens=min(max_tokens, MAX_OUTPUT_TOKENS), temperature=temperature
            )
            return resp, retries
        except Exception as exc:  # httpx.HTTPStatusError from LLMClient
            if _status_of(exc) not in RETRY_STATUS or delay is None:
                raise
            retries += 1
            sleep(delay)
    raise RuntimeError("unreachable")


def run_prompt(llm: Any, spec: PromptSpec, *, allow_format_repair: bool = True, **variables: Any) -> RunResult:
    messages = spec.render(model_name=getattr(llm, "model", None), **variables)
    started = time.perf_counter()
    usage: List[Dict[str, Any]] = []

    resp, retries = call_with_retry(llm, messages, spec.max_tokens, spec.temperature)
    usage.append(resp.get("usage") or {})
    raw = message_text(resp)
    parsed = parse_output(raw, spec)
    calls = 1

    if not parsed.ok and allow_format_repair:
        retry = messages + [{"role": "assistant", "content": raw}, json_fix_message(parsed.error or "")]
        resp, more = call_with_retry(llm, retry, spec.max_tokens, 0.0)
        retries += more
        usage.append(resp.get("usage") or {})
        parsed = parse_output(message_text(resp), spec)
        calls = 2

    return RunResult(spec.prompt_id, parsed, calls, time.perf_counter() - started, usage, retries)

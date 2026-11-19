"""Robust parsing + validation of model JSON output.

Small models wrap JSON in prose, code fences or Qwen3 <think> blocks. This
module extracts the first JSON object, checks required keys and enum values,
and returns an error string usable by the single bounded repair attempt.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import PromptSpec

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


@dataclass
class ParseResult:
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    raw: str

    @property
    def ok(self) -> bool:
        return self.error is None


def strip_reasoning(text: str) -> str:
    text = _THINK.sub("", text)
    # An unterminated <think> (truncated by max_tokens) leaves nothing usable after it.
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
    return _FENCE.sub("", text).strip()


def extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced top-level {...} span, string-aware."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_output(raw: str, spec: PromptSpec) -> ParseResult:
    cleaned = strip_reasoning(raw or "")
    span = extract_json_object(cleaned)
    if span is None:
        return ParseResult(None, "No JSON object found in model output.", raw)
    try:
        data = json.loads(span)
    except json.JSONDecodeError as exc:
        return ParseResult(None, f"Invalid JSON: {exc.msg} at char {exc.pos}.", raw)
    if not isinstance(data, dict):
        return ParseResult(None, "Top-level JSON value must be an object.", raw)

    missing = [k for k in spec.required_keys if k not in data]
    if missing:
        return ParseResult(data, f"Missing required keys: {missing}.", raw)
    for key, allowed in spec.enums.items():
        if key in data and data[key] not in allowed:
            return ParseResult(data, f"Field '{key}' must be one of {list(allowed)}, got {data[key]!r}.", raw)
    if spec.validator is not None:
        problem = spec.validator(data)
        if problem:
            return ParseResult(data, problem, raw)
    return ParseResult(data, None, raw)


def message_text(response: Dict[str, Any]) -> str:
    """Pull assistant text from an OpenAI-compatible response (LLMClient output)."""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def json_fix_message(error: str) -> Dict[str, str]:
    """Follow-up user turn for the single allowed format repair."""
    return {
        "role": "user",
        "content": f"Your previous reply could not be used: {error} "
        "Reply again with only the corrected JSON object, no other text.",
    }

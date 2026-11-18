"""Finsight prompt library.

Usage::

    from app.prompts import Node, get_prompt, parse_output, format_evidence

    spec = get_prompt(Node.CLASSIFY)                 # default technique for the node
    spec = get_prompt("classify_and_plan", "cot")    # or pick one explicitly
    messages = spec.render(model_name=llm.model, company="none", period="none", question=q)
    resp = llm.create_chat_completion(messages, max_tokens=spec.max_tokens, temperature=spec.temperature)
    result = parse_output(message_text(resp), spec)

See README.md in this folder for the technique matrix and evaluation workflow.
"""
from __future__ import annotations

# Importing the node modules registers their prompts.
from . import answer, evidence, routing  # noqa: F401
from .base import (
    DEFAULT_TECHNIQUE,
    PROMPT_SET_VERSION,
    FewShotExample,
    Node,
    PromptSpec,
    Technique,
    all_prompts,
    available,
    get_prompt,
)
from .parsing import ParseResult, json_fix_message, message_text, parse_output
from .refusals import refusal_message, refusal_sentence
from .runner import run_prompt
from .shared import RenderedAnswer, citation_label, format_evidence, format_tool_results, render_citations

__all__ = [
    "DEFAULT_TECHNIQUE",
    "PROMPT_SET_VERSION",
    "FewShotExample",
    "Node",
    "ParseResult",
    "PromptSpec",
    "Technique",
    "all_prompts",
    "available",
    "format_evidence",
    "format_tool_results",
    "get_prompt",
    "json_fix_message",
    "message_text",
    "parse_output",
    "RenderedAnswer",
    "citation_label",
    "refusal_message",
    "refusal_sentence",
    "render_citations",
    "run_prompt",
]

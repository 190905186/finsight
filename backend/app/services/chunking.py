from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from app.config import settings

STATEMENT_HINTS = (
    "consolidated statement",
    "consolidated balance sheet",
    "consolidated statements",
    "cash flow",
    "income statement",
    "operations",
    "financial position",
    "stockholders",
    "shareholders",
)
TABLE_HINTS = ("(millions)", "(thousands)", "in millions", "in thousands", "december 31", "year ended")
SECTION_PATTERNS = [
    re.compile(r"(item\s+\d+[a-z]?[.\s].{0,80})", re.IGNORECASE),
    re.compile(r"(consolidated\s+(?:statements?|balance sheet)[^\n]{0,80})", re.IGNORECASE),
]


@dataclass
class PageText:
    page_index: int  # 0-based
    text: str


@dataclass
class ChunkDraft:
    section: str
    page_start: int
    page_end: int
    content: str
    token_estimate: int
    content_hash: str


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def detect_section(text: str, fallback: str = "narrative") -> str:
    for pattern in SECTION_PATTERNS:
        match = pattern.search(text[:1200])
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()[:180]
    return fallback


def is_statement_or_table_page(text: str) -> bool:
    lowered = text.lower()
    if any(hint in lowered for hint in STATEMENT_HINTS):
        return True
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.08 and any(hint in lowered for hint in TABLE_HINTS):
        return True
    return False


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunk_pages(pages: Iterable[PageText], token_budget: int | None = None) -> list[ChunkDraft]:
    budget = token_budget or settings.chunk_token_budget
    drafts: list[ChunkDraft] = []
    buffer_pages: list[PageText] = []
    buffer_section = "narrative"

    def flush() -> None:
        nonlocal buffer_pages, buffer_section
        if not buffer_pages:
            return
        content = "\n\n".join(page.text.strip() for page in buffer_pages if page.text.strip())
        if not content.strip():
            buffer_pages = []
            return
        drafts.append(
            ChunkDraft(
                section=buffer_section,
                page_start=buffer_pages[0].page_index,
                page_end=buffer_pages[-1].page_index,
                content=content,
                token_estimate=estimate_tokens(content),
                content_hash=_hash_content(content),
            )
        )
        buffer_pages = []

    for page in pages:
        text = (page.text or "").strip()
        if not text:
            continue
        section = detect_section(text)
        tokens = estimate_tokens(text)
        keep_whole = is_statement_or_table_page(text) or tokens >= budget * 0.7

        if keep_whole:
            flush()
            drafts.append(
                ChunkDraft(
                    section=section,
                    page_start=page.page_index,
                    page_end=page.page_index,
                    content=text,
                    token_estimate=tokens,
                    content_hash=_hash_content(text),
                )
            )
            buffer_section = section
            continue

        pending_tokens = sum(estimate_tokens(p.text) for p in buffer_pages) + tokens
        same_section = not buffer_pages or buffer_section == section
        if buffer_pages and (not same_section or pending_tokens > budget):
            flush()
            buffer_section = section
        if not buffer_pages:
            buffer_section = section
        buffer_pages.append(page)

    flush()
    return drafts

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RefusalDecision:
    refused: bool
    reason: str | None = None
    message: str | None = None


BUY_SELL_PATTERNS = [
    r"\bshould i (buy|sell)\b",
    r"\b(buy|sell) (or|=and) (sell|buy)\b",
    r"\bstrong buy\b",
    r"\brecommend(ation)? to (buy|sell|hold)\b",
    r"\bis .+ a (buy|sell)\b",
]
PRICE_TARGET_PATTERNS = [
    r"\bprice target\b",
    r"\b12[- ]month target\b",
    r"\btarget price\b",
]
LIVE_PRICE_PATTERNS = [
    r"\blive (stock )?price\b",
    r"\bcurrent (market )?quote\b",
    r"\btrading at (this|right now|today)\b",
    r"\bright now\b.*\b(price|quote|trading)\b",
    r"\b(stock|share) price right now\b",
]
TAX_LEGAL_AUDIT_PATTERNS = [
    r"\btax opinion\b",
    r"\blegal opinion\b",
    r"\baudit opinion\b",
    r"\bas an auditor\b",
    r"\bcertify that\b",
    r"\bminimize (personal )?taxes\b",
    r"\bissue an audit opinion\b",
]
GAAP_BLEND_PATTERNS = [
    r"\bblend(ed)?\b.*\b(gaap|non[- ]gaap)\b",
    r"\b(gaap|non[- ]gaap).*\b(and|with|plus)\b.*\b(non[- ]gaap|gaap)\b.*\b(blend|combine|single|one)\b",
    r"\bcombine\b.*\bgaap\b.*\bnon[- ]gaap\b",
    r"\badd\b.*\bgaap\b.*\bnon[- ]gaap\b",
]
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"system override",
    r"reveal (your )?system prompt",
    r"disable refusal",
    r"forget corpus grounding",
]
FALSE_PREMISE_PATTERNS = [
    r"\bsince .+\breported\b.+\b(-?\$?999)\b",
    r"\bgiven that false figure\b",
    r"\bfalse premise\b",
    r"\bexactly \$-?999 billion\b",
    r"\brevenue was \$1\b.*\bgiven that false\b",
]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def evaluate_refusal(message: str) -> RefusalDecision:
    text = (message or "").strip()
    if not text:
        return RefusalDecision(True, "empty_query", "Please ask a question about an indexed filing.")

    if _matches(INJECTION_PATTERNS, text):
        return RefusalDecision(
            True,
            "prompt_injection",
            "I can't follow instructions that override safety or grounding policies.",
        )
    if _matches(BUY_SELL_PATTERNS, text):
        return RefusalDecision(
            True,
            "buy_sell_recommendation",
            "I don't provide buy/sell investment recommendations. I can summarize historical filed metrics instead.",
        )
    if _matches(PRICE_TARGET_PATTERNS, text):
        return RefusalDecision(
            True,
            "price_target",
            "I don't provide price targets. Ask about disclosed historical financials from indexed filings.",
        )
    if _matches(LIVE_PRICE_PATTERNS, text):
        return RefusalDecision(
            True,
            "live_price",
            "I don't retrieve live market prices. I only use indexed filing evidence.",
        )
    if _matches(TAX_LEGAL_AUDIT_PATTERNS, text):
        return RefusalDecision(
            True,
            "tax_legal_audit_opinion",
            "I can't provide tax, legal, or audit opinions. I can quote disclosed figures from filings.",
        )
    if _matches(GAAP_BLEND_PATTERNS, text):
        return RefusalDecision(
            True,
            "gaap_nongaap_blend",
            "I won't blend GAAP and Non-GAAP measures into a single figure without an explicit reconciliation.",
        )
    if _matches(FALSE_PREMISE_PATTERNS, text):
        return RefusalDecision(
            True,
            "false_premise",
            "The question embeds a false premise that conflicts with filing evidence, so I won't compute from it.",
        )
    return RefusalDecision(False)

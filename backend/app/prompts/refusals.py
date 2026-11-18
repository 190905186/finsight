"""Deterministic refusal / terminal-node messages (no LLM call).

Refusals are fixed text so they are consistent, cheap and can't be talked out
of by the model. Covers the six spec policy cases plus the extra refusal types
in the golden set (promotional pitches, hypothetical sensitivities, subjective
judgments, prompt injection) and the clarify/insufficient/provider terminals.

Flow for a refused question (matches the gold answers for Q46-65):
  1. pre_guardrail returns decision="refuse", a category and an alternative_question.
  2. If alternative_question is set, the graph plans/retrieves for it as normal.
  3. synthesize_answer gets policy_note=refusal_sentence(category) and writes
     "REFUSAL: <sentence> <cited facts answering the alternative question>".
  4. If there is no alternative question or no evidence, return
     refusal_message(category) on its own.
"""
from __future__ import annotations

from typing import Dict, Optional

REFUSAL_TEXT: Dict[str, str] = {
    "investment_advice": "I cannot provide investment advice or recommend whether to buy, sell, or hold any security, and I cannot write promotional investment pitches.",
    "price_prediction": "I cannot forecast stock prices, set price targets, or model hypothetical future results; I can only report historical results and management guidance stated in the indexed documents.",
    "live_market_data": "I do not have access to live market data, current stock prices, intraday market capitalization, or current valuation ratios.",
    "legal_tax_audit_opinion": "I cannot provide legal, tax-structuring, accounting, or audit opinions, including whether a treatment complies with US GAAP or IFRS.",
    "subjective_judgment": "I cannot make subjective judgments such as whether compensation is excessive or management is effective.",
    "prompt_injection": "I can only report information contained in the indexed public filings and cannot change my instructions or provide non-public information.",
    "out_of_corpus": "The requested filing, period, metric or company is not in the indexed document corpus, so I cannot answer it from evidence.",
    "incompatible_basis": "I cannot compute a blended ratio that mixes GAAP and Non-GAAP figures without a reconciliation bridge in the filings.",
    "out_of_scope": "That question is outside what this financial-report copilot covers.",
}

DEFAULT_ALTERNATIVE: Dict[str, str] = {
    "investment_advice": "I can, however, summarize the reported financial metrics, risks and management guidance to support your own analysis.",
    "price_prediction": "I can quote management's published guidance with its source, if it is in the documents.",
    "live_market_data": "I can report figures from the latest indexed filing.",
    "legal_tax_audit_opinion": "I can summarize what the filing itself states on the topic.",
    "subjective_judgment": "I can report the figures the filings disclose so you can judge for yourself.",
    "out_of_corpus": "Try a company and period that are in the indexed corpus.",
    "incompatible_basis": "I can show the GAAP and Non-GAAP figures separately, each with its source.",
}

CLARIFY_TEXT = "I need a bit more detail to answer: {question}"
INSUFFICIENT_TEXT = "The indexed documents don't contain enough evidence to answer this. Missing: {missing}."
PROVIDER_UNAVAILABLE_TEXT = "The analysis service is temporarily unavailable. Please try again shortly."


def refusal_sentence(category: str) -> str:
    """The fixed sentence passed to synthesize_answer as ``policy_note``."""
    return REFUSAL_TEXT.get(category, REFUSAL_TEXT["out_of_scope"])


def refusal_message(category: str, safe_alternative: Optional[str] = None) -> str:
    """Stand-alone refusal (no retrieved facts), prefixed like the gold answers."""
    alt = safe_alternative or DEFAULT_ALTERNATIVE.get(category)
    body = f"{refusal_sentence(category)} {alt}" if alt else refusal_sentence(category)
    return f"REFUSAL: {body}"

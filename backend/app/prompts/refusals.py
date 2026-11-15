"""Deterministic refusal / terminal-node messages (no LLM call).

Refusals are fixed text so they are consistent, cheap and can't be talked out
of by the model. Covers the six DOCX policy cases plus clarify/insufficient/
provider_unavailable terminals.
"""
from __future__ import annotations

from typing import Dict, Optional

REFUSAL_TEXT: Dict[str, str] = {
    "investment_advice": "I can't give investment recommendations such as whether to buy, sell or hold a security.",
    "price_prediction": "I can't predict stock prices, set price targets, or forecast results beyond management guidance disclosed in the indexed documents.",
    "live_market_data": "I don't have live or current market data (prices, market capitalization, P/E or news); I only use the indexed filings and reports.",
    "legal_tax_audit_opinion": "I can't give legal, tax or audit opinions, including whether statements comply with GAAP or IFRS.",
    "out_of_corpus": "The requested company, document, period or metric isn't in the indexed documents, so I can't answer it from evidence.",
    "incompatible_basis": "I can't combine GAAP and Non-GAAP figures in one calculation without a disclosed reconciliation.",
    "prompt_injection": "I can't change my instructions or reveal system configuration.",
    "out_of_scope": "That question is outside what this financial-report copilot covers.",
}

DEFAULT_ALTERNATIVE: Dict[str, str] = {
    "investment_advice": "I can summarize reported results, trends and disclosed risk factors instead.",
    "price_prediction": "I can quote management's published guidance with its source, if it's in the documents.",
    "live_market_data": "I can report figures from the latest indexed filing.",
    "legal_tax_audit_opinion": "I can summarize what the filing itself states about the topic.",
    "out_of_corpus": "Try a company and period that are in the indexed corpus.",
    "incompatible_basis": "I can show the GAAP and Non-GAAP figures separately, each with its source.",
}

CLARIFY_TEXT = "I need a bit more detail to answer: {question}"
INSUFFICIENT_TEXT = "The indexed documents don't contain enough evidence to answer this. Missing: {missing}."
PROVIDER_UNAVAILABLE_TEXT = "The analysis service is temporarily unavailable. Please try again shortly."


def refusal_message(category: str, safe_alternative: Optional[str] = None) -> str:
    base = REFUSAL_TEXT.get(category, REFUSAL_TEXT["out_of_scope"])
    alt = safe_alternative or DEFAULT_ALTERNATIVE.get(category)
    return f"{base} {alt}" if alt else base

"""Evidence-side prompts: extract_financial_fact and analyze_financial_report.

Extraction turns retrieved table/text chunks into normalized facts that the
deterministic calculator consumes. Values are returned as STRINGS exactly as
printed so Decimal parsing happens in code, not in the model.
"""
from __future__ import annotations

import json

from .base import FewShotExample, Node, PromptSpec, Technique, register
from .shared import format_evidence, system_prompt

V = "1.0"


def _j(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


# =========================================================================== extract

_EX_TASK = """extract the requested financial facts from the evidence so a deterministic calculator can use them.
For each requested metric and period, return the value exactly as printed (digits, commas, parentheses for negatives), plus unit/scale, currency, basis and the source_id it came from, and a short verbatim quote that contains the number.
Read column headers carefully: match the requested period to the correct column (current vs prior year, three vs nine months).
If a requested metric/period is not in the evidence, list it under "missing". Never estimate, convert scale, or compute."""

_EX_SCHEMA = '{"facts": [{"metric": "...", "period": "...", "value": "<as printed>", "scale": "units|thousands|millions|billions", "currency": "...", "basis": "GAAP|non_GAAP", "source_id": "S#", "quote": "..."}], "missing": ["<metric @ period>"]}'
_EX_COT_SCHEMA = '{"reasoning": ["<at most 4 short steps: locate table, identify columns, read row, check scale>"], "facts": [...same shape as below...], "missing": [...]}\nFact shape: ' + _EX_SCHEMA

_EX_USER = """Requested metrics: {{metrics}}
Requested periods: {{periods}}

{{evidence}}

Return JSON: """

_EX_DEMO_EVIDENCE = format_evidence([
    {
        "source_id": "S1",
        "document_name": "Q2 FY25 10-Q",
        "section": "Condensed Consolidated Statements of Operations",
        "page": 4,
        "period": "Q2 FY2025",
        "text": "(in millions, except per share data)  Three Months Ended June 30,  2025  2024\n"
        "Net sales  $ 1,284  $ 1,150\nCost of sales  742  689\nGross profit  542  461\n"
        "Operating income (loss)  118  (12)",
    }
])

_EX_EXAMPLES = (
    FewShotExample(
        "Requested metrics: net sales, operating income\nRequested periods: Q2 FY2025, Q2 FY2024\n\n" + _EX_DEMO_EVIDENCE + "\n\nReturn JSON: " + _EX_SCHEMA,
        _j({
            "facts": [
                {"metric": "net sales", "period": "Q2 FY2025", "value": "1,284", "scale": "millions", "currency": "USD", "basis": "GAAP", "source_id": "S1", "quote": "Net sales  $ 1,284  $ 1,150"},
                {"metric": "net sales", "period": "Q2 FY2024", "value": "1,150", "scale": "millions", "currency": "USD", "basis": "GAAP", "source_id": "S1", "quote": "Net sales  $ 1,284  $ 1,150"},
                {"metric": "operating income", "period": "Q2 FY2025", "value": "118", "scale": "millions", "currency": "USD", "basis": "GAAP", "source_id": "S1", "quote": "Operating income (loss)  118  (12)"},
                {"metric": "operating income", "period": "Q2 FY2024", "value": "(12)", "scale": "millions", "currency": "USD", "basis": "GAAP", "source_id": "S1", "quote": "Operating income (loss)  118  (12)"},
            ],
            "missing": [],
        }),
    ),
    FewShotExample(
        "Requested metrics: capital expenditures\nRequested periods: Q2 FY2025\n\n" + _EX_DEMO_EVIDENCE + "\n\nReturn JSON: " + _EX_SCHEMA,
        _j({"facts": [], "missing": ["capital expenditures @ Q2 FY2025"]}),
    ),
)

_EX_COMMON = dict(node=Node.EXTRACT, version=V, required_keys=("facts", "missing"), max_tokens=700)

register(
    PromptSpec(technique=Technique.ZERO_SHOT, system=system_prompt(_EX_TASK), user_template=_EX_USER + _EX_SCHEMA, **_EX_COMMON),
    PromptSpec(technique=Technique.FEW_SHOT, system=system_prompt(_EX_TASK), user_template=_EX_USER + _EX_SCHEMA, examples=_EX_EXAMPLES, **_EX_COMMON),
    PromptSpec(
        technique=Technique.COT,
        system=system_prompt(_EX_TASK + "\n\nBefore extracting, reason briefly in the \"reasoning\" list: which table holds the metric, which column is which period, what the stated scale is."),
        user_template=_EX_USER + _EX_COT_SCHEMA,
        **{**_EX_COMMON, "max_tokens": 900},
    ),
)


# =========================================================================== analyze (multi-hop / narrative)

_AN_TASK = """analyze retrieved evidence for a multi-document or narrative question and list grounded findings for the answer writer.
Each finding is one factual statement supported by one or more source_ids. Note any evidence gaps (something the question needs that is absent) and conflicts (two sources disagree, or a figure was restated; keep both with their sources).
Distinguish reported GAAP figures from Non-GAAP/adjusted figures, and management guidance (forward-looking, attributed) from reported results."""

_AN_SCHEMA = '{"findings": [{"statement": "...", "source_ids": ["S#"]}], "gaps": ["..."], "conflicts": [{"description": "...", "source_ids": ["S#", "S#"]}]}'
_AN_COT_SCHEMA = '{"reasoning": ["<at most 5 short steps>"], "findings": [{"statement": "...", "source_ids": ["S#"]}], "gaps": ["..."], "conflicts": [{"description": "...", "source_ids": ["S#", "S#"]}]}'

_AN_USER = """Question: {{question}}
Plan: {{plan}}

{{evidence}}

Return JSON: """

_AN_EXAMPLES = (
    FewShotExample(
        "Question: What did Northwind say drove the Q3 FY2025 gross margin change?\nPlan: {\"intent\": \"multi_hop\", \"periods\": [\"Q3 FY2025\"]}\n\n"
        + format_evidence([
            {"source_id": "S1", "document_name": "Q3 FY25 Earnings Release", "page": 1, "period": "Q3 FY2025", "text": "GAAP gross margin was 41.2%, compared with 38.9% in the prior-year quarter."},
            {"source_id": "S2", "document_name": "Q3 FY25 Earnings Call Transcript", "page": 6, "period": "Q3 FY2025", "text": "CFO: The improvement was mainly mix, as higher-margin sensor modules grew faster, partly offset by freight costs."},
        ])
        + "\n\nReturn JSON: " + _AN_SCHEMA,
        _j({
            "findings": [
                {"statement": "GAAP gross margin was 41.2% in Q3 FY2025 versus 38.9% in Q3 FY2024.", "source_ids": ["S1"]},
                {"statement": "The CFO attributed the improvement mainly to product mix toward higher-margin sensor modules, partly offset by freight costs.", "source_ids": ["S2"]},
            ],
            "gaps": [],
            "conflicts": [],
        }),
    ),
)

_AN_COMMON = dict(node=Node.ANALYZE, version=V, required_keys=("findings", "gaps", "conflicts"), max_tokens=800)

register(
    PromptSpec(technique=Technique.ZERO_SHOT, system=system_prompt(_AN_TASK), user_template=_AN_USER + _AN_SCHEMA, **_AN_COMMON),
    PromptSpec(technique=Technique.FEW_SHOT, system=system_prompt(_AN_TASK), user_template=_AN_USER + _AN_SCHEMA, examples=_AN_EXAMPLES, **_AN_COMMON),
    PromptSpec(
        technique=Technique.COT,
        system=system_prompt(
            _AN_TASK
            + "\n\nReason step by step in the \"reasoning\" list: (1) split the question into the sub-facts it needs, (2) find each sub-fact and its source, (3) check periods and basis match, (4) note what is missing or conflicting, (5) write findings."
        ),
        user_template=_AN_USER + _AN_COT_SCHEMA,
        **{**_AN_COMMON, "max_tokens": 1100},
    ),
)

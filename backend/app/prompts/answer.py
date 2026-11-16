"""Answer-side prompts: synthesize_answer, verify_answer (critic), repair_once.

Citation contract: the model writes inline markers [S1] for evidence and [C1]
for calculator results; ``claims`` lists the IDs per sentence. Code then
(a) checks every ID was actually supplied for this request and
(b) renders S-markers to "[Doc_Name | Section/Page | Period]".
"""
from __future__ import annotations

import json

from .base import FewShotExample, Node, PromptSpec, Technique, register
from .shared import format_evidence, format_tool_results, system_prompt

V = "1.0"

SYNTH_STATUSES = ("answered", "partially_answered", "insufficient_evidence")
VERDICTS = ("pass", "fail")
ISSUE_TYPES = (
    "unsupported_claim",
    "wrong_citation",
    "number_mismatch",
    "period_mismatch",
    "unit_or_scale_mismatch",
    "missing_basis_label",
    "model_arithmetic",
    "policy_violation",
    "incomplete",
)


def _j(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


# =========================================================================== synthesize

_SY_TASK = """write the final answer from the evidence and calculator results.
Format of "answer":
- First sentence answers the question directly.
- Then, only if useful, 1-4 short supporting sentences or "- " bullets.
- Every sentence with a fact or number ends with its markers, e.g. [S1] or [S1][C1].
- For a derived figure, show the formula and inputs exactly as given in the tool_result, e.g. "($1,284M - $1,150M) / $1,150M = 11.65% [C1]".
- Label Non-GAAP/adjusted figures as such. Never blend GAAP and Non-GAAP figures in one calculation.
- Quote guidance as management's ("Management guided ..."), never as your own forecast.
"claims" lists each factual sentence of the answer with the IDs it relies on.
Use status "insufficient_evidence" (and say what is missing) if the evidence cannot answer; "partially_answered" if only part can be answered."""

_SY_SCHEMA = '{"status": "answered|partially_answered|insufficient_evidence", "answer": "...", "claims": [{"text": "...", "source_ids": ["S#"], "calc_ids": ["C#"]}]}'
_SY_COT_SCHEMA = '{"reasoning": ["<at most 4 short steps>"], "status": "answered|partially_answered|insufficient_evidence", "answer": "...", "claims": [{"text": "...", "source_ids": ["S#"], "calc_ids": ["C#"]}]}'

_SY_USER = """Question: {{question}}
Plan: {{plan}}

{{evidence}}

{{tool_results}}

Return JSON: """

_SY_EX1_EVID = format_evidence([
    {"source_id": "S1", "document_name": "Q2 FY25 10-Q", "section": "Condensed Consolidated Statements of Operations", "page": 4, "period": "Q2 FY2025",
     "text": "(in millions) Three Months Ended June 30, 2025 2024\nNet sales $ 1,284 $ 1,150"},
])
_SY_EX1_TOOLS = format_tool_results([
    {"calc_id": "C1", "tool": "calculate_growth", "formula": "(current - prior) / prior * 100", "inputs": {"current": "1284", "prior": "1150", "scale": "millions", "currency": "USD"}, "result": "11.65", "unit": "percent", "source_ids": ["S1"], "status": "valid"},
])

_SY_EX2_EVID = format_evidence([
    {"source_id": "S1", "document_name": "FY24 10-K", "section": "Item 7 MD&A", "page": 38, "period": "FY2024", "basis": "GAAP",
     "text": "Operating income was $412 million in fiscal 2024."},
    {"source_id": "S2", "document_name": "Q4 FY24 Earnings Release", "page": 2, "period": "FY2024", "basis": "non_GAAP",
     "text": "Adjusted EBITDA for fiscal 2024 was $655 million. See reconciliation tables."},
])

_SY_EXAMPLES = (
    FewShotExample(
        'Question: What was Northwind\'s year-over-year net sales growth in Q2 FY2025?\nPlan: {"intent": "calculation"}\n\n' + _SY_EX1_EVID + "\n\n" + _SY_EX1_TOOLS + "\n\nReturn JSON: " + _SY_SCHEMA,
        _j({
            "status": "answered",
            "answer": "Northwind's net sales grew 11.65% year over year in Q2 FY2025 [S1][C1].\n- Q2 FY2025 net sales were $1,284M versus $1,150M in Q2 FY2024 [S1].\n- Calculation: ($1,284M - $1,150M) / $1,150M = 11.65% [C1].",
            "claims": [
                {"text": "Northwind's net sales grew 11.65% year over year in Q2 FY2025.", "source_ids": ["S1"], "calc_ids": ["C1"]},
                {"text": "Q2 FY2025 net sales were $1,284M versus $1,150M in Q2 FY2024.", "source_ids": ["S1"], "calc_ids": []},
                {"text": "Calculation: ($1,284M - $1,150M) / $1,150M = 11.65%.", "source_ids": [], "calc_ids": ["C1"]},
            ],
        }),
    ),
    FewShotExample(
        'Question: What was Bluefin\'s FY2024 adjusted EBITDA margin using operating income as the base?\nPlan: {"intent": "calculation", "basis": "both"}\n\n' + _SY_EX2_EVID + "\n\n" + format_tool_results([]) + "\n\nReturn JSON: " + _SY_SCHEMA,
        _j({
            "status": "partially_answered",
            "answer": "I can't compute that ratio because it mixes a Non-GAAP figure (adjusted EBITDA) with a GAAP figure (operating income), and no bridge between them was retrieved.\n- FY2024 GAAP operating income was $412M [S1].\n- FY2024 adjusted EBITDA (Non-GAAP) was $655M [S2].\nI can compute a like-for-like margin if FY2024 revenue is provided in the same basis.",
            "claims": [
                {"text": "FY2024 GAAP operating income was $412M.", "source_ids": ["S1"], "calc_ids": []},
                {"text": "FY2024 adjusted EBITDA (Non-GAAP) was $655M.", "source_ids": ["S2"], "calc_ids": []},
            ],
        }),
    ),
    FewShotExample(
        'Question: What was Northwind\'s Q4 FY2025 revenue?\nPlan: {"intent": "extraction", "periods": ["Q4 FY2025"]}\n\n' + _SY_EX1_EVID + "\n\n" + format_tool_results([]) + "\n\nReturn JSON: " + _SY_SCHEMA,
        _j({
            "status": "insufficient_evidence",
            "answer": "The indexed documents do not contain Northwind's Q4 FY2025 revenue; the latest retrieved figures are for Q2 FY2025 [S1].",
            "claims": [{"text": "The latest retrieved figures are for Q2 FY2025.", "source_ids": ["S1"], "calc_ids": []}],
        }),
    ),
)

_SY_COMMON = dict(
    node=Node.SYNTHESIZE,
    version=V,
    required_keys=("status", "answer", "claims"),
    enums={"status": SYNTH_STATUSES},
    max_tokens=700,
    temperature=0.1,
)

register(
    PromptSpec(technique=Technique.ZERO_SHOT, system=system_prompt(_SY_TASK), user_template=_SY_USER + _SY_SCHEMA, **_SY_COMMON),
    PromptSpec(technique=Technique.FEW_SHOT, system=system_prompt(_SY_TASK), user_template=_SY_USER + _SY_SCHEMA, examples=_SY_EXAMPLES, **_SY_COMMON),
    PromptSpec(
        technique=Technique.COT,
        system=system_prompt(
            _SY_TASK
            + "\n\nPlan in the \"reasoning\" list before writing: (1) what exactly is asked, (2) which S/C IDs answer each part, (3) any period/basis/scale issue, (4) answered vs partial vs insufficient."
        ),
        user_template=_SY_USER + _SY_COT_SCHEMA,
        **{**_SY_COMMON, "max_tokens": 900},
    ),
)


# =========================================================================== verify (critic)

_VE_TASK = f"""audit a draft answer against the evidence and calculator results. You are a strict reviewer; you do not rewrite the answer.
Check each claim:
- Is every fact stated in the cited evidence? (unsupported_claim / wrong_citation)
- Do numbers, periods, units and scale match the source or tool_result exactly? (number_mismatch / period_mismatch / unit_or_scale_mismatch)
- Is every derived number taken from a tool_result rather than computed in the text? (model_arithmetic)
- Are Non-GAAP figures labelled and never mixed with GAAP in one calculation? (missing_basis_label)
- Does it avoid advice, price targets, unattributed forecasts and legal/tax/audit opinions? (policy_violation)
- Does it answer every part of the question it claims to answer? (incomplete)
Issue types: {", ".join(ISSUE_TYPES)}.
verdict is "pass" only if issues is empty."""

_VE_SCHEMA = '{"verdict": "pass|fail", "issues": [{"claim_index": 0, "type": "<issue type>", "detail": "<what is wrong and the correct value/source if known>"}]}'
_VE_COT_SCHEMA = '{"reasoning": ["<one short line per claim: claim_index -> checked against S#/C# -> ok or problem>"], "verdict": "pass|fail", "issues": [{"claim_index": 0, "type": "<issue type>", "detail": "..."}]}'

_VE_USER = """Question: {{question}}

{{evidence}}

{{tool_results}}

Draft answer JSON:
{{draft}}

Return JSON: """

_VE_EXAMPLES = (
    FewShotExample(
        "Question: What was Northwind's year-over-year net sales growth in Q2 FY2025?\n\n" + _SY_EX1_EVID + "\n\n" + _SY_EX1_TOOLS
        + '\n\nDraft answer JSON:\n{"status": "answered", "answer": "Net sales grew 12.1% to $1,284M in Q2 FY2025 [S1][C1].", "claims": [{"text": "Net sales grew 12.1% to $1,284M in Q2 FY2025.", "source_ids": ["S1"], "calc_ids": ["C1"]}]}\n\nReturn JSON: '
        + _VE_SCHEMA,
        _j({"verdict": "fail", "issues": [{"claim_index": 0, "type": "number_mismatch", "detail": "Growth must be 11.65% as in C1, not 12.1%."}]}),
    ),
)

_VE_COMMON = dict(
    node=Node.VERIFY,
    version=V,
    required_keys=("verdict", "issues"),
    enums={"verdict": VERDICTS},
    max_tokens=500,
)

register(
    PromptSpec(technique=Technique.ZERO_SHOT, system=system_prompt(_VE_TASK), user_template=_VE_USER + _VE_SCHEMA, **_VE_COMMON),
    PromptSpec(technique=Technique.FEW_SHOT, system=system_prompt(_VE_TASK), user_template=_VE_USER + _VE_SCHEMA, examples=_VE_EXAMPLES, **_VE_COMMON),
    PromptSpec(
        technique=Technique.COT,
        system=system_prompt(_VE_TASK + "\n\nWork claim by claim in the \"reasoning\" list before giving the verdict."),
        user_template=_VE_USER + _VE_COT_SCHEMA,
        **{**_VE_COMMON, "max_tokens": 800},
    ),
)


# =========================================================================== repair (single bounded attempt)

_RP_TASK = """fix a draft answer using the reviewer's issues. Make the smallest change that resolves each issue:
- Replace a wrong number/period/unit with the value from the cited evidence or tool_result.
- If a claim cannot be supported by the supplied evidence, delete it; do not search for new facts.
- Do not add new sources, tools or claims. Do not use IDs that are not in this message.
If removing claims leaves the question unanswered, set status to "partially_answered" or "insufficient_evidence" and say what is missing."""

_RP_USER = """Question: {{question}}

{{evidence}}

{{tool_results}}

Draft answer JSON:
{{draft}}

Reviewer issues JSON:
{{issues}}

Return the corrected answer as JSON: """ + _SY_SCHEMA

register(
    PromptSpec(
        node=Node.REPAIR,
        technique=Technique.ZERO_SHOT,
        version=V,
        system=system_prompt(_RP_TASK),
        user_template=_RP_USER,
        required_keys=("status", "answer", "claims"),
        enums={"status": SYNTH_STATUSES},
        max_tokens=700,
    ),
)

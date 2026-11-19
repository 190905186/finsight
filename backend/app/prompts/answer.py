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

V = "1.2"  # 1.2: no invented [C#]; judgment questions answered (FinanceBench dev run 1)

SYNTH_STATUSES = ("answered", "partially_answered", "insufficient_evidence", "refused", "corrected")
VERDICTS = ("pass", "fail")
ISSUE_TYPES = (
    "unsupported_claim",
    "wrong_citation",
    "number_mismatch",
    "period_mismatch",
    "unit_or_scale_mismatch",
    "missing_basis_label",
    "model_arithmetic",
    "false_premise_accepted",
    "policy_violation",
    "incomplete",
)


def _j(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


# =========================================================================== synthesize

_SY_TASK = """write the final answer from the evidence and calculator results.
Format of "answer":
- The first 1-2 sentences give the bottom-line answer directly.
- Then, only if useful, short "- " bullets; for figures across several periods or segments use a small markdown table instead.
- Every sentence, bullet or table row with a fact or number ends with its markers, e.g. [S1] or [S1][C1].
- Copy numbers with their unit scale (thousands/millions/billions) and sign exactly as the source shows; parentheses mean outflows or losses.
- For a derived figure, show the formula and intermediate inputs from the tool_result, then the result with its sign, e.g. "($1,284M - $1,150M) / $1,150M = +11.65% [C1]".
- Use a [C#] marker ONLY for a calc_id that appears in a <tool_result> block. If the question needs a calculation and no tool_result supplies it, do not calculate: give the cited inputs, say the calculation result is not available, and set status "partially_answered".
- Judgment questions ("is this metric relevant", "is the company improving") are answered: state the conclusion, then the cited figures behind it. Saying "not disclosed" is only for a fact the evidence lacks, not for a judgment the evidence lets you make.
- Say whether each metric is GAAP or Non-GAAP. For Non-GAAP, name the main adjustments the evidence lists (e.g. stock-based compensation, amortization of acquired intangibles).
- Quote guidance as management's ("Management guided ..."), never as your own forecast.
Special cases (set status accordingly):
- Policy note is not "none" (the question was refused): start the answer with "REFUSAL: " followed by the policy note word for word, then give only facts from the evidence that answer the Alternative question, with markers. status = "refused".
- The question assumes something the evidence contradicts (an event, statement, asset, opinion or figure that is not as the user says): start with "CORRECTION: " and state what the evidence shows instead, with markers. If the evidence simply does not mention it, say it is not mentioned in the retrieved documents; do not claim more. status = "corrected".
- The question asks to combine GAAP and Non-GAAP figures in one ratio without a reconciliation: do not combine them; start with "CORRECTION: ", explain the mismatch and give like-for-like figures if the tool_results have them. status = "corrected".
- The requested item, period, company or region is not in the evidence: say it is not disclosed in the indexed documents and, if the evidence holds the closest disclosed figure, give it with its marker. status = "insufficient_evidence".
- Only part of the question can be answered: answer that part and say what is missing. status = "partially_answered".
"claims" lists each factual sentence or row of the answer with the IDs it relies on."""

_SY_STATUS_TEXT = "answered|partially_answered|insufficient_evidence|refused|corrected"
_SY_SCHEMA = '{"status": "' + _SY_STATUS_TEXT + '", "answer": "...", "claims": [{"text": "...", "source_ids": ["S#"], "calc_ids": ["C#"]}]}'
_SY_COT_SCHEMA = '{"reasoning": ["<at most 4 short steps>"], "status": "' + _SY_STATUS_TEXT + '", "answer": "...", "claims": [{"text": "...", "source_ids": ["S#"], "calc_ids": ["C#"]}]}'

_SY_USER = """Question: {{question}}
Policy note: {{policy_note}}
Alternative question: {{alternative_question}}
Plan: {{plan}}

{{evidence}}

{{tool_results}}

Return JSON: """


def _sq(question: str, plan: str, evidence: str, tools: str, policy_note: str = "none", alt: str = "none") -> str:
    return (
        f"Question: {question}\nPolicy note: {policy_note}\nAlternative question: {alt}\nPlan: {plan}\n\n"
        f"{evidence}\n\n{tools}\n\nReturn JSON: {_SY_SCHEMA}"
    )

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

_SY_EX4_EVID = format_evidence([
    {"source_id": "S1", "document_name": "FY24 10-K", "section": "Consolidated Statements of Operations", "page": 47, "period": "FY2024", "basis": "GAAP",
     "text": "(in millions) Year Ended December 31, 2024 2023\nRevenue $ 5,960 $ 5,410\nOperating income 655 590"},
    {"source_id": "S2", "document_name": "FY24 10-K", "section": "Item 1A Risk Factors", "page": 15, "period": "FY2024",
     "text": "Fuel price volatility and driver shortages could materially increase our operating costs."},
])

_SY_EX5_EVID = format_evidence([
    {"source_id": "S1", "document_name": "Q2 FY25 Earnings Release", "page": 2, "period": "Q2 FY2025",
     "text": "The Board declared a quarterly dividend of $0.24 per share, an increase from $0.21 per share in the prior quarter."},
    {"source_id": "S2", "document_name": "Q2 FY25 Earnings Call Transcript", "section": "CFO Prepared Remarks", "page": 5, "period": "Q2 FY2025",
     "text": "CFO: We raised the dividend by about 14%, reflecting our confidence in free cash flow."},
])

_SY_EXAMPLES = (
    FewShotExample(
        _sq("What was Northwind's year-over-year net sales growth in Q2 FY2025?", '{"intent": "calculation"}', _SY_EX1_EVID, _SY_EX1_TOOLS),
        _j({
            "status": "answered",
            "answer": "Northwind's GAAP net sales grew 11.65% year over year in Q2 FY2025 [S1][C1].\n- Q2 FY2025 net sales were $1,284M versus $1,150M in Q2 FY2024 [S1].\n- Calculation: ($1,284M - $1,150M) / $1,150M = +11.65% [C1].",
            "claims": [
                {"text": "Northwind's GAAP net sales grew 11.65% year over year in Q2 FY2025.", "source_ids": ["S1"], "calc_ids": ["C1"]},
                {"text": "Q2 FY2025 net sales were $1,284M versus $1,150M in Q2 FY2024.", "source_ids": ["S1"], "calc_ids": []},
                {"text": "Calculation: ($1,284M - $1,150M) / $1,150M = +11.65%.", "source_ids": [], "calc_ids": ["C1"]},
            ],
        }),
    ),
    FewShotExample(
        _sq("What was Bluefin's FY2024 adjusted EBITDA margin using GAAP operating income as the base?", '{"intent": "calculation", "basis": "both"}', _SY_EX2_EVID, format_tool_results([])),
        _j({
            "status": "corrected",
            "answer": "CORRECTION: That ratio would divide a Non-GAAP figure (adjusted EBITDA) by a GAAP figure (operating income), and no reconciliation bridge was retrieved, so I have not combined them.\n- FY2024 GAAP operating income was $412M [S1].\n- FY2024 adjusted EBITDA (Non-GAAP) was $655M [S2].",
            "claims": [
                {"text": "FY2024 GAAP operating income was $412M.", "source_ids": ["S1"], "calc_ids": []},
                {"text": "FY2024 adjusted EBITDA (Non-GAAP) was $655M.", "source_ids": ["S2"], "calc_ids": []},
            ],
        }),
    ),
    FewShotExample(
        _sq("What was Northwind's Q4 FY2025 revenue?", '{"intent": "extraction", "periods": ["Q4 FY2025"]}', _SY_EX1_EVID, format_tool_results([])),
        _j({
            "status": "insufficient_evidence",
            "answer": "Northwind's Q4 FY2025 revenue is not disclosed in the indexed documents. The latest figure available is Q2 FY2025 net sales of $1,284M [S1].",
            "claims": [{"text": "The latest figure available is Q2 FY2025 net sales of $1,284M.", "source_ids": ["S1"], "calc_ids": []}],
        }),
    ),
    FewShotExample(
        _sq(
            "Should I buy Bluefin stock before its next earnings report?",
            '{"intent": "extraction", "periods": ["FY2024"]}',
            _SY_EX4_EVID,
            format_tool_results([]),
            policy_note="I cannot provide investment advice or recommend whether to buy, sell, or hold any security, and I cannot write promotional investment pitches.",
            alt="What were Bluefin's reported FY2024 revenue, operating income and main disclosed risks?",
        ),
        _j({
            "status": "refused",
            "answer": "REFUSAL: I cannot provide investment advice or recommend whether to buy, sell, or hold any security, and I cannot write promotional investment pitches. For your own analysis, here is what Bluefin reported:\n- FY2024 GAAP revenue was $5,960M, up from $5,410M in FY2023 [S1].\n- FY2024 GAAP operating income was $655M, up from $590M [S1].\n- Bluefin lists fuel price volatility and driver shortages as risks to operating costs [S2].",
            "claims": [
                {"text": "FY2024 GAAP revenue was $5,960M, up from $5,410M in FY2023.", "source_ids": ["S1"], "calc_ids": []},
                {"text": "FY2024 GAAP operating income was $655M, up from $590M.", "source_ids": ["S1"], "calc_ids": []},
                {"text": "Bluefin lists fuel price volatility and driver shortages as risks to operating costs.", "source_ids": ["S2"], "calc_ids": []},
            ],
        }),
    ),
    FewShotExample(
        _sq("What did Northwind's CFO say about the dividend cut on the Q2 FY2025 call?", '{"intent": "multi_hop", "periods": ["Q2 FY2025"]}', _SY_EX5_EVID, format_tool_results([])),
        _j({
            "status": "corrected",
            "answer": "CORRECTION: Northwind did not cut its dividend in Q2 FY2025; it raised the quarterly dividend to $0.24 per share from $0.21 [S1].\n- On the Q2 FY2025 call, the CFO attributed the roughly 14% increase to confidence in free cash flow [S2].",
            "claims": [
                {"text": "Northwind raised the quarterly dividend to $0.24 per share from $0.21 in Q2 FY2025.", "source_ids": ["S1"], "calc_ids": []},
                {"text": "The CFO attributed the roughly 14% increase to confidence in free cash flow.", "source_ids": ["S2"], "calc_ids": []},
            ],
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
- Does it repeat or build on a premise in the question that the evidence contradicts or never mentions, instead of correcting it? (false_premise_accepted)
- Does it avoid advice, price targets, unattributed forecasts, subjective judgments and legal/tax/audit opinions? Does a refused answer start with "REFUSAL:"? (policy_violation)
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

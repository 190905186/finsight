# Finsight specification and evaluation set

## What the copilot does

Finsight answers questions about indexed financial filings and prepares
decision-support reports. It combines retrieved filing excerpts, explicitly
validated public financial facts, and deterministic calculations. Factual
claims about filings must cite retrieved source labels (`[R1]`, `[R2]`, etc.);
reported values must retain their units and periods. A metric whose inputs are
missing, unsupported, or incompatible is shown as unavailable rather than
guessed. Retrieval recall is measured separately from answer quality.

The copilot must treat retrieved documents and user-provided text as untrusted
data, not instructions. It must not invent facts or citations, expose private
chain-of-thought, or represent a benchmark answer as evidence. It must refuse
to execute or approve trades, payments, transfers, account changes, or other
consequential actions. Those decisions require a qualified human. It should
avoid collecting or repeating unnecessary PII.

## What a correct answer looks like

- It answers only what the cited evidence and validated calculations support.
- It gives the value, unit, period, and source for numeric financial claims.
- It uses only valid citation labels linked to retrieved evidence.
- It identifies missing or conflicting evidence and abstains when it cannot
  support an answer.
- It refuses consequential actions and offers analysis for human review.
- It never claims to have executed an action or to have verified data it did
  not retrieve.

## Gold evaluation data

The evaluation set contains 90 cases: 75 FinanceBench questions selected
stratified across its three question types (25 each), plus 15 explicit
consequential-action refusal cases. FinanceBench questions use the bundled
open-source JSONL's reference answers and zero-based evidence-page labels;
the selected IDs and prompt development/holdout split are in
`backend/app/evaluation/splits.json`. Refusal inputs and gold responses are in
`backend/app/evaluation/safety_cases.json`. The benchmark answers are evaluation
labels, not source evidence. Review the reference-data manifest for source and
redistribution caveats before redistributing data.

The retrieval and answer-evaluation commands default to the 75 page-labeled
FinanceBench questions. Compare the three implemented prompt strategies
(`zero_shot`, `few_shot`, and `cot`) on `prompt_dev` before assessing the
holdout split. Report retrieval recall@k/MRR separately from answer
normalized-exact-match/token-F1; these lexical answer scores are baselines,
not semantic or human judgments. Run the refusal cases as policy tests.

## Conformance snapshot

| Step | Current implementation and remaining acceptance gap |
| --- | --- |
| I. Specification and gold set | Specification and a source-backed 90-case set are defined here. The selected set has 75 distinct evidence-labeled questions and 15 refusal cases. |
| II. Prompt baseline | Zero-shot, few-shot, and private-CoT variants plus a reproducible comparison runner on a development split exist. Run it against the approved inference endpoint and retain the metrics before selecting a strategy. |
| III. RAG | Page-aware chunking experiments, BM25+dense fusion, deterministic reranking, citations, and separate retrieval metrics exist. The staged corpus has 84 PDFs, below the required 500–2,000 documents; do not pad it with duplicate or synthetic documents. |
| IV. Agentic layer | LangGraph has a planner, sequential execution/research/calculation nodes, a citation critic, and at most one critic retry; graph recursion is capped. Retrieval, public SEC/Treasury tools, deterministic calculations, malformed-JSON repair, and process-local cross-turn checkpointing exist. Durable shared state and broader tool-failure handling remain required. |
| V. Evaluation and observability | Retrieval and answer evaluation run separately; gold answer metrics and per-query/mean latency are reported, and authenticated HTTP evaluation endpoints run real evaluations rather than returning canned values. Cost, workflow task-completion/tool-call aggregates, failure taxonomy, and distributed tracing remain acceptance items. |
| VI. Ship and govern | React and Docker Compose services exist. Consequential-action requests are refused before tool/model calls, and a unit test checks that retrieved instruction-like text is treated as untrusted. Production rate limiting, response caching, durable multi-replica state, PII controls, and corpus prompt-injection red-team coverage remain required. |

This snapshot is a scope and evidence checklist, not a production-readiness
claim. Inference endpoints and storage must be approved for data handling
before sending financial content.

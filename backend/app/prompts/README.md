# Finsight prompt library (`app/prompts`)

Prompts for every LLM node in the LangGraph workflow. Each node has **zero-shot**, **few-shot** and **chain-of-thought (CoT)** variants, all sharing the same Qwen3-8B endpoint. They're versioned, tested and ready to A/B on the dev split.

## What's in here

| File | Contents |
|---|---|
| `base.py` | `PromptSpec`, `Technique`, `Node`, registry, `get_prompt()`, Qwen3 `/no_think` switch |
| `shared.py` | Persona + 7 grounding rules shared by every system prompt; `format_evidence()`, `format_tool_results()` |
| `routing.py` | `pre_guardrail`, `classify_and_plan`, `query_rewrite` |
| `evidence.py` | `extract_financial_fact`, `analyze_financial_report` |
| `answer.py` | `synthesize_answer`, `verify_answer` (critic), `repair_once` |
| `refusals.py` | Fixed refusal text for the six DOCX policy cases (no LLM call) |
| `parsing.py` | Pulls JSON out of `<think>` blocks, code fences and prose, then checks required keys and enum values |
| `runner.py` | `run_prompt()`: render → call → parse → at most **one** format-repair call |
| `experiment.py` | CLI to compare techniques on a labelled dev set (accuracy, JSON validity, latency, tokens) |
| `dev_sample.jsonl` | 16 synthetic dev questions (fictional companies) for smoke runs |

## Technique matrix

| Node | zero_shot | few_shot | cot | **Default** | Why |
|---|:-:|:-:|:-:|---|---|
| `pre_guardrail` | ✓ | ✓ (6 ex.) | ✓ | few_shot | Refusal edges (guidance vs prediction, injection) are learned best from examples |
| `classify_and_plan` | ✓ | ✓ (4 ex.) | ✓ | few_shot | Examples teach period pairs (Q3 FY25 + Q3 FY24) and tool choice |
| `query_rewrite` | ✓ | ✓ (1 ex.) | – | few_shot | Teaches filing vocabulary ("purchases of property and equipment") |
| `extract_financial_fact` | ✓ | ✓ (2 ex.) | ✓ | few_shot | Column/period alignment, parentheses = negative, "missing" handling |
| `analyze_financial_report` | ✓ | ✓ (1 ex.) | ✓ | cot | Multi-hop needs decomposition into sub-facts |
| `synthesize_answer` | ✓ | ✓ (3 ex.) | ✓ | few_shot | Shows citation markers, formula display, GAAP/Non-GAAP refusal, insufficient evidence |
| `verify_answer` | ✓ | ✓ (1 ex.) | ✓ | cot | Claim-by-claim checking catches more errors |
| `repair_once` | ✓ | – | – | zero_shot | Tightly scoped; examples add cost with little gain |

These defaults are **starting hypotheses**. Change them in `DEFAULT_TECHNIQUE` (or per run with an env var, e.g. `PROMPT_TECHNIQUE_CLASSIFY=cot`) based on dev-split results.

## Design decisions to keep (important for a ≤8B model)

1. **The model cites IDs only.** Evidence comes in as `<evidence source_id="S1" ...>` and calculator results as `<tool_result calc_id="C1">`. The model writes `[S1][C1]`; code renders `[Doc_Name | Section/Page | Period]` and rejects any ID that wasn't supplied. This makes the citation-integrity check deterministic.
2. **The model never does arithmetic.** Extraction returns values *as printed* (strings), `Decimal` code calculates, and synthesis copies the tool's formula. The verifier flags `model_arithmetic`.
3. **CoT goes in a JSON `reasoning` field, not in Qwen3's native thinking.** Native thinking isn't bounded and costs a lot of tokens on every node. Every prompt sent to a Qwen3 model gets `/no_think`, and CoT variants ask for 3–5 short steps instead. To try native thinking, set `PROMPT_QWEN3_THINKING=1`. Local `qwen2.5:3b` gets no suffix.
4. **Few-shot examples use fictional companies** (Northwind Instruments, Bluefin Logistics). Nothing from the Acme golden set appears in any prompt, and a test enforces this, so the 75-query evaluation isn't contaminated.
5. **Retrieved text is untrusted.** Grounding rule 2 plus the `<evidence>` tags form the prompt-injection boundary. ACL filtering still happens in code before retrieval.
6. **Refusals are deterministic text.** The guardrail prompt *classifies*, and `refusal_message(category)` *speaks*.
7. **Fail closed.** A missing template variable raises an error. A bad JSON reply gets one repair call, and after that the node should route to `degraded`.

## Using it in a graph node

```python
from app.prompts import Node, get_prompt, run_prompt, format_evidence, format_tool_results
from app.services.llm_client import LLMClient

with LLMClient() as llm:
    res = run_prompt(
        llm, get_prompt(Node.SYNTHESIZE),
        question=state.question,
        plan=json.dumps(state.plan),
        evidence=format_evidence(state.evidence),        # dicts with source_id, text, document_name, section, page, period, unit, basis
        tool_results=format_tool_results(state.calcs),   # dicts with calc_id, tool, formula, inputs, result, unit, source_ids, status
    )
    if not res.ok:
        return route_to("degraded")
    state.draft = res.parsed.data
    state.llm_calls += res.calls          # enforce LLM_MAX_CALLS_PER_REQUEST
    state.prompt_ids.append(res.prompt_id)  # e.g. "synthesize_answer/few_shot@1.0" -> log in traces/eval runs
```

Template variables per node:

| Node | Variables |
|---|---|
| guardrail, classify | `question`, `company`, `period` (use `"none"` if not selected) |
| rewrite | `question`, `plan` |
| extract | `metrics`, `periods`, `evidence` |
| analyze | `question`, `plan`, `evidence` |
| synthesize | `question`, `plan`, `evidence`, `tool_results` |
| verify | `question`, `evidence`, `tool_results`, `draft` |
| repair | `question`, `evidence`, `tool_results`, `draft`, `issues` |

## Comparing techniques

From `backend/`, with `.env` pointing at the endpoint:

```bash
python -m app.prompts.experiment --node pre_guardrail
python -m app.prompts.experiment --node classify_and_plan --techniques zero_shot few_shot cot --repeats 3 --out cls_results.json
```

The output is a markdown table with accuracy, JSON validity on the first try, validity after repair, average calls, p50 latency and average tokens for each technique. Paste it into the team report.

**Protocol (from design doc §9):**

1. Tune only on the **dev** split. Build a JSONL from the DOCX dev questions using the same fields as `dev_sample.jsonl`: `expected_decision`/`expected_category` for the guardrail and `expected_intent` for classify.
2. Pick a winner per node by accuracy first, then JSON validity, then tokens and latency.
3. Freeze the prompts, bump `PROMPT_SET_VERSION`, then run the held-out 75 queries end to end. Record `prompt_id` for every node.
4. Don't edit prompts after looking at held-out results.

Synthesis, verify and extract need retrieved evidence, so evaluate them end to end through the graph (citation precision/recall, key-number match) once retrieval works.

## Tests

```bash
cd backend && python -m pytest -q tests/test_prompts.py
```

52 offline tests. They check that every prompt renders, every few-shot answer parses against its own schema, there's no Acme leakage, the Qwen3 switch works, the parser handles `<think>`, fences and prose, the single repair works, and experiment scoring works with a fake LLM.

## Optional upgrades

- If the hosted endpoint is vLLM, pass `response_format={"type": "json_object"}` (or `guided_json`) through `LLMClient` to force valid JSON. `parse_output` still validates enums.
- Self-consistency for `classify_and_plan`: run CoT 3× at temperature 0.5 and take a majority vote on `intent`. This only makes sense if dev results show the intent is unstable, and it triples the calls.

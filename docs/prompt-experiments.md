# Prompt experiments log

Each run compares prompting techniques on the **dev** questions (`backend/app/prompts/dev_sample.jsonl`, 24 fictional-company questions), never on the held-out 75-query golden set.
Model: `Qwen/Qwen3-8B` on the Track B endpoint (group 046-065), temperature 0, `/no_think`.
Full per-question outputs are in `data/evaluation/runs/`.

## Run 1: 2026-11-19, prompt set 2026.11.1

### pre_guardrail (24 questions)

| technique | accuracy | category acc | JSON 1st try | valid | avg calls | p50 latency s | avg tokens |
|---|---|---|---|---|---|---|---|
| zero_shot | 54.2% | 66.7% | 91.7% | 100.0% | 1.08 | 1.52 | 880 |
| **few_shot** | **66.7%** | **91.7%** | 100.0% | 100.0% | 1.00 | 1.08 | 1859 |
| cot | 58.3% | 75.0% | 100.0% | 100.0% | 1.00 | 3.13 | 936 |

### classify_and_plan (13 questions with an intent label)

| technique | accuracy | category acc | JSON 1st try | valid | avg calls | p50 latency s | avg tokens |
|---|---|---|---|---|---|---|---|
| zero_shot | 76.9% | - | 92.3% | 100.0% | 1.08 | 1.92 | 782 |
| **few_shot** | **76.9%** | - | **100.0%** | 100.0% | 1.00 | 1.64 | 1429 |
| cot | 53.8% | - | 100.0% | 100.0% | 1.00 | 3.96 | 873 |

### Findings

- **Few-shot wins both nodes.** CoT is slower (3-4 s vs 1-2 s) and no more accurate on these short routing tasks, so it is not used for routing.
- **Guardrail errors were all over-blocking** (allow → refuse/clarify). Few-shot got every must-refuse question right. Causes seen in the outputs:
  1. The shared evidence rules ("use only the evidence given", "never compute numbers") were in the routing prompts. With no documents retrieved yet, the model refused or clarified, e.g. "Calculate free cash flow" was refused as `price_prediction`.
  2. `Selected company: none` triggered clarify even when the question names the company.
  3. A factual audit question ("why did the auditor flag…") was treated as asking for an audit opinion.
  4. Some outputs said `refuse` with category `none`.
- **Classify errors:** old-year or undisclosed-figure questions → `clarify` (should plan the search); guidance ranges labelled `narrative` (should be `extraction`).

### Changes for run 2 (prompt set 2026.11.2)

- Routing nodes (guardrail, classify, rewrite) now use their own rules: route only, no documents yet is expected, calculations are allowed, never judge data availability, a company named in the question counts.
- Guardrail: factual audit/tax/legal questions are allowed; only judgments and tax-structuring advice are refused. A consistency check (refuse ⇔ category ≠ none) triggers the one format repair.
- Classify: clarify only when company or metric is undeterminable; guidance figures = extraction; two or more document types = multi_hop.

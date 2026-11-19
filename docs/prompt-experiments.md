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

## Run 2: 2026-11-19, prompt set 2026.11.2 (CoT dropped for routing)

### pre_guardrail (24 questions)

| technique | accuracy | category acc | JSON 1st try | valid | avg calls | p50 latency s | avg tokens |
|---|---|---|---|---|---|---|---|
| zero_shot | 75.0% | 70.8% | 100.0% | 100.0% | 1.00 | 1.30 | 823 |
| few_shot | 70.8% | 77.3% | 87.5% | 91.7% | 1.12 | 1.14 | 2149 |

### classify_and_plan (13 questions)

| technique | accuracy | category acc | JSON 1st try | valid | avg calls | p50 latency s | avg tokens |
|---|---|---|---|---|---|---|---|
| zero_shot | 84.6% | - | 100.0% | 100.0% | 1.00 | 1.81 | 762 |
| **few_shot** | **100.0%** | - | 100.0% | 100.0% | 1.00 | 1.68 | 1488 |

### Findings

- **classify_and_plan is settled: few-shot, 100%.**
- Guardrail improved (zero-shot 54% → 75%), but still over-blocks allowed questions: "total revenue", "summarize risk factors" and "management guidance range" were refused, mostly as `out_of_scope`.
- The new consistency check shows the bias: twice few-shot answered `refuse` with category `none` (no rule applied, but it refused anyway) and kept it after the repair turn.
- Likely causes: the few-shot set was 6 refuse / 2 allow / 1 clarify, so the model copies the refusal ratio; `out_of_scope` was vague ("unrelated to financial reports"); and the fictional companies may look unfamiliar.

### Changes for run 3 (prompt set 2026.11.3)

- Allow-by-default: refuse only when a question clearly matches a refusal type; an unfamiliar company is never a reason.
- `out_of_scope` narrowed to non-company, non-finance requests.
- Three more allow examples (calculation, disclosure summary, figures plus call commentary); the set is now 5 allow / 6 refuse / 1 clarify and ends on allow.
- The repair message for "refuse with no category" now says to switch to `allow`.
- Caution: 24 dev questions is a small set, so one question = 4.2 points. Before freezing, extend the dev set (written fresh, never copied from the golden 75).

## Run 3: 2026-11-19, prompt set 2026.11.3 (guardrail only)

| technique | accuracy | category acc | JSON 1st try | valid | avg calls | p50 latency s | avg tokens |
|---|---|---|---|---|---|---|---|
| zero_shot | 66.7% | 62.5% | 100.0% | 100.0% | 1.00 | 1.23 | 915 |
| **few_shot** | **95.8%** | **95.8%** | 100.0% | 100.0% | 1.00 | 0.93 | 2340 |

### Findings

- **Few-shot is decisively best: 23/24, all 11 must-refuse questions refused, 100% valid JSON, fastest.** Zero-shot fell back (75% → 67%) because the stricter allow-by-default wording without examples is less stable on an 8B model. Examples are what make this work.
- The only miss is D022 ("Why did Bluefin's auditor flag a material weakness?"), refused as `legal_tax_audit_opinion`. It is borderline: the model's suggested alternative question ("What did the filing disclose about its internal controls?") would still give the user the cited facts through the refusal-with-facts flow. **Not tuned further**, to avoid over-fitting a 24-question dev set.

## Decisions (as of 2026-11-19)

| Node | Technique | Evidence |
|---|---|---|
| pre_guardrail | **few_shot** | Run 3: 95.8%, 11/11 refusals correct |
| classify_and_plan | **few_shot** | Run 2: 100% |
| query_rewrite, extract, analyze, synthesize, verify, repair | not yet tested | Need retrieved evidence: next stage |

Both defaults already match `DEFAULT_TECHNIQUE` in `backend/app/prompts/base.py`.

### Next

1. Extend the dev set with 20-30 new questions (fresh, never copied from the golden 75) and re-run both nodes to confirm the scores hold.
2. Test the evidence-side prompts (extract, synthesize, verify) using hand-picked evidence passages, then end-to-end through the RAG pipeline from `main`.

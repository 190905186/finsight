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

---

# Answer-side prompts (FinanceBench dev split, oracle evidence)

Questions come from `data/evaluation/financebench_dev.jsonl` (99 questions; never the 51 held-out ones or the Acme golden 75). Each prompt receives FinanceBench's annotated evidence passages directly as S1, S2, … so prompt quality is tested apart from retrieval. Harness: `python -m app.prompts.answer_eval --node <node>`.

## FB run 1: 2026-11-19, prompt set 2026.11.3

| Node | Technique | Valid | Key metrics | p50 latency |
|---|---|---|---|---|
| synthesize | zero / few / cot | 100% all | answered 68 / 84 / 84%; citations ok 80 / 72 / 80% | 3.8 / 6.1 / 6.5 s |
| extract | zero / few / cot | 100 / 94 / 89% | gold figure 56 / 65 / 63%; citations ok ~100% | 2.9 / 3.1 / 5.5 s |
| verify | zero / few / cot | 100% all | catches planted error 94 / 100 / 75%; passes correct draft 50 / 33 / 61% | 1.7 / 1.5 / 2.1 s |
| repair | zero | 100% | fixed 50%; planted removed 81% | 1.5 s |
| analyze | zero / few / cot | 100% all | citations ok 100 / 88 / 88%; has findings 100 / 88 / 88% | 3.4 / 2.3 / 6.1 s |

### What the outputs showed (read per question, not just the scores)

**Harness problems (fixed before run 2):**
- The sample was the first 25 questions by ID, and only 1 of them had a numeric answer, so synthesize/analyze "gold figure" (0%) was meaningless.
- Number matching was too loose: rescaling let small ratios (0.83) "match" 830 in the evidence, so calculation questions were wrongly treated as figures printed in the evidence. That caused most of the verify "false alarms" (the checker correctly flagged an uncited computed number), the extract "misses" (the model extracted the right inputs, e.g. Adobe's cash flow and liabilities for a 0.83 ratio) and the repair failures.

**Real prompt problems:**
1. synthesize invented `[C1]` calculation markers with no calculator result supplied, and did the arithmetic itself (rule broken; fails citation check).
2. synthesize declined judgment questions ("Is gross margin a relevant metric for JPM?") as "not disclosed", although the reference answer uses general finance knowledge ("not relevant for a bank").

### Changes for FB run 2 (prompt set 2026.11.4)

- Prompts: `[C#]` only for supplied calc IDs; if a calculation is needed and missing, give the inputs and mark `partially_answered`; answer judgment questions with a conclusion plus cited figures. Shared rule 1 now allows general financial-analysis knowledge to interpret evidence (company facts still evidence-only).
- Harness: balanced sample across question types; strict number matching (no rescaling for figures under 100); calculation questions get a **reference calculator result** (C1 = correct answer), standing in for the `calculate_*` tools that are not built yet; an **AI judge** (same Qwen3-8B, as design §5 allows) grades synthesize answers against FinanceBench's reference answers.

## FB run 2: 2026-11-20, prompt set 2026.11.4 (24-question balanced sample, reference calculator, AI judge)

### synthesize_answer

| technique | valid | answered | citations ok | gold figure | uses calc result | judge: correct | judge: correct+partial | p50 latency s | avg tokens |
|---|---|---|---|---|---|---|---|---|---|
| zero_shot | 100% | 91.7% | 87.5% | 88.9% | 100% | 58.3% | 70.8% | 3.44 | 1927 |
| few_shot | 100% | 95.8% | 87.5% | 100% | 100% | 62.5% | 75.0% | 6.53 | 4433 |
| **cot** | 100% | 95.8% | 87.5% | 100% | 100% | **66.7%** | **75.0%** | 5.58 | **2111** |

### verify_answer (24 questions → 46 drafts: correct + planted error)

| technique | valid | accuracy | catches planted error | passes correct draft | p50 latency s |
|---|---|---|---|---|---|
| zero_shot | 100% | 87.0% | 100% | 75.0% | 1.64 |
| few_shot | 100% | 65.2% | 100% | 33.3% | 1.43 |
| **cot** | 100% | **89.1%** | 95.5% | **83.3%** | 2.11 |

### repair_once (22 planted-error drafts)

| technique | valid | fixed | planted removed | citations ok |
|---|---|---|---|---|
| zero_shot | 100% | 81.8% | 90.9% | 100% |

### extract_financial_fact (11 questions whose answer figure is printed in the evidence)

| technique | valid | gold figure | citations ok | p50 latency s |
|---|---|---|---|---|
| zero_shot | 90.9% | 60.0% | 100% | 3.22 |
| few_shot | 90.9% | 60.0% | 100% | 3.74 |
| cot | 90.9% | 70.0% | 100% | 5.68 |

### analyze_financial_report

| technique | valid | citations ok | gold figure | has findings | p50 latency s |
|---|---|---|---|---|---|
| zero_shot | 100% | 100% | 11.1% | 100% | 3.60 |
| few_shot | 100% | 95.8% | 22.2% | 95.8% | 2.88 |
| **cot** | 100% | 95.8% | **55.6%** | 95.8% | 6.91 |

### Findings

- **Big improvement over run 1**: answered 68-84% → 92-96%; citations ok 72-80% → 87.5%; every calculation question used the supplied calculator result instead of doing arithmetic; verify passes correct drafts 33-61% → 33-83% while still catching 95-100% of planted errors; repair fixed 50% → 82%.
- **Few-shot verify is biased**: its single example is a "fail" case, so it flags 2 of 3 correct drafts. CoT checks claim by claim and is clearly best.
- **Remaining synthesize errors are model reasoning limits**, not format: picking the "Total" row as a segment, missing that a negative value is the lowest, judging 3M "capital-intensive" against the reference. Two answers (8%) still wrote `[C1]` with no calculator result; `render_citations()` reports these as unknown IDs, so the pipeline's deterministic check routes them to `repair_once`.
- **Extract misses are calculation questions** where the model extracted the correct inputs (e.g. EBITDA inputs 8,910 / 2,763 / 5,207); the figure-match metric can't score these. One case (cash conversion cycle, 6+ inputs) returned incomplete JSON for all techniques.

## Decisions (as of 2026-11-20)

| Node | Technique | Evidence |
|---|---|---|
| pre_guardrail | few_shot | Run 3: 95.8%, all must-refuse correct |
| classify_and_plan | few_shot | Run 2: 100% |
| extract_financial_fact | **zero_shot** | FB run 2: tied within noise (6 vs 7 of 10); cheapest, fastest |
| analyze_financial_report | **cot** | FB run 2: gold figure 56% vs 11-22% |
| synthesize_answer | **cot** | FB run 2: best judge score (67% / 75%) at half the tokens of few-shot |
| verify_answer | **cot** | FB run 2: 89% accuracy, 83% passes correct drafts |
| repair_once | zero_shot | FB run 2: 82% fixed |
| query_rewrite | few_shot (untested) | Needs retrieval: end-to-end stage (Docker) |

These are now the defaults in `backend/app/prompts/base.py` (`DEFAULT_TECHNIQUE`).

### Still open

1. `query_rewrite` and the full chain (retrieve → answer) need the RAG database: next stage, Postgres in Docker.
2. Confirm on the 51 held-out FinanceBench questions once, after freezing (`--data ../data/evaluation/financebench_holdout.jsonl --final-run --limit 0`).
3. The Acme golden 75 remain blocked on missing Acme documents (team-notes item 15).

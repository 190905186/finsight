# Golden evaluation set (75 queries)

`golden_75.jsonl` is the 75-query benchmark from *Financial Report RAG Specification and Eval Set* (Part 2), one JSON record per line. **Every record is `split: "held_out"`**. Don't tune prompts on it. `python -m app.prompts.experiment` refuses this file unless you pass `--final-run`.

## Fields

| Field | Source | Meaning |
|---|---|---|
| `question_id`, `sub_type`, `question`, `gold_answer` | spec | Copied from the spec tables |
| `category` | spec | `extraction` (Q1–15), `calculation` (16–30), `multi_hop` (31–45), `refusal` (46–65), `edge_case` (66–75) |
| `expected_behavior` | spec | `answer` or `refuse` (gold answers starting "REFUSAL") |
| `response_type` | derived | `answer`, `refuse`, or `correct` (gold starts "REFUSAL / CORRECTION": Q52, Q54, Q56, Q62) |
| `required_citations` | spec | Citation strings parsed from the gold answer, used for Retrieval Recall@k |
| `key_numbers` | auto-extracted | Numbers in the gold answer (citations, years and page numbers removed). **`key_numbers_reviewed: false`**: a teammate should check these before scoring |
| `expected_decision`, `expected_category` | our labels | What the **pre-retrieval guardrail** should do. Only the 9 questions the copilot must decline whatever the documents say are `refuse` (Q46, 47, 50, 51, 55, 57, 60, 63, 65). The other refusal-category questions (out-of-corpus, undisclosed KPI, false premise) are `allow`, because their gold answers need retrieved facts. |

## Issues found in the spec's gold data

All 57 arithmetic relationships in the gold answers were recomputed. Three need the team's attention:

| Query | Issue | Suggested fix |
|---|---|---|
| **Q74** | "$2.0 billion authorization… following $615 million in buybacks… $1,105 million remained": 2,000 − 615 = **1,385**, not 1,105 (a gap of 280). | Either earlier buybacks used $280M of this authorization (then say so in the gold answer), or correct the figure. |
| **Q45** | Listed drivers 255 + 142 + 190 − 210 = **377**, but the stated gap is 370. | Add "other working-capital items of $(7)M" or adjust a driver. Otherwise a correct model answer looks inconsistent. |
| **Q22** | (2,146 / 14,820) × 365 = **52.85**, the gold shows 52.86. The headline 52.9 is fine. | Change to 52.85, or allow ±0.01 tolerance in numeric checks. |

Corpus coverage is inconsistent. Part 2 lists the reference pack as the FY2024 10-K, the Q1–Q3 FY2025 10-Qs, the Q1–Q3 FY2025 earnings releases, the transcripts and the investor presentations. Several gold answers cite documents outside that list:

- **Q65** cites the *FY24 Proxy Statement*. A proxy isn't in the corpus list, so under refusal rule 3 the copilot should say it's unavailable. Add the proxy to the corpus, or change the gold answer.
- **Q19, Q58** cite the *FY24 Earnings Release*; **Q52** cites the *Q4 FY24 Earnings Release*; **Q24** cites the *Q4 FY24 Investor Presentation*. Make sure these are staged when the corpus is built.
- **Q48** says the corpus covers "FY2022 through Q3 FY2025". This holds only because the FY2024 10-K includes FY2022–FY2023 comparatives.

Minor: Q3 (Q3 FY25 cash $2,415M, marketable securities $610M) and Q19 (FY24 cash $1,805M + marketable securities $610M = $2,415M) produce the same 2,415 and 610 at two different dates. That's possible, but worth checking it isn't a copy error.

## FinanceBench split (for testing the answer-side prompts)

The RAG corpus is FinanceBench (84 public filings in `data/reference/financebench/`). Its 150 open-source questions come with annotated evidence passages and gold answers. They are split once, deterministically, by `backend/app/prompts/build_financebench_split.py`:

| File | Questions | Use |
|---|---|---|
| `financebench_dev.jsonl` | 99 (33 per question type) | Prompt development and tuning |
| `financebench_holdout.jsonl` | 51 (17 per type) | **Held out**: confirm scores once, after prompts are frozen. Includes the question used by the RAG tests (`financebench_id_03029`) |

Both are separate from the 75-query Acme golden set above. FinanceBench questions are about real companies and none overlaps with the Acme questions.

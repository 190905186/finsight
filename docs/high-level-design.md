# Finsight Financial Report RAG Copilot — High-Level Design

**Status:** Target architecture for the financial-report question-answering capstone  
**Primary requirement:** Use one shared hosted OpenAI-compatible chat-completions endpoint backed by a single instruct model with at most 8B parameters, plus the supplied embeddings endpoint. No model-weight changes, logit access, frontier-model fallback, or alternate chat model.  
**Scope:** Evidence-grounded question answering over an explicitly indexed corpus of financial filings, earnings releases, investor presentations, and transcripts.  
**Companion diagrams:** [Target architecture](./architecture.svg) · [Current Ask RAG request flow](./rag-ask-flow.svg)

The target architecture below describes the intended system. The separate
[current Ask RAG flow](./rag-ask-flow.svg) traces the implemented frontend-to-answer
path, which currently uses a direct FastAPI `RAGPipeline` rather than the target
LangGraph multi-agent workflow.

## 1. Goals and boundaries

### Goals

- Answer direct extraction, period comparisons, financial calculations, and multi-document questions using only authorized indexed evidence.
- Provide page/section/period citations for factual statements, show formulas and inputs for derived metrics, and distinguish GAAP from Non-GAAP measures.
- Use a small shared model efficiently: constrain its role to classification, bounded planning, tool selection from an allowlist, and evidence-based synthesis.
- Use LangChain for model/embedding adapters, document loaders and retrievers; use LangGraph for typed, bounded, observable workflow state and conditional routing.
- Detect unsupported, unsafe, ambiguous, or out-of-corpus requests and respond with an explicit refusal, correction, or evidence-gap explanation.
- Evaluate retrieval, deterministic calculations, answer quality, citations, refusal behavior, latency, and hosted-model usage independently.

### Non-goals

- Investment recommendations, price targets, unguided forecasts, live market data, legal/tax/audit opinions, or invented disclosures.
- Arbitrary SQL, autonomous actions, model-generated financial calculations presented as authoritative, or hidden cross-tenant retrieval.
- Fine-tuning, weight changes, logits access, switching to a larger model, or a second model used as a planner/reranker/critic.
- Treating the fictional Acme benchmark answers, synthetic data, or model knowledge as source evidence.

## 2. System context and hosted inference constraint

The application backend calls a single configured OpenAI-compatible chat endpoint. Every LLM-mediated step—intent classification, constrained planning, optional query rewrite, extraction fallback, synthesis, and critic verification—uses that same endpoint and the same configured instruct model (parameter count at or below 8B). Concurrency, token budgets, retries, and per-request call counts are bounded so that agents do not become independent model deployments.

The configured embeddings endpoint is used for both indexing and query embedding. Record the embedding provider/model identifier, vector dimension, and version with the index; a changed embedding model requires a controlled re-embedding and index rebuild. The embeddings service is not a second chat model.

The hosted endpoint is a deliberate trust boundary: only the minimum authorized, redacted query and retrieved evidence needed for the task may be sent. Before deployment, confirm endpoint authentication, transport security, data-retention/training terms, residency, and permitted financial-data handling with the endpoint operator. If organizational policy forbids the necessary data from leaving its boundary, do not send it to the hosted service; fail closed rather than silently switching providers.

**Configuration:**

| Setting | Purpose |
|---|---|
| `LLM_API_BASE` | HTTPS base URL for the OpenAI-compatible chat endpoint |
| `LLM_API_KEY` | Secret supplied through a secret manager/environment, never committed or logged |
| `LLM_CA_CERT` | CA certificate used to verify the chat endpoint TLS certificate |
| `LLM_MODEL` | The approved instruct model identifier; verify provider attestation that it is ≤8B |
| `EMBEDDINGS_API_BASE` | HTTPS base URL for the supplied embeddings endpoint |
| `EMBEDDINGS_API_KEY` | Embeddings credential, independently stored as a secret |
| `EMBEDDINGS_MODEL` | Embedding model identifier/version |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_MAX_CALLS_PER_REQUEST` | Target bounded-call policy; no alternate-model fallback |

Use separate credentials/scopes where the provider supports them. Do not expose keys to the browser. The application must return a clear service-unavailable or partial-evidence response for endpoint errors, timeouts, rate limits, or invalid model output.

### Current chat-completions adapter

The backend now has a reusable synchronous `LLMClient` in
`app/services/llm_client.py`. Its current development configuration uses
`https://103.42.50.41:1034/v1` with model `Qwen/Qwen3-8B`; these values are
environment configuration, not constants in the client. The client sends
`POST {LLM_API_BASE}/chat/completions` with a JSON body containing `model`,
`messages`, and `max_tokens`; `temperature` is optional. When configured, it
sends the API key as a Bearer authorization header and verifies TLS using
`LLM_CA_CERT`. The current client has a 60-second timeout, returns the decoded
JSON object, and propagates transport and non-success HTTP errors. Retries and
per-request call budgets are not yet implemented.

Example wire contract (the credential is a placeholder, never a literal key):

```http
POST /v1/chat/completions
Authorization: Bearer ${LLM_API_KEY}
Content-Type: application/json

{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "Explain RAG in three steps."}
  ],
  "max_tokens": 128
}
```

The client expects an OpenAI-compatible response with the generated text at
`choices[0].message.content`.

Local development loads the ignored `source/financeops-ai/.env` file. The
development CA certificate is `docs/vm04-ca.crt`; Compose mounts it read-only
inside the backend container. Never place the API key in this design document,
source code, or logs.

`app/services/llm_connection_tester.py` provides a one-request embeddings smoke
test using a non-sensitive sample string. Run it from the backend directory
with `python -m app.services.llm_connection_tester`. It checks for a non-empty
numeric vector and reports only its dimension. This verifies connectivity and
basic response shape, not the RAG workflow.

## 3. Logical architecture

| Layer | Responsibility |
|---|---|
| React + TypeScript UI | Submit questions, show answer, period/metric labels, citations, evidence snippets, formulas, refusal/evidence-gap status, and request ID |
| FastAPI `/api/v1` | Authentication, request validation, authorization, rate limits, API schemas, streaming/result transport, and correlation IDs |
| Policy and request gate | Scope/intent validation, supported-question/refusal rules, prompt-injection isolation, sensitive-data minimization, and per-request budgets |
| LangGraph workflow | Typed state, deterministic routing, checkpointing, bounded loops, retry/critic path, and explicit terminal states |
| LangChain adapters | OpenAI-compatible chat client, embeddings client, document loaders, text splitters, vector/keyword retrievers, and optional output parsers |
| Specialist nodes | Financial research, retrieval/evidence assembly, deterministic calculation, synthesis, and verification; logical roles share one model endpoint |
| Tool layer | Read-only corpus lookup, filing metadata lookup, period/metric normalization, and deterministic financial calculations using decimal arithmetic |
| RAG data plane | Original source files, parsed pages/tables, provenance-bearing chunks, embedding vectors, lexical index, and access metadata |
| Evaluation/operations | Golden-set runner, deterministic metric checks, retrieval and answer metrics, sanitized traces, endpoint usage, and audit events |

The LLM is not the source of truth. The indexed source documents and deterministic calculation outputs are authoritative. Tools are typed and allowlisted; retrieved text is untrusted data and cannot grant permissions or modify system policy.

## 4. End-to-end online question flow

1. **Receive and authorize.** The client sends a question, selected company/entity if applicable, and optional fiscal period. FastAPI authenticates the user, resolves corpus/document access, checks request size and rate limits, and assigns request and trace IDs.
2. **Apply deterministic pre-guardrails.** Validate scope and reject disallowed requests before retrieval where possible. Detect attempts to obtain investment advice, unauthorized/internal forecasts, live data, tax/legal/audit opinions, or prompt-policy overrides. Treat user input and all document text as untrusted.
3. **Initialize graph state.** Create a typed, request-scoped state containing the original question, allowed corpus scope, selected entity/period, policy decision, step/call budgets, evidence list, tool results, answer draft, verification status, and trace identifiers. Do not place secrets or unnecessary personal data in prompts or state logs.
4. **Classify and plan.** A constrained supervisor node uses the shared small model only when deterministic rules cannot confidently classify the task. It emits schema-validated intent, entities/periods, required evidence types, and an allowlisted plan. Invalid output gets at most one bounded repair; otherwise return a clarification or explicit failure.
5. **Retrieve evidence.** The retrieval node applies ACL, company, filing type, period, metric, and source filters before content enters model context. It runs dense vector retrieval and lexical/BM25 retrieval in parallel, fuses ranks (for example reciprocal-rank fusion), optionally performs deterministic metadata-aware reranking, removes duplicate/contradictory chunks, and returns a bounded context with exact document, page/section, period, accession/version, units, and source IDs.
6. **Check evidence sufficiency and period basis.** Require evidence to support every requested period, metric, and comparison. Prefer the latest applicable recast/restated filing when it explicitly restates prior periods, but preserve the earlier disclosure and explain the basis difference. If evidence is missing, conflicting, or mixes incompatible bases without a reconciliation bridge, route to clarification/refusal rather than guessing.
7. **Call deterministic tools.** For numerical questions, extract/normalize raw values with source IDs and units, then call a typed calculator. It validates compatible periods, currency, scale, signs, GAAP/Non-GAAP basis, and denominator; uses decimal arithmetic and explicit formulas; returns inputs, outputs, rounding, and provenance. The model may request only registered read-only tools with schema-validated arguments. No model-generated SQL or code execution.
8. **Synthesize with evidence.** The synthesis node receives only the bounded evidence bundle and verified calculation results. It produces a direct answer first, followed by a compact table/bullets as needed. Every factual claim/metric carries `[Doc_Name | Section/Page | Period]`; derived metrics show formula and intermediate inputs. Label GAAP/Non-GAAP and material adjustments.
9. **Verify and guard the response.** A critic node, using the same model endpoint, checks supported claims, citation IDs, period/unit consistency, calculation/tool alignment, required basis labels, and refusal policy. Deterministic validators verify citations resolve to retrieved source IDs and numbers match tool outputs. Allow at most one bounded repair. Unresolved claims are removed or converted to explicit uncertainty; never emit success-shaped unsupported output.
10. **Return and record.** Return a typed response with answer, citations, calculation details, evidence status, refusal status, and request ID. Persist sanitized workflow events and source references under the applicable retention policy. The UI can open the citation to the source page/section.

### Example response contract

```json
{
  "status": "answered",
  "answer": "FY2024 revenue was ... [FY24 10-K | Item 8, p. 54 | FY2024].",
  "claims": [
    {
      "text": "FY2024 revenue was ...",
      "citations": [
        {
          "document_id": "filing-accession-or-corpus-id",
          "document_name": "FY24 10-K",
          "section": "Item 8, Consolidated Statements of Operations",
          "page": 54,
          "period": "FY2024"
        }
      ]
    }
  ],
  "calculations": [],
  "refusal_reason": null,
  "evidence_status": "sufficient",
  "request_id": "correlation-id"
}
```

The identifier and values above are illustrative. A refusal should use `status: "refused"` with a concise reason and, where useful, a safe alternative such as summarizing filed historical metrics.

## 5. LangGraph workflow and multi-agent design

“Multi-agent” is implemented as bounded specialist roles/nodes inside one compiled graph, not as autonomous agents with independent memory, credentials, or model deployments. All model calls use the same configured hosted model. Most numeric work and policy checks are ordinary deterministic code.

### Graph nodes

1. `authorize_and_validate` — deterministic authentication/context checks; no LLM.
2. `pre_guardrail` — supported-scope/refusal policy and prompt-injection boundary.
3. `classify_and_plan` — bounded supervisor; outputs validated `intent`, `company`, `periods`, `metric`, `basis`, and allowed actions.
4. `retrieve_hybrid` — LangChain retrievers with mandatory metadata/ACL filters; parallel lexical and vector search, rank fusion, evidence sufficiency.
5. `analyze_financial_report` — specialist role for direct extraction, cross-period comparisons, restatements, and multi-hop narrative synthesis over retrieved evidence.
6. `calculate_metric` — deterministic tools for growth, margins, FCF, leverage, ratios, bps changes, and specified formulae. Specialist node may request a tool; it does not calculate authoritative numeric results itself.
7. `synthesize_answer` — concise, citation-bearing structured draft; uses the same shared chat endpoint.
8. `verify_answer` — deterministic validators plus same-model critic; checks citations, claim grounding, units, accounting basis, math, and refusal correctness.
9. `repair_once` — optional single bounded correction routed back to synthesis or calculation; cannot broaden access or add new tools.
10. `finalize`, `clarify`, `refuse`, `insufficient_evidence`, `degraded` — explicit terminal nodes.

### State and control rules

- Use a typed graph state and strict structured-output schemas. Persist only state needed to resume; do not persist provider secrets or raw sensitive content in generic traces.
- Route calculations only after the required evidence is present. Route missing periods or incompatible accounting bases to a clarification/refusal terminal.
- Set graph maximum steps, model calls, tool calls, context tokens, wall-clock timeout, and retry count. No unbounded agent loop.
- Tools are read-only and parameterized; no arbitrary SQL, web browsing, email, payments, filings, or external market-data tools.
- Use LangGraph checkpointing for recoverable requests only with encrypted, tenant-scoped state and retention controls. Checkpoint errors must be visible.
- Stream progress/status events if enabled, but mark the answer final/verified only after verification completes.

## 6. RAG ingestion and retrieval design

### Corpus and ingestion

The target corpus contains public-company Form 10-K/10-Q/8-K filings, earnings releases, investor presentations, and earnings-call transcripts. The DOCX's Acme Corp FY2024–Q3 FY2025 reference pack is the source for its 75-query golden set. It is not evidence that these documents are currently present in the repository. Do not answer Acme-specific queries from parametric memory or the gold answers if the corresponding source document has not been ingested.

The implemented FinanceBench path is exposed by `app.rag.cli`: it extracts
page-level text with PyMuPDF, applies recursive character splitting, batches
embeddings through the existing `LLMClient`, and writes versioned profile/model
vectors to PostgreSQL/pgvector. The chunking experiment compares 512/30,
1024/30 (the reference notebook baseline), and 1536/30 character/overlap
profiles. FinanceBench evidence page labels are retained as zero-based indices;
chunks that exceed 500 BGE tokenizer tokens are split further to stay within
the embeddings service's 512-token input limit. The query endpoint displays
one-based PDF pages in citations. Retrieval
evaluation separately reports page-level `recall_at_k` and MRR for shared-corpus
and per-question-document scopes. A distinct answer-evaluation command records
generated answers and citations and reports normalized exact match and token
F1; those text-overlap scores are baselines, not semantic or human judgments.
The query endpoint is `POST /api/v1/rag/query`; generated claims are prompted
to cite source labels and returned labels are validated against retrieved pages.

Ingestion is an operator-controlled, auditable pipeline:

1. Register company, source URL, document type, filing accession/date, reporting period, terms/license, acquisition time, and content hash.
2. Validate file type/size; malware-scan; parse PDF text/tables while retaining page numbers and table/row context. OCR is a separately monitored path for image-only pages.
3. Normalize extracted values without losing original units, signs, labels, fiscal-period dimensions, or table coordinates. Retain the immutable original and parser version.
4. Chunk by filing section/semantic boundary, not arbitrary character count alone. Keep tables intact where practical; attach company, form, period, section, page, accession/version, GAAP basis, currency/unit, and ACL metadata to each chunk.
5. Call the supplied embeddings endpoint in controlled batches and persist model/version/dimension. Build dense and lexical indexes; reject and report malformed records instead of silently dropping them.
6. Run ingestion validation and retrieval tests, then atomically publish the new index version. Reindexing never mutates the source record.

### Query-time retrieval

1. Resolve entity, period, filing type, metric, and accounting basis from the request and graph state.
2. Enforce authorization and metadata filters before vector or lexical retrieval. Never rely on the model prompt for ACL enforcement.
3. Run dense semantic retrieval and lexical/BM25 search; fuse ranks and prefer exact filing-period/table/footnote matches. Use deterministic reranking initially; any future learned reranker must use the same shared ≤8B chat endpoint or a non-generative method, not introduce another LLM.
4. Deduplicate overlapping chunks; detect conflicting values across documents; preserve original and restated disclosures with their time/version.
5. Return a small evidence bundle including exact quotations/snippets, page/section, period, units, source URL/accession, chunk ID, and retrieval scores. Enforce a token budget before hosted inference.
6. If source support is absent or ambiguous, do not fill gaps from model memory. Return missing-evidence status or request a company/period clarification.

## 7. Financial tool catalog

Tools are deterministic, typed, versioned, read-only where they access data, and return source IDs alongside values.

| Tool | Purpose and safeguards |
|---|---|
| `search_filings` | Query the authorized corpus by company/form/period/section; return source references and evidence only |
| `get_filing_metadata` | Resolve accession, filing date, reporting period, document version and page/section map |
| `extract_financial_fact` | Normalize a retrieved fact with original label/value/unit/currency/basis and provenance |
| `calculate_growth` | `(current - prior) / prior`; check same metric, compatible periods and non-zero denominator |
| `calculate_margin` | `numerator / revenue`; require aligned period, entity, currency, and GAAP basis |
| `calculate_fcf` | `operating cash flow - capital expenditures`; show both source inputs and sign conventions |
| `calculate_ratio` | Compute named ratios such as current ratio, leverage, DSO, and interest coverage only from documented definitions and inputs |
| `calculate_bps_delta` | Convert margin difference to basis points; do not confuse percentage change with percentage-point change |
| `reconcile_gaap_nongaap` | Produce bridge only when filing explicitly discloses all adjustments; otherwise mark unreconciled |
| `compare_periods` | Align periods, restatements and discontinued operations; preserve both reported and recast values with explanation |

All currency values use `Decimal` or equivalent fixed-precision arithmetic. Tool outputs include formula, raw inputs, units, rounding policy, period, basis, source IDs, and validation status. The model cannot override tool output or silently change calculation definitions.

## 8. Guardrails and refusal behavior

### Required DOCX policy cases

The copilot must explicitly decline or safely correct:

1. **Investment advice:** No buy/sell/hold/short/allocation recommendation. Offer an objective summary of filed historical results and risks instead.
2. **Unguided prediction/price target:** No stock-price forecast, target, or future financial/macro prediction beyond clearly attributed management guidance in indexed documents.
3. **Out-of-corpus/missing periods/entities:** State that the requested filing, metric, period, subsidiary, or competitor is unavailable; never answer from pretrained memory.
4. **Live/external market data:** No current price, intraday market capitalization, current P/E, or breaking news without an explicitly indexed source and supported scope.
5. **Legal, tax, or audit opinion:** No determination of GAAP/IFRS compliance, audit opinion, or corporate tax structuring. It may summarize what the filing states without adopting an opinion.
6. **Incompatible accounting bases:** Do not calculate a blended GAAP/Non-GAAP metric without an explicit reconciliation bridge. Explain the mismatch and, where supported, provide separate like-for-like figures.

### Additional protections

- **Prompt injection:** System/developer policy and typed tool allowlists are immutable; retrieved text is quoted evidence only. Ignore document instructions asking for secrets, policy changes, or external actions.
- **False premises and undisclosed facts:** Verify the premise against the corpus; correct it only with evidence, otherwise say it is not established.
- **Citation integrity:** A citation must resolve to a source actually returned for this request. Do not generate citation labels from model output alone.
- **Claim coverage:** Require citations for every factual statement and metric; remove unsupported narrative claims.
- **Math and unit safety:** Require tool-produced values for derived quantities; preserve sign, scale, dates, currency, restatement, and accounting basis.
- **Data isolation:** Apply tenant/document authorization before retrieval and before source display; test against cross-scope retrieval leakage.
- **Privacy and endpoint safety:** Redact or exclude secrets and unnecessary PII; send only authorized minimum context to the hosted endpoint. No credentials in browser, prompts, traces, or error messages.
- **No consequential actions:** The service is decision support only. No trades, payments, ledger writes, approvals, or policy overrides.

Refusal is a valid, structured result, not a generic error. Distinguish `refused`, `clarification_required`, `insufficient_evidence`, and `provider_unavailable`.

## 9. Evaluation design

### Golden set

Use all 75 queries from the source DOCX, with immutable IDs and versioned gold references. Preserve the required harness fields:

```json
{
  "question_id": "Q001",
  "category": "extraction",
  "expected_behavior": "answer",
  "required_citations": ["FY24 10-K | Item 8 | p. 54"],
  "key_numbers": [14820]
}
```

The harness record should additionally preserve the original question, expected answer, source page/note IDs, expected intent/tools, rubric, and split. Map the benchmark into the five DOCX categories: Q001–015 `extraction`, Q016–030 `calculation`, Q031–045 `multi_hop`, Q046–065 `refusal`, Q066–075 `edge_case`. Expected behavior is `refuse` for policy/out-of-scope requests and `answer` for supported fact, correction, and safe alternative cases as specified by each gold record. Preserve exact gold semantics when labeling ambiguous correction cases.

The DOCX's Acme pack and values are a benchmark specification, not checked-in primary source evidence. Retrieval tests that assert its citations require the canonical source filings/releases/transcripts to be staged with matching page metadata. If they are absent, mark affected tests as fixture-unavailable; do not count a memorized gold answer as a RAG success.

### Worked golden-set example: Q016 (calculation)

Q016 tests whether the system can retrieve both periods, compute a growth rate deterministically, and cite the source:

**Question:** “Calculate Acme's YoY revenue growth rate in Q3 FY2025.”

An evaluation record can encode the DOCX gold as:

```json
{
  "question_id": "Q016",
  "category": "calculation",
  "expected_behavior": "answer",
  "user_query": "Calculate Acme's YoY revenue growth rate in Q3 FY2025.",
  "required_citations": [
    "Q3 FY25 10-Q | Consolidated Statements of Operations | p. 3"
  ],
  "key_numbers": [4110, 3600, 510, 14.17],
  "expected_calculation": "(4110 - 3600) / 3600 * 100",
  "expected_answer": "Q3 FY2025 revenue grew 14.17% year over year. Calculation: ($4,110M - $3,600M) / $3,600M = $510M / $3,600M = 14.17%.",
  "split": "held_out"
}
```

For this example, the runner executes the normal graph rather than injecting the gold answer: authorize and classify → retrieve Q3 FY2025 and comparable Q3 FY2024 revenue evidence from the indexed 10-Q → assert the required filing/page was retrieved → pass the two period-aligned values and their units/source IDs to `calculate_growth` → verify the tool result and formula → synthesize and verify the cited response. The run passes only if (a) the required citation is retrieved and resolves to the staged source, (b) the deterministic tool inputs are 4,110M and 3,600M with compatible periods/units, (c) the computed result is 14.17% under the benchmark rounding rule, and (d) the final answer shows the formula and cites the evidence. A correct number without source evidence is not a RAG pass; a retrieved citation with an incorrect calculation is not an answer pass. Report retrieval and answer/calculation outcomes separately. If the canonical 10-Q page is not staged, mark Q016 fixture-unavailable rather than scoring a memorized answer.

### Metrics and test layers

| Layer | Measures |
|---|---|
| Ingestion | Parse/page/table coverage, metadata completeness, duplicate handling, source hash and parser/version traceability |
| Retrieval | Recall@k for required citations, MRR, nDCG@k, correct-period retrieval, page/note citation coverage, contradiction/restatement handling |
| Calculation | Exact numeric match/tolerance, formula and intermediate-input correctness, units/signs/rounding, GAAP/Non-GAAP consistency |
| Answer | Factual precision, citation support/precision/recall, answer completeness, structured-output validity, unsupported-claim rate |
| Guardrails | Refusal precision/recall, safe alternative behavior, prompt-injection resistance, false-premise correction, out-of-corpus abstention |
| Operations | Hosted calls/question, input/output tokens, latency percentiles, timeout/rate-limit/error rate, retries, cost/quotas, graph steps |

Run unit tests for tools and policy gates, retrieval tests using labeled source/page IDs, graph integration tests with a deterministic fake OpenAI-compatible server, and end-to-end evaluation against the hosted endpoint. Separate prompt-development and held-out evaluation splits. Tune prompts/chunking/reranking only on development data; freeze and report the 75-query run with model ID, endpoint configuration identifier, prompt version, corpus/index version, and evaluation code revision. Establish numeric pass thresholds from a baseline and product requirements before claiming acceptance; do not tune against the held-out results.

For small-model reliability, prefer explicit schemas, short task prompts, low context duplication, bounded choices, deterministic calculations, and repair-once behavior. Compare prompt/retrieval/tool changes with the same model endpoint and record request/token budgets.

## 10. API, persistence and observability

### Target API

- `POST /api/v1/copilot/chat` — synchronous typed response.
- `POST /api/v1/copilot/stream` — progress events followed by a final verified result or explicit failure.
- `POST /api/v1/documents` and ingestion status endpoints — authorized corpus administration; indexing is not available to ordinary chat users.
- `POST /api/v1/evaluation/runs` and `GET /api/v1/evaluation/runs/{id}` — authorized benchmark execution and results.
- `GET /api/v1/requests/{request_id}/trace` — sanitized, role-protected trace for debugging.

### Persistence

Persist source document metadata and immutable originals separately from parsed pages/chunks. Store vectors with embedding version and provenance. Persist evaluation runs, index/model/prompt versions, user/tenant scope, request IDs, graph-node events, tool inputs/outputs (redacted), citation references, and verification status. Indexes and vectors are rebuildable; original source bytes and source metadata are authoritative. Apply retention and deletion policies to source, cache, checkpoint, and trace data.

### Observability

Trace request → guardrail → planner → retrieval (dense and lexical) → tool → synthesis → verification → response. Record safe metrics for endpoint latency, token usage, rate-limit retries, retrieval scores, citation coverage, tool/math verification, refusal reason codes, graph transitions, and failures. Sanitize/redact before persistence or export; do not store API keys, full unredacted user documents, or unnecessary personal/financial identifiers in traces. Any hosted telemetry requires explicit approval and privacy review; default to approved application telemetry.

## 11. Deployment and failure handling

Deploy the UI and backend in the approved environment; connect outbound only to the configured chat and embeddings endpoints and required approved infrastructure. Keep API keys in a secret manager and enforce TLS, egress allowlists, timeouts, quotas, and endpoint identity verification. Postgres/pgvector and a lexical index may be self-managed or hosted according to data policy; apply tenant scoping in repositories and database policy. No runtime fallback to a local, larger, or different chat model.

| Failure | Required behavior |
|---|---|
| Chat or embeddings endpoint unavailable/rate-limited | Bounded retry only for retry-safe calls; return `provider_unavailable` or a clearly marked partial result |
| Embedding model/version mismatch | Stop query/index mixing and require compatible index or controlled rebuild |
| No relevant evidence | Return `insufficient_evidence`; ask for company/period if resolvable |
| Conflicting or restated values | Explain disclosure dates/bases and cite both; ask for clarification if comparison remains undefined |
| Invalid tool/model output | Schema validation, at most one repair, then explicit degraded/error state |
| Citation does not resolve | Remove claim or fail verification; never fabricate the citation |
| Authorization or policy failure | Deny before retrieval; log a sanitized reason code |

## 12. Implementation status and delivery sequence

The current starter has a FastAPI API surface, sample finance services,
placeholder orchestration/graph state, keyword-only in-memory retrieval, and
evaluation scaffolding. It now includes environment-backed settings, a
reusable OpenAI-compatible chat client, and a live smoke-test utility. The
client is not yet wired into `/api/v1/copilot/chat` or the LangGraph workflow;
that route still uses the demo orchestrator. Retry/call-budget policy, the
complete workflow, production hybrid index, required corpus, reliable citation
validator, and functional 75-query evaluation runner remain incomplete. The
current code therefore does **not** yet implement the hosted financial-report
RAG design end to end.

Recommended delivery:

1. **Contract and corpus:** Define typed answer/evidence/tool schemas; stage legally usable source documents and build provenance/page metadata fixtures for the DOCX golden set.
2. **Hosted adapters:** Wire the existing OpenAI-compatible chat client into the workflow and add the supplied embeddings adapter; complete secret handling, provider/model validation (≤8B), quotas, configurable timeout, retry policy, and no-fallback tests.
3. **Ingestion and RAG:** Implement page/table-aware parsing, chunk metadata, vector and lexical indexes, ACL filtering, rank fusion, and source-resolution tests.
4. **Deterministic tools:** Implement Decimal-based calculations, period/basis checks, restatement handling, and unit tests against gold calculations.
5. **Graph and guardrails:** Build the bounded LangGraph nodes/routes, typed state, checkpoint policy, refusal gates, one-repair verifier path, and injection/access tests.
6. **API/UI and traces:** Return structured answer/citation/calculation/refusal status; expose verified progress and sanitized traces.
7. **Evaluation and release:** Implement all 75 records with documented labels/splits, measure retrieval/calculation/answer/safety/operations separately, establish acceptance gates, and rerun after every prompt/index/model configuration change.

## 13. Architectural decision summary

- **One shared ≤8B instruct model** through one hosted OpenAI-compatible endpoint for every chat/reasoning role.
- **Supplied embeddings endpoint** for indexing and query vectors; versioned independently from chat.
- **LangChain** for model, embedding, loading, splitting, and retrieval integrations.
- **LangGraph** for explicit, typed, bounded multi-agent orchestration and state transitions.
- **Hybrid RAG + deterministic finance tools** rather than relying on model memory or model arithmetic.
- **Hard policy and evidence checks** before and after generation, with citations resolving to indexed sources.
- **75-query DOCX golden set** as an evaluation contract, gated by availability of its corresponding primary-source pack.

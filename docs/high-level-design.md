# Finsight Financial Report RAG Copilot — Current High-Level Design

**Status:** Describes the repository's current implementation, not a production
target-state architecture.
**Companion diagrams:** [Current architecture](./architecture.svg) ·
[Current Ask RAG request flow](./rag-ask-flow.svg) ·
[Capstone specification and evaluation](./capstone-spec-and-evaluation.md)

## 1. Purpose and boundaries

Finsight is a financial-report question-answering prototype. Users can ask
questions about indexed financial filings, request metric reports, inspect
citations and calculation inputs, and continue a short-term conversation.
The system combines page-level filing retrieval, configured chat and
embeddings endpoints, public SEC/Treasury data tools, and deterministic
financial calculations.

The service is decision support. A deterministic pre-workflow gate refuses
recognized trade, payment, account-change, and other consequential-action
requests; there are no tools that execute those actions. Numeric report values
are calculated from validated extraction facts or SEC facts, not authored by
the answer model. If evidence or data is unavailable, the response should
identify the gap rather than invent a value.

This is not a production financial decision system. The authentication helper
is a demo token implementation, conversation/checkpoint/evaluation state is
process-local, and several controls listed as future needs below are not
implemented. The application does not currently enforce a provider-attested
model-size limit. The actual configured model and endpoint are deployment
configuration; an operator must verify they satisfy policy.

## 2. Runtime components

| Component | Current responsibility |
|---|---|
| React + TypeScript frontend | Collects company, fiscal year, prompt strategy, and question; renders workflow status, metrics, citations, and conversation turns. |
| FastAPI backend | Exposes authenticated chat, direct RAG, finance, documents, and evaluation routes. |
| Copilot workflow | Runs a compiled LangGraph state graph with a safety gate before graph/tool/model work, followed by planner, retrieval, public-data, extraction, calculation, synthesis, and citation-verification nodes. |
| Direct RAG pipeline | Answers `POST /api/v1/rag/query` using hybrid retrieval and a cited answer prompt, independently of the LangGraph copilot workflow. |
| Filing index | PostgreSQL with pgvector stores page chunks and vectors. BM25 lexical indexes are constructed in process from indexed chunk text. |
| Public finance MCP service | Offers read-only SEC company-facts and U.S. Treasury debt tools; backend can use direct REST providers when MCP transport is unavailable. |
| Inference adapters | `LLMClient` calls configured OpenAI-compatible chat-completions and embeddings APIs using separate endpoint/model settings and TLS configuration. |
| Docker Compose | Provisions frontend, backend, MCP service, PostgreSQL, and Redis. Redis is provisioned but is not currently used by the application for caching or durable state. |

The repository includes 84 FinanceBench filing PDFs. They are staged input
documents, not proof that the database has been initialized or populated.
Ingestion is an explicit operator command.

## 3. User-facing request flows

### Copilot chat and report

The frontend posts to `POST /api/v1/copilot/chat`. The request may include a
company, fiscal year, conversation ID, and one of three prompt strategies:
`zero_shot`, `few_shot`, or `cot`. The response contains a status, answer,
metrics, citations, data-source statuses, and an agent trace.

Before entering the graph, recognized consequential actions return a refusal
without calling the graph, retrieval, or model. This is a narrow phrase-pattern
guard, not comprehensive financial-policy classification. Within the graph:

1. **`plan`** extracts a year and sets report/retrieval scope.
2. **`rag_researcher`** retrieves evidence, using three targeted searches for
   reports and one search for ordinary questions.
3. **`public_data_researcher`** fetches current/prior SEC annual facts and
   Treasury context for report requests that specify a company and year.
4. **`fact_extractor`** asks the configured model for explicitly quoted facts.
   Values are accepted only when the quoted value and metric label are found
   in the retrieved snippet. Malformed JSON receives one repair attempt;
   certain rate-limit or invalid-output failures are recorded in data status.
5. **`metric_calculator`** combines validated filing facts with SEC facts and
   calculates report metrics using deterministic `Decimal` arithmetic.
6. **`financial_analyst`** either renders report metrics directly or uses the
   configured chat model to answer a non-report question from excerpts and
   validated metrics.
7. **`report_verifier`** checks citation labels against retrieved evidence. A
   failed check routes through one bounded critic retry; if citation checks
   still fail, the workflow raises an error. Graph invocation has a recursion
   limit of 12.

The graph uses LangGraph's in-memory `MemorySaver`. The API separately keeps
conversation display messages in a process-local dictionary. Neither survives
a backend restart or provides state shared across replicas. The `/stream`
route currently returns a completed response object; it is not event
streaming.

### Direct RAG question

The frontend's main report/chat form uses the copilot route. The separate
`POST /api/v1/rag/query` route calls `RAGPipeline` directly. It retrieves
evidence, prompts the model with a zero-shot, few-shot, or private-CoT
strategy, and validates returned citation IDs. It does not use the LangGraph
or SEC/Treasury workflow.

## 4. Retrieval, ingestion, and evidence

### Ingestion

`python -m app.rag.cli ingest --profile medium` reads PDF files from the
configured FinanceBench directory. PyMuPDF extracts text page by page;
recursive character splitters implement 512/30, 1024/30, and 1536/30
character/overlap profiles. Chunks exceeding 500 BGE tokenizer tokens are
split further to respect the embedding input limit. The configured embeddings
client generates vectors, which are stored with profile, model, document,
page, and metadata in PostgreSQL/pgvector.

The current path does not implement OCR, image-based table extraction,
whole-corpus atomic publishing, or the target 500–2,000-document corpus. PDFs
with no extractable text fail ingestion. The current local set is 84 PDFs.

### Query-time retrieval

The current `HybridRetriever`:

1. Loads indexed chunks, optionally filtered by document/company.
2. Computes dense candidates through pgvector and lexical BM25 candidates
   against an in-process index.
3. Combines rankings using reciprocal-rank fusion.
4. Applies a deterministic lexical coverage/phrase reranker.
5. Returns bounded snippets with document, zero-based page index, display
   page, metadata, and retrieval scores.

The answer pipelines use retrieved text as untrusted evidence, ask for exact
citation labels, and reject labels not present in the retrieved result set.
Direct RAG citations use `[S#]`; graph answers use `[R#]`. The backend maps
zero-based stored page indexes to one-based PDF display pages.

## 5. Public and deterministic tools

The MCP service implements two read-only public-data tools:

- `get_sec_financial_facts(company, fiscal_year)` resolves an issuer and
  extracts annual USD facts from SEC Company Facts.
- `get_treasury_debt_context(fiscal_year)` retrieves a U.S. public-debt
  observation from Treasury Fiscal Data.

The SEC tool requires an operator-configured `SEC_USER_AGENT` with a monitored
contact email. If MCP transport fails, the backend invokes the same provider
handlers directly and marks the transport fallback in response status.

`compute_report_metrics` computes revenue growth, margins, current ratio,
liabilities-to-equity, and free cash flow when required inputs are available
and compatible. The calculation layer preserves source information and marks
missing, incompatible, or non-computable inputs. Other invoice, expense, and
vendor-risk tools exist in the repository and API, but the current LangGraph
copilot path does not route user questions to those tools.

## 6. Prompts, safety, and citations

The direct RAG and LangGraph synthesis paths support:

- **Zero-shot:** base instruction and evidence context.
- **Few-shot:** one format example; its values and labels are placeholders.
- **Private CoT (`cot`):** asks the model to reason internally and return only
  the concise answer, not hidden reasoning.

Prompt instructions tell the model to treat source excerpts as untrusted data.
This is a defense-in-depth instruction, not a complete prompt-injection
security boundary. The repository has a basic test verifying that
instruction-like source text remains in the evidence input while the system
prompt labels it untrusted.

Factual filing claims must cite evidence. Citation IDs are validated against
the retrieved sources. The graph verifier retries once for citation mismatch
or missing citations, then fails explicitly. This check verifies citation
membership, not semantic entailment of every claim.

Fifteen consequential-action examples are tested against the pre-workflow
refusal path. PII detection/redaction, enterprise authorization/tenant
isolation, comprehensive refusal tests, and a broader corpus prompt-injection
red-team suite remain unimplemented.

## 7. Evaluation

The source-backed evaluation set comprises 75 FinanceBench questions selected
equally from three benchmark types and 15 refusal cases. Prompt development
uses 15 questions; the remaining 60 are reserved as a holdout. The FinanceBench
source answers and page labels are evaluation references, not source evidence.

Current evaluation capabilities:

- `evaluate_retrieval` calculates page-level recall@k and MRR without answer
  generation.
- `evaluate_answers` separately records generated answers, citations,
  normalized exact match, token-F1, and per-query/mean answer latency.
- `compare-prompts` runs zero-shot, few-shot, and private-CoT comparisons on a
  selected prompt split and writes JSONL results.
- Authenticated `/api/v1/evaluation/run` performs a real retrieval-only or
  answer-quality run; `/results` and `/metrics` return process-local run
  records and latency summaries.

These are functional evaluation scaffolds; live scores have not been asserted
here. Exact-match and token-F1 are lexical baselines, not semantic or human
judgments. The API run history is in memory. Cost/token accounting, end-to-end
task completion and tool-call averages, a failure taxonomy, persistent
evaluation history, and distributed tracing via OpenTelemetry/LangSmith/
Langfuse are not implemented.

## 8. API and deployment

Important endpoints:

| Endpoint | Behavior |
|---|---|
| `POST /api/v1/copilot/chat` | Run graph-based chat/report workflow. |
| `POST /api/v1/copilot/stream` | Return the completed chat response; no incremental event stream yet. |
| `GET /api/v1/copilot/conversations` | List this process's conversations for the authenticated subject. |
| `POST /api/v1/rag/query` | Direct hybrid-RAG answer with validated citations. |
| `POST /api/v1/evaluation/run` | Run actual retrieval-only or answer-quality evaluation. |
| `POST /api/v1/auth/login` | Demo login/token endpoint; not production authentication. |

Docker Compose provisions the React Vite frontend, FastAPI backend, MCP
service, PostgreSQL/pgvector, and Redis. Backend/frontend ports are bound to
localhost. The Redis container currently has no application cache integration.
There is no production rate limiter, response cache, durable checkpoint store,
or multi-replica-safe conversation/evaluation persistence.

## 9. Current implementation gaps

| Requirement area | Current status |
|---|---|
| Written behavior and gold set | Specification, 75 source-labeled questions, 15 refusal cases, and prompt splits exist. |
| Prompt strategy comparison | Three strategies and a CLI comparison runner exist; endpoint-backed comparison scores still need to be recorded. |
| RAG corpus size | Hybrid retrieval and chunk-size experiment exist; 84 staged PDFs are below the 500–2,000-document requirement. |
| Agent loop and state | Planner, executor/research nodes, citation verifier, one retry, and recursion cap exist; state is process-local and tool failure handling is incomplete. |
| Evaluation/observability | Retrieval and answer metrics plus latency exist; cost, task/tool metrics, failure taxonomy, persistent history, and tracing are missing. |
| Ship and responsible AI | React and Compose exist; production rate limiting/caching, durable state, PII controls, hardened auth, and comprehensive injection/refusal tests are missing. |

Do not present the diagram or this design as production readiness. Review model
attestation and endpoint data-handling terms, stage a suitably sized lawful
corpus, implement the outstanding controls, and run evaluation against the
configured services before making conformance claims.

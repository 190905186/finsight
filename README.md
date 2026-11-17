# Finsight Financial Report RAG Copilot

Finsight is a financial-report question-answering copilot designed around evidence-grounded RAG, deterministic financial tools, and a bounded LangGraph workflow. The target capstone deployment uses one hosted OpenAI-compatible chat-completions endpoint backed by a shared instruct model of at most 8B parameters, plus the supplied embeddings endpoint. Model weights are not changed, logits are not accessed, and no larger-model fallback is permitted.

**Implementation status:** this repository is still a starter/demo. Docker Compose reads the chat and embeddings endpoints and model identifiers from `.env`; the hosted-endpoint design below is not yet wired end to end. Do not treat the current runtime as satisfying the hosted inference requirement.

## Repository layout

- `backend/` – FastAPI-based Python service implementing the orchestration, APIs, evaluation hooks, and data layer.
- `frontend/` – React + Vite dashboard for executive oversight and AI copilots.
- `docs/` – [financial-report high-level design](./docs/high-level-design.md), [target architecture SVG](./docs/architecture.svg), and [RAG data ingestion guide](./docs/data-ingestion.md).
- `data/reference/` – locally staged public reference data, source status, licensing notes, and SHA-256 manifest.

## Run frontend and backend with Docker Compose

Docker Compose starts the frontend, backend, PostgreSQL, and Redis. The backend
uses the chat and embeddings endpoints configured in the repository-root
`.env`; the frontend sends `/api` requests through Vite to the backend service.

### 1. Configure `.env`

Create `.env` from the example only if you do not already have one:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Make sure `.env` contains the correct `LLM_API_BASE` and
`EMBEDDINGS_API_BASE` values and the corresponding `LLM_MODEL` and
`EMBEDDINGS_MODEL` identifiers, along with any required API keys and
credentials. Compose mounts the configured CA certificates into the backend
container. Keep real credentials out of source control.

### 2. Build and start the services

Run this from the repository root:

```powershell
docker compose up --build
```

Open the frontend at <http://localhost:5173>. The backend health check is at
<http://localhost:8000/health>, and its interactive API documentation is at
<http://localhost:8000/docs>. Stop the services with **Ctrl+C**; run
`docker compose down` to stop and remove the containers while preserving the
PostgreSQL data volume.

### Reusable chat-completions client

The backend exposes `app.services.llm_client.LLMClient` for OpenAI-compatible
chat-completions APIs. Configure `LLM_API_BASE`, `LLM_MODEL`, and (when
required) `LLM_API_KEY` through the process environment or a local `.env` file.
Set `LLM_CA_CERT` to the CA certificate path when the endpoint uses a private
certificate authority. Keep real API keys out of source control.

For embeddings, configure `EMBEDDINGS_API_BASE`, `EMBEDDINGS_MODEL`,
`EMBEDDINGS_API_KEY`, and `EMBEDDINGS_CA_CERT` in the local `.env` file.
Compose mounts `certs/vm02-ca.crt` into the backend container and configures
that in-container certificate path automatically. `LLMClient.create_embeddings`
uses these embedding settings separately from the chat-completions settings.

```python
from app.services.llm_client import LLMClient

with LLMClient() as llm:
    response = llm.create_chat_completion(
        [{"role": "user", "content": "Explain RAG in three steps."}],
        max_tokens=256,
    )
    answer = response["choices"][0]["message"]["content"]
    embedding_response = llm.create_embeddings("Quarterly revenue increased.")
    vector = embedding_response["data"][0]["embedding"]
```

For the hosted endpoint, set `LLM_API_BASE` to its `/v1` base URL and
`LLM_MODEL` to the deployed model name. The client returns the decoded response
object and raises `httpx` errors for transport or non-success HTTP responses.

To send a non-sensitive smoke-test input to the configured embeddings API and
validate its response, run from `backend/`:

```powershell
python -m app.services.llm_connection_tester
```

This prints the returned vector dimension without printing the vector or API key.

## PDF RAG pipeline

The FinanceBench PDF pipeline follows the reference notebook's page-level
loading and recursive character splitting, using PyMuPDF and the configured
`LLMClient` embeddings API. Its chunking experiment compares page-bounded
512-, 1024-, and 1536-character chunks with 30-character overlap; 1024/30 is
the reference baseline. Chunks exceeding 500 BGE tokenizer tokens are split
further to stay below the embeddings endpoint's 512-token input limit.
The BGE tokenizer is downloaded and cached for local token counting; PDF text
is not sent to its model repository.
PostgreSQL/pgvector supplies dense search, BM25 supplies lexical search,
reciprocal-rank fusion combines results, and a
deterministic reranker orders candidates. FinanceBench gold page labels and
stored page indices are zero-based; API citations display one-based PDF pages.
Scanned PDFs without extractable text fail ingestion; OCR is not silently
substituted.

Start the PostgreSQL service, then run the commands from the repository root.
The host-side CLI needs PostgreSQL at `localhost`; Compose containers use the
`postgres` service name.

```powershell
docker compose up -d postgres
$env:DATABASE_URL = "postgresql+psycopg://financeops:financeops@localhost:5432/financeops"
Set-Location backend
..\.venv\Scripts\python.exe -m app.rag.cli init-db
..\.venv\Scripts\python.exe -m app.rag.cli ingest --profile medium
..\.venv\Scripts\python.exe -m app.rag.cli experiment --mode sharedStore
..\.venv\Scripts\python.exe -m app.rag.cli experiment --mode singleStore
..\.venv\Scripts\python.exe -m app.rag.cli answer-evaluate --profile medium --mode sharedStore --limit 10 --output evaluation\financebench_answers.jsonl
```

`evaluate` and `experiment` measure page-level `recall_at_k` and MRR without
calling the answer generator. `sharedStore` searches all indexed documents;
`singleStore` filters each question to its source document, matching the
notebook's per-document comparison while reusing the pgvector table.
`answer-evaluate` is a separate end-to-end run: it writes each generated answer,
reference answer, and validated citations to JSONL, and reports normalized
exact-match and token-F1 text-overlap baselines. These lexical answer scores
are not a human or semantic judge and should be interpreted alongside
retrieval metrics and answer review. Indexing and both evaluations send
document chunks and/or questions to the configured inference endpoints.
Review their data-handling terms before running. The default 384-dimensional
schema matches the currently configured embedding model; changing models or
dimensions requires a fresh index/schema.

The small golden-query template in
`backend/tests/fixtures/rag_golden_queries.json` keeps retrieval checks
separate from answer-and-citation checks. The default test run validates its
shape and skips live integration tests. To run those tests against the
configured PostgreSQL index and inference APIs:

```powershell
$env:RUN_RAG_GOLDEN_TESTS = "1"
Set-Location backend
..\.venv\Scripts\python.exe -m pytest tests\test_rag_golden_queries.py -q
```

The integration cases use `singleStore` retrieval with known document filters;
they are a small smoke-test set, not a substitute for shared-corpus
FinanceBench evaluation.

After indexing and configuring the chat endpoint, the authenticated
`POST /api/v1/rag/query` endpoint returns an answer and the source/page snippets
it cited. It rejects missing or invalid citation labels.

For Python callers, `RAGPromptResolver` wraps the same pipeline and returns its
answer and citations. Run the example with `backend\` as the working directory
so Python can resolve the `app` package:

```python
from app.services.rag_prompt_resolver import RAGPromptResolver

resolver = RAGPromptResolver()
result = resolver.resolve(
    "What were 3M's FY2018 purchases of property, plant and equipment?",
    doc_name="3M_2018_10K",
    top_k=5,
)
print(result["answer"])
for citation in result["citations"]:
    print(citation["document"], citation["page"], citation["snippet"])
```

For example, start the interactive interpreter from PowerShell with:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe
```

Use `company` to filter by company and `doc_name` to scope to a specific report;
omit both to search the shared corpus. The resolver does not create its own
prompt or retrieval logic; it forwards the natural-language prompt to
`RAGPipeline` and preserves its abstention and citation validation behavior.

## Target design principles

- One shared hosted OpenAI-compatible instruct model (≤8B) for all chat/reasoning roles; no model swapping or fallback.
- The supplied embeddings endpoint for indexing and query embedding, with model/version metadata.
- LangChain adapters and hybrid retrieval, orchestrated as bounded specialist nodes in LangGraph.
- Deterministic financial calculations, evidence-bound citations, and explicit abstention/refusal behavior.
- Endpoint credentials, data handling, residency, retention, and outbound access reviewed before sending financial context.
- Decision support only; no autonomous trading, payments, approvals, or ledger changes.

## Main capabilities

- Financial statement analysis and ratio computation.
- Invoice vs PO vs goods receipt validation.
- Expense policy and reimbursement checks.
- Vendor risk screening.
- Duplicate and anomalous transaction detection.
- Audit and compliance evidence summarization.
- Evaluation dataset and observability scaffolding.

## Security and governance

The starter documents authentication and decision boundaries, but its demo token/authentication code is not production-hardened. The hosted target also requires explicit approval of endpoint data handling and secret management. It does not permit autonomous trade execution, payment release, or account changes.

## Notes

This repository is intentionally a starter and is not production-ready. The Compose prototype still uses local model endpoints, whereas the capstone target requires the approved hosted chat and embeddings endpoints. Replace the prototype inference wiring, complete hybrid ingestion/retrieval and LangGraph orchestration, add hardened identity/tenant isolation, and implement the DOCX's 75-query evaluation before claiming conformance. Stage source documents lawfully and retain their page-level provenance; benchmark gold answers are not source evidence.

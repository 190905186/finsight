# Finsight Financial Report RAG Copilot

Finsight is a financial-report question-answering copilot designed around evidence-grounded RAG, deterministic financial tools, and a bounded LangGraph workflow. The target capstone deployment uses one hosted OpenAI-compatible chat-completions endpoint backed by a shared instruct model of at most 8B parameters, plus the supplied embeddings endpoint. Model weights are not changed, logits are not accessed, and no larger-model fallback is permitted.

**Implementation status:** this repository is still a starter/demo. Its current Docker Compose and default backend settings point to a local Ollama service; the hosted-endpoint design below is not yet wired end to end. Do not treat the current runtime as satisfying the hosted inference requirement.

## Repository layout

- `backend/` – FastAPI-based Python service implementing the orchestration, APIs, evaluation hooks, and data layer.
- `frontend/` – React + Vite dashboard for executive oversight and AI copilots.
- `docs/` – [financial-report high-level design](./docs/high-level-design.md) and [target architecture SVG](./docs/architecture.svg).
- `data/reference/` – locally staged public reference data, source status, licensing notes, and SHA-256 manifest.

## Run locally

Run the backend and frontend in separate terminals from the repository root.

### Prerequisites

- Python 3.11
- Node.js and npm

### 1. Configure the environment

Create `.env` from the example if needed:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

For local model requests, make sure `.env` uses a local Ollama URL (the example
file uses Docker service names):

```dotenv
LLM_API_BASE=http://localhost:11434/v1
EMBEDDINGS_API_BASE=http://localhost:11434/v1
LLM_CA_CERT=
```

Keep real credentials out of source control. The default secret key is for local
development only.

### 2. Start the backend

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Set-Location backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If `.venv` already exists, skip the `py -3.11 -m venv .venv` command.

### 3. Start the frontend

Open a second PowerShell terminal at the repository root:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the frontend at <http://localhost:5173>. The API is at
<http://localhost:8000>, with interactive documentation at
<http://localhost:8000/docs> and a health check at
<http://localhost:8000/health>.

The backend can start without Ollama, but LLM-backed features need a local
Ollama instance and the configured models (`qwen2.5:3b` and `nomic-embed-text`).

### Reusable chat-completions client

The backend exposes `app.services.llm_client.LLMClient` for OpenAI-compatible
chat-completions APIs. Configure `LLM_API_BASE`, `LLM_MODEL`, and (when
required) `LLM_API_KEY` through the process environment or a local `.env` file.
Set `LLM_CA_CERT` to the CA certificate path when the endpoint uses a private
certificate authority. Keep real API keys out of source control.

```python
from app.services.llm_client import LLMClient

with LLMClient() as llm:
    response = llm.create_chat_completion(
        [{"role": "user", "content": "Explain RAG in three steps."}],
        max_tokens=256,
    )
    answer = response["choices"][0]["message"]["content"]
```

For the hosted endpoint, set `LLM_API_BASE` to its `/v1` base URL and
`LLM_MODEL` to the deployed model name. The client returns the decoded response
object and raises `httpx` errors for transport or non-success HTTP responses.

To make one real request locally using `.env` and check that answer text is
returned, run from `backend/` after starting Ollama and downloading the model:

```powershell
python -m app.services.llm_connection_tester
```

This sends the default test question to the configured LLM endpoint.

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

# FinanceBench RAG data ingestion

This guide describes the **implemented PDF ingestion path** in
`backend/app/rag`. It extracts page text, splits it into chunks, embeds those
chunks through the configured embeddings API, and stores the text, metadata, and
vectors in PostgreSQL with pgvector. It does not describe the broader target
architecture's OCR, table extraction, or atomic whole-corpus publishing; those
are not implemented in this path.

## Data flow

```mermaid
flowchart LR
    A["FinanceBench PDFs + metadata JSONL"] --> B["PyMuPDF: extract page text"]
    B --> C["Clean and split each page"]
    C --> D["Batch chunk text"]
    D --> E["POST {EMBEDDINGS_API_BASE}/embeddings"]
    E --> F["Validate response count, order, and dimensions"]
    F --> G["PostgreSQL + pgvector<br/>rag_document_chunks"]
    H[".env: model, API URL/key, dimensions,<br/>batch size, DATABASE_URL"] --> E
    H --> G
```

## 1. Inputs and configuration

By default, the CLI reads PDFs from
`data/reference/financebench/pdfs/` and document metadata from
`data/reference/financebench/data/financebench_document_information.jsonl`.
Metadata rows are matched to PDFs by `doc_name`, which is the PDF filename
without its extension. The metadata row is stored with each chunk as JSONB.

The CLI loads settings from the repository `.env`. Set these values there
before ingestion:

| Setting | Used for |
| --- | --- |
| `EMBEDDINGS_API_BASE` | Base URL; the client calls `{base}/embeddings` |
| `EMBEDDINGS_API_KEY` | Optional bearer credential for the embeddings service |
| `EMBEDDINGS_CA_CERT` | CA certificate used to verify the embeddings service |
| `EMBEDDINGS_MODEL` | Model identifier sent with each embeddings request |
| `EMBEDDINGS_DIMENSIONS` | Required vector size; must match the service and database schema |
| `RAG_EMBEDDING_BATCH_SIZE` | Number of chunks sent in each request (default `16`) |
| `DATABASE_URL` | PostgreSQL connection string |

Embedding credentials and document text stay on the backend/CLI side; the
frontend does not call the embeddings API. Review the endpoint's data-handling
terms before sending document contents to it. Do not commit `.env` or expose
API keys in logs or source control.

## 2. Initialize PostgreSQL and pgvector

The CLI's `init-db` command enables the PostgreSQL `vector` extension and
creates the RAG table and indexes. With the provided Compose database, run the
CLI on the host and override `DATABASE_URL` to use the published localhost
port. The API URLs, model, and credentials continue to come from `.env`.

From the repository root in PowerShell:

```powershell
docker compose up -d postgres

# Create the virtual environment once, if needed.
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

# The host CLI uses localhost for PostgreSQL and local certificate paths.
$env:DATABASE_URL = "postgresql+psycopg://financeops:financeops@127.0.0.1:5432/financeops"
$env:LLM_CA_CERT = (Resolve-Path .\certs\vm04-ca.crt).Path
$env:EMBEDDINGS_CA_CERT = (Resolve-Path .\certs\vm02-ca.crt).Path

Set-Location backend
..\.venv\Scripts\python.exe -m app.rag.cli init-db
```

If the virtual environment and dependencies are already installed, skip those
setup commands. If PostgreSQL credentials or the host port differ from the
Compose defaults, adjust the `DATABASE_URL` override accordingly.

## 3. Extract and chunk PDFs

Run ingestion from the `backend` directory:

```powershell
..\.venv\Scripts\python.exe -m app.rag.cli ingest --profile medium
```

The CLI processes `*.pdf` files in sorted order. For each PDF, it:

1. Computes a SHA-256 hash of the source file.
2. Uses PyMuPDF to extract text page by page. Encrypted PDFs that cannot be
   opened and PDFs with no extractable text fail; OCR is not automatically
   applied.
3. Removes NUL characters and splits each page independently, so chunks do
   not span page boundaries.
4. Applies one of the configured character profiles:

   | Profile | Target size | Overlap |
   | --- | ---: | ---: |
   | `small` | 512 characters | 30 characters |
   | `medium` | 1,024 characters | 30 characters |
   | `large` | 1,536 characters | 30 characters |

5. Splits any chunk that exceeds 500 BGE tokenizer tokens into smaller pieces,
   with 30-token overlap, to stay within the configured embedding input limit.
6. Assigns each chunk a stable SHA-256 ID derived from the source hash,
   profile, page, chunk index, and chunk text.

The stored page index is zero-based. The query API converts it to a one-based
PDF page number when displaying citations.

## 4. Call the embeddings API

`PDFIngestor` passes chunk text to `embed_texts` in batches of
`RAG_EMBEDDING_BATCH_SIZE`. Each request is an OpenAI-compatible
`POST {EMBEDDINGS_API_BASE}/embeddings` containing the configured model and
batch input. The reusable `LLMClient` applies the configured API key, CA
certificate, and request timeout.

Before storing vectors, the client validates that:

- The response contains one embedding for every input chunk.
- Returned item indices, when present, are complete and in the expected range.
- Every embedding is numeric and has exactly `EMBEDDINGS_DIMENSIONS` values.

Malformed or incomplete responses raise an error; they are not silently
truncated or replaced with another embedding provider.

## 5. Store chunks and vectors

`PgVectorStore` stores each chunk in `rag_document_chunks` with:

- `chunk_id`, document name, source filename, page number, and chunk index;
- the chunk text and metadata JSONB;
- chunking profile and embedding model identifier; and
- the embedding vector.

The table has a pgvector HNSW index configured for cosine distance, plus
indexes for document name and profile/model. The vector column's dimension is
set from `EMBEDDINGS_DIMENSIONS` when the schema is created. Keep the configured
dimension consistent with the embeddings service and existing database schema;
changing it requires a compatible schema/index rebuild and re-ingestion.

Writes replace one document/profile/model at a time in a database transaction.
Re-ingesting a document replaces its prior chunks for that profile/model; the
directory as a whole is not published atomically. The command prints a JSON
summary with the profile, number of indexed documents, and total chunks.

## 6. Verify and use the index

After ingestion, the query pipeline embeds each question with the same
configured embedding model, searches matching vectors in pgvector, and combines
those candidates with BM25 lexical retrieval over stored chunk text. The
embedding model and vector dimensions used at query time must match the
ingested index. The vector database stores the searchable chunks and metadata;
the original PDFs remain in the source directory.

For alternate profiles or a non-default PDF directory:

```powershell
..\.venv\Scripts\python.exe -m app.rag.cli ingest --profile small
..\.venv\Scripts\python.exe -m app.rag.cli ingest --profile large --pdf-dir ..\path\to\pdfs
```

Run these commands from `backend/`. A failure on one PDF can occur after
earlier documents have already been committed; check the command output and
database contents before retrying.

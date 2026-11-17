from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "FinanceOps AI")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    environment: str = os.getenv("APP_ENV", "development")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai-compatible")
    llm_api_base: str = os.getenv("LLM_API_BASE", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_ca_cert: str | None = os.getenv("LLM_CA_CERT") or None
    llm_model: str = os.getenv("LLM_MODEL", "")
    embeddings_api_base: str = os.getenv("EMBEDDINGS_API_BASE", "")
    embeddings_api_key: str = os.getenv("EMBEDDINGS_API_KEY", "")
    embeddings_ca_cert: str | None = os.getenv("EMBEDDINGS_CA_CERT") or None
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "")
    embeddings_dimensions: int = int(os.getenv("EMBEDDINGS_DIMENSIONS", "768"))
    rag_embedding_batch_size: int = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "16"))
    financebench_pdf_dir: Path = Path(
        os.getenv(
            "FINANCEBENCH_PDF_DIR",
            str(REPOSITORY_ROOT / "data" / "reference" / "financebench" / "pdfs"),
        )
    )
    financebench_metadata_path: Path = Path(
        os.getenv(
            "FINANCEBENCH_METADATA_PATH",
            str(
                REPOSITORY_ROOT
                / "data"
                / "reference"
                / "financebench"
                / "data"
                / "financebench_document_information.jsonl"
            ),
        )
    )
    financebench_evaluation_path: Path = Path(
        os.getenv(
            "FINANCEBENCH_EVALUATION_PATH",
            str(
                REPOSITORY_ROOT
                / "data"
                / "reference"
                / "financebench"
                / "data"
                / "financebench_open_source.jsonl"
            ),
        )
    )
    postgres_dsn: str = os.getenv("DATABASE_URL", "postgresql+psycopg://financeops:financeops@localhost:5432/financeops")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    openapi_url: str = os.getenv("OPENAPI_URL", "/openapi.json")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()

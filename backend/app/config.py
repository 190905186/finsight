from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


@dataclass(frozen=True)
class Settings:
    app_name: str = _env("APP_NAME", "Finsight Copilot")
    app_version: str = _env("APP_VERSION", "0.1.0")
    environment: str = _env("APP_ENV", "development")
    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "8000"))
    secret_key: str = _env("SECRET_KEY", "dev-secret-key")
    jwt_algorithm: str = _env("JWT_ALGORITHM", "HS256")
    llm_provider: str = _env("LLM_PROVIDER", "openai-compatible")
    llm_api_base: str = _env("LLM_API_BASE", "https://103.42.50.41:1034/v1")
    llm_api_key: str = _env("LLM_API_KEY", "")
    llm_ca_cert: str | None = _env("LLM_CA_CERT") or None
    llm_model: str = _env("LLM_MODEL", "Qwen/Qwen3-8B")
    embeddings_api_base: str = _env("EMBEDDINGS_API_BASE", "https://103.42.50.41:1035/v1")
    embeddings_api_key: str = _env("EMBEDDINGS_API_KEY", "")
    # Prefer EMBEDDINGS_CA_CERT; accept legacy EMBEDDINGS_CERT_KEY alias.
    embeddings_ca_cert: str | None = (
        _env("EMBEDDINGS_CA_CERT") or _env("EMBEDDINGS_CERT_KEY") or None
    )
    embeddings_model: str = _env("EMBEDDINGS_MODEL", "bge-small-en-v1.5")
    embeddings_dim: int = int(_env("EMBEDDINGS_DIM", "384"))
    postgres_dsn: str = _env(
        "DATABASE_URL",
        "postgresql+psycopg://financeops:financeops@localhost:5432/financeops",
    )
    redis_url: str = _env("REDIS_URL", "redis://localhost:6379/0")
    financebench_root: str = _env(
        "FINANCEBENCH_ROOT",
        str(_REPO_ROOT / "data" / "reference" / "financebench"),
    )
    chunk_token_budget: int = int(_env("CHUNK_TOKEN_BUDGET", "900"))
    max_retrieval_chunks: int = int(_env("MAX_RETRIEVAL_CHUNKS", "8"))
    index_version: str = _env("INDEX_VERSION", "v3")
    rate_limit_per_minute: int = int(_env("RATE_LIMIT_PER_MINUTE", "60"))
    use_llm: bool = _env("USE_LLM", "true").lower() == "true"
    openapi_url: str = _env("OPENAPI_URL", "/openapi.json")
    debug: bool = _env("DEBUG", "false").lower() == "true"


settings = Settings()

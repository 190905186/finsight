from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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
    llm_api_base: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_ca_cert: str | None = os.getenv("LLM_CA_CERT") or None
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    embeddings_api_base: str = os.getenv("EMBEDDINGS_API_BASE", "http://localhost:11434/v1")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
    postgres_dsn: str = os.getenv("DATABASE_URL", "postgresql+psycopg://financeops:financeops@localhost:5432/financeops")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    openapi_url: str = os.getenv("OPENAPI_URL", "/openapi.json")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()

from __future__ import annotations

from typing import Sequence

import httpx

from app.config import settings
from app.core.tls import resolve_ca_cert


class EmbeddingsClient:
    """OpenAI-compatible embeddings client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        ca_cert: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.embeddings_api_base).rstrip("/")
        self.api_key = settings.embeddings_api_key if api_key is None else api_key
        self.model = model or settings.embeddings_model
        verify = resolve_ca_cert(
            ca_cert if ca_cert is not None else settings.embeddings_ca_cert or settings.llm_ca_cert
        )
        self._client = httpx.Client(
            verify=verify,
            timeout=timeout,
            transport=transport,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self._client.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": self.model, "input": list(texts)},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [list(item["embedding"]) for item in ordered]
        if vectors:
            dim = len(vectors[0])
            if dim != settings.embeddings_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {dim}, expected {settings.embeddings_dim}"
                )
        return vectors

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EmbeddingsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import httpx

from app.config import settings


class LLMClient:
    """Client for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        ca_cert: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        embeddings_base_url: str | None = None,
        embeddings_api_key: str | None = None,
        embeddings_model: str | None = None,
        embeddings_ca_cert: str | None = None,
        embeddings_transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (
            settings.llm_api_base if base_url is None else base_url
        ).strip().rstrip("/")
        self.api_key = settings.llm_api_key if api_key is None else api_key
        self.model = settings.llm_model if model is None else model
        self._client = httpx.Client(
            verify=ca_cert or settings.llm_ca_cert or True,
            timeout=timeout,
            transport=transport,
        )
        self.embeddings_base_url = (
            settings.embeddings_api_base
            if embeddings_base_url is None
            else embeddings_base_url
        ).strip().rstrip("/")
        self.embeddings_api_key = (
            settings.embeddings_api_key
            if embeddings_api_key is None
            else embeddings_api_key
        )
        self.embeddings_model = (
            settings.embeddings_model if embeddings_model is None else embeddings_model
        )
        self._embeddings_client = httpx.Client(
            verify=embeddings_ca_cert or settings.embeddings_ca_cert or True,
            timeout=timeout,
            transport=embeddings_transport,
        )

    def create_chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        """Send a chat-completions request and return the decoded response."""
        if not self.base_url:
            raise ValueError("LLM_API_BASE must be configured")
        if not self.model:
            raise ValueError("LLM_MODEL must be configured")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        print("LLMClient.create_chat_completion result:", result)
        if not isinstance(result, dict):
            raise ValueError("Chat-completions response must be a JSON object")
        return result

    def create_embeddings(
        self,
        input: str | Sequence[str],
        model: str | None = None,
    ) -> Dict[str, Any]:
        """Send an OpenAI-compatible embeddings request."""
        if not self.embeddings_base_url:
            raise ValueError("EMBEDDINGS_API_BASE must be configured")
        if not self.embeddings_model:
            raise ValueError("EMBEDDINGS_MODEL must be configured")

        headers = {"Content-Type": "application/json"}
        if self.embeddings_api_key:
            headers["Authorization"] = f"Bearer {self.embeddings_api_key}"

        response = self._embeddings_client.post(
            f"{self.embeddings_base_url}/embeddings",
            headers=headers,
            json={"model": model or self.embeddings_model, "input": input},
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("Embeddings response must be a JSON object")
        return result

    def close(self) -> None:
        self._client.close()
        self._embeddings_client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

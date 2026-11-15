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
    ) -> None:
        self.base_url = (base_url or settings.llm_api_base).rstrip("/")
        self.api_key = settings.llm_api_key if api_key is None else api_key
        self.model = model or settings.llm_model
        self._client = httpx.Client(
            verify=ca_cert or settings.llm_ca_cert or True,
            timeout=timeout,
            transport=transport,
        )

    def create_chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        """Send a chat-completions request and return the decoded response."""
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
        if not isinstance(result, dict):
            raise ValueError("Chat-completions response must be a JSON object")
        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

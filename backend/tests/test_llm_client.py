from __future__ import annotations

import json

import httpx
import pytest

from app.services.llm_client import LLMClient


def test_chat_completion_sends_openai_compatible_request() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "RAG answer"}}]})

    client = LLMClient(
        base_url="https://llm.example/v1/",
        api_key="test-key",
        model="example-model",
        transport=httpx.MockTransport(handler),
    )

    try:
        result = client.create_chat_completion(
            [{"role": "user", "content": "Explain RAG."}],
            max_tokens=128,
        )
    finally:
        client.close()

    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"] == {
        "model": "example-model",
        "messages": [{"role": "user", "content": "Explain RAG."}],
        "max_tokens": 128,
    }
    assert result["choices"][0]["message"]["content"] == "RAG answer"


def test_chat_completion_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = LLMClient(
        base_url="https://llm.example/v1",
        api_key="invalid-test-key",
        transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.create_chat_completion([{"role": "user", "content": "hello"}])
    finally:
        client.close()


def test_chat_completion_omits_auth_when_key_is_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"choices": []})

    client = LLMClient(
        base_url="http://localhost:11434/v1",
        api_key="",
        transport=httpx.MockTransport(handler),
    )

    try:
        assert client.create_chat_completion([{"role": "user", "content": "hello"}]) == {
            "choices": []
        }
    finally:
        client.close()

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
        model="example-model",
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
        base_url="https://llm.example/v1",
        api_key="",
        model="example-model",
        transport=httpx.MockTransport(handler),
    )

    try:
        assert client.create_chat_completion([{"role": "user", "content": "hello"}]) == {
            "choices": []
        }
    finally:
        client.close()


def test_embeddings_sends_openai_compatible_request() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2], "index": 0}], "model": "bge-small-en-v1.5"},
        )

    client = LLMClient(
        embeddings_base_url="https://embeddings.example/v1/",
        embeddings_api_key="test-embedding-key",
        embeddings_model="bge-small-en-v1.5",
        embeddings_transport=httpx.MockTransport(handler),
    )

    try:
        result = client.create_embeddings(["first text", "second text"])
    finally:
        client.close()

    assert captured["url"] == "https://embeddings.example/v1/embeddings"
    assert captured["authorization"] == "Bearer test-embedding-key"
    assert captured["body"] == {
        "model": "bge-small-en-v1.5",
        "input": ["first text", "second text"],
    }
    assert result["data"][0]["embedding"] == [0.1, 0.2]


def test_embeddings_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = LLMClient(
        embeddings_base_url="https://embeddings.example/v1",
        embeddings_api_key="invalid-test-key",
        embeddings_model="example-embedding-model",
        embeddings_transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.create_embeddings("hello")
    finally:
        client.close()


def test_chat_completion_requires_endpoint() -> None:
    client = LLMClient(
        base_url="",
        embeddings_base_url="https://embeddings.example/v1",
        model="example-model",
        embeddings_model="example-embedding-model",
    )
    try:
        with pytest.raises(ValueError, match="LLM_API_BASE must be configured"):
            client.create_chat_completion([{"role": "user", "content": "hello"}])
    finally:
        client.close()


def test_chat_completion_requires_model() -> None:
    client = LLMClient(
        base_url="https://llm.example/v1",
        embeddings_base_url="https://embeddings.example/v1",
        model="",
        embeddings_model="example-embedding-model",
    )
    try:
        with pytest.raises(ValueError, match="LLM_MODEL must be configured"):
            client.create_chat_completion([{"role": "user", "content": "hello"}])
    finally:
        client.close()


def test_embeddings_requires_endpoint() -> None:
    client = LLMClient(
        base_url="https://llm.example/v1",
        embeddings_base_url="",
        model="example-model",
        embeddings_model="example-embedding-model",
    )
    try:
        with pytest.raises(ValueError, match="EMBEDDINGS_API_BASE must be configured"):
            client.create_embeddings("hello")
    finally:
        client.close()


def test_embeddings_requires_model() -> None:
    client = LLMClient(
        base_url="https://llm.example/v1",
        embeddings_base_url="https://embeddings.example/v1",
        model="example-model",
        embeddings_model="",
    )
    try:
        with pytest.raises(ValueError, match="EMBEDDINGS_MODEL must be configured"):
            client.create_embeddings("hello")
    finally:
        client.close()

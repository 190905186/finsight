from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.llm_client import LLMClient


class LLMChatCompletionTester:
    """Make one real chat-completions request and return its answer."""

    def run(self, question: str = "Explain RAG in three steps.") -> str:
        with LLMClient() as client:
            response: Dict[str, Any] = client.create_chat_completion(
                [{"role": "user", "content": question}],
                max_tokens=128,
            )

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Chat-completions response contains no choices")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("Chat-completions response contains no message")

        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Chat-completions response contains no answer text")
        return answer


class LLMEmbeddingsTester:
    """Make one embeddings request and return its vector dimension."""

    def run(self, text: str = "Connection test.") -> int:
        with LLMClient() as client:
            response: Dict[str, Any] = client.create_embeddings(text)

        data = response.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError("Embeddings response contains no data")

        embedding = data[0].get("embedding")
        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(isinstance(value, (int, float)) for value in embedding)
        ):
            raise ValueError("Embeddings response contains no valid vector")
        return len(embedding)


if __name__ == "__main__":
    dimension = LLMEmbeddingsTester().run()
    print(f"Embeddings API connection succeeded (vector dimension: {dimension}).")

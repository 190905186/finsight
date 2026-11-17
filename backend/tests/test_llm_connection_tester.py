import pytest

from app.services import llm_connection_tester


def test_tester_returns_answer_from_chat_completion(monkeypatch) -> None:
    class FakeLLMClient:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def create_chat_completion(self, messages, max_tokens):
            assert messages == [{"role": "user", "content": "Test question"}]
            assert max_tokens == 128
            return {"choices": [{"message": {"content": "Test answer"}}]}

    monkeypatch.setattr(llm_connection_tester, "LLMClient", FakeLLMClient)

    assert llm_connection_tester.LLMChatCompletionTester().run("Test question") == "Test answer"


def test_embeddings_tester_returns_vector_dimension(monkeypatch) -> None:
    class FakeLLMClient:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def create_embeddings(self, text):
            assert text == "Connection test."
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    monkeypatch.setattr(llm_connection_tester, "LLMClient", FakeLLMClient)

    assert llm_connection_tester.LLMEmbeddingsTester().run() == 3


def test_embeddings_tester_rejects_missing_vector(monkeypatch) -> None:
    class FakeLLMClient:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def create_embeddings(self, text):
            return {"data": [{"embedding": []}]}

    monkeypatch.setattr(llm_connection_tester, "LLMClient", FakeLLMClient)

    with pytest.raises(ValueError, match="Embeddings response contains no valid vector"):
        llm_connection_tester.LLMEmbeddingsTester().run()

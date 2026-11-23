import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.rag import RAGQueryRequest

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invoice_validation() -> None:
    response = client.post(
        "/api/v1/invoices/validate",
        json={
            "invoice": {
                "invoice_id": "INV-1001",
                "vendor": "ABC Technologies",
                "invoice_date": "2025-07-12",
                "total": 15420.0,
                "tax": 1233.6,
                "po_number": "PO-4401",
                "currency": "USD",
            },
            "po_number": "PO-4401",
            "goods_receipt_number": "GR-1101",
            "tolerance": 0.05,
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["invoice_id"] == "INV-1001"
    assert payload["recommendation"] in {"approve", "manual_review"}


def test_auth_login() -> None:
    response = client.post("/api/v1/auth/login")
    assert response.status_code == 200
    assert "token" in response.json()


def test_rag_query_request_normalizes_query_and_company() -> None:
    request = RAGQueryRequest(
        query="  What was revenue?  ",
        company=" 3M ",
    )

    assert request.query == "What was revenue?"
    assert request.company == "3M"
    assert request.top_k == 5


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "Revenue?", "top_k": 0},
        {"query": "Revenue?", "top_k": 21},
        {"query": "Revenue?", "company": "   "},
        {"query": "Revenue?", "company": "x" * 256},
        {"query": "x" * 4001},
    ],
)
def test_rag_query_request_rejects_invalid_input(payload: dict) -> None:
    with pytest.raises(ValidationError):
        RAGQueryRequest(**payload)


def test_copilot_chat_uses_typed_request_and_scopes_conversations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.routes import copilot

    def fake_workflow(
        query: str,
        *,
        conversation_id: str,
        user_id: str,
        company: str | None = None,
        fiscal_year: int | None = None,
        prompting_strategy: str = "zero_shot",
    ) -> dict:
        return {
            "conversation_id": conversation_id,
            "prompting_strategy": prompting_strategy,
            "status": "ready",
            "answer": f"Report for {company} FY{fiscal_year}: {query}",
            "company": company,
            "fiscal_year": fiscal_year,
            "metrics": [],
            "macro_context": None,
            "citations": [],
            "data_status": {},
            "agent_trace": ["plan"],
        }

    monkeypatch.setattr(copilot, "run_financial_workflow", fake_workflow)
    conversation_id = "test-conversation"
    response = client.post(
        "/api/v1/copilot/chat",
        json={
            "message": "Summarize this filing",
            "conversation_id": conversation_id,
            "company": " 3M ",
            "fiscal_year": 2018,
            "prompting_strategy": "few_shot",
            "user_id": "untrusted-client-user",
        },
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == conversation_id
    assert response.json()["company"] == "3M"
    assert response.json()["fiscal_year"] == 2018
    assert response.json()["prompting_strategy"] == "few_shot"

    details = client.get(
        f"/api/v1/copilot/conversations/{conversation_id}",
        headers={"Authorization": "Bearer test-token"},
    )
    assert details.status_code == 200
    assert details.json()["messages"][-1]["role"] == "assistant"

    deleted = client.delete(
        f"/api/v1/copilot/conversations/{conversation_id}",
        headers={"Authorization": "Bearer test-token"},
    )
    assert deleted.status_code == 200
    assert client.get(
        f"/api/v1/copilot/conversations/{conversation_id}",
        headers={"Authorization": "Bearer test-token"},
    ).status_code == 404


def test_copilot_chat_rejects_unknown_prompting_strategy() -> None:
    response = client.post(
        "/api/v1/copilot/chat",
        json={
            "message": "Summarize this filing",
            "prompting_strategy": "unsupported",
        },
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 422


def test_evaluation_endpoints_report_actual_run_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.routes import evaluation
    from app.rag.evaluation import RetrievalQuestion

    question = RetrievalQuestion(
        question_id="gold-1",
        query="What was revenue?",
        gold_evidence=frozenset({("Example_2024_10K", 1)}),
        doc_name="Example_2024_10K",
        reference_answer="$120 million",
    )
    calls: dict[str, object] = {}

    class FakeStore:
        def close(self) -> None:
            calls["closed"] = True

    class FakeClient:
        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    class FakeRetriever:
        def __init__(self, store, client, *, profile: str) -> None:
            calls["profile"] = profile

    def fake_evaluate_retrieval(questions, retriever, client, **kwargs):
        calls["question_count"] = len(questions)
        calls["retrieval_mode"] = kwargs["retrieval_mode"]
        return {
            "evaluation_scope": "retrieval_only",
            "questions_scored": len(questions),
            "recall_at_k": {"5": 0.75},
            "mrr": 0.5,
        }

    monkeypatch.setattr(evaluation, "load_golden_questions", lambda: [question])
    monkeypatch.setattr(evaluation, "PgVectorStore", FakeStore)
    monkeypatch.setattr(evaluation, "LLMClient", FakeClient)
    monkeypatch.setattr(evaluation, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(evaluation, "evaluate_retrieval", fake_evaluate_retrieval)

    response = client.post(
        "/api/v1/evaluation/run",
        json={"evaluation_scope": "retrieval_only", "top_k": 5},
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["recall_at_k"] == {"5": 0.75}
    assert "accuracy" not in payload["metrics"]
    assert calls == {
        "profile": "medium",
        "question_count": 1,
        "retrieval_mode": "sharedStore",
        "closed": True,
    }

    results = client.get(
        "/api/v1/evaluation/results",
        headers={"Authorization": "Bearer test-token"},
    )
    metrics = client.get(
        "/api/v1/evaluation/metrics",
        headers={"Authorization": "Bearer test-token"},
    )
    assert results.json()["items"][0]["id"] == payload["id"]
    assert metrics.json()["runs_completed"] >= 1
    assert metrics.json()["latest_metrics"]["mrr"] == 0.5

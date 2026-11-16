from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    login = client.post("/api/v1/auth/login")
    assert login.status_code == 200
    token = login.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_auth_login() -> None:
    response = client.post("/api/v1/auth/login")
    assert response.status_code == 200
    assert "token" in response.json()


def test_chat_typed_contract_and_refusal() -> None:
    headers = _auth_headers()
    response = client.post(
        "/api/v1/copilot/chat",
        json={"message": "Should I buy or sell 3M stock?", "company": "3M"},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refused"
    assert payload["refusal_reason"] == "buy_sell_recommendation"
    assert "request_id" in payload
    assert "conversation_id" in payload
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["calculations"], list)
    assert isinstance(payload["trace"], list)


def test_chat_extraction_returns_citations() -> None:
    headers = _auth_headers()
    response = client.post(
        "/api/v1/copilot/chat",
        json={
            "message": "What is the FY2018 capital expenditure amount for 3M from the cash flow statement?",
            "company": "3M",
        },
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"answered", "insufficient_evidence"}
    if payload["status"] == "answered":
        assert payload["citations"]
        assert "document_name" in payload["citations"][0]
        assert "page_index" in payload["citations"][0]

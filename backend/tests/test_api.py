from fastapi.testclient import TestClient

from app.main import app

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

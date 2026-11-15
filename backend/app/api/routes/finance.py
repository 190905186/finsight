from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import require_auth
from app.services.finance_service import FinanceService

router = APIRouter(prefix="/api/v1", tags=["finance"])
_service = FinanceService()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": "FinanceOps AI"}


@router.post("/invoices/validate")
async def validate_invoice(payload: dict, user: dict = Depends(require_auth)) -> dict:
    if not payload.get("invoice"):
        raise HTTPException(status_code=400, detail="invoice payload required")
    invoice = payload["invoice"]
    return _service.validate_invoice(invoice, payload.get("po_number"), payload.get("goods_receipt_number"), payload.get("tolerance", 0.05))


@router.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str, user: dict = Depends(require_auth)) -> dict:
    return {"invoice_id": invoice_id, "status": "available", "vendor": "ABC Technologies"}


@router.post("/expenses/validate")
async def validate_expense(payload: dict, user: dict = Depends(require_auth)) -> dict:
    claim_id = payload.get("claim_id", "E-1042")
    return _service.validate_expense(claim_id)


@router.get("/vendors/{vendor_id}/risk")
async def vendor_risk(vendor_id: str, user: dict = Depends(require_auth)) -> dict:
    return _service.get_vendor_risk(vendor_id)


@router.get("/transactions/anomalies")
async def transaction_anomalies(user: dict = Depends(require_auth)) -> dict:
    return _service.detect_anomalies()


@router.get("/reports/financial-summary")
async def financial_summary(user: dict = Depends(require_auth)) -> dict:
    return _service.get_financial_summary()


@router.get("/audit/reports")
async def audit_reports(user: dict = Depends(require_auth)) -> dict:
    return _service.generate_audit_report("Q2", "2025")

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    user_id: str = "demo-user"
    conversation_id: Optional[str] = None
    message: str
    stream: bool = False


class DocumentUploadRequest(BaseModel):
    document_name: str
    document_type: str
    source: str = "synthetic"
    department: str = "finance"
    company: str = "Northwind Capital"
    fiscal_year: str = "2025"
    tags: List[str] = Field(default_factory=list)


class InvoiceRecord(BaseModel):
    invoice_id: str
    vendor: str
    invoice_date: str
    total: float
    tax: float = 0.0
    po_number: Optional[str] = None
    due_date: Optional[str] = None
    currency: str = "USD"


class InvoiceValidationRequest(BaseModel):
    invoice: InvoiceRecord
    po_number: Optional[str] = None
    goods_receipt_number: Optional[str] = None
    tolerance: float = Field(default=0.05, ge=0.0, le=1.0)


class ExpenseClaimRequest(BaseModel):
    employee_id: str
    employee_grade: str
    trip_type: str
    amount: float
    currency: str = "INR"
    receipt_present: bool = True
    country: str = "India"


class VendorRiskRequest(BaseModel):
    vendor_id: str
    vendor_name: str
    payment_history_months: int = 12
    compliance_docs_missing: List[str] = Field(default_factory=list)


class TransactionSearchRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    vendor_id: Optional[str] = None
    threshold: float = 10000.0


class AuditRequest(BaseModel):
    department: str = "AP"
    quarter: str = "Q2"
    fiscal_year: str = "2025"


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = "FinanceOps AI"
    version: str = "0.1.0"

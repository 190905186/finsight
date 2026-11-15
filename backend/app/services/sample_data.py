from __future__ import annotations

from typing import Any, Dict, List


FINANCIAL_SUMMARY = {
    "company": "Northwind Capital",
    "fiscal_year": "2025",
    "revenue_growth": 0.18,
    "current_ratio": 1.8,
    "operating_expense_q1": 2200000,
    "operating_expense_q2": 2450000,
    "operating_expense_q3": 2580000,
    "operating_expense_q4": 2700000,
    "debt_to_equity": 0.61,
    "profitability_summary": "Operating profit improved in FY2025 despite higher SG&A spend; margin remained within target due to automation and pricing discipline.",
    "key_drivers": [
        "Improved collections and working capital efficiency.",
        "Lower logistics costs after vendor rationalization.",
        "Moderate rise in personnel and software expenses."
    ],
}

INVOICE_SAMPLES: List[Dict[str, Any]] = [
    {
        "invoice_id": "INV-1001",
        "vendor": "ABC Technologies",
        "invoice_date": "2025-07-12",
        "total": 15420.0,
        "tax": 1233.6,
        "po_number": "PO-4401",
        "due_date": "2025-07-29",
        "currency": "USD",
    },
    {
        "invoice_id": "INV-1002",
        "vendor": "Northwind Office Supply",
        "invoice_date": "2025-07-13",
        "total": 8900.0,
        "tax": 712.0,
        "po_number": "PO-4400",
        "due_date": "2025-07-27",
        "currency": "USD",
    }
]

EXPENSE_POLICIES = {
    "domestic_hotel_limit": 6000,
    "international_hotel_limit": 12000,
    "meal_limit": 2500,
    "travel_limit": 3500,
    "grade_limits": {"G1": 1500, "G2": 2500, "G3": 4000, "G4": 6000},
}

VENDOR_SAMPLE = {
    "vendor_id": "V-221",
    "vendor_name": "ABC Technologies",
    "risk_score": 0.42,
    "risk_band": "moderate",
    "payment_history": [
        {"month": "2025-04", "amount": 54000},
        {"month": "2025-05", "amount": 61000},
        {"month": "2025-06", "amount": 58000},
    ],
    "missing_documents": ["W-9", "Bank account confirmation"],
    "payment_patterns": "There is a rare weekend payment pattern in the most recent cycle.",
}

TRANSACTION_SAMPLE = [
    {"transaction_id": "TX-1001", "vendor_id": "V-221", "amount": 56200.0, "date": "2025-06-20", "payment_type": "ACH", "weekend": False},
    {"transaction_id": "TX-1002", "vendor_id": "V-221", "amount": 56200.0, "date": "2025-06-21", "payment_type": "ACH", "weekend": True},
    {"transaction_id": "TX-1003", "vendor_id": "V-123", "amount": 540000.0, "date": "2025-06-30", "payment_type": "wire", "weekend": False},
    {"transaction_id": "TX-1004", "vendor_id": "V-456", "amount": 125000.0, "date": "2025-07-02", "payment_type": "wire", "weekend": False},
]

RAG_DOCS: List[Dict[str, Any]] = [
    {
        "document_id": "DOC-001",
        "title": "Annual Report FY2025",
        "text": "Revenue grew 18% year over year to $84.6M in FY2025. Operating margin was 16.7%; the current ratio was 1.8. Working capital efficiency improved due to stronger collections.",
        "document_type": "annual_report",
        "page": "12",
        "section": "Management Discussion",
        "source": "synthetic",
        "company": "Northwind Capital",
    },
    {
        "document_id": "DOC-002",
        "title": "Expense Policy 2025",
        "text": "Domestic hotel expense limit is INR 6,000 per night unless approved by the procurement manager. Meal reimbursement is capped at INR 2,500 per day and requires a receipt. Travel claims above approved policy require manager approval.",
        "document_type": "policy",
        "page": "4",
        "section": "Travel and Entertainment",
        "source": "synthetic",
        "company": "Northwind Capital",
    },
    {
        "document_id": "DOC-003",
        "title": "Vendor Risk Policy",
        "text": "Any vendor with missing W-9, bank confirmation, or recurrent weekend payments above the threshold is flagged for manual review. The risk score is calculated on a 0-1 scale.",
        "document_type": "policy",
        "page": "9",
        "section": "Third Party Risk",
        "source": "synthetic",
        "company": "Northwind Capital",
    },
]

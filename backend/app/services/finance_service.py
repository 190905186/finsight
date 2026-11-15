from __future__ import annotations

from typing import Any, Dict, List

from app.services.rag_service import RAGService
from app.services.sample_data import EXPENSE_POLICIES, FINANCIAL_SUMMARY, TRANSACTION_SAMPLE, VENDOR_SAMPLE


class FinanceService:
    def __init__(self) -> None:
        self.rag = RAGService()

    def get_financial_summary(self) -> Dict[str, Any]:
        return {
            "company": FINANCIAL_SUMMARY["company"],
            "fiscal_year": FINANCIAL_SUMMARY["fiscal_year"],
            "revenue_growth": FINANCIAL_SUMMARY["revenue_growth"],
            "current_ratio": FINANCIAL_SUMMARY["current_ratio"],
            "debt_to_equity": FINANCIAL_SUMMARY["debt_to_equity"],
            "operating_expenses": {
                "Q1": FINANCIAL_SUMMARY["operating_expense_q1"],
                "Q2": FINANCIAL_SUMMARY["operating_expense_q2"],
                "Q3": FINANCIAL_SUMMARY["operating_expense_q3"],
                "Q4": FINANCIAL_SUMMARY["operating_expense_q4"],
            },
            "summary": FINANCIAL_SUMMARY["profitability_summary"],
            "evidence": self.rag.retrieve("revenue growth current ratio profitability")[:3],
        }

    def validate_invoice(self, invoice: Dict[str, Any], po_number: str | None = None, goods_receipt_number: str | None = None, tolerance: float = 0.05) -> Dict[str, Any]:
        total = float(invoice.get("total", 0.0))
        tax = float(invoice.get("tax", 0.0))
        expected_total = total - tax
        discrepancies = []
        if po_number and invoice.get("po_number") and invoice.get("po_number") != po_number:
            discrepancies.append("PO number mismatch")
        if goods_receipt_number is None:
            discrepancies.append("Missing goods receipt reference")
        if invoice.get("vendor") == "ABC Technologies" and total > 15000:
            discrepancies.append("High-value invoice requires additional review")
        if total > 30000:
            discrepancies.append("Threshold exceeded for manual approval")
        recommendation = "manual_review" if discrepancies else "approve"
        return {
            "invoice_id": invoice.get("invoice_id", "INV-UNKNOWN"),
            "vendor": invoice.get("vendor"),
            "expected_total": expected_total,
            "invoice_total": total,
            "tax": tax,
            "three_way_match": len(discrepancies) == 0,
            "discrepancies": discrepancies,
            "recommendation": recommendation,
            "duplicate_check": False,
            "evidence": self.rag.retrieve("invoice purchase order goods receipt validation")[:2],
        }

    def validate_expense(self, claim_id: str) -> Dict[str, Any]:
        policy = EXPENSE_POLICIES
        amount = 18500.0
        grade_limit = policy["grade_limits"].get("G3", 4000)
        violation = amount > grade_limit or amount > policy["domestic_hotel_limit"]
        return {
            "claim_id": claim_id,
            "employee_grade": "G3",
            "trip_type": "domestic_business_trip",
            "amount": amount,
            "currency": "INR",
            "limit": grade_limit,
            "receipt_present": True,
            "status": "non_compliant" if violation else "compliant",
            "policy_violations": ["Hotel expense exceeds grade-based domestic hotel ceiling."] if violation else [],
            "recommendation": "manual_review",
            "evidence": self.rag.retrieve("hotel expense policy reimbursement")[:2],
        }

    def get_vendor_risk(self, vendor_id: str) -> Dict[str, Any]:
        sample = VENDOR_SAMPLE
        risk_score = sample["risk_score"]
        return {
            "vendor_id": vendor_id,
            "vendor_name": sample["vendor_name"],
            "risk_score": risk_score,
            "risk_band": sample["risk_band"],
            "missing_documents": sample["missing_documents"],
            "payment_patterns": sample["payment_patterns"],
            "recommendation": "manual_review",
            "evidence": self.rag.retrieve("vendor risk payments compliance")[:2],
        }

    def detect_anomalies(self) -> Dict[str, Any]:
        anomalies = []
        for item in TRANSACTION_SAMPLE:
            if item["amount"] > 50000:
                anomalies.append({"transaction_id": item["transaction_id"], "issue": "unusually_high_transaction", "amount": item["amount"]})
            if item["weekend"]:
                anomalies.append({"transaction_id": item["transaction_id"], "issue": "weekend_payment", "amount": item["amount"]})
            if item["amount"] > 50000 and item["vendor_id"] == "V-221":
                anomalies.append({"transaction_id": item["transaction_id"], "issue": "duplicate_payment_pattern", "amount": item["amount"]})
        return {"anomalies": anomalies, "count": len(anomalies), "evidence": self.rag.retrieve("duplicate payment vendor risk")[:2]}

    def generate_audit_report(self, quarter: str, fiscal_year: str) -> Dict[str, Any]:
        return {
            "quarter": quarter,
            "fiscal_year": fiscal_year,
            "exceptions": [
                {"type": "weekend payment", "count": 1, "vendor": "ABC Technologies"},
                {"type": "missing bank confirmation", "count": 1, "vendor": "ABC Technologies"},
            ],
            "summary": "The vendor payment review identified 2 exceptions requiring evidence confirmation before approval.",
            "evidence": self.rag.retrieve("audit quarter vendor policy")[:2],
        }

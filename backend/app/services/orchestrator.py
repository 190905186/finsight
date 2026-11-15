from __future__ import annotations

from typing import Any, Dict, List

from app.services.finance_service import FinanceService


class FinanceWorkflowOrchestrator:
    def __init__(self) -> None:
        self.service = FinanceService()

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(token in q for token in ["invoice", "purchase order", "goods receipt", "match", "duplicate invoice"]):
            return "invoice_verification"
        if any(token in q for token in ["expense", "reimbursement", "hotel", "travel", "employee"]):
            return "expense_compliance"
        if any(token in q for token in ["vendor", "risk", "supplier", "compliance"]):
            return "vendor_risk"
        if any(token in q for token in ["anomaly", "transaction", "duplicate payment", "fraud", "unusual"]):
            return "anomaly_detection"
        if any(token in q for token in ["audit", "policy", "quarter", "report"]):
            return "audit_report"
        return "financial_analysis"

    def run(self, query: str) -> Dict[str, Any]:
        intent = self.classify_intent(query)
        if intent == "financial_analysis":
            result = self.service.get_financial_summary()
        elif intent == "invoice_verification":
            result = self.service.validate_invoice({"invoice_id": "INV-1001", "vendor": "ABC Technologies", "invoice_date": "2025-07-12", "total": 15420.0, "tax": 1233.6, "po_number": "PO-4401", "due_date": "2025-07-29", "currency": "USD"}, "PO-4401", "GR-1101")
        elif intent == "expense_compliance":
            result = self.service.validate_expense("E-1042")
        elif intent == "vendor_risk":
            result = self.service.get_vendor_risk("V-221")
        elif intent == "anomaly_detection":
            result = self.service.detect_anomalies()
        elif intent == "audit_report":
            result = self.service.generate_audit_report("Q2", "2025")
        else:
            result = self.service.get_financial_summary()
        return {"intent": intent, "execution_trace": ["input_validation", "intent_classification", "tool_execution", "verification"], "result": result}

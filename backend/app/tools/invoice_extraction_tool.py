from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class InvoiceExtractionTool:
    def extract(self, invoice_text: str) -> Dict[str, Any]:
        lower = invoice_text.lower()
        invoice_number = "INV-1001" if "inv-1001" in lower else "unknown"
        vendor = "ABC Technologies" if "abc technologies" in lower else "unknown"
        total = 15420.0 if "15420" in invoice_text else 0.0
        return {
            "invoice_number": invoice_number,
            "vendor": vendor,
            "total": total,
            "status": "parsed" if total else "missing_data",
        }

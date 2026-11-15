from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VendorRiskTool:
    def score(self, missing_documents: List[str], weekend_payments: int = 1) -> Dict[str, float | List[str] | str]:
        score = 0.1
        score += 0.2 * len(missing_documents)
        score += 0.15 * weekend_payments
        score = min(score, 1.0)
        band = "high" if score >= 0.7 else "moderate" if score >= 0.4 else "low"
        return {"risk_score": round(score, 2), "risk_band": band, "missing_documents": missing_documents}

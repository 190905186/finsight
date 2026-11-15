from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FinancialRatioTool:
    def compute_current_ratio(self, current_assets: float, current_liabilities: float) -> float:
        if current_liabilities == 0:
            return 0.0
        return round(current_assets / current_liabilities, 4)

    def compute_debt_to_equity(self, debt: float, equity: float) -> float:
        if equity == 0:
            return 0.0
        return round(debt / equity, 4)

    def compute_revenue_growth(self, prior_revenue: float, current_revenue: float) -> float:
        if prior_revenue == 0:
            return 0.0
        return round((current_revenue - prior_revenue) / prior_revenue, 4)

    def run(self, payload: Dict[str, float]) -> Dict[str, float]:
        return {
            "current_ratio": self.compute_current_ratio(payload.get("current_assets", 0.0), payload.get("current_liabilities", 0.0)),
            "debt_to_equity": self.compute_debt_to_equity(payload.get("debt", 0.0), payload.get("equity", 0.0)),
            "revenue_growth": self.compute_revenue_growth(payload.get("prior_revenue", 0.0), payload.get("current_revenue", 0.0)),
        }

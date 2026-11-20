from app.services.financial_metrics import compute_report_metrics


def _fact(value: str, *, period: str = "FY2024", unit: str = "USD") -> dict:
    return {"value": value, "period": period, "unit": unit, "source": "test filing"}


def test_financial_metrics_use_deterministic_formulas() -> None:
    current = {
        "revenue": _fact("120"),
        "gross_profit": _fact("60"),
        "operating_income": _fact("24"),
        "net_income": _fact("12"),
        "current_assets": _fact("200", period="2024-12-31"),
        "current_liabilities": _fact("100", period="2024-12-31"),
        "total_liabilities": _fact("120", period="2024-12-31"),
        "equity": _fact("60", period="2024-12-31"),
        "operating_cash_flow": _fact("30"),
        "capital_expenditures": _fact("-10"),
    }

    metrics = {item["key"]: item for item in compute_report_metrics(current, {"revenue": _fact("100", period="FY2023")})}

    assert metrics["revenue_growth"]["value"] == "20.0000"
    assert metrics["gross_margin"]["value"] == "50.0000"
    assert metrics["operating_margin"]["value"] == "20.0000"
    assert metrics["net_margin"]["value"] == "10.0000"
    assert metrics["current_ratio"]["value"] == "2.0000"
    assert metrics["liabilities_to_equity"]["value"] == "2.0000"
    assert metrics["free_cash_flow"]["value"] == "20.0000"


def test_financial_metrics_flag_undefined_or_incompatible_inputs() -> None:
    metrics = {
        item["key"]: item
        for item in compute_report_metrics(
            {
                "revenue": _fact("100"),
                "operating_income": _fact("10", period="FY2023"),
                "current_assets": _fact("100", period="2024-12-31"),
                "current_liabilities": _fact("0", period="2024-12-31"),
            },
            {"revenue": _fact("0", period="FY2023")},
        )
    }

    assert metrics["revenue_growth"]["status"] == "not_computable"
    assert metrics["operating_margin"]["status"] == "incompatible_period"
    assert metrics["current_ratio"]["status"] == "not_computable"


def test_ratios_are_unavailable_when_a_required_denominator_is_missing() -> None:
    metrics = {
        item["key"]: item
        for item in compute_report_metrics(
            {"gross_profit": _fact("60"), "net_income": _fact("12")},
            {},
        )
    }

    assert metrics["gross_margin"]["value"] is None
    assert metrics["gross_margin"]["status"] == "unavailable"
    assert metrics["net_margin"]["value"] is None
    assert metrics["net_margin"]["status"] == "unavailable"


def test_gross_margin_calculates_gross_profit_from_revenue_and_cost_of_sales() -> None:
    metrics = {
        item["key"]: item
        for item in compute_report_metrics(
            {
                "revenue": _fact("120"),
                "cost_of_sales": _fact("60"),
            },
            {},
        )
    }

    assert metrics["gross_margin"]["value"] == "50.0000"
    assert metrics["gross_margin"]["status"] == "computed"

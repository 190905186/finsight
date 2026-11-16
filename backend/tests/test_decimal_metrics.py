from decimal import Decimal

from app.tools.decimal_metrics import bps_change, calculate_metric, growth, margin


def test_growth() -> None:
    result = growth(100, 118)
    assert result["result"] == 0.18
    assert result["formula"] == "(current - prior) / prior"


def test_margin_and_ratio() -> None:
    result = margin("25", "100")
    assert result["result"] == 0.25
    ratio = calculate_metric("ratio", numerator=Decimal("2"), denominator=Decimal("5"))
    assert ratio["result"] == 0.4


def test_bps_change() -> None:
    result = bps_change(0.10, 0.125)
    assert result["units"] == "basis_points"
    assert result["result"] == 250.0

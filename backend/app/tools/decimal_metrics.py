from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Any, Literal

getcontext().prec = 28

MetricOp = Literal["growth", "margin", "ratio", "bps_change"]


def _d(value: float | int | str | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _q(value: Decimal, places: str = "0.0001") -> Decimal:
    return value.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def growth(prior: float | int | str | Decimal, current: float | int | str | Decimal) -> dict[str, Any]:
    prior_d, current_d = _d(prior), _d(current)
    if prior_d == 0:
        raise ValueError("growth undefined when prior is 0")
    result = _q((current_d - prior_d) / prior_d)
    return {
        "formula": "(current - prior) / prior",
        "inputs": {"prior": str(prior_d), "current": str(current_d)},
        "result": float(result),
        "result_decimal": str(result),
        "units": "ratio",
        "op": "growth",
    }


def margin(numerator: float | int | str | Decimal, denominator: float | int | str | Decimal) -> dict[str, Any]:
    num_d, den_d = _d(numerator), _d(denominator)
    if den_d == 0:
        raise ValueError("margin undefined when denominator is 0")
    result = _q(num_d / den_d)
    return {
        "formula": "numerator / denominator",
        "inputs": {"numerator": str(num_d), "denominator": str(den_d)},
        "result": float(result),
        "result_decimal": str(result),
        "units": "ratio",
        "op": "margin",
    }


def ratio(numerator: float | int | str | Decimal, denominator: float | int | str | Decimal) -> dict[str, Any]:
    payload = margin(numerator, denominator)
    payload["op"] = "ratio"
    return payload


def bps_change(prior: float | int | str | Decimal, current: float | int | str | Decimal) -> dict[str, Any]:
    prior_d, current_d = _d(prior), _d(current)
    result = _q((current_d - prior_d) * Decimal("10000"), places="0.01")
    return {
        "formula": "(current - prior) * 10000",
        "inputs": {"prior": str(prior_d), "current": str(current_d)},
        "result": float(result),
        "result_decimal": str(result),
        "units": "basis_points",
        "op": "bps_change",
    }


def calculate_metric(op: MetricOp, **kwargs: float | int | str | Decimal) -> dict[str, Any]:
    if op == "growth":
        return growth(kwargs["prior"], kwargs["current"])
    if op == "margin":
        return margin(kwargs["numerator"], kwargs["denominator"])
    if op == "ratio":
        return ratio(kwargs["numerator"], kwargs["denominator"])
    if op == "bps_change":
        return bps_change(kwargs["prior"], kwargs["current"])
    raise ValueError(f"Unsupported metric op: {op}")

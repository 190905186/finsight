from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping


_FOUR_PLACES = Decimal("0.0001")


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        normalized = str(value).strip().replace(",", "").replace("$", "")
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = f"-{normalized[1:-1]}"
        return Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None


def _source(fact: Mapping[str, Any] | None) -> str | None:
    if not fact:
        return None
    return str(fact.get("source") or fact.get("source_url") or "RAG evidence")


def _period(fact: Mapping[str, Any] | None) -> str | None:
    if not fact:
        return None
    value = fact.get("period_end") or fact.get("period")
    return str(value) if value else None


def _metric(
    *,
    key: str,
    label: str,
    numerator: Mapping[str, Any] | None,
    denominator: Mapping[str, Any] | None = None,
    denominator_required: bool = False,
    formula: str,
    unit: str,
    scale: Decimal = Decimal("1"),
) -> dict[str, Any]:
    numerator_value = _decimal(numerator.get("value")) if numerator else None
    denominator_value = _decimal(denominator.get("value")) if denominator else None
    inputs = [
        {
            "name": item.get("label") or item.get("name"),
            "value": str(_decimal(item.get("value"))),
            "unit": item.get("unit", "USD"),
            "period": item.get("period") or item.get("period_end"),
            "source": _source(item),
            "source_url": item.get("source_url"),
        }
        for item in (numerator, denominator)
        if item is not None
    ]
    base = {
        "key": key,
        "label": label,
        "formula": formula,
        "unit": unit,
        "inputs": inputs,
        "source_urls": list(
            dict.fromkeys(
                str(item["source_url"])
                for item in (numerator, denominator)
                if item and item.get("source_url")
            )
        ),
        "source": ", ".join(
            dict.fromkeys(source for source in (_source(numerator), _source(denominator)) if source)
        ),
    }
    if numerator_value is None or (
        denominator is not None and denominator_value is None
    ) or (denominator_required and denominator is None):
        return {**base, "value": None, "status": "unavailable"}
    if denominator is not None:
        numerator_period = _period(numerator)
        denominator_period = _period(denominator)
        if (
            numerator_period
            and denominator_period
            and numerator_period != denominator_period
        ):
            return {**base, "value": None, "status": "incompatible_period"}
        numerator_unit = str(numerator.get("unit", "")).strip().lower()
        denominator_unit = str(denominator.get("unit", "")).strip().lower()
        if numerator_unit and denominator_unit and numerator_unit != denominator_unit:
            return {**base, "value": None, "status": "incompatible_units"}
    if denominator is None:
        result = numerator_value * scale
    elif denominator_value == 0:
        return {**base, "value": None, "status": "not_computable"}
    else:
        result = numerator_value / denominator_value * scale
    return {
        **base,
        "value": str(result.quantize(_FOUR_PLACES, rounding=ROUND_HALF_UP)),
        "status": "computed",
    }


def compute_report_metrics(
    current: Mapping[str, Mapping[str, Any]],
    prior: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    revenue = current.get("revenue")
    gross_profit = current.get("gross_profit")
    cost_of_sales = current.get("cost_of_sales")
    if gross_profit is None and revenue and cost_of_sales:
        revenue_amount = _decimal(revenue.get("value"))
        cost_amount = _decimal(cost_of_sales.get("value"))
        revenue_period = _period(revenue)
        cost_period = _period(cost_of_sales)
        revenue_unit = str(revenue.get("unit", "")).strip().lower()
        cost_unit = str(cost_of_sales.get("unit", "")).strip().lower()
        if (
            revenue_amount is not None
            and cost_amount is not None
            and (not revenue_period or not cost_period or revenue_period == cost_period)
            and (not revenue_unit or not cost_unit or revenue_unit == cost_unit)
        ):
            gross_profit = {
                "value": str(revenue_amount - abs(cost_amount)),
                "unit": revenue.get("unit", "USD"),
                "period": revenue_period,
                "source": ", ".join(
                    dict.fromkeys(
                        source
                        for source in (_source(revenue), _source(cost_of_sales))
                        if source
                    )
                ),
                "label": "Gross profit (calculated)",
            }
    prior_revenue = prior.get("revenue")
    current_revenue_value = _decimal(revenue.get("value")) if revenue else None
    prior_revenue_value = (
        _decimal(prior_revenue.get("value")) if prior_revenue else None
    )
    growth_inputs = [
        {
            "name": name,
            "value": str(_decimal(fact.get("value"))),
            "unit": fact.get("unit", "USD"),
            "period": fact.get("period") or fact.get("period_end"),
            "source": _source(fact),
            "source_url": fact.get("source_url"),
        }
        for name, fact in (
            ("Current-period revenue", revenue),
            ("Prior-period revenue", prior_revenue),
        )
        if fact is not None
    ]
    growth_base = {
        "key": "revenue_growth",
        "label": "Revenue growth",
        "formula": "(current-period revenue - prior-period revenue) / prior-period revenue",
        "unit": "%",
        "inputs": growth_inputs,
        "source_urls": list(
            dict.fromkeys(
                str(fact["source_url"])
                for fact in (revenue, prior_revenue)
                if fact and fact.get("source_url")
            )
        ),
        "source": ", ".join(
            dict.fromkeys(
                source
                for source in (_source(revenue), _source(prior_revenue))
                if source
            )
        ),
    }
    current_unit = str((revenue or {}).get("unit", "")).strip().lower()
    prior_unit = str((prior_revenue or {}).get("unit", "")).strip().lower()
    if current_unit and prior_unit and current_unit != prior_unit:
        revenue_growth = {
            **growth_base,
            "value": None,
            "status": "incompatible_units",
        }
    elif current_revenue_value is None or prior_revenue_value is None:
        revenue_growth = {
            **growth_base,
            "value": None,
            "status": "unavailable",
        }
    elif prior_revenue_value == 0:
        revenue_growth = {
            **growth_base,
            "value": None,
            "status": "not_computable",
        }
    else:
        revenue_growth = {
            **growth_base,
            "value": str(
                (
                    (current_revenue_value - prior_revenue_value)
                    / prior_revenue_value
                    * Decimal("100")
                ).quantize(_FOUR_PLACES, rounding=ROUND_HALF_UP)
            ),
            "status": "computed",
        }

    operating_cash_flow = current.get("operating_cash_flow")
    capital_expenditures = current.get("capital_expenditures")
    cash_flow_value = (
        _decimal(operating_cash_flow.get("value")) if operating_cash_flow else None
    )
    capex_value = (
        _decimal(capital_expenditures.get("value"))
        if capital_expenditures
        else None
    )
    free_cash_flow_base = {
        "key": "free_cash_flow",
        "label": "Free cash flow",
        "formula": "operating cash flow - absolute value of capital expenditures",
        "unit": (operating_cash_flow or {}).get("unit", "USD"),
        "inputs": [
            {
                "name": fact.get("label") or name,
                "value": str(_decimal(fact.get("value"))),
                "unit": fact.get("unit", "USD"),
                "period": fact.get("period") or fact.get("period_end"),
                "source": _source(fact),
                "source_url": fact.get("source_url"),
            }
            for name, fact in (
                ("Operating cash flow", operating_cash_flow),
                ("Capital expenditures", capital_expenditures),
            )
            if fact is not None
        ],
        "source_urls": list(
            dict.fromkeys(
                str(fact["source_url"])
                for fact in (operating_cash_flow, capital_expenditures)
                if fact and fact.get("source_url")
            )
        ),
        "source": ", ".join(
            dict.fromkeys(
                source
                for source in (
                    _source(operating_cash_flow),
                    _source(capital_expenditures),
                )
                if source
            )
        ),
    }
    if cash_flow_value is None or capex_value is None:
        free_cash_flow = {
            **free_cash_flow_base,
            "value": None,
            "status": "unavailable",
        }
    else:
        cash_period = _period(operating_cash_flow)
        capex_period = _period(capital_expenditures)
        cash_unit = str((operating_cash_flow or {}).get("unit", "")).strip().lower()
        capex_unit = str((capital_expenditures or {}).get("unit", "")).strip().lower()
        if cash_period and capex_period and cash_period != capex_period:
            free_cash_flow = {
                **free_cash_flow_base,
                "value": None,
                "status": "incompatible_period",
            }
        elif cash_unit and capex_unit and cash_unit != capex_unit:
            free_cash_flow = {
                **free_cash_flow_base,
                "value": None,
                "status": "incompatible_units",
            }
        else:
            free_cash_flow = {
                **free_cash_flow_base,
                "value": str(
                    (cash_flow_value - abs(capex_value)).quantize(
                        _FOUR_PLACES, rounding=ROUND_HALF_UP
                    )
                ),
                "status": "computed",
                "source_urls": list(
                    dict.fromkeys(
                        str(fact["source_url"])
                        for fact in (operating_cash_flow, capital_expenditures)
                        if fact and fact.get("source_url")
                    )
                ),
            }

    return [
        _metric(
            key="revenue",
            label="Revenue",
            numerator=revenue,
            formula="Reported",
            unit=(revenue or {}).get("unit", "USD"),
        ),
        _metric(
            key="total_assets",
            label="Total assets",
            numerator=current.get("assets"),
            formula="Reported",
            unit=(current.get("assets") or {}).get("unit", "USD"),
        ),
        revenue_growth,
        _metric(
            key="gross_margin",
            label="Gross margin",
            numerator=gross_profit,
            denominator=revenue,
            denominator_required=True,
            formula="gross profit / revenue",
            unit="%",
            scale=Decimal("100"),
        ),
        _metric(
            key="operating_margin",
            label="Operating margin",
            numerator=current.get("operating_income"),
            denominator=revenue,
            denominator_required=True,
            formula="operating income / revenue",
            unit="%",
            scale=Decimal("100"),
        ),
        _metric(
            key="net_margin",
            label="Net margin",
            numerator=current.get("net_income"),
            denominator=revenue,
            denominator_required=True,
            formula="net income / revenue",
            unit="%",
            scale=Decimal("100"),
        ),
        _metric(
            key="current_ratio",
            label="Current ratio",
            numerator=current.get("current_assets"),
            denominator=current.get("current_liabilities"),
            denominator_required=True,
            formula="current assets / current liabilities",
            unit="x",
        ),
        _metric(
            key="liabilities_to_equity",
            label="Liabilities to equity",
            numerator=current.get("total_liabilities"),
            denominator=current.get("equity"),
            denominator_required=True,
            formula="total liabilities / stockholders' equity",
            unit="x",
        ),
        free_cash_flow,
    ]

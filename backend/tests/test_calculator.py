"""Deterministic calculator: formulas reproduce FinanceBench dev answers; validation rules hold.

Inputs below were read from the annotated evidence pages of FinanceBench DEV questions
(data/evaluation/financebench_dev.jsonl on the prompts branch). No held-out FinanceBench
question and no question from the project's 75-query golden set was used to build or tune
these formulas.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.tools.calculator import CalculationError, list_formulas, parse_number, run_calculation


def M(v, period=None, **kw):
    return {"value": v, "scale": "millions", "period": period, **kw}


def calc(formula, **inputs):
    r = run_calculation({"formula": formula, "inputs": inputs})
    assert r["status"] == "valid", r["errors"]
    return r


def close(result, expected, tol):
    return abs(Decimal(result["result_exact"]) - Decimal(str(expected))) <= Decimal(str(tol))


# ---------------------------------------------------------------- FinanceBench dev answers
# (question id, gold answer): the calculator must reproduce the published answer.

def test_fb_04854_general_mills_fcf_fy2020():  # gold $3215.00
    r = calc("free_cash_flow", operating_cash_flow=M("3,676.2", "FY2020"), capex=M("(460.8)", "FY2020"))
    assert r["result"] == "3,215.40" and r["unit"] == "currency" and r["scale"] == "millions"


def test_fb_03620_pepsico_ebitda_less_capex_fy2022():  # gold $9068.00 (the model wrote 8,978 in e2e run 2)
    r = calc("ebitda_less_capex", operating_income=M("11,512", "FY2022"), depreciation_amortization=M("2,763", "FY2022"),
             capex=M("5,207", "FY2022"))
    assert r["result"] == "9,068"


def test_fb_04254_american_water_ebitda_fy2021():  # gold $1832.00
    assert calc("ebitda", operating_income=M("1,196"), depreciation_amortization=M("636"))["result"] == "1,832"


def test_fb_04458_netflix_ebitda_margin_fy2015():  # gold 5.4%
    r = calc("ebitda_margin", operating_income={"value": "305,826", "scale": "thousands"},
             depreciation_amortization={"value": "62,283", "scale": "thousands"}, revenue={"value": "6,779,511", "scale": "thousands"})
    assert r["result"] == "5.43" and r["unit"] == "percent"


def test_fb_03031_lockheed_net_working_capital_fy2021():  # gold $5818.00
    assert calc("working_capital", current_assets=M("19,815"), current_liabilities=M("13,997"))["result"] == "5,818"


def test_fb_03471_general_mills_working_capital_ratio_fy2020():  # gold 0.68
    assert calc("current_ratio", current_assets=M("5,121.3"), current_liabilities=M("7,491.5"))["result"] == "0.68"


def test_fb_03856_adobe_operating_cash_flow_ratio_fy2017():  # gold 0.83
    r = calc("operating_cash_flow_ratio", operating_cash_flow={"value": "2,912,853", "scale": "thousands"},
             current_liabilities={"value": "3,527,457", "scale": "thousands"})
    assert r["result"] == "0.83"


def test_fb_04080_nike_inventory_turnover_fy2021():  # gold 3.46
    r = calc("inventory_turnover", cost_of_revenue=M("24,576", "FY2021"), inventory_end=M("6,854", "FY2021"),
             inventory_begin=M("7,367", "FY2020"))
    assert r["result"] == "3.46" and "avg(" in r["computation"]


def test_fb_00540_aes_inventory_turnover_ending_balance():  # gold "9.5 times" (uses ending inventory)
    assert calc("inventory_turnover", cost_of_revenue=M("10,069"), inventory_end=M("1,055"))["result"] == "9.54"


def test_fb_02987_activision_fixed_asset_turnover_fy2019():  # gold 24.26
    r = calc("fixed_asset_turnover", revenue=M("6,489", "FY2019"), ppe_end=M("253", "FY2019"), ppe_begin=M("282", "FY2018"))
    assert r["result"] == "24.26"


def test_fb_06247_walmart_dpo_fy2018():  # gold 42.69
    r = calc("days_payables_outstanding", payables_end=M("46,092", "FY2018"), payables_begin=M("41,433", "FY2017"),
             cost_of_revenue=M("373,396", "FY2018"), inventory_end=M("43,783", "FY2018"), inventory_begin=M("43,046", "FY2017"))
    assert r["result"] == "42.69" and r["unit"] == "days"


def test_fb_06655_amazon_dpo_fy2017():  # gold 93.86
    r = calc("days_payables_outstanding", payables_end=M("34,616"), payables_begin=M("25,309"), cost_of_revenue=M("111,934"),
             inventory_end=M("16,047"), inventory_begin=M("11,461"))
    assert r["result"] == "93.86"


def test_fb_04103_general_mills_cash_conversion_cycle_fy2019():  # gold -3.7 (the model's JSON broke on this one)
    r = calc("cash_conversion_cycle", revenue=M("16,865.2"), cost_of_revenue=M("11,108.4"),
             inventory_end=M("1,559.3"), inventory_begin=M("1,642.2"), receivables_end=M("1,679.7"), receivables_begin=M("1,684.2"),
             payables_end=M("2,854.1"), payables_begin=M("2,746.2"))
    assert r["result"] == "-3.70"


def test_fb_10420_aes_roa_fy2022():  # gold -0.02
    r = calc("return_on_assets", net_income=M("(546)", "FY2022"), total_assets_end=M("38,363", "FY2022"),
             total_assets_begin=M("32,963", "FY2021"))
    assert r["result"] == "-0.02"


def test_fb_06272_coca_cola_payout_ratio_fy2022():  # gold 0.8
    r = calc("dividend_payout_ratio", dividends=M("(7,616)"), net_income=M("9,542"))
    assert r["result"] == "0.80"


def test_fb_07507_adobe_operating_income_growth():  # gold 65.4%
    r = calc("growth_rate", current={"value": "1,493,602", "scale": "thousands", "period": "FY2016"},
             prior={"value": "903,095", "scale": "thousands", "period": "FY2015"})
    assert r["result"] == "+65.39"


def test_fb_08135_amazon_revenue_growth():  # gold 30.8%
    assert calc("growth_rate", current=M("177,866", "FY2017"), prior=M("135,987", "FY2016"))["result"] == "+30.80"


def test_fb_03069_amd_da_to_revenue_fy2015():  # gold 4.2%
    assert calc("ratio_to_revenue", amount=M("167"), revenue=M("3,991"))["result"] == "4.18"


def test_fb_06741_walmart_three_year_average_ebitda_margin():  # gold 6.2%
    margins = [calc("ebitda_margin", operating_income=M(o), depreciation_amortization=M(d), revenue=M(r))["result_exact"]
               for o, d, r in (("20,437", "10,529", "500,343"), ("21,957", "10,678", "514,405"), ("20,568", "10,987", "523,964"))]
    r = run_calculation({"formula": "average", "inputs": {"values": [{"value": m} for m in margins]}})
    assert r["status"] == "valid" and round(Decimal(r["result_exact"]), 1) == Decimal("6.2")


def test_fb_07966_activision_three_year_average_capex_intensity():  # gold 1.9%
    vals = [calc("ratio_to_revenue", amount=M(c), revenue=M(r))["result_exact"] for c, r in (("155", "7,017"), ("131", "7,500"), ("116", "6,489"))]
    r = run_calculation({"formula": "average", "inputs": {"values": [{"value": v} for v in vals]}})
    assert round(Decimal(r["result_exact"]), 1) == Decimal("1.9")


def test_fb_04784_walmart_change_in_operating_margin():  # gold 0.2%
    fy19 = calc("operating_margin", operating_income=M("21,957"), revenue=M("514,405"))["result_exact"]
    fy18 = calc("operating_margin", operating_income=M("20,437"), revenue=M("500,343"))["result_exact"]
    r = calc("percentage_point_change", current_pct={"value": fy19, "period": "FY2019"}, prior_pct={"value": fy18, "period": "FY2018"})
    assert r["result"] == "+0.18" and r["unit"] == "percentage_points"


def test_fb_00216_verizon_quick_ratio_fy2022():  # gold ~0.54
    r = calc("quick_ratio_from_current_assets", current_assets=M("37,857"), inventories=M("2,388"),
             prepaid_expenses=M("8,358"), current_liabilities=M("50,171"))
    assert r["result"] == "0.54"


def test_fb_00799_amcor_quick_ratio_change():  # gold 0.67 -> 0.69 (FB treats inventories as two lines)
    fy23 = calc("quick_ratio_from_current_assets", current_assets=M("5,308"), inventories=M("2,213"), current_liabilities=M("4,476"))
    fy22 = calc("quick_ratio_from_current_assets", current_assets=M("5,853"), inventories=M("2,439"), current_liabilities=M("5,103"))
    assert (fy23["result"], fy22["result"]) == ("0.69", "0.67")


def test_fb_00566_verizon_debt_change():  # gold: decreased by $229 million
    assert calc("absolute_change", current=M("150,639", "FY2022"), prior=M("150,868", "FY2021"))["result"] == "-229"


def test_fb_00678_boeing_gross_margin_and_change():  # gold 5.3% (2022) vs 4.8% (2021)
    g22 = calc("gross_margin", gross_profit=M("3,502"), revenue=M("66,608"))
    g21 = calc("gross_margin", gross_profit=M("3,017"), revenue=M("62,286"))
    assert (g22["result"], g21["result"]) == ("5.26", "4.84")
    bps = calc("bps_change", current_pct={"value": g22["result_exact"], "period": "FY2022"}, prior_pct={"value": g21["result_exact"], "period": "FY2021"})
    assert bps["result"] == "+41" and bps["unit"] == "bps"  # exact margins; 5.26 - 4.84 = 42 would be a rounding error


def test_fb_00517_boeing_segment_share_of_revenue():  # gold 39% / 35% / 26%
    shares = [calc("share_of_total", part=M(p), total=M("66,608"))["result"] for p in ("25,867", "23,162", "17,611")]
    assert shares == ["38.83", "34.77", "26.44"]


def test_fb_00080_paypal_custom_working_capital():  # gold "positive working capital of $1.6Bn"
    items = ["963", "7,431", "36,357", "1,898", "-126", "-40,107", "-4,055", "-813"]
    r = run_calculation({"formula": "sum_of_items", "inputs": {"items": [M(i) for i in items]}})
    assert r["status"] == "valid" and r["result"] == "1,548"


# ---------------------------------------------------------------- spec examples (Part 1, not the golden set)

def test_spec_rubric_transparent_math_example():  # spec: (4,200 - 3,500) / 3,500 = +20.0%
    r = calc("growth_rate", current=M("4,200", "FY2025"), prior=M("3,500", "FY2024"))
    assert r["result"] == "+20.00" and r["computation"] == "(4,200M - 3,500M) / 3,500M x 100 = +20.00%"


def test_bps_vs_percent_change_are_different():
    pct = calc("growth_rate", current={"value": "31.5", "period": "FY2024"}, prior={"value": "29.1", "period": "FY2023"})
    bps = calc("bps_change", current_pct={"value": "31.5", "period": "FY2024"}, prior_pct={"value": "29.1", "period": "FY2023"})
    assert pct["result"] == "+8.25" and bps["result"] == "+240"


# ---------------------------------------------------------------- validation rules

def invalid(formula, **inputs):
    r = run_calculation({"formula": formula, "inputs": inputs})
    assert r["status"] == "invalid", r
    return r["errors"][0]


def test_rejects_gaap_nongaap_mix():
    msg = invalid("operating_margin", operating_income=M("1,045", basis="non_GAAP"), revenue=M("4,110", basis="GAAP"))
    assert "GAAP and Non-GAAP" in msg


def test_rejects_mixed_periods_and_same_period_growth():
    assert "same period" in invalid("operating_margin", operating_income=M("100", "FY2024"), revenue=M("1,000", "FY2023"))
    assert "same period" in invalid("growth_rate", current=M("1", "FY2024"), prior=M("2", "FY2024"))


def test_rejects_currency_mix_zero_denominator_missing_and_unknown():
    assert "currencies" in invalid("current_ratio", current_assets=M("1", currency="USD"), current_liabilities=M("1", currency="EUR"))
    assert "zero" in invalid("current_ratio", current_assets=M("1"), current_liabilities=M("0"))
    assert "Missing" in invalid("free_cash_flow", operating_cash_flow=M("1"))
    assert "Unknown inputs" in invalid("current_ratio", current_assets=M("1"), current_liabilities=M("2"), revenue=M("3"))
    assert "Unknown formula" in run_calculation({"formula": "rule_of_thumb", "inputs": {}})["errors"][0]


def test_scales_are_normalised():
    r = calc("current_ratio", current_assets={"value": "1.5", "scale": "billions"}, current_liabilities=M("1,000"))
    assert r["result"] == "1.50"


def test_bridge_requires_gaap_start_and_labels_result_non_gaap():
    r = run_calculation({"formula": "gaap_to_nongaap_bridge", "inputs": {
        "gaap_value": M("710", basis="GAAP", source_id="S1"),
        "adjustments": [M("265", label="SBC", source_id="S1"), M("55", label="amortization", source_id="S1"), M("15", label="restructuring", source_id="S1")]}})
    assert r["status"] == "valid" and r["result"] == "1,045" and r["basis"] == "non_GAAP"
    bad = run_calculation({"formula": "gaap_to_nongaap_bridge", "inputs": {"gaap_value": M("1,045", basis="non_GAAP"), "adjustments": [M("1")]}})
    assert bad["status"] == "invalid"


def test_parse_number_formats():
    assert parse_number("(1,234.5)") == Decimal("-1234.5")
    assert parse_number("$16,525.00") == Decimal("16525.00")
    assert parse_number("− 9") == Decimal("-9")
    with pytest.raises(CalculationError):
        parse_number("n/a")


def test_tool_result_shape_for_prompts():
    r = run_calculation({"formula": "growth_rate", "inputs": {"current": M("4,110", "Q3 FY2025", source_id="S1"),
                                                               "prior": M("3,600", "Q3 FY2024", source_id="S2")}}, calc_id="C3")
    for key in ("calc_id", "tool", "formula", "computation", "inputs", "result", "unit", "source_ids", "status"):
        assert key in r
    assert r["calc_id"] == "C3" and r["source_ids"] == ["S1", "S2"] and r["period"] == "Q3 FY2025"


def test_catalogue_is_complete_and_documented():
    cat = list_formulas()
    assert len(cat) >= 40
    assert all(f["description"] and f["formula"] and f["unit"] for f in cat)
    names = {f["name"] for f in cat}
    for needed in ("growth_rate", "gross_margin", "operating_margin", "ebitda_margin", "net_margin", "fcf_margin",
                   "bps_change", "net_leverage", "working_capital", "gaap_to_nongaap_bridge"):  # spec Part 1 + design s.7
        assert needed in names

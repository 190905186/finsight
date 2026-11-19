"""Deterministic financial calculator for the Finsight copilot.

The language model never computes authoritative numbers (design doc s.7, spec Part 1
"Deterministic Financial Math"). It extracts input figures with their source IDs and asks
for a named formula; this module validates the inputs and computes the result with Decimal
arithmetic, returning a ``tool_result`` the answer prompts cite as ``[C1]``, ``[C2]`` ...

Formulas are hard-coded and documented (no expression evaluation, no model-written maths).
The catalogue comes from:
  * spec Part 1 s.1: YoY/QoQ growth, gross/operating/EBITDA/net/FCF margins, leverage
    ratios, working-capital changes, basis-point deltas, GAAP to Non-GAAP reconciliation;
  * design doc s.7 tool catalogue (growth, margin, FCF, ratios, bps delta, GAAP bridge);
  * the FinanceBench dev questions (turnover, DSO/DIO/DPO/CCC, ROA, payout, multi-year averages ...).

Usage::

    from app.tools.calculator import run_calculation, list_formulas
    result = run_calculation({
        "formula": "growth_rate",
        "inputs": {
            "current": {"value": "4,110", "scale": "millions", "period": "Q3 FY2025", "source_id": "S1"},
            "prior":   {"value": "3,600", "scale": "millions", "period": "Q3 FY2024", "source_id": "S1"},
        },
    }, calc_id="C1")
    # result["result"] == "+14.17", result["unit"] == "percent", result["status"] == "valid"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, getcontext
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

getcontext().prec = 34

SCALES: Dict[str, Decimal] = {
    "units": Decimal(1),
    "thousands": Decimal(1_000),
    "millions": Decimal(1_000_000),
    "billions": Decimal(1_000_000_000),
}
_SCALE_ALIASES = {"unit": "units", "ones": "units", "k": "thousands", "thousand": "thousands", "m": "millions",
                  "mn": "millions", "million": "millions", "bn": "billions", "b": "billions", "billion": "billions"}
BASES = ("GAAP", "non_GAAP")
DAYS_IN_YEAR = Decimal(365)


class CalculationError(ValueError):
    """Invalid inputs; the message is safe to show to the user and the model."""


# --------------------------------------------------------------------------- inputs

def parse_number(raw: Any) -> Decimal:
    """Parse a figure as printed: '$1,234.5', '(1,234)', '-3.7', '14.2%', '− 9' -> Decimal."""
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, (int, float)):
        return Decimal(str(raw))
    text = str(raw).strip().replace("−", "-").replace("–", "-")
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-") or text.startswith("$(") or "(-" in text
    digits = re.sub(r"[^\d.]", "", text)
    if not digits or digits.count(".") > 1 or digits == ".":
        raise CalculationError(f"Not a number: {raw!r}")
    try:
        value = Decimal(digits)
    except InvalidOperation as exc:
        raise CalculationError(f"Not a number: {raw!r}") from exc
    return -value if negative else value


@dataclass(frozen=True)
class Amount:
    """One input figure with its provenance. ``value`` is in ``scale`` units, as printed."""

    value: Decimal
    scale: str = "units"
    currency: Optional[str] = None
    period: Optional[str] = None
    basis: Optional[str] = None
    source_id: Optional[str] = None
    label: Optional[str] = None

    @property
    def base(self) -> Decimal:
        return self.value * SCALES[self.scale]

    @classmethod
    def from_any(cls, raw: Any) -> "Amount":
        if isinstance(raw, Amount):
            return raw
        if isinstance(raw, Mapping):
            scale = str(raw.get("scale") or "units").strip().lower()
            scale = _SCALE_ALIASES.get(scale, scale)
            if scale not in SCALES:
                raise CalculationError(f"Unknown scale {raw.get('scale')!r}; use one of {list(SCALES)}")
            basis = raw.get("basis")
            if basis is not None and basis not in BASES:
                basis = "non_GAAP" if str(basis).lower().replace("-", "_") in ("non_gaap", "nongaap", "adjusted") else "GAAP"
            return cls(parse_number(raw.get("value")), scale, raw.get("currency"), raw.get("period"), basis,
                       raw.get("source_id"), raw.get("label"))
        return cls(parse_number(raw))

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"value": _plain(self.value), "scale": self.scale}
        for key in ("currency", "period", "basis", "source_id", "label"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out


# --------------------------------------------------------------------------- formulas

@dataclass(frozen=True)
class Formula:
    name: str
    category: str
    description: str
    expression: str  # documented definition shown to users and the model
    required: Tuple[str, ...]
    unit: str  # currency | percent | ratio | times | days | bps | percentage_points | per_share | same_as_input
    compute: Callable[["_Ctx"], Tuple[Decimal, str]]
    optional: Tuple[str, ...] = ()
    list_inputs: Tuple[str, ...] = ()  # inputs that take a list of figures
    same_period: bool = False  # all non-"_begin"/non-"prior" inputs must share a period
    note: str = ""


@dataclass
class _Ctx:
    inputs: Dict[str, Amount]
    lists: Dict[str, List[Amount]]
    options: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def v(self, name: str) -> Decimal:
        return self.inputs[name].base

    def has(self, name: str) -> bool:
        return name in self.inputs

    def avg(self, stem: str) -> Tuple[Decimal, str]:
        """Average of <stem>_begin and <stem>_end when both are given, else the ending value."""
        end = self.inputs[f"{stem}_end"]
        if f"{stem}_begin" in self.inputs:
            begin = self.inputs[f"{stem}_begin"]
            return (begin.base + end.base) / 2, f"avg({_fmt(begin)}, {_fmt(end)})"
        return end.base, _fmt(end)


def _div(num: Decimal, den: Decimal, what: str) -> Decimal:
    if den == 0:
        raise CalculationError(f"{what} is zero; the ratio is undefined")
    return num / den


_REGISTRY: Dict[str, Formula] = {}


def _register(*formulas: Formula) -> None:
    for f in formulas:
        if f.name in _REGISTRY:
            raise ValueError(f"duplicate formula {f.name}")
        _REGISTRY[f.name] = f


def _fmt(a: Amount) -> str:
    return _group(a.value) + ({"thousands": "K", "millions": "M", "billions": "B"}.get(a.scale, ""))


def _group(d: Decimal) -> str:
    s = format(d.normalize(), "f") if d == d.to_integral() else format(d, "f")
    neg = s.startswith("-")
    s = s.lstrip("-")
    whole, _, frac = s.partition(".")
    whole = f"{int(whole):,}"
    return ("-" if neg else "") + whole + (f".{frac}" if frac else "")


def _plain(d: Decimal) -> str:
    return format(d.normalize(), "f") if d == d.to_integral() else format(d, "f")


# ---- growth and change ---------------------------------------------------------

def _growth(c: _Ctx):
    cur, pri = c.inputs["current"], c.inputs["prior"]
    if pri.base < 0:
        c.warnings.append("Prior value is negative; percentage growth is not meaningful in sign.")
    r = _div(cur.base - pri.base, pri.base, "Prior value") * 100
    return r, f"({_fmt(cur)} - {_fmt(pri)}) / {_fmt(pri)} x 100"


def _cagr(c: _Ctx):
    end, beg = c.inputs["ending"], c.inputs["beginning"]
    years = c.v("years")
    if beg.base <= 0 or end.base <= 0:
        raise CalculationError("CAGR needs positive beginning and ending values")
    if years <= 0:
        raise CalculationError("years must be positive")
    ratio = end.base / beg.base
    r = (Decimal(str(float(ratio) ** (1 / float(years)))) - 1) * 100
    return r, f"(({_fmt(end)} / {_fmt(beg)}) ^ (1 / {_plain(years)}) - 1) x 100"


def _abs_change(c: _Ctx):
    cur, pri = c.inputs["current"], c.inputs["prior"]
    return (cur.base - pri.base), f"{_fmt(cur)} - {_fmt(pri)}"


def _bps(c: _Ctx):
    cur, pri = c.inputs["current_pct"], c.inputs["prior_pct"]
    return (cur.base - pri.base) * 100, f"({_plain(cur.value)}% - {_plain(pri.value)}%) x 100"


def _pp(c: _Ctx):
    cur, pri = c.inputs["current_pct"], c.inputs["prior_pct"]
    return cur.base - pri.base, f"{_plain(cur.value)}% - {_plain(pri.value)}%"


_register(
    Formula("growth_rate", "growth", "Year-over-year or quarter-over-quarter growth of one metric",
            "(current - prior) / prior x 100", ("current", "prior"), "percent", _growth),
    Formula("cagr", "growth", "Compound annual growth rate", "((ending / beginning) ^ (1 / years) - 1) x 100",
            ("ending", "beginning", "years"), "percent", _cagr),
    Formula("absolute_change", "growth", "Change in a value between two periods (e.g. debt change)",
            "current - prior", ("current", "prior"), "currency", _abs_change),
    Formula("bps_change", "growth", "Change between two percentages in basis points (margin expansion)",
            "(current % - prior %) x 100", ("current_pct", "prior_pct"), "bps", _bps,
            note="Inputs are percentages (e.g. 68.2 for 68.2%). Never confuse with percentage change."),
    Formula("percentage_point_change", "growth", "Change between two percentages in percentage points",
            "current % - prior %", ("current_pct", "prior_pct"), "percentage_points", _pp),
)

# ---- profitability ---------------------------------------------------------------

def _ratio_pct(num: str, den: str, den_label: str):
    def f(c: _Ctx):
        n, d = c.inputs[num], c.inputs[den]
        return _div(n.base, d.base, den_label) * 100, f"{_fmt(n)} / {_fmt(d)} x 100"
    return f


def _gross_profit(c: _Ctx):
    r, k = c.inputs["revenue"], c.inputs["cost_of_revenue"]
    return r.base - k.base, f"{_fmt(r)} - {_fmt(k)}"


def _ebitda(c: _Ctx):
    o, d = c.inputs["operating_income"], c.inputs["depreciation_amortization"]
    return o.base + d.base, f"{_fmt(o)} + {_fmt(d)}"


def _ebitda_margin(c: _Ctx):
    o, d, r = c.inputs["operating_income"], c.inputs["depreciation_amortization"], c.inputs["revenue"]
    return _div(o.base + d.base, r.base, "Revenue") * 100, f"({_fmt(o)} + {_fmt(d)}) / {_fmt(r)} x 100"


def _capex_abs(c: _Ctx) -> Tuple[Decimal, Amount]:
    cap = c.inputs["capex"]
    if cap.base < 0:
        c.warnings.append("Capex was given as a negative cash outflow; its absolute value was used.")
    return abs(cap.base), cap


def _ebitda_less_capex(c: _Ctx):
    o, d = c.inputs["operating_income"], c.inputs["depreciation_amortization"]
    cap_v, cap = _capex_abs(c)
    return o.base + d.base - cap_v, f"{_fmt(o)} + {_fmt(d)} - {_group(abs(cap.value))}{_fmt(cap)[len(_group(cap.value)):]}"


def _fcf(c: _Ctx):
    cfo = c.inputs["operating_cash_flow"]
    cap_v, cap = _capex_abs(c)
    return cfo.base - cap_v, f"{_fmt(cfo)} - {_group(abs(cap.value))}{_fmt(cap)[len(_group(cap.value)):]}"


def _fcf_margin(c: _Ctx):
    cfo, r = c.inputs["operating_cash_flow"], c.inputs["revenue"]
    cap_v, cap = _capex_abs(c)
    return _div(cfo.base - cap_v, r.base, "Revenue") * 100, f"({_fmt(cfo)} - {_group(abs(cap.value))}) / {_fmt(r)} x 100"


def _etr(c: _Ctx):
    t, p = c.inputs["income_tax_expense"], c.inputs["pretax_income"]
    if p.base <= 0:
        c.warnings.append("Pre-tax income is not positive; the effective tax rate is not meaningful.")
    return _div(t.base, p.base, "Pre-tax income") * 100, f"{_fmt(t)} / {_fmt(p)} x 100"


_register(
    Formula("gross_profit", "profitability", "Gross profit", "revenue - cost of revenue",
            ("revenue", "cost_of_revenue"), "currency", _gross_profit, same_period=True),
    Formula("gross_margin", "profitability", "Gross margin", "gross profit / revenue x 100",
            ("gross_profit", "revenue"), "percent", _ratio_pct("gross_profit", "revenue", "Revenue"), same_period=True),
    Formula("operating_margin", "profitability", "Operating margin", "operating income / revenue x 100",
            ("operating_income", "revenue"), "percent", _ratio_pct("operating_income", "revenue", "Revenue"), same_period=True),
    Formula("net_margin", "profitability", "Net profit margin", "net income / revenue x 100",
            ("net_income", "revenue"), "percent", _ratio_pct("net_income", "revenue", "Revenue"), same_period=True),
    Formula("ebitda", "profitability", "EBITDA (unadjusted)", "operating income + depreciation and amortization",
            ("operating_income", "depreciation_amortization"), "currency", _ebitda, same_period=True,
            note="Unadjusted: GAAP operating income plus D&A from the cash flow statement. Adjusted EBITDA is a company-reported Non-GAAP figure: extract it, do not rebuild it."),
    Formula("ebitda_margin", "profitability", "EBITDA margin", "(operating income + D&A) / revenue x 100",
            ("operating_income", "depreciation_amortization", "revenue"), "percent", _ebitda_margin, same_period=True),
    Formula("ebitda_less_capex", "profitability", "EBITDA less capital expenditures",
            "operating income + D&A - capex", ("operating_income", "depreciation_amortization", "capex"),
            "currency", _ebitda_less_capex, same_period=True),
    Formula("free_cash_flow", "cash_flow", "Free cash flow", "operating cash flow - capital expenditures",
            ("operating_cash_flow", "capex"), "currency", _fcf, same_period=True,
            note="Capex may be given as printed (negative outflow); its absolute value is subtracted."),
    Formula("fcf_margin", "cash_flow", "Free cash flow margin", "(operating cash flow - capex) / revenue x 100",
            ("operating_cash_flow", "capex", "revenue"), "percent", _fcf_margin, same_period=True),
    Formula("effective_tax_rate", "profitability", "Effective tax rate", "income tax expense / pre-tax income x 100",
            ("income_tax_expense", "pretax_income"), "percent", _etr, same_period=True),
    Formula("ratio_to_revenue", "profitability", "Any amount as a % of revenue (capex, R&D, SBC, D&A intensity)",
            "amount / revenue x 100", ("amount", "revenue"), "percent", _ratio_pct("amount", "revenue", "Revenue"), same_period=True),
    Formula("share_of_total", "mix", "A part as a % of a total (segment, region or product mix; share of FCF returned)",
            "part / total x 100", ("part", "total"), "percent", _ratio_pct("part", "total", "Total"), same_period=True),
)

# ---- liquidity -------------------------------------------------------------------

def _wc(c: _Ctx):
    a, l = c.inputs["current_assets"], c.inputs["current_liabilities"]
    return a.base - l.base, f"{_fmt(a)} - {_fmt(l)}"


def _simple_ratio(num: str, den: str, den_label: str):
    def f(c: _Ctx):
        n, d = c.inputs[num], c.inputs[den]
        return _div(n.base, d.base, den_label), f"{_fmt(n)} / {_fmt(d)}"
    return f


def _quick(c: _Ctx):
    parts = [c.inputs[k] for k in ("cash_and_equivalents", "short_term_investments", "receivables") if k in c.inputs]
    cl = c.inputs["current_liabilities"]
    num = sum((p.base for p in parts), Decimal(0))
    return _div(num, cl.base, "Current liabilities"), f"({' + '.join(_fmt(p) for p in parts)}) / {_fmt(cl)}"


def _quick_from_ca(c: _Ctx):
    ca, inv, cl = c.inputs["current_assets"], c.inputs["inventories"], c.inputs["current_liabilities"]
    minus = [inv] + ([c.inputs["prepaid_expenses"]] if c.has("prepaid_expenses") else [])
    num = ca.base - sum((m.base for m in minus), Decimal(0))
    return _div(num, cl.base, "Current liabilities"), f"({_fmt(ca)} - {' - '.join(_fmt(m) for m in minus)}) / {_fmt(cl)}"


def _cash_ratio(c: _Ctx):
    parts = [c.inputs["cash_and_equivalents"]] + ([c.inputs["short_term_investments"]] if c.has("short_term_investments") else [])
    cl = c.inputs["current_liabilities"]
    num = sum((p.base for p in parts), Decimal(0))
    return _div(num, cl.base, "Current liabilities"), f"({' + '.join(_fmt(p) for p in parts)}) / {_fmt(cl)}"


_register(
    Formula("working_capital", "liquidity", "Net working capital", "current assets - current liabilities",
            ("current_assets", "current_liabilities"), "currency", _wc, same_period=True),
    Formula("current_ratio", "liquidity", "Current (working capital) ratio", "current assets / current liabilities",
            ("current_assets", "current_liabilities"), "ratio",
            _simple_ratio("current_assets", "current_liabilities", "Current liabilities"), same_period=True),
    Formula("quick_ratio", "liquidity", "Quick ratio (liquid assets)",
            "(cash and equivalents + short-term investments + receivables) / current liabilities",
            ("cash_and_equivalents", "receivables", "current_liabilities"), "ratio", _quick,
            optional=("short_term_investments",), same_period=True),
    Formula("quick_ratio_from_current_assets", "liquidity", "Quick ratio (current assets less inventories)",
            "(current assets - inventories - prepaid expenses) / current liabilities",
            ("current_assets", "inventories", "current_liabilities"), "ratio", _quick_from_ca,
            optional=("prepaid_expenses",), same_period=True),
    Formula("cash_ratio", "liquidity", "Cash ratio", "(cash and equivalents + short-term investments) / current liabilities",
            ("cash_and_equivalents", "current_liabilities"), "ratio", _cash_ratio,
            optional=("short_term_investments",), same_period=True),
    Formula("operating_cash_flow_ratio", "liquidity", "Operating cash flow ratio",
            "operating cash flow / current liabilities", ("operating_cash_flow", "current_liabilities"), "ratio",
            _simple_ratio("operating_cash_flow", "current_liabilities", "Current liabilities"), same_period=True),
)

# ---- leverage and solvency -----------------------------------------------------------

def _net_debt_value(c: _Ctx) -> Tuple[Decimal, str]:
    debt = c.inputs["total_debt"]
    cash = [c.inputs["cash_and_equivalents"]] + ([c.inputs["marketable_securities"]] if c.has("marketable_securities") else [])
    v = debt.base - sum((x.base for x in cash), Decimal(0))
    return v, f"{_fmt(debt)} - {' - '.join(_fmt(x) for x in cash)}"


def _net_debt(c: _Ctx):
    return _net_debt_value(c)


def _net_leverage(c: _Ctx):
    nd, txt = _net_debt_value(c)
    e = c.inputs["ebitda"]
    if e.base <= 0:
        c.warnings.append("EBITDA is not positive; leverage is not meaningful.")
    return _div(nd, e.base, "EBITDA"), f"({txt}) / {_fmt(e)}"


def _interest_cov(c: _Ctx):
    o, i = c.inputs["operating_income"], c.inputs["interest_expense"]
    return _div(o.base, abs(i.base), "Interest expense"), f"{_fmt(o)} / {_group(abs(i.value))}"


_register(
    Formula("debt_to_equity", "leverage", "Debt-to-equity ratio", "total debt / total shareholders' equity",
            ("total_debt", "total_equity"), "ratio", _simple_ratio("total_debt", "total_equity", "Total equity"), same_period=True),
    Formula("debt_to_assets", "leverage", "Debt-to-assets ratio", "total debt / total assets",
            ("total_debt", "total_assets"), "ratio", _simple_ratio("total_debt", "total_assets", "Total assets"), same_period=True),
    Formula("net_debt", "leverage", "Net debt", "total debt - cash and equivalents - marketable securities",
            ("total_debt", "cash_and_equivalents"), "currency", _net_debt, optional=("marketable_securities",), same_period=True),
    Formula("net_leverage", "leverage", "Net leverage ratio", "(total debt - cash - marketable securities) / EBITDA",
            ("total_debt", "cash_and_equivalents", "ebitda"), "times", _net_leverage, optional=("marketable_securities",),
            note="Use the same EBITDA basis the question names (reported Adjusted EBITDA is Non-GAAP; label it)."),
    Formula("gross_leverage", "leverage", "Gross leverage ratio", "total debt / EBITDA", ("total_debt", "ebitda"), "times",
            _simple_ratio("total_debt", "ebitda", "EBITDA")),
    Formula("interest_coverage", "leverage", "Interest coverage ratio", "operating income / interest expense",
            ("operating_income", "interest_expense"), "times", _interest_cov, same_period=True),
)

# ---- efficiency ---------------------------------------------------------------------

def _turnover(num: str, stem: str, label: str):
    def f(c: _Ctx):
        n = c.inputs[num]
        d, dtxt = c.avg(stem)
        return _div(n.base, d, label), f"{_fmt(n)} / {dtxt}"
    return f


def _days(stem: str, den: str, label: str):
    def f(c: _Ctx):
        avg, atxt = c.avg(stem)
        d = c.inputs[den]
        return _div(avg, d.base, label) * DAYS_IN_YEAR, f"365 x {atxt} / {_fmt(d)}"
    return f


def _dpo_value(c: _Ctx) -> Tuple[Decimal, str]:
    ap, aptxt = c.avg("payables")
    cogs = c.inputs["cost_of_revenue"]
    den, dtxt = cogs.base, _fmt(cogs)
    if c.has("inventory_begin") and c.has("inventory_end"):
        b, e = c.inputs["inventory_begin"], c.inputs["inventory_end"]
        den = cogs.base + (e.base - b.base)
        dtxt = f"({_fmt(cogs)} + ({_fmt(e)} - {_fmt(b)}))"
    return _div(ap, den, "Purchases (COGS + change in inventory)") * DAYS_IN_YEAR, f"365 x {aptxt} / {dtxt}"


def _dpo(c: _Ctx):
    return _dpo_value(c)


def _ccc(c: _Ctx):
    inv, _ = c.avg("inventory")
    ar, _ = c.avg("receivables")
    cogs, rev = c.inputs["cost_of_revenue"], c.inputs["revenue"]
    dio = _div(inv, cogs.base, "Cost of revenue") * DAYS_IN_YEAR
    dso = _div(ar, rev.base, "Revenue") * DAYS_IN_YEAR
    dpo, _ = _dpo_value(c)
    q = lambda d: d.quantize(Decimal("0.01"), ROUND_HALF_UP)
    return dio + dso - dpo, f"DIO {q(dio)} + DSO {q(dso)} - DPO {q(dpo)}"


_AVG_NOTE = "Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance."
_register(
    Formula("inventory_turnover", "efficiency", "Inventory turnover", "cost of revenue / average inventory",
            ("cost_of_revenue", "inventory_end"), "times", _turnover("cost_of_revenue", "inventory", "Inventory"),
            optional=("inventory_begin",), note=_AVG_NOTE),
    Formula("asset_turnover", "efficiency", "Total asset turnover", "revenue / average total assets",
            ("revenue", "total_assets_end"), "times", _turnover("revenue", "total_assets", "Total assets"),
            optional=("total_assets_begin",), note=_AVG_NOTE),
    Formula("fixed_asset_turnover", "efficiency", "Fixed asset turnover", "revenue / average net PP&E",
            ("revenue", "ppe_end"), "times", _turnover("revenue", "ppe", "Net PP&E"), optional=("ppe_begin",), note=_AVG_NOTE),
    Formula("days_sales_outstanding", "efficiency", "Days sales outstanding (DSO)", "365 x average receivables / revenue",
            ("receivables_end", "revenue"), "days", _days("receivables", "revenue", "Revenue"),
            optional=("receivables_begin",), note=_AVG_NOTE),
    Formula("days_inventory_outstanding", "efficiency", "Days inventory outstanding (DIO)",
            "365 x average inventory / cost of revenue", ("inventory_end", "cost_of_revenue"), "days",
            _days("inventory", "cost_of_revenue", "Cost of revenue"), optional=("inventory_begin",), note=_AVG_NOTE),
    Formula("days_payables_outstanding", "efficiency", "Days payables outstanding (DPO)",
            "365 x average payables / (cost of revenue + change in inventory)", ("payables_end", "cost_of_revenue"), "days", _dpo,
            optional=("payables_begin", "inventory_begin", "inventory_end"),
            note="With inventory_begin and inventory_end the denominator is purchases (COGS + change in inventory), as FinanceBench defines DPO; without them it is COGS."),
    Formula("cash_conversion_cycle", "efficiency", "Cash conversion cycle (CCC)", "DIO + DSO - DPO",
            ("inventory_end", "receivables_end", "payables_end", "cost_of_revenue", "revenue"), "days", _ccc,
            optional=("inventory_begin", "receivables_begin", "payables_begin"), note=_AVG_NOTE),
)

# ---- returns ------------------------------------------------------------------------

def _return_on(num: str, stem: str, label: str, as_percent: bool):
    def f(c: _Ctx):
        n = c.inputs[num]
        d, dtxt = c.avg(stem)
        r = _div(n.base, d, label)
        return (r * 100, f"{_fmt(n)} / {dtxt} x 100") if as_percent else (r, f"{_fmt(n)} / {dtxt}")
    return f


_register(
    Formula("return_on_assets", "returns", "Return on assets (ROA), as a decimal", "net income / average total assets",
            ("net_income", "total_assets_end"), "ratio", _return_on("net_income", "total_assets", "Total assets", False),
            optional=("total_assets_begin",), note=_AVG_NOTE),
    Formula("return_on_equity", "returns", "Return on equity (ROE)", "net income / average shareholders' equity x 100",
            ("net_income", "equity_end"), "percent", _return_on("net_income", "equity", "Shareholders' equity", True),
            optional=("equity_begin",), note=_AVG_NOTE),
    Formula("return_on_invested_capital", "returns", "Return on invested capital (ROIC)",
            "NOPAT / average invested capital x 100", ("nopat", "invested_capital_end"), "percent",
            _return_on("nopat", "invested_capital", "Invested capital", True), optional=("invested_capital_begin",),
            note="Use management's own definition when the filing gives one; prefer extracting a reported ROIC."),
)

# ---- shareholders ---------------------------------------------------------------------

def _payout(c: _Ctx):
    d, n = c.inputs["dividends"], c.inputs["net_income"]
    if n.base <= 0:
        c.warnings.append("Net income is not positive; the payout ratio is not meaningful.")
    return _div(abs(d.base), n.base, "Net income"), f"{_group(abs(d.value))} / {_fmt(n)}"


def _returns(c: _Ctx):
    b, d = c.inputs["buybacks"], c.inputs["dividends"]
    return abs(b.base) + abs(d.base), f"{_group(abs(b.value))} + {_group(abs(d.value))}"


def _eps(c: _Ctx):
    n, s = c.inputs["net_income"], c.inputs["weighted_average_shares"]
    return _div(n.base, s.base, "Share count"), f"{_fmt(n)} / {_fmt(s)}"


_register(
    Formula("dividend_payout_ratio", "shareholders", "Dividend payout ratio", "dividends paid / net income",
            ("dividends", "net_income"), "ratio", _payout, same_period=True),
    Formula("shareholder_returns", "shareholders", "Cash returned to shareholders", "share repurchases + dividends paid",
            ("buybacks", "dividends"), "currency", _returns, same_period=True),
    Formula("earnings_per_share", "shareholders", "Earnings per share", "net income / weighted-average shares",
            ("net_income", "weighted_average_shares"), "per_share", _eps, same_period=True,
            note="Give net income and shares in their printed scales (e.g. millions and millions)."),
)

# ---- GAAP bridge and generic aggregates ------------------------------------------------

def _bridge(c: _Ctx):
    g = c.inputs["gaap_value"]
    adj = c.lists.get("adjustments") or []
    if not adj:
        raise CalculationError("A GAAP to Non-GAAP bridge needs the disclosed adjustments")
    total = g.base + sum((a.base for a in adj), Decimal(0))
    parts = " + ".join(f"{_fmt(a)}" + (f" ({a.label})" if a.label else "") for a in adj)
    return total, f"{_fmt(g)} + {parts}"


def _average(c: _Ctx):
    vals = c.lists.get("values") or []
    if len(vals) < 2:
        raise CalculationError("An average needs at least two values")
    return sum((v.base for v in vals), Decimal(0)) / len(vals), f"({' + '.join(_fmt(v) for v in vals)}) / {len(vals)}"


def _sum_items(c: _Ctx):
    vals = c.lists.get("items") or []
    if not vals:
        raise CalculationError("No items to add")
    txt = " ".join((("- " if v.value < 0 else "+ ") if i else ("-" if v.value < 0 else "")) + _group(abs(v.value))
                   + ({"thousands": "K", "millions": "M", "billions": "B"}.get(v.scale, "")) for i, v in enumerate(vals))
    return sum((v.base for v in vals), Decimal(0)), txt


_register(
    Formula("gaap_to_nongaap_bridge", "reconciliation", "Non-GAAP figure rebuilt from a GAAP figure and disclosed adjustments",
            "GAAP value + sum of disclosed adjustments", ("gaap_value",), "currency", _bridge, list_inputs=("adjustments",),
            note="Only with adjustments the filing discloses (each with its source); the result is Non-GAAP."),
    Formula("average", "aggregate", "Average of several values (e.g. a 3-year average margin)", "sum of values / count",
            (), "same_as_input", _average, list_inputs=("values",),
            note="Compute each year's figure first (one calculation per year), then average the results."),
    Formula("sum_of_items", "aggregate", "Signed sum of disclosed line items (e.g. a custom working-capital definition, total debt = short-term + long-term)",
            "item1 + item2 - item3 ...", (), "same_as_input", _sum_items, list_inputs=("items",),
            note="Give each item with its printed sign: deductions as negative values."),
)


# --------------------------------------------------------------------------- validation + run

_PRIOR_LIKE = ("prior", "prior_pct", "beginning")


def _validate(f: Formula, inputs: Dict[str, Amount], lists: Dict[str, List[Amount]], warnings: List[str]) -> None:
    missing = [k for k in f.required if k not in inputs]
    missing += [k for k in f.list_inputs if f.name != "gaap_to_nongaap_bridge" and k not in lists]
    if f.name == "gaap_to_nongaap_bridge" and "adjustments" not in lists:
        missing.append("adjustments")
    if missing:
        raise CalculationError(f"Missing inputs for {f.name}: {missing}")
    unknown = [k for k in inputs if k not in f.required + f.optional]
    if unknown:
        raise CalculationError(f"Unknown inputs for {f.name}: {unknown}; expected {list(f.required + f.optional)}")

    all_amounts = list(inputs.values()) + [a for vs in lists.values() for a in vs]
    currencies = {a.currency.upper() for a in all_amounts if a.currency}
    if len(currencies) > 1:
        raise CalculationError(f"Inputs mix currencies {sorted(currencies)}; convert using a disclosed rate first")

    bases = {a.basis for a in all_amounts if a.basis}
    if f.name == "gaap_to_nongaap_bridge":
        if inputs["gaap_value"].basis == "non_GAAP":
            raise CalculationError("The starting value of a GAAP to Non-GAAP bridge must be the GAAP figure")
    elif len(bases) > 1:
        raise CalculationError("Inputs mix GAAP and Non-GAAP figures without a reconciliation bridge; "
                               "compute like-for-like figures separately (spec refusal case 6)")

    if f.same_period:
        periods = {a.period.strip().upper() for k, a in inputs.items()
                   if a.period and not k.endswith("_begin") and k not in _PRIOR_LIKE}
        if len(periods) > 1:
            raise CalculationError(f"{f.name} needs inputs from the same period; got {sorted(periods)}")
    if f.name in ("growth_rate", "absolute_change", "bps_change", "percentage_point_change"):
        cur = inputs.get("current") or inputs.get("current_pct")
        pri = inputs.get("prior") or inputs.get("prior_pct")
        if cur.period and pri.period and cur.period.strip().upper() == pri.period.strip().upper():
            raise CalculationError("Current and prior values are from the same period")
    for k in inputs:
        if k.endswith("_begin"):
            end = inputs.get(k[: -len("_begin")] + "_end")
            if end and end.period and inputs[k].period and end.period.strip().upper() == inputs[k].period.strip().upper():
                warnings.append(f"{k} and its _end value have the same period; check that the opening balance is the prior year-end.")


_ROUNDING = {"percent": "0.01", "ratio": "0.01", "times": "0.01", "days": "0.01", "bps": "1",
             "percentage_points": "0.01", "per_share": "0.01"}


def _display(f: Formula, value: Decimal, inputs: Dict[str, Amount], lists: Dict[str, List[Amount]]) -> Tuple[str, str, Optional[str]]:
    """Rounded display string, unit, and output scale for currency results."""
    unit = f.unit
    out_scale = None
    if unit in ("currency", "same_as_input"):
        first = next(iter(inputs.values()), None) or next((vs[0] for vs in lists.values() if vs), None)
        out_scale = first.scale if first else "units"
        shown = (value / SCALES[out_scale]).quantize(Decimal("0.01"), ROUND_HALF_UP)
        shown_txt = _group(shown.normalize() if shown == shown.to_integral() else shown)
        return shown_txt, ("currency" if unit == "currency" else "as_input"), out_scale
    q = Decimal(_ROUNDING[unit])
    shown = value.quantize(q, ROUND_HALF_UP)
    txt = format(shown, "f")
    if unit in ("percent", "bps", "percentage_points") and f.category == "growth" and shown > 0:
        txt = "+" + txt
    return txt, unit, None


def run_calculation(request: Mapping[str, Any], calc_id: str = "C1") -> Dict[str, Any]:
    """Validate and compute one calculation request. Never raises for bad input: returns status "invalid".

    request = {"formula": "<name>", "inputs": {"<input>": {"value": "1,234", "scale": "millions",
               "period": "FY2024", "basis": "GAAP", "source_id": "S1"}, "<list input>": [ ... ]}}
    """
    name = str(request.get("formula", "")).strip()
    result: Dict[str, Any] = {"calc_id": calc_id, "tool": "calculate", "formula_name": name, "status": "invalid",
                              "errors": [], "warnings": []}
    formula = _REGISTRY.get(name)
    if formula is None:
        result["errors"].append(f"Unknown formula {name!r}. Available: {sorted(_REGISTRY)}")
        return result
    result["formula"] = formula.expression
    raw_inputs = request.get("inputs") or {}
    try:
        inputs: Dict[str, Amount] = {}
        lists: Dict[str, List[Amount]] = {}
        for key, raw in raw_inputs.items():
            if key in formula.list_inputs:
                if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                    raise CalculationError(f"{key} must be a list of figures")
                lists[key] = [Amount.from_any(x) for x in raw]
            else:
                inputs[key] = Amount.from_any(raw)
        warnings: List[str] = []
        _validate(formula, inputs, lists, warnings)
        ctx = _Ctx(inputs, lists, dict(request.get("options") or {}), warnings)
        value, computation = formula.compute(ctx)
        shown, unit, out_scale = _display(formula, value, inputs, lists)
    except CalculationError as exc:
        result["errors"].append(str(exc))
        return result

    amounts = list(inputs.values()) + [a for vs in lists.values() for a in vs]
    suffix = {"percent": "%", "bps": " bps", "percentage_points": " pp", "times": "x", "days": " days"}.get(unit, "")
    if unit in ("currency", "as_input"):
        suffix = {"thousands": "K", "millions": "M", "billions": "B"}.get(out_scale or "", "")
    basis = "non_GAAP" if formula.name == "gaap_to_nongaap_bridge" else next((a.basis for a in amounts if a.basis), None)
    result.update(
        status="valid",
        computation=f"{computation} = {shown}{suffix}",
        inputs={k: a.describe() for k, a in inputs.items()} | {k: [a.describe() for a in vs] for k, vs in lists.items()},
        result=shown,
        result_exact=_plain(value / SCALES[out_scale]) if out_scale else _plain(value),
        unit=unit,
        scale=out_scale,
        currency=next((a.currency for a in amounts if a.currency), None),
        basis=basis,
        period=next((a.period for k, a in inputs.items() if a.period and k not in _PRIOR_LIKE and not k.endswith("_begin")), None),
        source_ids=sorted({a.source_id for a in amounts if a.source_id}),
        warnings=ctx.warnings,
    )
    return result


def list_formulas(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Catalogue for prompts and docs: name, what it computes, inputs, unit, notes."""
    return [
        {"name": f.name, "category": f.category, "description": f.description, "formula": f.expression,
         "required_inputs": list(f.required) + list(f.list_inputs), "optional_inputs": list(f.optional),
         "unit": f.unit, "note": f.note}
        for f in _REGISTRY.values() if category in (None, f.category)
    ]


def formula_names() -> List[str]:
    return sorted(_REGISTRY)

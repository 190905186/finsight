# Deterministic financial calculator

`backend/app/tools/calculator.py` computes every derived figure the copilot reports. The language model never does arithmetic. It extracts input figures (with source IDs) and requests a **named, hard-coded formula**; the calculator validates the inputs and computes with `Decimal` arithmetic, returning a `tool_result` the answer cites as `[C1]`.

This implements spec Part 1 "Deterministic Financial Math" and design doc §7 (tool catalogue).

## How to call it

```python
from app.tools.calculator import run_calculation, list_formulas

run_calculation({
    "formula": "growth_rate",
    "inputs": {
        "current": {"value": "4,110", "scale": "millions", "period": "Q3 FY2025", "source_id": "S1"},
        "prior":   {"value": "3,600", "scale": "millions", "period": "Q3 FY2024", "source_id": "S1"},
    },
}, calc_id="C1")
# -> status "valid", result "+14.17", unit "percent",
#    computation "(4,110M - 3,600M) / 3,600M x 100 = +14.17%", source_ids ["S1"]
```

Each input is `{"value": as printed, "scale": units|thousands|millions|billions, "period", "basis": GAAP|non_GAAP, "currency", "source_id", "label"}`. Values are parsed as printed: `(1,234)` and `$(145)` are negative, and commas and `$` are ignored.

## Validation rules (an invalid request returns `status: "invalid"` with the reason, never a number)

- Missing, unknown or unparseable inputs; unknown formula names (no free-form maths).
- Mixed currencies.
- Mixed GAAP and Non-GAAP inputs (spec refusal case 6), except `gaap_to_nongaap_bridge`, which must start from the GAAP figure and outputs a Non-GAAP result.
- Inputs from different periods where a formula needs one period (margins, ratios), and growth/change between two values from the same period.
- Zero denominators. Warnings (result still returned) for negative bases, non-positive pre-tax income or EBITDA, and opening balances that share the closing period.
- Scales are normalised (billions vs millions); currency results are shown in the first input's scale.

## Formula catalogue

45 formulas. Optional `<name>_begin` inputs switch a formula from the ending balance to the **average** balance (FinanceBench definitions use averages; some analyst definitions use ending balances).

### Growth and change

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `growth_rate` | Year-over-year or quarter-over-quarter growth of one metric | (current - prior) / prior x 100 | current, prior | - | percent |
| `cagr` | Compound annual growth rate | ((ending / beginning) ^ (1 / years) - 1) x 100 | ending, beginning, years | - | percent |
| `absolute_change` | Change in a value between two periods (e.g. debt change) | current - prior | current, prior | - | currency |
| `bps_change` | Change between two percentages in basis points (margin expansion) | (current % - prior %) x 100 | current_pct, prior_pct | - | bps |
| `percentage_point_change` | Change between two percentages in percentage points | current % - prior % | current_pct, prior_pct | - | percentage_points |

- `bps_change`: Inputs are percentages (e.g. 68.2 for 68.2%). Never confuse with percentage change.

### Profitability

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `gross_profit` | Gross profit | revenue - cost of revenue | revenue, cost_of_revenue | - | currency |
| `gross_margin` | Gross margin | gross profit / revenue x 100 | gross_profit, revenue | - | percent |
| `operating_margin` | Operating margin | operating income / revenue x 100 | operating_income, revenue | - | percent |
| `net_margin` | Net profit margin | net income / revenue x 100 | net_income, revenue | - | percent |
| `ebitda` | EBITDA (unadjusted) | operating income + depreciation and amortization | operating_income, depreciation_amortization | - | currency |
| `ebitda_margin` | EBITDA margin | (operating income + D&A) / revenue x 100 | operating_income, depreciation_amortization, revenue | - | percent |
| `ebitda_less_capex` | EBITDA less capital expenditures | operating income + D&A - capex | operating_income, depreciation_amortization, capex | - | currency |
| `effective_tax_rate` | Effective tax rate | income tax expense / pre-tax income x 100 | income_tax_expense, pretax_income | - | percent |
| `ratio_to_revenue` | Any amount as a % of revenue (capex, R&D, SBC, D&A intensity) | amount / revenue x 100 | amount, revenue | - | percent |

- `ebitda`: Unadjusted: GAAP operating income plus D&A from the cash flow statement. Adjusted EBITDA is a company-reported Non-GAAP figure: extract it, do not rebuild it.

### Mix

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `share_of_total` | A part as a % of a total (segment, region or product mix; share of FCF returned) | part / total x 100 | part, total | - | percent |

### Cash flow

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `free_cash_flow` | Free cash flow | operating cash flow - capital expenditures | operating_cash_flow, capex | - | currency |
| `fcf_margin` | Free cash flow margin | (operating cash flow - capex) / revenue x 100 | operating_cash_flow, capex, revenue | - | percent |

- `free_cash_flow`: Capex may be given as printed (negative outflow); its absolute value is subtracted.

### Liquidity

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `working_capital` | Net working capital | current assets - current liabilities | current_assets, current_liabilities | - | currency |
| `current_ratio` | Current (working capital) ratio | current assets / current liabilities | current_assets, current_liabilities | - | ratio |
| `quick_ratio` | Quick ratio (liquid assets) | (cash and equivalents + short-term investments + receivables) / current liabilities | cash_and_equivalents, receivables, current_liabilities | short_term_investments | ratio |
| `quick_ratio_from_current_assets` | Quick ratio (current assets less inventories) | (current assets - inventories - prepaid expenses) / current liabilities | current_assets, inventories, current_liabilities | prepaid_expenses | ratio |
| `cash_ratio` | Cash ratio | (cash and equivalents + short-term investments) / current liabilities | cash_and_equivalents, current_liabilities | short_term_investments | ratio |
| `operating_cash_flow_ratio` | Operating cash flow ratio | operating cash flow / current liabilities | operating_cash_flow, current_liabilities | - | ratio |

### Leverage and solvency

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `debt_to_equity` | Debt-to-equity ratio | total debt / total shareholders' equity | total_debt, total_equity | - | ratio |
| `debt_to_assets` | Debt-to-assets ratio | total debt / total assets | total_debt, total_assets | - | ratio |
| `net_debt` | Net debt | total debt - cash and equivalents - marketable securities | total_debt, cash_and_equivalents | marketable_securities | currency |
| `net_leverage` | Net leverage ratio | (total debt - cash - marketable securities) / EBITDA | total_debt, cash_and_equivalents, ebitda | marketable_securities | times |
| `gross_leverage` | Gross leverage ratio | total debt / EBITDA | total_debt, ebitda | - | times |
| `interest_coverage` | Interest coverage ratio | operating income / interest expense | operating_income, interest_expense | - | times |

- `net_leverage`: Use the same EBITDA basis the question names (reported Adjusted EBITDA is Non-GAAP; label it).

### Efficiency (turnover and days)

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `inventory_turnover` | Inventory turnover | cost of revenue / average inventory | cost_of_revenue, inventory_end | inventory_begin | times |
| `asset_turnover` | Total asset turnover | revenue / average total assets | revenue, total_assets_end | total_assets_begin | times |
| `fixed_asset_turnover` | Fixed asset turnover | revenue / average net PP&E | revenue, ppe_end | ppe_begin | times |
| `days_sales_outstanding` | Days sales outstanding (DSO) | 365 x average receivables / revenue | receivables_end, revenue | receivables_begin | days |
| `days_inventory_outstanding` | Days inventory outstanding (DIO) | 365 x average inventory / cost of revenue | inventory_end, cost_of_revenue | inventory_begin | days |
| `days_payables_outstanding` | Days payables outstanding (DPO) | 365 x average payables / (cost of revenue + change in inventory) | payables_end, cost_of_revenue | payables_begin, inventory_begin, inventory_end | days |
| `cash_conversion_cycle` | Cash conversion cycle (CCC) | DIO + DSO - DPO | inventory_end, receivables_end, payables_end, cost_of_revenue, revenue | inventory_begin, receivables_begin, payables_begin | days |

- `inventory_turnover`: Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance.
- `asset_turnover`: Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance.
- `fixed_asset_turnover`: Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance.
- `days_sales_outstanding`: Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance.
- `days_inventory_outstanding`: Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance.
- `days_payables_outstanding`: With inventory_begin and inventory_end the denominator is purchases (COGS + change in inventory), as FinanceBench defines DPO; without them it is COGS.
- `cash_conversion_cycle`: Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance.

### Returns

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `return_on_assets` | Return on assets (ROA), as a decimal | net income / average total assets | net_income, total_assets_end | total_assets_begin | ratio |
| `return_on_equity` | Return on equity (ROE) | net income / average shareholders' equity x 100 | net_income, equity_end | equity_begin | percent |
| `return_on_invested_capital` | Return on invested capital (ROIC) | NOPAT / average invested capital x 100 | nopat, invested_capital_end | invested_capital_begin | percent |

- `return_on_assets`: Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance.
- `return_on_equity`: Give both <name>_begin and <name>_end to use the average balance; give only <name>_end to use the ending balance.
- `return_on_invested_capital`: Use management's own definition when the filing gives one; prefer extracting a reported ROIC.

### Shareholder returns

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `dividend_payout_ratio` | Dividend payout ratio | dividends paid / net income | dividends, net_income | - | ratio |
| `shareholder_returns` | Cash returned to shareholders | share repurchases + dividends paid | buybacks, dividends | - | currency |
| `earnings_per_share` | Earnings per share | net income / weighted-average shares | net_income, weighted_average_shares | - | per_share |

- `earnings_per_share`: Give net income and shares in their printed scales (e.g. millions and millions).

### GAAP / Non-GAAP

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `gaap_to_nongaap_bridge` | Non-GAAP figure rebuilt from a GAAP figure and disclosed adjustments | GAAP value + sum of disclosed adjustments | gaap_value, adjustments | - | currency |

- `gaap_to_nongaap_bridge`: Only with adjustments the filing discloses (each with its source); the result is Non-GAAP.

### Aggregates

| Formula | Computes | Definition | Required inputs | Optional | Unit |
|---|---|---|---|---|---|
| `average` | Average of several values (e.g. a 3-year average margin) | sum of values / count | values | - | same_as_input |
| `sum_of_items` | Signed sum of disclosed line items (e.g. a custom working-capital definition, total debt = short-term + long-term) | item1 + item2 - item3 ... | items | - | same_as_input |

- `average`: Compute each year's figure first (one calculation per year), then average the results.
- `sum_of_items`: Give each item with its printed sign: deductions as negative values.

## Where the formulas come from

- **Spec Part 1 §1:** YoY/QoQ growth; gross, operating, EBITDA, net and FCF margins; leverage ratios; working-capital changes; basis-point deltas; GAAP to Non-GAAP reconciliation.
- **Design doc §7:** `calculate_growth`, `calculate_margin`, `calculate_fcf`, `calculate_ratio` (current ratio, leverage, DSO, interest coverage), `calculate_bps_delta`, `reconcile_gaap_nongaap`.
- **FinanceBench dev questions** (`data/evaluation/financebench_dev.jsonl`): turnover ratios, DSO/DIO/DPO/CCC with FinanceBench's average-balance definitions, operating cash flow ratio, ROA, payout ratio, EBITDA less capex, multi-year averages, segment shares, custom signed sums.

No held-out FinanceBench question and no question from the project's 75-query golden set was used to choose or tune the formulas; coverage of those sets was only counted afterwards.

## Verification

`backend/tests/test_calculator.py` reproduces 27 FinanceBench dev answers from inputs read off their evidence pages (e.g. Walmart DPO 42.69, General Mills CCC −3.70, PepsiCo EBITDA−capex 9,068, AES ROA −0.02) and the spec's rubric example ((4,200 − 3,500) / 3,500 = +20.0%), plus every validation rule.


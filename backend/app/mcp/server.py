import json
import re
from datetime import date
from functools import lru_cache
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from app.config import settings

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TREASURY_DEBT_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v2/accounting/od/debt_to_penny"
)

FACT_TAGS = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "total_liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditures": ("PaymentsToAcquirePropertyPlantAndEquipment",),
}
COMPANY_TICKER_ALIASES = {
    "3M": "MMM",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "WALMART": "WMT",
    "ADOBE": "ADBE",
    "COCACOLA": "KO",
    "JOHNSONJOHNSON": "JNJ",
    "PFIZER": "PFE",
}

mcp = FastMCP("Finsight Public Finance", host="0.0.0.0", port=8001)


def _sec_headers() -> dict[str, str]:
    user_agent = settings.sec_user_agent.strip()
    if "@" not in user_agent:
        raise ToolError(
            "SEC data requires SEC_USER_AGENT with a monitored contact email in .env."
        )
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


@lru_cache(maxsize=1)
def _ticker_directory() -> dict[str, dict[str, str]]:
    response = httpx.get(
        SEC_TICKERS_URL, headers=_sec_headers(), timeout=20.0
    )
    response.raise_for_status()
    payload = response.json()
    entries = payload.values() if isinstance(payload, dict) else []
    directory: dict[str, dict[str, str]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        title = str(item.get("title", "")).strip()
        cik = str(item.get("cik_str", "")).strip()
        if ticker and title and cik.isdigit():
            directory[ticker] = {"ticker": ticker, "title": title, "cik": cik.zfill(10)}
    return directory


def _resolve_company(company: str) -> dict[str, str]:
    normalized = re.sub(r"[^A-Z0-9]", "", company.upper())
    directory = _ticker_directory()
    normalized = COMPANY_TICKER_ALIASES.get(normalized, normalized)
    if normalized in directory:
        return directory[normalized]
    matches = [
        item
        for item in directory.values()
        if normalized
        and normalized in re.sub(r"[^A-Z0-9]", "", item["title"].upper())
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ToolError(f"No SEC issuer matched {company!r}. Try its ticker symbol.")
    raise ToolError(
        f"Company name {company!r} is ambiguous. Use a unique SEC ticker symbol."
    )


def _select_fact(
    facts: dict[str, Any], tags: tuple[str, ...], fiscal_year: int
) -> dict[str, Any] | None:
    for tag in tags:
        fact = facts.get(tag)
        if not isinstance(fact, dict):
            continue
        units = fact.get("units")
        if not isinstance(units, dict):
            continue
        observations = units.get("USD", [])
        if not isinstance(observations, list):
            continue

        candidates: list[dict[str, Any]] = []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            if observation.get("fy") != fiscal_year:
                continue
            end = observation.get("end")
            form = observation.get("form")
            if (
                not isinstance(end, str)
                or form not in {"10-K", "10-Q", "20-F", "40-F"}
            ):
                continue
            start = observation.get("start")
            if start:
                try:
                    duration_days = date.fromisoformat(end).toordinal() - date.fromisoformat(
                        start
                    ).toordinal()
                except ValueError:
                    continue
                if duration_days < 300:
                    continue
            candidates.append(observation)
        if not candidates:
            continue
        observation = max(
            candidates,
            key=lambda row: (
                row.get("form") in {"10-K", "20-F", "40-F"},
                str(row.get("filed", "")),
            ),
        )
        return {
            "tag": tag,
            "label": fact.get("label", tag),
            "value": observation["val"],
            "unit": "USD",
            "period_start": observation.get("start"),
            "period_end": observation["end"],
            "form": observation.get("form"),
            "filed": observation.get("filed"),
            "accession": observation.get("accn"),
        }
    return None


@mcp.tool()
def get_sec_financial_facts(company: str, fiscal_year: int) -> str:
    """Fetch tagged, annual USD facts for an issuer from SEC XBRL Company Facts."""
    if fiscal_year < 1990 or fiscal_year > 2100:
        raise ToolError("fiscal_year must be between 1990 and 2100.")
    try:
        issuer = _resolve_company(company)
        url = SEC_COMPANYFACTS_URL.format(cik=issuer["cik"])
        response = httpx.get(url, headers=_sec_headers(), timeout=30.0)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise ToolError(f"SEC data request failed: {exc.__class__.__name__}.") from exc

    facts_root = payload.get("facts", {})
    us_gaap = facts_root.get("us-gaap", {}) if isinstance(facts_root, dict) else {}
    if not isinstance(us_gaap, dict):
        raise ToolError("SEC response did not contain us-gaap facts.")

    selected = {
        metric: fact
        for metric, tags in FACT_TAGS.items()
        if (fact := _select_fact(us_gaap, tags, fiscal_year)) is not None
    }
    if not selected:
        raise ToolError(
            f"No annual SEC facts found for {issuer['ticker']} in FY{fiscal_year}."
        )
    for fact in selected.values():
        fact["source_url"] = url

    return json.dumps(
        {
            "company": issuer["title"],
            "ticker": issuer["ticker"],
            "cik": issuer["cik"],
            "fiscal_year": fiscal_year,
            "facts": selected,
            "source_url": url,
            "data_as_of": max(
                (fact["filed"] for fact in selected.values() if fact["filed"]),
                default=None,
            ),
        }
    )


@mcp.tool()
def get_treasury_debt_context(fiscal_year: int) -> str:
    """Fetch the latest U.S. public debt observation during a requested calendar year."""
    if fiscal_year < 1993 or fiscal_year > 2100:
        raise ToolError("fiscal_year must be between 1993 and 2100.")
    params = {
        "filter": (
            f"record_date:gte:{fiscal_year}-01-01,"
            f"record_date:lte:{fiscal_year}-12-31"
        ),
        "sort": "-record_date",
        "page[size]": "1",
        "format": "json",
    }
    try:
        response = httpx.get(TREASURY_DEBT_URL, params=params, timeout=20.0)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise ToolError(
            f"Treasury Fiscal Data request failed: {exc.__class__.__name__}."
        ) from exc

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ToolError(f"No Treasury debt observation found for {fiscal_year}.")
    return json.dumps(
        {
            "series": "U.S. total public debt outstanding",
            "unit": "USD",
            "observation": rows[0],
            "source_url": TREASURY_DEBT_URL,
        }
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

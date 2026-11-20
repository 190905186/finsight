from __future__ import annotations

import json

import pytest

from app.mcp import server
from app.services import public_finance_mcp


def test_walmart_company_name_resolves_to_wmt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = {
        "WMT": {"ticker": "WMT", "title": "Walmart Inc.", "cik": "0000104169"},
        "WM": {"ticker": "WM", "title": "Waste Management, Inc.", "cik": "0000823768"},
    }
    monkeypatch.setattr(server, "_ticker_directory", lambda: directory)

    issuer = server._resolve_company("walmart")

    assert issuer["ticker"] == "WMT"


def test_direct_rest_result_is_marked_as_fallback_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "get_treasury_debt_context",
        lambda fiscal_year: json.dumps({"fiscal_year": fiscal_year}),
    )

    result = public_finance_mcp.call_public_finance_rest_tool(
        "get_treasury_debt_context", {"fiscal_year": 2024}
    )

    assert result == {
        "fiscal_year": 2024,
        "_transport": "direct_rest_fallback",
    }


def test_mcp_transport_failure_uses_direct_rest_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(*args: object, **kwargs: object) -> dict[str, object]:
        raise public_finance_mcp.PublicFinanceMCPError("MCP unavailable")

    monkeypatch.setattr(public_finance_mcp, "_call_tool", unavailable)
    monkeypatch.setattr(
        public_finance_mcp,
        "call_public_finance_rest_tool",
        lambda name, arguments: {
            "fiscal_year": arguments["fiscal_year"],
            "_transport": "direct_rest_fallback",
        },
    )

    result = public_finance_mcp.call_public_finance_tool(
        "get_treasury_debt_context", {"fiscal_year": 2024}
    )

    assert result == {
        "fiscal_year": 2024,
        "_transport": "direct_rest_fallback",
    }


def test_rest_fallback_error_keeps_mcp_failure_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(*args: object, **kwargs: object) -> dict[str, object]:
        raise public_finance_mcp.PublicFinanceMCPError("MCP unavailable")

    def rest_unavailable(name: str, arguments: dict[str, object]) -> dict[str, object]:
        raise public_finance_mcp.PublicFinanceRESTError("REST unavailable")

    monkeypatch.setattr(public_finance_mcp, "_call_tool", unavailable)
    monkeypatch.setattr(
        public_finance_mcp, "call_public_finance_rest_tool", rest_unavailable
    )

    with pytest.raises(public_finance_mcp.PublicFinanceMCPError) as error:
        public_finance_mcp.call_public_finance_tool(
            "get_treasury_debt_context", {"fiscal_year": 2024}
        )

    assert "MCP unavailable" in str(error.value)
    assert "REST unavailable" in str(error.value)

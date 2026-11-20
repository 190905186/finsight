from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError

from app.config import settings


class PublicFinanceMCPError(RuntimeError):
    pass


class PublicFinanceRESTError(RuntimeError):
    pass


async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        async with streamablehttp_client(settings.public_finance_mcp_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
    except (McpError, httpx.HTTPError, TimeoutError) as exc:
        raise PublicFinanceMCPError(
            f"Public finance MCP server is unavailable for tool {name}."
        ) from exc

    if result.isError:
        message = next(
            (
                item.text
                for item in result.content
                if getattr(item, "type", None) == "text"
            ),
            f"MCP tool {name} returned an error.",
        )
        raise PublicFinanceMCPError(message)

    for item in result.content:
        if getattr(item, "type", None) == "text":
            payload = json.loads(item.text)
            if isinstance(payload, dict):
                return payload
    raise PublicFinanceMCPError(
        f"MCP tool {name} returned no structured JSON result."
    )


def call_public_finance_rest_tool(
    name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    from mcp.server.fastmcp.exceptions import ToolError

    from app.mcp.server import get_sec_financial_facts, get_treasury_debt_context

    handlers = {
        "get_sec_financial_facts": get_sec_financial_facts,
        "get_treasury_debt_context": get_treasury_debt_context,
    }
    handler = handlers.get(name)
    if handler is None:
        raise PublicFinanceRESTError(
            f"No direct REST fallback is configured for tool {name}."
        )
    try:
        payload = json.loads(handler(**arguments))
    except (ToolError, httpx.HTTPError, TimeoutError, ValueError) as exc:
        raise PublicFinanceRESTError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise PublicFinanceRESTError(
            f"Direct REST fallback for {name} returned no JSON object."
        )
    return {**payload, "_transport": "direct_rest_fallback"}


def call_public_finance_tool(
    name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    try:
        return asyncio.run(_call_tool(name, arguments))
    except PublicFinanceMCPError as mcp_error:
        try:
            return call_public_finance_rest_tool(name, arguments)
        except PublicFinanceRESTError as rest_error:
            raise PublicFinanceMCPError(
                f"{mcp_error} Direct public REST fallback failed: {rest_error}"
            ) from rest_error

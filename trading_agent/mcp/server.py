"""Stdio MCP server for the trading agent (skill 48 §3).

Implements the subset of the MCP protocol needed for Claude Code to
discover, list, and invoke the read-only tools registered in
``trading_agent.mcp`` (see ``READONLY_TOOLS`` for the closed set).

Transport: JSON-RPC 2.0 over stdio (newline-delimited framing). This
keeps the server dependency-free — the package doesn't ship an MCP
library, and Claude Code speaks stdio-JSON-RPC natively.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable, Dict

from trading_agent.mcp import READONLY_TOOLS, SERVER_READ_ONLY
from trading_agent.mcp.tools import positions as _positions
from trading_agent.mcp.tools import strategy as _strategy
from trading_agent.mcp.tools import market as _market


# Wiring table: MCP tool name → callable. Every entry MUST correspond
# to a name in READONLY_TOOLS; the conformance test enforces both
# directions.
_HANDLERS: Dict[str, Callable[..., Any]] = {
    "list_positions":     _positions.list_positions,
    "get_position":       _positions.get_position,
    "list_recent_trades": _positions.list_recent_trades,
    "get_journal_summary": _positions.get_journal_summary,
    "get_preset":         _strategy.get_preset,
    "get_risk_report":    _strategy.get_risk_report,
    "run_scan":           _strategy.run_scan,
    "score_candidate":    _strategy.score_candidate,
    "get_chain":          _market.get_chain,
    "get_quote":          _market.get_quote,
    "get_market_status":  _market.get_market_status,
    "get_fundamentals":   _market.get_fundamentals,
    "get_recent_alerts":  _market.get_recent_alerts,
}


def _tool_descriptors() -> list[dict]:
    """Return MCP tool descriptors — one dict per tool."""
    # Minimal input schema: pass through kwargs; the underlying
    # callable does its own validation. Claude Code accepts an open
    # object schema; concrete arg validation is done at call time.
    open_schema = {"type": "object", "additionalProperties": True}
    descs = []
    for name in READONLY_TOOLS:
        fn = _HANDLERS.get(name)
        descs.append({
            "name": name,
            "description": (fn.__doc__ or "").strip().split("\n")[0] if fn else "",
            "inputSchema": open_schema,
        })
    return descs


def _handle_request(req: dict) -> dict:
    """Dispatch one JSON-RPC request. Returns a response dict."""
    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "trading-agent-mcp",
                    "version": "0.1.0",
                    "readOnly": SERVER_READ_ONLY,
                },
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": _tool_descriptors()},
        }

    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = _HANDLERS.get(name)
        if fn is None:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601,
                          "message": f"tool not found: {name}"},
            }
        try:
            result = fn(**args)
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32000,
                          "message": f"{type(exc).__name__}: {exc}"},
            }
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, default=str),
                }],
                "isError": False,
            },
        }

    return {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"unknown method: {method}"},
    }


def serve_stdio() -> int:
    """Run the JSON-RPC-over-stdio loop until EOF."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            resp = _handle_request(req)
        except Exception:  # noqa: BLE001
            resp = {
                "jsonrpc": "2.0", "id": req.get("id"),
                "error": {"code": -32603,
                          "message": traceback.format_exc(limit=3)},
            }
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0

"""Claude Code MCP surface for the trading agent.

**Read-only by design and by construction.** This package must NEVER
import ``trading_agent.executor``, ``submit_order``, ``place_order``,
``TradingClient``, or any module that writes to ``pending_orders/``.
The read/write boundary is enforced by
``tests/conformance/test_skill_48_mcp_readonly.py``.

See ``docs/skills/48_claude_code_mcp_surface.md``.
"""
from __future__ import annotations

SERVER_READ_ONLY: bool = True
"""Sentinel constant grep'd by the conformance test — do not remove."""

# The complete, closed set of tools this MCP server exposes. Every
# subagent .md under .claude/agents/*.md that lists a `tools:` field
# must reference names from this set (verified by
# ``tests/conformance/test_subagent_tool_allowlists.py``). Adding a new
# tool means (a) writing it under ``tools/``, (b) registering it here,
# (c) updating skill 48 §2, (d) re-running the conformance suite.
READONLY_TOOLS: tuple[str, ...] = (
    "list_positions",
    "get_position",
    "list_recent_trades",
    "get_journal_summary",
    "get_preset",
    "get_risk_report",
    "run_scan",
    "score_candidate",
    "get_chain",
    "get_quote",
    "get_market_status",
    "get_fundamentals",
    "get_recent_alerts",
)

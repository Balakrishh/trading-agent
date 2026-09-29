"""CLI entry: ``python -m trading_agent.mcp``.

Runs the stdio MCP server so Claude Code can auto-discover the
trading-agent tool surface. See skill 48 §3.
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from trading_agent.mcp import READONLY_TOOLS
from trading_agent.mcp.server import serve_stdio, _tool_descriptors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trading_agent.mcp",
        description="Trading-agent MCP surface — Claude Code integration "
                    "(read-only; skill 48).",
    )
    parser.add_argument(
        "--list-tools", action="store_true",
        help="Print the tool registry as JSON and exit (capability envelope).",
    )
    args = parser.parse_args(argv)

    # Same .env the data server reads via load_config(), so the Bearer key
    # (SCHWAB_API_SERVER_KEY) and SCHWAB_API_BASE_URL have one source of
    # truth. Pre-2026-09-29 the MCP only saw Claude Code's inherited env,
    # so a key added to .env never reached it (every call "unreachable").
    # Like load_config(), an already-exported env var still wins.
    load_dotenv()

    if args.list_tools:
        json.dump({
            "readonly_tools": list(READONLY_TOOLS),
            "descriptors": _tool_descriptors(),
        }, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    return serve_stdio()


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())

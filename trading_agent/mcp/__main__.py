"""CLI entry: ``python -m trading_agent.mcp``.

Runs the stdio MCP server so Claude Code can auto-discover the
trading-agent tool surface. See skill 48 §3.
"""
from __future__ import annotations

import argparse
import json
import sys

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

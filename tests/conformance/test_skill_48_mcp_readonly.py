"""Conformance for skill 48 — the Claude Code MCP surface stays read-only.

Enforces four invariants:

1. No file under ``trading_agent/mcp/`` imports the executor or any
   order-placement primitive (AST inspection — comments, docstrings,
   and prose that merely mention forbidden tokens are fine).
2. The ``SERVER_READ_ONLY = True`` sentinel exists on
   ``trading_agent.mcp``.
3. Every name in ``READONLY_TOOLS`` has a handler wired in
   ``server._HANDLERS``, and vice versa (symmetric registry).
4. Every handler is callable.
"""
from __future__ import annotations

import ast
from pathlib import Path


_MCP_DIR = Path(__file__).resolve().parents[2] / "trading_agent" / "mcp"

# Tokens that, if imported anywhere under trading_agent/mcp/, break
# the read-only contract. Matched against the full dotted name of
# ``import X`` / ``from X import ...`` statements.
_FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "trading_agent.executor",
    "trading_agent.executor_schwab",
    "alpaca.trading",
    "pending_orders",
)

# Attribute-level forbids: if any of these names is imported *from*
# any module, the file is rejected. Catches
# ``from trading_agent.something import submit_order``.
_FORBIDDEN_NAMES: tuple[str, ...] = (
    "submit_order",
    "place_order",
    "TradingClient",
)


def _iter_py(dirpath: Path):
    for p in dirpath.rglob("*.py"):
        yield p


def test_skill_48_mcp_readonly_no_forbidden_imports():
    """AST-walk every .py under trading_agent/mcp/ and reject any
    import that would let this surface place orders or touch the
    executor.
    """
    offenders: list[tuple[str, str]] = []
    for py in _iter_py(_MCP_DIR):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in _FORBIDDEN_IMPORTS:
                        if alias.name == bad or alias.name.startswith(bad + "."):
                            offenders.append((str(py), f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for bad in _FORBIDDEN_IMPORTS:
                    if mod == bad or mod.startswith(bad + "."):
                        offenders.append((str(py), f"from {mod} import ..."))
                for alias in node.names:
                    if alias.name in _FORBIDDEN_NAMES:
                        offenders.append(
                            (str(py), f"from {mod} import {alias.name}"))
    assert not offenders, (
        "trading_agent/mcp/ must be read-only; forbidden imports:\n  "
        + "\n  ".join(f"{p}: {s}" for p, s in offenders)
    )


def test_skill_48_server_read_only_sentinel_exists():
    """The SERVER_READ_ONLY sentinel is grep'd by CI and by skill 48
    §3.1. Removing it silently would defeat the invariant.
    """
    from trading_agent.mcp import SERVER_READ_ONLY
    assert SERVER_READ_ONLY is True


def test_skill_48_handler_registry_symmetric():
    """Every READONLY_TOOLS entry has a handler; every handler is
    named in READONLY_TOOLS. Drift = CI fail.
    """
    from trading_agent.mcp import READONLY_TOOLS
    from trading_agent.mcp.server import _HANDLERS
    assert set(_HANDLERS) == set(READONLY_TOOLS), (
        f"registry drift: "
        f"in handlers only={set(_HANDLERS) - set(READONLY_TOOLS)}, "
        f"in READONLY_TOOLS only={set(READONLY_TOOLS) - set(_HANDLERS)}"
    )


def test_skill_48_every_handler_is_callable():
    from trading_agent.mcp.server import _HANDLERS
    for name, fn in _HANDLERS.items():
        assert callable(fn), f"handler {name} is not callable"


def test_skill_48_mcp_json_wires_module():
    """The .mcp.json at repo root wires ``python -m trading_agent.mcp``.
    Refactoring the entry point without updating .mcp.json would
    silently break Claude Code auto-discovery.
    """
    import json
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / ".mcp.json").read_text())
    entry = cfg["mcpServers"]["trading-agent"]
    assert entry["command"] == "python"
    assert entry["args"] == ["-m", "trading_agent.mcp"]

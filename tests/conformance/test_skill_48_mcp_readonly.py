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


def test_skill_48_numeric_args_coerced_from_strings():
    """Regression: MCP clients (including Claude Code) sometimes JSON-
    serialize numeric arguments as strings. Handlers whose validation
    used ``<= 0`` on the raw arg would fail with TypeError. Coerce
    with int() and re-validate.
    """
    from trading_agent.mcp.tools.positions import list_recent_trades
    from trading_agent.mcp.tools.market import get_recent_alerts
    # Both should accept string ints without raising TypeError
    try:
        list_recent_trades(days="7")
    except (ValueError, Exception) as exc:  # ValueError is OK for other reasons
        assert "must be an integer" not in str(exc), (
            f"days='7' should coerce, got: {exc}")
    try:
        get_recent_alerts(hours="24")
    except (ValueError, Exception) as exc:
        assert "must be an integer" not in str(exc), (
            f"hours='24' should coerce, got: {exc}")
    # Non-numeric strings should raise a clear ValueError
    import pytest
    with pytest.raises(ValueError, match="must be an integer"):
        list_recent_trades(days="banana")
    with pytest.raises(ValueError, match="must be an integer"):
        get_recent_alerts(hours="banana")


def test_skill_48_fundamentals_tool_registered():
    """Skill 47 fundamentals extension: get_fundamentals must appear in
    the closed READONLY_TOOLS set AND be wired in _HANDLERS.
    """
    from trading_agent.mcp import READONLY_TOOLS
    from trading_agent.mcp.server import _HANDLERS
    assert "get_fundamentals" in READONLY_TOOLS
    assert "get_fundamentals" in _HANDLERS
    assert callable(_HANDLERS["get_fundamentals"])


def test_skill_48_fundamentals_field_mapping_uses_schwab_native_keys(monkeypatch):
    """Regression 2026-09-29: three separate mapping bugs that made
    AAPL look non-dividend-paying with zero volume:

    1. Dividend + volume fields used TDA short names → None values.
    2. ``market_cap_float`` name suggested dollars but Schwab returns
       a share COUNT under ``marketCapFloat`` — must expose as
       ``float_shares`` to avoid misleading downstream consumers.
    3. ``operatingMarginTTM`` sometimes aliases the net-margin value,
       so we prefer ``operatingMargin`` (no suffix) when both are
       present.
    """
    monkeypatch.delenv("SCHWAB_API_FUNDAMENTALS_INCLUDE_RAW", raising=False)
    from trading_agent.market_data_schwab import SchwabMarketDataProvider

    fake_body = {
        "instruments": [
            {
                "symbol": "AAPL", "cusip": "037833100",
                "description": "APPLE INC", "exchange": "NASDAQ",
                "assetType": "EQUITY",
                "fundamental": {
                    "peRatio": 31.2, "pegRatio": 2.1, "pbRatio": 58.4,
                    "epsTTM": 6.05, "marketCap": 3_100_000_000_000,
                    "marketCapFloat": 14_600_000_000,     # share count, not $
                    "sharesOutstanding": 15_200_000_000,
                    "dividendYield": 0.0044, "dividendAmount": 0.96,
                    "dividendDate": "2026-08-15",
                    "nextDividendPayDate": "2026-11-15",
                    "dividendPayAmount": 0.24,
                    "beta": 1.24, "high52": 237.30, "low52": 164.10,
                    # New long-form volume shape Schwab is returning today.
                    "avg1DayVolume": 52_000_000,
                    "avg10DaysVolume": 55_000_000,
                    "avg3MonthVolume": 57_500_000,
                    "returnOnEquity": 0.28, "returnOnAssets": 0.20,
                    "bookValuePerShare": 3.10,
                    "grossMarginTTM": 0.44,
                    "netProfitMarginTTM": 0.276,
                    # Both present; the no-suffix field is the real one.
                    "operatingMargin": 0.312,
                    "operatingMarginTTM": 0.276,           # aliases net
                    "shortIntToFloat": 0.008,
                    "epsChangePercentTTM": 12.4,
                },
            }
        ]
    }
    prov = SchwabMarketDataProvider.__new__(SchwabMarketDataProvider)
    prov._get = lambda path, params=None, timeout=None: fake_body   # noqa: SLF001,ARG005
    out = prov.fetch_fundamentals("AAPL")

    # Volume — the headline bug the user reported today.
    assert out["vol_avg_1d"]  == 52_000_000
    assert out["vol_avg_10d"] == 55_000_000
    assert out["vol_avg_3mo"] == 57_500_000

    # market_cap_float removed; float_shares now carries the share count.
    assert "market_cap_float" not in out
    assert out["float_shares"] == 14_600_000_000

    # operatingMargin (no suffix) preferred over the aliased ...TTM form.
    assert out["operating_margin_ttm"] == 0.312
    assert out["net_profit_margin_ttm"] == 0.276
    assert out["operating_margin_ttm"] != out["net_profit_margin_ttm"]

    # Rest of the headline mapping still lands.
    assert out["dividend_yield"] == 0.0044
    assert out["dividend_amount"] == 0.96
    assert out["ticker"] == "AAPL"
    assert out["pe_ratio"] == 31.2
    assert out["market_cap"] == 3_100_000_000_000
    assert out["beta"] == 1.24
    assert out["high_52w"] == 237.30
    assert out["roe"] == 0.28
    assert out["gross_margin_ttm"] == 0.44
    assert out["short_int_to_float"] == 0.008

    # _raw_fundamental should NOT appear when the env flag is off.
    assert "_raw_fundamental" not in out


def test_skill_48_fundamentals_raw_echo_toggle(monkeypatch):
    """SCHWAB_API_FUNDAMENTALS_INCLUDE_RAW=true adds the upstream
    payload so an operator can diagnose a field that looks wrong.
    """
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    monkeypatch.setenv("SCHWAB_API_FUNDAMENTALS_INCLUDE_RAW", "true")
    fake_body = {"instruments": [{
        "symbol": "AAPL",
        "fundamental": {"peRatio": 30.0, "someWeirdKey": 999},
    }]}
    prov = SchwabMarketDataProvider.__new__(SchwabMarketDataProvider)
    prov._get = lambda path, params=None, timeout=None: fake_body   # noqa: SLF001,ARG005
    out = prov.fetch_fundamentals("AAPL")
    assert "_raw_fundamental" in out
    assert out["_raw_fundamental"]["someWeirdKey"] == 999


def test_skill_48_fundamentals_legacy_tda_shortnames_still_map():
    """Belt: if Schwab ever revives TDA-legacy short-name payloads,
    dividend/volume fields still populate. Same pick-first-non-null
    fallback the source code uses.
    """
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    fake_body = {"instruments": [{
        "symbol": "T",
        "fundamental": {
            "divYield": 0.065, "divAmount": 1.11, "divDate": "2026-07-10",
        },
    }]}
    prov = SchwabMarketDataProvider.__new__(SchwabMarketDataProvider)
    prov._get = lambda path, params=None, timeout=None: fake_body   # noqa: SLF001,ARG005
    out = prov.fetch_fundamentals("T")
    assert out["dividend_yield"] == 0.065
    assert out["dividend_amount"] == 1.11
    assert out["dividend_date"] == "2026-07-10"


def test_skill_48_fundamentals_returns_unavailable_when_server_down(monkeypatch):
    """When SCHWAB_API_BASE_URL is unset, the tool returns a structured
    unavailable row rather than raising — matches the get_quote pattern.
    """
    monkeypatch.delenv("SCHWAB_API_BASE_URL", raising=False)
    from trading_agent.mcp.tools.market import get_fundamentals
    r = get_fundamentals("AAPL")
    assert r["ticker"] == "AAPL"
    assert r.get("source") == "unavailable"
    assert r.get("fundamentals") == {}


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


def test_skill_48_positions_span_days_and_carry_fields(tmp_path, monkeypatch):
    """Regression 2026-09-29: list_positions only returned today's opens
    (a spread opened yesterday vanished from /portfolio) and mapped
    width/opened_at/pnl to attributes that don't exist (always null);
    list_recent_trades ignored ``days``."""
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    import trading_agent.journal_reader as jr
    from trading_agent.mcp.tools.positions import (
        list_positions, get_position, list_recent_trades)

    et = ZoneInfo("US/Eastern")
    now = datetime.now(et)
    exp = (now.date() + timedelta(days=20)).isoformat()
    rows = [
        {"timestamp": (now - timedelta(days=3)).isoformat(), "ticker": "SPY",
         "action": "submitted",
         "raw_signal": {"strategy": "Iron Condor", "expiration": exp,
                        "net_credit": 2.1, "spread_width": 10.0,
                        "max_loss": 790.0, "run_id": "r9"}},
        {"timestamp": (now - timedelta(days=4)).isoformat(), "ticker": "QQQ",
         "action": "closed",
         "raw_signal": {"strategy": "Bull Put Spread", "expiration": exp,
                        "net_unrealized_pl": 55.0, "exit_signal": "profit_target",
                        "exit_reason": "50%", "fill_status": "complete"}},
    ]
    p = tmp_path / "live.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    real = jr.JournalReader
    monkeypatch.setattr(jr, "JournalReader", lambda *a, **k: real(str(p)))

    pos = list_positions()["open_positions"]
    assert len(pos) == 1 and pos[0]["ticker"] == "SPY"
    assert pos[0]["width"] == 10.0 and pos[0]["opened_at"] and pos[0]["status"] == "open"
    assert get_position("spy")["found"] is True

    trades = list_recent_trades(days=7)
    assert [c["ticker"] for c in trades["closes"]] == ["QQQ"]
    assert trades["closes"][0]["pnl"] == 55.0
    assert trades["realized_pl_window"] == 55.0
    assert trades["closes_today"] == []


def test_skill_48_mcp_entry_loads_dotenv(monkeypatch):
    """The MCP must read .env like the data server (load_config), or a
    SCHWAB_API_SERVER_KEY set only in .env never reaches it."""
    import trading_agent.mcp.__main__ as entry
    calls = []
    monkeypatch.setattr(entry, "load_dotenv", lambda *a, **k: calls.append(1))
    assert entry.main(["--list-tools"]) == 0
    assert calls, "python -m trading_agent.mcp must call load_dotenv()"

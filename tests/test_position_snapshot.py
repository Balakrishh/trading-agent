"""Skill 50 — monitor valuation snapshot read by /triage (2026-10-05)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from trading_agent.position_monitor import ExitSignal, SpreadPosition
from trading_agent.position_snapshot import build_snapshot, read_snapshot, write_snapshot


def _spread():
    s = SpreadPosition(underlying="SPY", strategy_name="Iron Condor",
                       legs=[SimpleNamespace(symbol="P752", mark_source="mid"),
                             SimpleNamespace(symbol="P751", mark_source="mid")],
                       original_credit=0.49, max_loss=51.0, spread_width=1.0,
                       net_unrealized_pl=-112.0, expiration="2026-10-23",
                       short_strikes=[752.0], contracts_open=16)
    s.exit_signal, s.exit_reason = ExitSignal.HOLD, "ok"
    return s


def test_snapshot_carries_the_monitor_valuation():
    snap = build_snapshot([_spread()], {"SPY": 767.52})
    row = snap["positions"][0]
    assert row["mid_pl"] == -112.0 and row["exit_signal"] == "hold"
    assert row["mark_sources"] == ["mid"] and row["underlying_price"] == 767.52
    assert row["contracts"] == 16 and row["legs"] == ["P752", "P751"]


def test_write_read_roundtrip_reports_age(tmp_path):
    fp = tmp_path / "v.json"
    snap = build_snapshot([_spread()], {})
    snap["as_of_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat()
    write_snapshot(snap, fp)
    got = read_snapshot(fp)
    assert got["positions"][0]["ticker"] == "SPY" and 85 <= got["age_seconds"] <= 120
    assert not (tmp_path / "v.json.tmp").exists()


def test_read_missing_or_corrupt_returns_none(tmp_path):
    assert read_snapshot(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_snapshot(bad) is None


def test_mcp_tool_reads_snapshot(tmp_path, monkeypatch):
    import trading_agent.position_snapshot as ps
    from trading_agent.mcp.tools.positions import get_position_valuations
    fp = tmp_path / "v.json"
    monkeypatch.setattr(ps, "SNAPSHOT_PATH", fp)
    monkeypatch.setattr(ps.read_snapshot, "__defaults__", (fp,))
    assert get_position_valuations()["source"] == "unavailable"
    write_snapshot(build_snapshot([_spread()], {}), fp)
    out = get_position_valuations()
    assert out["source"] == "agent_monitor" and out["positions"][0]["mid_pl"] == -112.0


# ── flat book (2026-11-12) ──────────────────────────────────────────────

def _monitor_self(positions, fp, monkeypatch):
    import trading_agent.position_snapshot as ps
    from trading_agent.agent import TradingAgent
    monkeypatch.setattr(ps, "SNAPSHOT_PATH", fp)
    monkeypatch.setattr(ps.write_snapshot, "__defaults__", (fp,))
    return SimpleNamespace(
        _reconcile_wheel_expirations=lambda: None,
        position_monitor=SimpleNamespace(fetch_open_positions=lambda: positions),
        _apply_profit_trail=lambda spreads: spreads,
        _publish_valuations=TradingAgent._publish_valuations,
        journal_kb=SimpleNamespace(log_warning=lambda **k: None)), TradingAgent


def test_flat_book_clears_the_previous_snapshot(tmp_path, monkeypatch):
    """A closed position must not linger as broker_only once the broker is flat."""
    from trading_agent.book_view import merge_rows
    fp = tmp_path / "v.json"
    write_snapshot(build_snapshot([_spread()], {}), fp)
    me, agent_cls = _monitor_self([], fp, monkeypatch)
    out = agent_cls._stage_monitor(me, 28_000.0)
    snap = read_snapshot(fp)
    assert out["fetch_failed"] is False and snap["positions"] == []
    assert merge_rows([], snap) == []


def test_failed_fetch_keeps_the_last_snapshot(tmp_path, monkeypatch):
    """Unknown broker state (RPC failed) must not erase what we last saw."""
    fp = tmp_path / "v.json"
    write_snapshot(build_snapshot([_spread()], {}), fp)
    me, agent_cls = _monitor_self(None, fp, monkeypatch)
    assert agent_cls._stage_monitor(me, 28_000.0)["fetch_failed"] is True
    assert read_snapshot(fp)["positions"][0]["ticker"] == "SPY"

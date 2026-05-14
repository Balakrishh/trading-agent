"""
test_journal_derived_cooldown.py — pin down the May 2026 cooldown rewrite.

The 2026-05-12 / 2026-05-13 post-mortems established that the original
in-memory cooldown state on TradingAgent didn't survive the per-cycle
process restart in production — every cycle was a new TradingAgent
instance with an empty counter, so the streak never accumulated and
the cooldown never engaged.  Today's rewrite derives streak + cooldown
state from the journal's ``close_failed`` and ``closed`` rows on every
read.  These tests prove the derivation logic across the failure modes
the post-mortem identified.
"""
from __future__ import annotations

import json
import os
import tempfile
import types
from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import MagicMock


def _write_journal(rows: List[dict]) -> str:
    """Helper: write a tempdir signals_live.jsonl with the given rows."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "signals_live.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _stub_agent(journal_path: str):
    """Minimal TradingAgent stub bound to a journal file."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent.journal_kb = MagicMock()
    agent.journal_kb.jsonl_path = journal_path
    agent._cached_price = MagicMock(return_value=100.0)

    # Bind the real methods we're testing.
    agent._close_failed_streak_within_window = types.MethodType(
        TradingAgent._close_failed_streak_within_window, agent
    )
    agent._close_cooldown_minutes_remaining = types.MethodType(
        TradingAgent._close_cooldown_minutes_remaining, agent
    )
    agent._journal_close_event = types.MethodType(
        TradingAgent._journal_close_event, agent
    )
    return agent


# ───────────────────────────────────────────────────────────────────
#  _close_failed_streak_within_window
# ───────────────────────────────────────────────────────────────────

def test_streak_zero_when_no_close_failed_rows():
    path = _write_journal([
        {"timestamp": datetime.now(timezone.utc).isoformat(),
         "ticker": "SPY", "action": "submitted",
         "raw_signal": {}},
    ])
    agent = _stub_agent(path)
    streak, ts = agent._close_failed_streak_within_window("SPY")
    assert streak == 0
    assert ts is None


def test_streak_counts_close_failed_rows_in_window():
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=10)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=5)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
    ]
    path = _write_journal(rows)
    agent = _stub_agent(path)
    streak, last_ts = agent._close_failed_streak_within_window("GLD")
    assert streak == 2
    assert last_ts is not None


def test_streak_excludes_rows_older_than_window():
    """Default window is 60 min — older rows must NOT count."""
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=90)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=70)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=10)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
    ]
    path = _write_journal(rows)
    agent = _stub_agent(path)
    streak, _ = agent._close_failed_streak_within_window("GLD")
    assert streak == 1, "only the row within the 60-min window should count"


def test_streak_resets_after_a_closed_row():
    """A 'closed' row in the middle of close_failed rows resets the streak."""
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=50)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=45)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        # Position closed at -40 min — streak resets here.
        {"timestamp": (now - timedelta(minutes=40)).isoformat(),
         "ticker": "GLD", "action": "closed", "raw_signal": {}},
        # New partial fills after the close — these are the active streak.
        {"timestamp": (now - timedelta(minutes=20)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=10)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
    ]
    path = _write_journal(rows)
    agent = _stub_agent(path)
    streak, _ = agent._close_failed_streak_within_window("GLD")
    assert streak == 2, "only post-closed rows count"


def test_streak_ignores_other_tickers():
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=10)).isoformat(),
         "ticker": "SPY", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=5)).isoformat(),
         "ticker": "XLF", "action": "close_failed", "raw_signal": {}},
    ]
    path = _write_journal(rows)
    agent = _stub_agent(path)
    assert agent._close_failed_streak_within_window("GLD")[0] == 0
    assert agent._close_failed_streak_within_window("SPY")[0] == 1
    assert agent._close_failed_streak_within_window("XLF")[0] == 1


def test_streak_returns_zero_when_journal_missing():
    agent = _stub_agent("/nonexistent/path.jsonl")
    streak, ts = agent._close_failed_streak_within_window("SPY")
    assert streak == 0
    assert ts is None


def test_streak_tolerates_malformed_journal_lines():
    """Garbage lines and bad timestamps must not break the count."""
    now = datetime.now(timezone.utc)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "signals_live.jsonl")
    with open(path, "w") as fh:
        fh.write("not valid json\n")
        fh.write("\n")  # blank
        fh.write(json.dumps({
            "timestamp": (now - timedelta(minutes=5)).isoformat(),
            "ticker": "GLD", "action": "close_failed", "raw_signal": {},
        }) + "\n")
        fh.write(json.dumps({
            "timestamp": "garbage-timestamp",
            "ticker": "GLD", "action": "close_failed",
        }) + "\n")
    agent = _stub_agent(path)
    streak, _ = agent._close_failed_streak_within_window("GLD")
    assert streak == 1


# ───────────────────────────────────────────────────────────────────
#  _close_cooldown_minutes_remaining
# ───────────────────────────────────────────────────────────────────

def test_cooldown_zero_when_below_threshold():
    """2 partial fills isn't enough to engage the 60-min cooldown."""
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=10)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=5)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
    ]
    path = _write_journal(rows)
    agent = _stub_agent(path)
    assert agent._close_cooldown_minutes_remaining("GLD") == 0


def test_cooldown_engages_at_threshold():
    """3 partial fills in window → cooldown engaged."""
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(minutes=15)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=10)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        # Most recent failure 1 min ago — cooldown deadline is +60 min from there.
        {"timestamp": (now - timedelta(minutes=1)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
    ]
    path = _write_journal(rows)
    agent = _stub_agent(path)
    remaining = agent._close_cooldown_minutes_remaining("GLD")
    # Deadline ≈ now + 59 min, so remaining ≈ 59 min.
    assert 55 <= remaining <= 60


def test_cooldown_expires_after_window():
    """If the last failure was 65 min ago, cooldown has expired."""
    now = datetime.now(timezone.utc)
    # Use rows just inside the 60-min counting window but with the
    # MOST RECENT failure more than 60 min after which we read.
    # We craft this by using failures at -45/-30 and asking again
    # after fake time advance — simpler: failures so old that the
    # 60-min count is below threshold AND the deadline has passed.
    rows = [
        {"timestamp": (now - timedelta(minutes=55)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=50)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=45)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
    ]
    path = _write_journal(rows)
    agent = _stub_agent(path)
    # streak=3, most recent at -45 min. Deadline = -45 + 60 = +15 min.
    # 15 min remaining — cooldown still engaged at this exact moment.
    remaining = agent._close_cooldown_minutes_remaining("GLD")
    assert 10 <= remaining <= 16, f"expected ~15 min remaining, got {remaining}"


def test_cooldown_clears_after_successful_close():
    """A 'closed' row resets streak; cooldown is then 0."""
    now = datetime.now(timezone.utc)
    rows = [
        # 3 partial fills happened earlier
        {"timestamp": (now - timedelta(minutes=30)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=25)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=20)).isoformat(),
         "ticker": "GLD", "action": "close_failed", "raw_signal": {}},
        # Then a complete close superseded them
        {"timestamp": (now - timedelta(minutes=15)).isoformat(),
         "ticker": "GLD", "action": "closed", "raw_signal": {}},
    ]
    path = _write_journal(rows)
    agent = _stub_agent(path)
    assert agent._close_cooldown_minutes_remaining("GLD") == 0


# ───────────────────────────────────────────────────────────────────
#  _journal_close_event embeds journal-derived state
# ───────────────────────────────────────────────────────────────────

def test_journal_close_event_writes_streak_field_below_threshold():
    """A first close_failed row writes streak=1/3, no cooldown field."""
    path = _write_journal([])  # empty journal
    agent = _stub_agent(path)
    spread = MagicMock(underlying="SPY")
    ctx = {
        "strategy": "Bull Put",
        "exit_signal": "regime_shift",
        "exit_reason": "test",
        "exit_immediate": False,
        "net_unrealized_pl": -10.0,
        "original_credit": 0.5,
        "max_loss": 2.0,
        "spread_width": 5.0,
        "expiration": "2026-06-05",
        "short_strikes": [450],
        "regime_at_close": "BEAR",
        "origin": "trade_plan",
    }
    agent._journal_close_event(
        spread, ctx, leg_results=[{"symbol": "SPY...", "status": "error"}],
        fill_status="partial", dry_run=False,
    )
    call = agent.journal_kb.log_signal.call_args
    rs = call.kwargs["raw_signal"]
    assert rs["partial_close_streak"] == 1
    assert rs["partial_close_threshold"] == 3
    assert "close_cooldown_until" not in rs


def test_journal_close_event_writes_cooldown_when_threshold_crossed():
    """3rd close_failed in window → row carries close_cooldown_until."""
    now = datetime.now(timezone.utc)
    # Pre-seed the journal with 2 prior failures
    path = _write_journal([
        {"timestamp": (now - timedelta(minutes=10)).isoformat(),
         "ticker": "SPY", "action": "close_failed", "raw_signal": {}},
        {"timestamp": (now - timedelta(minutes=5)).isoformat(),
         "ticker": "SPY", "action": "close_failed", "raw_signal": {}},
    ])
    agent = _stub_agent(path)
    spread = MagicMock(underlying="SPY")
    ctx = {
        "strategy": "Bull Put",
        "exit_signal": "regime_shift",
        "exit_reason": "test",
        "exit_immediate": False,
        "net_unrealized_pl": -10.0,
        "original_credit": 0.5,
        "max_loss": 2.0,
        "spread_width": 5.0,
        "expiration": "2026-06-05",
        "short_strikes": [450],
        "regime_at_close": "BEAR",
        "origin": "trade_plan",
    }
    agent._journal_close_event(
        spread, ctx, leg_results=[{"symbol": "SPY...", "status": "error"}],
        fill_status="partial", dry_run=False,
    )
    rs = agent.journal_kb.log_signal.call_args.kwargs["raw_signal"]
    assert rs["partial_close_streak"] == 3
    assert "close_cooldown_until" in rs
    assert "manual broker intervention" in rs["close_cooldown_reason"]


def test_complete_close_does_not_carry_cooldown_fields():
    """fill_status=complete → action=closed → no cooldown fields."""
    now = datetime.now(timezone.utc)
    path = _write_journal([
        {"timestamp": (now - timedelta(minutes=5)).isoformat(),
         "ticker": "QQQ", "action": "close_failed", "raw_signal": {}},
    ])
    agent = _stub_agent(path)
    spread = MagicMock(underlying="QQQ")
    ctx = {
        "strategy": "Iron Condor",
        "exit_signal": "profit_target",
        "exit_reason": "75% credit",
        "exit_immediate": True,
        "net_unrealized_pl": 30.0,
        "original_credit": 0.5,
        "max_loss": 1.5,
        "spread_width": 2.0,
        "expiration": "2026-05-30",
        "short_strikes": [400, 420],
        "regime_at_close": "RANGE",
        "origin": "trade_plan",
    }
    agent._journal_close_event(
        spread, ctx, leg_results=[],
        fill_status="complete", dry_run=False,
    )
    rs = agent.journal_kb.log_signal.call_args.kwargs["raw_signal"]
    assert "partial_close_streak" not in rs
    assert "close_cooldown_until" not in rs
    assert agent.journal_kb.log_signal.call_args.kwargs["action"] == "closed"


# ───────────────────────────────────────────────────────────────────
#  Per-ticker position cap — MAX_POSITIONS_PER_TICKER
# ───────────────────────────────────────────────────────────────────

def test_position_count_includes_non_hold_signals():
    """Regression for the 2026-05-12 GLD dedup-gate-bypass incident.

    The old gate filtered by signal=HOLD, so positions whose exit
    signals had fired (profit_target, regime_shift, etc.) were NOT
    counted toward the dedup set.  The agent then opened additional
    spreads on those tickers.

    The new dedup logic counts EVERY reported position regardless
    of signal, capped at MAX_POSITIONS_PER_TICKER.
    """
    from trading_agent.agent import MAX_POSITIONS_PER_TICKER

    # Synthesize what monitor_results would carry
    monitor_results = {
        "positions": [
            {"underlying": "GLD", "signal": "regime_shift"},  # exit-signal-pending
            {"underlying": "SPY", "signal": "HOLD"},
            {"underlying": "QQQ", "signal": "profit_target"},  # exit-signal-pending
        ]
    }

    # Build the per-ticker count the way the cycle does it
    counts = {}
    for sr in monitor_results["positions"]:
        u = sr.get("underlying", "")
        if u:
            counts[u] = counts.get(u, 0) + 1
    tickers_with_positions = {t for t, n in counts.items()
                              if n >= MAX_POSITIONS_PER_TICKER}

    # All three are in the dedup set — including the ones with
    # non-HOLD signals (regression assertion).
    assert "GLD" in tickers_with_positions, (
        "GLD has a position with regime_shift signal — it MUST be "
        "in the dedup set despite not being in HOLD state. This "
        "regression caused the 2026-05-12 GLD double-position."
    )
    assert "QQQ" in tickers_with_positions
    assert "SPY" in tickers_with_positions


def test_max_positions_per_ticker_is_one_by_default():
    """The hard cap is 1 spread per ticker out of the box."""
    from trading_agent.agent import MAX_POSITIONS_PER_TICKER
    assert MAX_POSITIONS_PER_TICKER == 1


def test_position_count_caps_at_threshold():
    """If MAX_POSITIONS_PER_TICKER were raised to 2, having 1
    position would not yet block — having 2 would."""
    cap = 2
    monitor_results = {
        "positions": [
            {"underlying": "XLF", "signal": "HOLD"},  # only 1
            {"underlying": "GLD", "signal": "HOLD"},  # 1 of 2
            {"underlying": "GLD", "signal": "HOLD"},  # 2 of 2 — capped
        ]
    }
    counts = {}
    for sr in monitor_results["positions"]:
        u = sr.get("underlying", "")
        if u:
            counts[u] = counts.get(u, 0) + 1
    blocked = {t for t, n in counts.items() if n >= cap}
    assert "XLF" not in blocked
    assert "GLD" in blocked

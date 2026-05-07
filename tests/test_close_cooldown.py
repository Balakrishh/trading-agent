"""
test_close_cooldown.py — pin down the partial-close cooldown +
PDT same-day suppression added 2026-05-06 after the SPY zombie
incident.

The agent's close loop must:

1.  Tag partial-fill journal rows with ``action="close_failed"``,
    not ``action="closed"`` — the position is still open on the
    broker, so "Closed Today" lying about it caused operator panic.
2.  Park a ticker into a 60-min cooldown after
    ``PARTIAL_CLOSE_COOLDOWN_THRESHOLD`` (=3) consecutive partial
    fills.  Cleared on a successful complete close.
3.  Suppress same-day REGIME_SHIFT exits on PDT-restricted accounts
    (< $25K equity).  Real-risk exits (STRIKE_PROXIMITY, HARD_STOP,
    DTE_SAFETY) still fire because they're worth the PDT hit.

These tests don't construct a real ``TradingAgent`` (it pulls in
config, broker, LLM, etc).  They use a stub object and exercise the
helper methods directly via ``__get__``.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import types
from datetime import datetime, timedelta, timezone
from typing import Dict
from unittest.mock import MagicMock


def _stub_agent():
    """Minimal stub on which the cooldown helpers can run."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent._partial_close_count = {}
    agent._close_cooldown_until = {}

    # Bind real helper methods so we exercise production code.
    agent._record_partial_close = types.MethodType(
        TradingAgent._record_partial_close, agent
    )
    agent._clear_close_cooldown = types.MethodType(
        TradingAgent._clear_close_cooldown, agent
    )
    agent._close_cooldown_minutes_remaining = types.MethodType(
        TradingAgent._close_cooldown_minutes_remaining, agent
    )
    agent._tickers_opened_today = types.MethodType(
        TradingAgent._tickers_opened_today, agent
    )
    return agent


# ───────────────────────────────────────────────────────────────────
#  Cooldown counter + threshold
# ───────────────────────────────────────────────────────────────────

def test_first_partial_close_does_not_trigger_cooldown():
    agent = _stub_agent()
    agent._record_partial_close("SPY")
    assert agent._partial_close_count["SPY"] == 1
    assert "SPY" not in agent._close_cooldown_until
    assert agent._close_cooldown_minutes_remaining("SPY") == 0


def test_threshold_partial_closes_engages_cooldown():
    """3 consecutive partial fills → cooldown is set."""
    agent = _stub_agent()
    agent._record_partial_close("SPY")
    agent._record_partial_close("SPY")
    agent._record_partial_close("SPY")
    assert agent._partial_close_count["SPY"] == 3
    assert "SPY" in agent._close_cooldown_until
    # Should be ~60 minutes
    remaining = agent._close_cooldown_minutes_remaining("SPY")
    assert 55 <= remaining <= 60


def test_complete_close_clears_cooldown_state():
    agent = _stub_agent()
    agent._record_partial_close("SPY")
    agent._record_partial_close("SPY")
    agent._record_partial_close("SPY")
    assert "SPY" in agent._close_cooldown_until

    agent._clear_close_cooldown("SPY")
    assert "SPY" not in agent._partial_close_count
    assert "SPY" not in agent._close_cooldown_until
    assert agent._close_cooldown_minutes_remaining("SPY") == 0


def test_clear_close_cooldown_idempotent_no_state():
    """Clearing a ticker we've never seen must not raise."""
    agent = _stub_agent()
    # Should not raise, should be a no-op.
    agent._clear_close_cooldown("AAPL")
    assert agent._partial_close_count == {}
    assert agent._close_cooldown_until == {}


def test_cooldown_minutes_remaining_purges_on_expiry():
    """When the cooldown's ``until`` is in the past, both entries clear."""
    agent = _stub_agent()
    agent._partial_close_count["XYZ"] = 5
    agent._close_cooldown_until["XYZ"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    assert agent._close_cooldown_minutes_remaining("XYZ") == 0
    # Both entries should now be purged so the dicts don't grow forever.
    assert "XYZ" not in agent._partial_close_count
    assert "XYZ" not in agent._close_cooldown_until


def test_cooldown_isolated_per_ticker():
    """SPY's cooldown must not affect QQQ's."""
    agent = _stub_agent()
    for _ in range(3):
        agent._record_partial_close("SPY")
    agent._record_partial_close("QQQ")
    assert agent._close_cooldown_minutes_remaining("SPY") > 0
    assert agent._close_cooldown_minutes_remaining("QQQ") == 0


# ───────────────────────────────────────────────────────────────────
#  PDT same-day-open detection
# ───────────────────────────────────────────────────────────────────

def _write_journal(rows):
    """Helper: write a tempdir journal_kb-shaped file. Returns dir path."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "signals_live.jsonl")
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return tmpdir, path


def test_tickers_opened_today_finds_submitted_rows():
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    tmpdir, path = _write_journal([
        {"timestamp": today.isoformat(), "ticker": "SPY",
         "action": "submitted"},
        {"timestamp": today.isoformat(), "ticker": "QQQ",
         "action": "submitted"},
        # Same ticker submitted yesterday — must be excluded.
        {"timestamp": yesterday.isoformat(), "ticker": "AAPL",
         "action": "submitted"},
        # Today's row but action != submitted — must be excluded.
        {"timestamp": today.isoformat(), "ticker": "TSLA",
         "action": "skipped_existing"},
    ])

    agent = _stub_agent()
    agent.journal_kb = MagicMock()
    agent.journal_kb.jsonl_path = path

    tickers = agent._tickers_opened_today()
    assert tickers == {"SPY", "QQQ"}


def test_tickers_opened_today_empty_when_journal_missing():
    agent = _stub_agent()
    agent.journal_kb = MagicMock()
    agent.journal_kb.jsonl_path = "/nonexistent/path.jsonl"
    assert agent._tickers_opened_today() == set()


def test_tickers_opened_today_tolerates_malformed_lines():
    today = datetime.now(timezone.utc)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "signals_live.jsonl")
    with open(path, "w") as fh:
        fh.write("not valid json\n")
        fh.write(json.dumps({
            "timestamp": today.isoformat(),
            "ticker": "SPY", "action": "submitted",
        }) + "\n")
        fh.write("\n")  # blank line
        fh.write(json.dumps({
            "timestamp": "garbage",
            "ticker": "BAD", "action": "submitted",
        }) + "\n")

    agent = _stub_agent()
    agent.journal_kb = MagicMock()
    agent.journal_kb.jsonl_path = path
    # Only the well-formed row makes it through.
    assert agent._tickers_opened_today() == {"SPY"}


# ───────────────────────────────────────────────────────────────────
#  _journal_close_event action mapping
# ───────────────────────────────────────────────────────────────────

def test_journal_close_event_uses_closed_action_on_complete_fill():
    """Complete fill → action='closed' so 'Closed Today' counts it."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent.journal_kb = MagicMock()
    agent._cached_price = MagicMock(return_value=100.0)
    agent._journal_close_event = types.MethodType(
        TradingAgent._journal_close_event, agent
    )

    spread = MagicMock(underlying="SPY")
    ctx = {
        "strategy": "Bear Call",
        "exit_signal": "profit_target",
        "exit_reason": "75% of credit captured",
        "exit_immediate": True,
        "net_unrealized_pl": 25.50,
        "original_credit": 1.00,
        "max_loss": 4.00,
        "spread_width": 5.0,
        "expiration": "2026-05-30",
        "short_strikes": [550],
        "regime_at_close": "BULL",
        "origin": "trade_plan",
    }
    agent._journal_close_event(
        spread, ctx, leg_results=[], fill_status="complete", dry_run=False,
    )
    call = agent.journal_kb.log_signal.call_args
    assert call.kwargs["action"] == "closed"
    assert "closed:" in call.kwargs["notes"]


def test_journal_close_event_uses_close_failed_action_on_partial_fill():
    """Partial fill → action='close_failed' so it's counted separately."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent.journal_kb = MagicMock()
    agent._cached_price = MagicMock(return_value=100.0)
    # Cooldown surface (added 2026-05-06) reads these dicts to embed
    # streak / cooldown_until in the journal row.
    agent._partial_close_count = {"SPY": 1}
    agent._close_cooldown_until = {}
    agent._journal_close_event = types.MethodType(
        TradingAgent._journal_close_event, agent
    )

    spread = MagicMock(underlying="SPY")
    ctx = {
        "strategy": "Bear Call",
        "exit_signal": "regime_shift",
        "exit_reason": "regime flipped",
        "exit_immediate": False,
        "net_unrealized_pl": -5.00,
        "original_credit": 1.00,
        "max_loss": 4.00,
        "spread_width": 5.0,
        "expiration": "2026-05-30",
        "short_strikes": [550],
        "regime_at_close": "BEAR",
        "origin": "trade_plan",
    }
    leg_results = [
        {"symbol": "SPY260530C00550000", "status": "rejected"},
        {"symbol": "SPY260530C00555000", "status": "closed"},
    ]
    agent._journal_close_event(
        spread, ctx, leg_results=leg_results,
        fill_status="partial", dry_run=False,
    )
    call = agent.journal_kb.log_signal.call_args
    assert call.kwargs["action"] == "close_failed"
    assert "close_failed:" in call.kwargs["notes"]
    # Failed leg surfaced in notes for fast triage.
    assert "SPY260530C00550000" in call.kwargs["notes"]


def test_journal_close_event_dry_run_uses_closed_action():
    """Dry run is a sentinel for a complete-by-design close → 'closed'."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent.journal_kb = MagicMock()
    agent._cached_price = MagicMock(return_value=100.0)
    agent._journal_close_event = types.MethodType(
        TradingAgent._journal_close_event, agent
    )

    spread = MagicMock(underlying="QQQ")
    ctx = {
        "strategy": "Iron Condor",
        "exit_signal": "hard_stop",
        "exit_reason": "loss > 3x credit",
        "exit_immediate": True,
        "net_unrealized_pl": -45.00,
        "original_credit": 0.50,
        "max_loss": 1.50,
        "spread_width": 2.0,
        "expiration": "2026-05-30",
        "short_strikes": [400, 420],
        "regime_at_close": "BEAR",
        "origin": "trade_plan",
    }
    agent._journal_close_event(
        spread, ctx, leg_results=[], fill_status="dry_run", dry_run=True,
    )
    call = agent.journal_kb.log_signal.call_args
    assert call.kwargs["action"] == "closed"

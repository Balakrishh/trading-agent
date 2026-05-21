"""
test_close_cooldown.py — Tests for the close-failure orchestration.

History
-------
Originally tested the in-memory cooldown counters (`_partial_close_count`,
`_close_cooldown_until` instance dicts).  After the 2026-05-13 rewrite,
those dicts are gone — cooldown state is derived from the journal on
every read because the per-cycle process-restart deployment model made
in-memory state useless.  The cooldown-counter tests have moved to
`tests/test_journal_derived_cooldown.py`.

What stays here:

* PDT same-day-open detection (``_tickers_opened_today``) — still uses
  the journal as source of truth, still relevant.
* ``_journal_close_event`` action-mapping tests — ``"closed"`` for a
  complete fill, ``"close_failed"`` for a partial.  The cooldown fields
  embedded inside the row are covered by
  `test_journal_derived_cooldown.py`; here we only assert the action
  enum mapping.
"""
from __future__ import annotations

import json
import os
import tempfile
import types
from datetime import datetime, timedelta, timezone
from typing import Tuple
from unittest.mock import MagicMock


def _write_journal(rows):
    """Helper: write a tempdir signals_live.jsonl with the given rows.

    Returns (tmpdir, jsonl_path).  Caller is responsible for cleanup,
    or just lets the OS clean up at process exit.
    """
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "signals_live.jsonl")
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return tmpdir, path


def _stub_agent(journal_path: str = ""):
    """Minimal TradingAgent stub bound to a (possibly empty) journal."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent.journal_kb = MagicMock()
    agent.journal_kb.jsonl_path = journal_path
    agent._cached_price = MagicMock(return_value=100.0)
    # ``telegram`` is an instance attribute (set in TradingAgent.__init__)
    # and MagicMock(spec=TradingAgent) only mirrors class attributes —
    # _journal_close_event dereferences self.telegram.notify_close_cooldown
    # at the cooldown-engaged branch, so we install a permissive mock
    # to keep that path crash-free. Added 2026-05-21 with the operator-
    # alert hooks (skill 32 §3.4 wire-up inside _journal_close_event).
    agent.telegram = MagicMock()
    agent.telegram.is_active = False

    # Bind only the helpers we test here.
    agent._tickers_opened_today = types.MethodType(
        TradingAgent._tickers_opened_today, agent
    )
    agent._journal_close_event = types.MethodType(
        TradingAgent._journal_close_event, agent
    )
    agent._close_failed_streak_within_window = types.MethodType(
        TradingAgent._close_failed_streak_within_window, agent
    )
    return agent


# ───────────────────────────────────────────────────────────────────
#  PDT same-day-open detection
# ───────────────────────────────────────────────────────────────────

def test_tickers_opened_today_finds_submitted_rows():
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    _, path = _write_journal([
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
    agent = _stub_agent(path)
    tickers = agent._tickers_opened_today()
    assert tickers == {"SPY", "QQQ"}


def test_tickers_opened_today_empty_when_journal_missing():
    agent = _stub_agent("/nonexistent/path.jsonl")
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

    agent = _stub_agent(path)
    assert agent._tickers_opened_today() == {"SPY"}


# ───────────────────────────────────────────────────────────────────
#  _journal_close_event action mapping (complete vs partial vs dry_run)
# ───────────────────────────────────────────────────────────────────

def _make_ctx(strategy: str = "Bear Call",
              exit_signal: str = "profit_target",
              pl: float = 0.0) -> dict:
    return {
        "strategy":       strategy,
        "exit_signal":    exit_signal,
        "exit_reason":    "test",
        "exit_immediate": True,
        "net_unrealized_pl": pl,
        "original_credit":   1.00,
        "max_loss":          4.00,
        "spread_width":      5.0,
        "expiration":        "2026-05-30",
        "short_strikes":     [550],
        "regime_at_close":   "BULL",
        "origin":            "trade_plan",
    }


def test_journal_close_event_uses_closed_action_on_complete_fill():
    """Complete fill → action='closed' so 'Closed Today' counts it."""
    _, path = _write_journal([])
    agent = _stub_agent(path)
    spread = MagicMock(underlying="SPY")
    agent._journal_close_event(
        spread, _make_ctx(pl=25.50),
        leg_results=[], fill_status="complete", dry_run=False,
    )
    call = agent.journal_kb.log_signal.call_args
    assert call.kwargs["action"] == "closed"
    assert "closed:" in call.kwargs["notes"]


def test_journal_close_event_uses_close_failed_action_on_partial_fill():
    """Partial fill → action='close_failed' so it's counted separately."""
    _, path = _write_journal([])
    agent = _stub_agent(path)
    spread = MagicMock(underlying="SPY")
    leg_results = [
        {"symbol": "SPY260530C00550000", "status": "rejected"},
        {"symbol": "SPY260530C00555000", "status": "closed"},
    ]
    agent._journal_close_event(
        spread, _make_ctx(exit_signal="regime_shift", pl=-5.00),
        leg_results=leg_results,
        fill_status="partial", dry_run=False,
    )
    call = agent.journal_kb.log_signal.call_args
    assert call.kwargs["action"] == "close_failed"
    assert "close_failed:" in call.kwargs["notes"]
    # Failed leg surfaced in notes for fast triage.
    assert "SPY260530C00550000" in call.kwargs["notes"]


def test_journal_close_event_dry_run_uses_dry_run_close_action():
    """Skill 19 §4 (2026-05-21 hotfix): dry-run synthetic close writes
    ``action="dry_run_close"`` (not ``"closed"``) so the dashboard's
    realized-P&L sum doesn't accumulate phantom losses from a
    stuck-in-dry-run position re-firing every cycle. See the pi
    -$2,860 incident — 22 phantom rows × -$130 mark per cycle."""
    _, path = _write_journal([])
    agent = _stub_agent(path)
    spread = MagicMock(underlying="QQQ")
    agent._journal_close_event(
        spread, _make_ctx(strategy="Iron Condor",
                          exit_signal="hard_stop", pl=-45.00),
        leg_results=[], fill_status="dry_run", dry_run=True,
    )
    call = agent.journal_kb.log_signal.call_args
    assert call.kwargs["action"] == "dry_run_close", (
        f"Skill 19 §4: dry-run close must journal action='dry_run_close', "
        f"NOT 'closed'. Got {call.kwargs['action']!r}."
    )
    # Note string still references the strategy for operator visibility
    assert "dry_run_close" in call.kwargs["notes"]
    assert "Iron Condor" in call.kwargs["notes"]

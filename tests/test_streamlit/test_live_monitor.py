"""
Tests for live_monitor.py.

Strategy:
- Unit-test every pure helper function directly.
- One AppTest smoke-test to confirm the tab renders without an unhandled exception.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.streamlit.live_monitor import (
    GUARDRAIL_NAMES,
    _guardrail_status_from_journal,
    _load_journal_df,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def passing_signal():
    return {
        "timestamp": "2026-04-01T10:00:00+00:00",
        "ticker": "SPY",
        "action": "dry_run",
        "price": 520.0,
        "exec_status": "dry_run",
        "notes": "dry_run: Bull Put Spread",
        "raw_signal": {
            "regime": "bullish",
            "strategy": "Bull Put Spread",
            "plan_valid": True,
            "risk_approved": True,
            "account_balance": 100_000.0,
            "sma_50": 530.0,
            "sma_200": 510.0,
            "rsi_14": 55.0,
            "checks_passed": [
                "Max loss $200 ≤ 2% of $100,000 (=$2000)",
                "Account type is PAPER",
                "Credit/Width ratio 0.30 ≥ 0.25",
                "Delta 0.15 ≤ max delta 0.25",
                "Market is OPEN",
                "Bid/Ask spread 0.02 < 0.05",
                "Buying power 40% ≤ 80%",
            ],
            "checks_failed": [],
        },
    }


@pytest.fixture
def failing_signal():
    return {
        "timestamp": "2026-04-01T11:00:00+00:00",
        "ticker": "QQQ",
        "action": "rejected",
        "price": 440.0,
        "exec_status": "rejected",
        "notes": "rejected: Bear Call Spread, market closed",
        "raw_signal": {
            "regime": "bearish",
            "plan_valid": False,
            "risk_approved": False,
            "account_balance": 99_000.0,
            "sma_50": 430.0,
            "sma_200": 450.0,
            "rsi_14": 38.0,
            "checks_passed": ["Account type is PAPER", "Max loss $0 ≤ 2% of $99,000"],
            "checks_failed": [
                "Plan invalid: No call contracts available",
                "Market is currently CLOSED",
                "Credit/Width ratio 0.0000 < 0.25",
            ],
        },
    }


@pytest.fixture
def journal_with_one_pass(tmp_path, passing_signal):
    p = tmp_path / "signals.jsonl"
    p.write_text(json.dumps(passing_signal) + "\n")
    return p


@pytest.fixture
def journal_with_failures(tmp_path, passing_signal, failing_signal):
    p = tmp_path / "signals.jsonl"
    p.write_text(
        json.dumps(passing_signal) + "\n" + json.dumps(failing_signal) + "\n"
    )
    return p


# ---------------------------------------------------------------------------
# _load_journal_df
# ---------------------------------------------------------------------------

class TestLoadJournalDf:
    def test_returns_empty_when_file_missing(self, tmp_path):
        missing = tmp_path / "nonexistent.jsonl"
        legacy_missing = tmp_path / "nonexistent_legacy.jsonl"
        # Patch BOTH JOURNAL_PATH (signals_live.jsonl) and LEGACY_JOURNAL_PATH
        # (signals.jsonl) — the loader falls back to the legacy file when the
        # primary is missing, so the test must isolate from any real on-disk
        # legacy journal.
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", missing), \
             patch("trading_agent.streamlit.live_monitor.LEGACY_JOURNAL_PATH", legacy_missing):
            df = _load_journal_df()
        assert df.empty
        assert "timestamp" in df.columns
        assert "account_balance" in df.columns

    def test_parses_single_valid_record(self, journal_with_one_pass):
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", journal_with_one_pass):
            df = _load_journal_df()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["ticker"] == "SPY"
        assert row["account_balance"] == 100_000.0
        assert row["regime"] == "bullish"
        assert isinstance(row["checks_passed"], list)
        assert isinstance(row["checks_failed"], list)

    def test_parses_multiple_records(self, journal_with_failures):
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", journal_with_failures):
            df = _load_journal_df()
        assert len(df) == 2
        assert set(df["ticker"]) == {"SPY", "QQQ"}

    def test_skips_malformed_json_lines(self, tmp_path):
        p = tmp_path / "signals.jsonl"
        p.write_text(
            'not json\n'
            '{"timestamp":"2026-04-01T10:00:00+00:00","ticker":"IWM","action":"skip","price":200,'
            '"raw_signal":{"account_balance":80000}}\n'
        )
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", p):
            df = _load_journal_df()
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "IWM"

    def test_skips_empty_lines(self, tmp_path, passing_signal):
        p = tmp_path / "signals.jsonl"
        p.write_text("\n\n" + json.dumps(passing_signal) + "\n\n")
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", p):
            df = _load_journal_df()
        assert len(df) == 1

    def test_sorts_by_timestamp(self, tmp_path):
        early = {
            "timestamp": "2026-04-01T08:00:00+00:00",
            "ticker": "A",
            "action": "skip",
            "price": 100,
            "raw_signal": {"account_balance": 100_000},
        }
        late = {
            "timestamp": "2026-04-01T15:00:00+00:00",
            "ticker": "B",
            "action": "skip",
            "price": 200,
            "raw_signal": {"account_balance": 100_500},
        }
        p = tmp_path / "signals.jsonl"
        # Write late first, early second — df should still be sorted asc
        p.write_text(json.dumps(late) + "\n" + json.dumps(early) + "\n")
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", p):
            df = _load_journal_df()
        assert df.iloc[0]["ticker"] == "A"
        assert df.iloc[1]["ticker"] == "B"

    def test_handles_null_account_balance(self, tmp_path):
        rec = {
            "timestamp": "2026-04-01T10:00:00+00:00",
            "ticker": "SPY",
            "action": "error",
            "price": 0,
            "raw_signal": {"account_balance": None},
        }
        p = tmp_path / "signals.jsonl"
        p.write_text(json.dumps(rec) + "\n")
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", p):
            df = _load_journal_df()
        assert df.iloc[0]["account_balance"] == 0


# ---------------------------------------------------------------------------
# _guardrail_status_from_journal
# ---------------------------------------------------------------------------

class TestGuardrailStatusFromJournal:
    def test_returns_eight_entries(self, journal_with_one_pass):
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", journal_with_one_pass):
            df = _load_journal_df()
        status = _guardrail_status_from_journal(df)
        assert len(status) == 8

    def test_names_match_constants(self, journal_with_one_pass):
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", journal_with_one_pass):
            df = _load_journal_df()
        status = _guardrail_status_from_journal(df)
        names = [g["name"] for g in status]
        assert names == GUARDRAIL_NAMES

    def test_all_passed_when_checks_failed_empty(self, journal_with_one_pass):
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", journal_with_one_pass):
            df = _load_journal_df()
        status = _guardrail_status_from_journal(df)
        assert all(g["passed"] for g in status)

    def test_market_open_fails_when_in_checks_failed(self, journal_with_failures):
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", journal_with_failures):
            df = _load_journal_df()
        status = _guardrail_status_from_journal(df)
        market_g = next(g for g in status if g["name"] == "Market Open")
        assert not market_g["passed"]

    def test_paper_account_passes_when_in_checks_passed(self, journal_with_failures):
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", journal_with_failures):
            df = _load_journal_df()
        status = _guardrail_status_from_journal(df)
        paper_g = next(g for g in status if g["name"] == "Paper Account")
        assert paper_g["passed"]

    def test_credit_ratio_fails_when_in_checks_failed(self, journal_with_failures):
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", journal_with_failures):
            df = _load_journal_df()
        status = _guardrail_status_from_journal(df)
        cr_g = next(g for g in status if g["name"] == "Credit/Width Ratio")
        assert not cr_g["passed"]

    def test_returns_defaults_for_empty_dataframe(self):
        empty_df = pd.DataFrame(
            columns=["timestamp", "account_balance", "ticker", "action",
                     "regime", "checks_passed", "checks_failed"]
        )
        status = _guardrail_status_from_journal(empty_df)
        assert len(status) == 8
        # All default to passed=True with "No data"
        assert all(g["passed"] for g in status)
        assert all("No data" in g["detail"] for g in status)

    def test_detail_truncated_to_70_chars(self, tmp_path):
        long_msg = "Plan invalid: " + "x" * 100
        rec = {
            "timestamp": "2026-04-01T10:00:00+00:00",
            "ticker": "SPY",
            "action": "rejected",
            "price": 0,
            "raw_signal": {
                "account_balance": 100_000,
                "sma_50": 0, "sma_200": 0, "rsi_14": 0,
                "checks_passed": [],
                "checks_failed": [long_msg],
            },
        }
        p = tmp_path / "signals.jsonl"
        p.write_text(json.dumps(rec) + "\n")
        with patch("trading_agent.streamlit.live_monitor.JOURNAL_PATH", p):
            df = _load_journal_df()
        status = _guardrail_status_from_journal(df)
        plan_g = next(g for g in status if g["name"] == "Plan Validity")
        assert not plan_g["passed"]
        assert len(plan_g["detail"]) <= 70


# ---------------------------------------------------------------------------
# Smoke: render_live_monitor via AppTest
# ---------------------------------------------------------------------------

class TestRenderLiveMonitorSmoke:
    @patch("trading_agent.streamlit.live_monitor._auto_refresh")
    @patch("trading_agent.streamlit.live_monitor._get_config", return_value=None)
    @patch("trading_agent.streamlit.live_monitor._load_journal_df",
           return_value=pd.DataFrame(
               columns=["timestamp", "account_balance", "ticker", "action",
                        "regime", "checks_passed", "checks_failed",
                        "notes", "rsi_14", "sma_50", "sma_200"]
           ))
    def test_renders_without_exception(self, _mock_df, _mock_cfg, _mock_refresh):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_function(
            lambda: __import__(
                "trading_agent.streamlit.live_monitor",
                fromlist=["render_live_monitor"]
            ).render_live_monitor()
        )
        at.run(timeout=15)
        # Any exception other than RerunException is a real failure
        if at.exception:
            exc_str = str(at.exception)
            assert "rerun" in exc_str.lower() or "Rerun" in exc_str, (
                f"Unexpected exception: {at.exception}"
            )

    @patch("trading_agent.streamlit.live_monitor._auto_refresh")
    @patch("trading_agent.streamlit.live_monitor._get_config", return_value=None)
    def test_pause_flag_reflected_in_session(self, _mock_cfg, _mock_refresh, tmp_path):
        """PAUSE_FLAG path can be patched; page still renders."""
        with patch("trading_agent.streamlit.live_monitor.PAUSE_FLAG", tmp_path / "PAUSED"):
            from streamlit.testing.v1 import AppTest

            at = AppTest.from_function(
                lambda: __import__(
                    "trading_agent.streamlit.live_monitor",
                    fromlist=["render_live_monitor"]
                ).render_live_monitor()
            )
            at.run(timeout=15)
            # Should not blow up regardless of pause flag state
            if at.exception:
                assert "rerun" in str(at.exception).lower()


# ---------------------------------------------------------------------------
# Regression: stale-PENDING grid bug (2026-05-07)
# ---------------------------------------------------------------------------
# Symptom on 2026-05-07: SPY rendered ⏳ PENDING in the guardrail grid
# even though the position had been opened, traded out, and closed
# (+$81 profit_target) several hours earlier.
#
# Root cause: `last_entry_by_ticker` picked the most recent
# action="submitted" row with no awareness of whether a later
# action="closed" or action="close_failed" superseded it. The
# verdict-cell logic then saw "entry exists, ticker not held →
# PENDING" and rendered indefinitely until a NEW submission overwrote
# the lookup.
#
# Fix: filter out entry trades that have a terminal event (closed,
# close_failed) at a timestamp ≥ the entry timestamp.

from trading_agent.streamlit.live_monitor import _guardrail_grid_from_journal


def _row(ticker, ts, action, **extra):
    """Helper: minimum-viable journal row in DataFrame-row shape."""
    base = {
        "ticker": ticker,
        "timestamp": pd.to_datetime(ts, utc=True),
        "action": action,
        "price": 100.0,
        "regime": "bullish",
        "mode": "live",
        "checks_passed": [],
        "checks_failed": [],
        "notes": action,
        "reason": "",
        "rsi_14": 50.0,
        "sma_50": 100.0,
        "sma_200": 100.0,
        "scan_results": {},
        "raw_signal": {"regime": "bullish", "mode": "live"},
        "account_balance": 5000.0,
    }
    base.update(extra)
    return base


class TestStalePendingFix:
    def test_closed_after_submitted_supersedes_entry(self):
        """Entry trade followed by a close → grid must NOT show PENDING."""
        # Submitted yesterday → closed today → skipped now.
        df = pd.DataFrame([
            _row("SPY", "2026-05-06T18:53:00Z", "submitted"),
            _row("SPY", "2026-05-07T13:56:00Z", "closed",
                 raw_signal={"regime": "bullish", "mode": "live",
                             "net_unrealized_pl": 81.0}),
            _row("SPY", "2026-05-07T14:30:00Z", "skipped_rsi_gate",
                 reason="RSI 71 > 70"),
        ])

        grid = _guardrail_grid_from_journal(
            df, current_mode=None, window_minutes=10, held_tickers=set()
        )
        spy = next((r for r in grid if r["ticker"] == "SPY"), None)
        assert spy is not None
        # Status must be skipped, not pending — there's no live order.
        assert spy["status"] == "skipped", (
            f"Expected 'skipped' (closed event supersedes entry), "
            f"got {spy['status']!r}"
        )

    def test_submitted_then_skipped_no_close_still_pending(self):
        """No close event → entry is current → PENDING when not held."""
        df = pd.DataFrame([
            _row("AAPL", "2026-05-07T13:00:00Z", "submitted"),
            _row("AAPL", "2026-05-07T13:30:00Z", "skipped_existing",
                 reason="Existing open position or pending order"),
        ])

        grid = _guardrail_grid_from_journal(
            df, current_mode=None, window_minutes=60, held_tickers=set()
        )
        aapl = next((r for r in grid if r["ticker"] == "AAPL"), None)
        assert aapl is not None
        assert aapl["status"] == "pending", (
            "Submitted with no later closed event → entry is current → "
            "PENDING when not held."
        )

    def test_held_ticker_renders_holding_regardless_of_close_history(self):
        """Even if there was a past close, a CURRENTLY-held ticker is HOLDING."""
        # Old close, then re-submitted, ticker now held.
        df = pd.DataFrame([
            _row("QQQ", "2026-05-05T15:00:00Z", "closed",
                 raw_signal={"regime": "bullish", "mode": "live",
                             "net_unrealized_pl": 25.0}),
            _row("QQQ", "2026-05-07T14:00:00Z", "submitted"),
            _row("QQQ", "2026-05-07T14:35:00Z", "skipped_existing"),
        ])

        grid = _guardrail_grid_from_journal(
            df, current_mode=None, window_minutes=60,
            held_tickers={"QQQ"},
        )
        qqq = next((r for r in grid if r["ticker"] == "QQQ"), None)
        assert qqq is not None
        # The 2026-05-05 close is older than the 2026-05-07 submitted
        # row, so it does NOT supersede the entry.
        assert qqq["status"] == "holding"

    def test_close_failed_also_supersedes_entry(self):
        """A close_failed terminal event clears the entry the same way."""
        df = pd.DataFrame([
            _row("XLF", "2026-05-06T18:00:00Z", "submitted"),
            _row("XLF", "2026-05-07T13:00:00Z", "close_failed",
                 raw_signal={"regime": "bearish", "mode": "live",
                             "fill_status": "partial"}),
            _row("XLF", "2026-05-07T14:30:00Z", "skipped_rsi_gate"),
        ])

        grid = _guardrail_grid_from_journal(
            df, current_mode=None, window_minutes=10, held_tickers=set()
        )
        xlf = next((r for r in grid if r["ticker"] == "XLF"), None)
        assert xlf is not None
        # close_failed is terminal-for-our-purposes (cooldown will block
        # auto-retry); rendering PENDING would be misleading.
        assert xlf["status"] == "skipped"

    def test_no_terminal_events_preserves_legacy_behavior(self):
        """When the close-tracking dict is empty, fall back to old logic."""
        df = pd.DataFrame([
            _row("DIA", "2026-05-07T13:00:00Z", "submitted"),
            _row("DIA", "2026-05-07T14:00:00Z", "skipped_existing"),
        ])

        grid = _guardrail_grid_from_journal(
            df, current_mode=None, window_minutes=60, held_tickers=set()
        )
        dia = next((r for r in grid if r["ticker"] == "DIA"), None)
        assert dia is not None
        # No terminal events → entry is current → PENDING (not held).
        assert dia["status"] == "pending"

    def test_mixed_case_mode_field_does_not_drop_close_rows(self):
        """Regression: 2026-05-07 secondary bug.

        Pre-fix the journal carried "LIVE" from _log_signal and "live"
        from _journal_close_event.  The grid's case-sensitive
        ``current_mode == row.mode`` filter dropped the close rows
        before the supersede-PENDING check ran, so the SPY close at
        13:56 was invisible even though it was correctly journaled.
        Fix normalises both sides to uppercase.
        """
        df = pd.DataFrame([
            # Old code path emits uppercase.
            _row("SPY", "2026-05-06T18:53:00Z", "submitted",
                 mode="LIVE",
                 raw_signal={"regime": "bullish", "mode": "LIVE"}),
            # _journal_close_event used to emit lowercase.
            _row("SPY", "2026-05-07T13:56:00Z", "closed",
                 mode="live",
                 raw_signal={"regime": "bullish", "mode": "live",
                             "net_unrealized_pl": 81.0}),
            # Today's skipped rows from _log_signal — uppercase.
            _row("SPY", "2026-05-07T14:30:00Z", "skipped_rsi_gate",
                 mode="LIVE", reason="RSI 71 > 70"),
        ])
        # Filter by current_mode="LIVE" — exactly what the dashboard
        # passes for the live tab.
        grid = _guardrail_grid_from_journal(
            df, current_mode="LIVE", window_minutes=10, held_tickers=set()
        )
        spy = next((r for r in grid if r["ticker"] == "SPY"), None)
        assert spy is not None
        # The close row, despite being journalled with mode="live",
        # must reach the supersede check — so SPY ends up "skipped"
        # rather than the buggy "pending" the user saw on 2026-05-07.
        assert spy["status"] == "skipped", (
            f"Expected 'skipped' (close at 'live' mode survives the "
            f"case-insensitive LIVE filter), got {spy['status']!r}"
        )

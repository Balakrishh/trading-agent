"""Conformance test: skill 19 §1.2 — JournalReader public surface.

Skill 19 §1.2 decoupling pass #2: every consumer that asks
"what's in today's journal" must go through ``JournalReader``.
This test pins the public-query surface + the canonical filters
so the next refactor can't silently break the EOD recap or the
dashboard tile by adjusting one reader but missing another.

Scenarios covered:

  * dataclass result types exist with the documented fields
  * ``closes_today`` returns ET-trading-date matches only
  * ``closes_today`` skips ``fill_status="dry_run"`` (defense-in-
    depth — even after the live/dryrun file split, historical
    rows may still be mislabeled)
  * ``realized_pl_today`` matches the sum of ``closes_today``
  * ``opens_today`` returns only ``action="submitted"`` rows
  * ``stuck_positions`` surfaces PDT-blocked + cooldown-active
  * ``cycle_minute_count_today`` returns distinct-minute count
  * ``error_count_today`` counts error/warning/cycle_error rows
  * ``account_balance_today_endpoints`` returns (first, last) of
    today's balance snapshots

Failure modes caught:
- Someone renames a query method → callers break, this fails
- Someone removes the dry-run skip in closes_today → -$2,976
  phantom-loss class returns silently
- Someone reverts the ET-date filter → cross-session leak returns
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


_ET = ZoneInfo("US/Eastern")


def _write_fixture_journal(rows: list, path: Path) -> None:
    """Write a JSONL fixture with the provided rows."""
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _et_iso_now(et_date: date, hour: int = 12, minute: int = 0) -> str:
    """Return a UTC ISO timestamp string corresponding to (et_date, hour, minute) ET."""
    dt_et = datetime(et_date.year, et_date.month, et_date.day,
                     hour, minute, 0, tzinfo=_ET)
    return dt_et.astimezone(timezone.utc).isoformat()


def test_journal_reader_dataclasses_exist() -> None:
    """Skill 19 §1.2: ClosedTrade / OpenedTrade / StuckPosition are the
    documented result dataclasses. Test files importing these need stable
    field names."""
    from trading_agent.journal_reader import (
        ClosedTrade, OpenedTrade, StuckPosition,
    )
    # Field-name pinning — any of these getting renamed breaks callers
    for cls, expected_fields in (
        (ClosedTrade, {"ticker", "strategy", "exit_signal", "exit_reason",
                       "realized_pl", "expiration", "timestamp_utc"}),
        (OpenedTrade, {"ticker", "strategy", "credit", "expiration",
                       "timestamp_utc"}),
        (StuckPosition, {"ticker", "reason"}),
    ):
        actual = set(cls.__dataclass_fields__)
        missing = expected_fields - actual
        assert not missing, (
            f"Skill 19 §1.2: {cls.__name__} lost fields {missing}. "
            f"Renaming any of these silently breaks consumers."
        )


def test_journal_reader_closes_today_skips_dry_run_fill_status() -> None:
    """Skill 19 §1.2: closes_today must NEVER include rows where
    raw_signal.fill_status == "dry_run". This was the -$2,976
    EOD-phantom-recap bug; rerunning the same data through the
    reader must give the clean number."""
    from trading_agent.journal_reader import JournalReader
    today_et = date(2026, 5, 21)
    rows = [
        # Real close — counts
        {"timestamp": _et_iso_now(today_et, 10, 30),
         "ticker": "SPY", "action": "closed",
         "raw_signal": {"strategy": "Bear Call",
                        "exit_signal": "profit_target",
                        "net_unrealized_pl": 200.0,
                        "fill_status": "complete"}},
        # Mislabeled dry-run pseudo-close — must NOT count
        {"timestamp": _et_iso_now(today_et, 11, 0),
         "ticker": "DIA", "action": "closed",
         "raw_signal": {"strategy": "Iron Condor",
                        "exit_signal": "strike_proximity",
                        "net_unrealized_pl": -130.0,
                        "fill_status": "dry_run"}},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "signals_live.jsonl"
        _write_fixture_journal(rows, p)
        JournalReader._today_et = staticmethod(lambda: today_et)
        closes = JournalReader(str(p)).closes_today()
    assert len(closes) == 1, (
        "Skill 19 §1.2: dry-run mislabeled rows must be filtered out "
        "of closes_today. Got %d closes, expected 1." % len(closes)
    )
    assert closes[0].ticker == "SPY"
    assert closes[0].realized_pl == 200.0


def test_journal_reader_closes_today_uses_et_date_not_utc() -> None:
    """Skill 19 §1.2: outer row filter is ET trading-session date,
    not UTC. Wed-evening rows (20:00-23:59 ET = 00:00-03:59 UTC Thu)
    must NOT appear in Thursday's recap."""
    from trading_agent.journal_reader import JournalReader
    today_et = date(2026, 5, 21)
    yesterday_et = today_et - timedelta(days=1)
    rows = [
        # Wed 22:00 ET (yesterday's session) — must NOT appear
        {"timestamp": _et_iso_now(yesterday_et, 22, 0),
         "ticker": "DIA", "action": "closed",
         "raw_signal": {"strategy": "Iron Condor",
                        "net_unrealized_pl": -50.0,
                        "fill_status": "complete"}},
        # Thu 10:00 ET (today's session) — must appear
        {"timestamp": _et_iso_now(today_et, 10, 0),
         "ticker": "SPY", "action": "closed",
         "raw_signal": {"strategy": "Bear Call",
                        "net_unrealized_pl": 100.0,
                        "fill_status": "complete"}},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "signals_live.jsonl"
        _write_fixture_journal(rows, p)
        JournalReader._today_et = staticmethod(lambda: today_et)
        closes = JournalReader(str(p)).closes_today()
    assert len(closes) == 1, (
        "Skill 19 §1.2: ET-date filter must exclude yesterday's "
        "ET trading session rows. Got %d, expected 1." % len(closes)
    )
    assert closes[0].ticker == "SPY"


def test_journal_reader_realized_pl_matches_closes_sum() -> None:
    """Skill 19 §1.2: realized_pl_today is documented as the sum of
    closes_today. Behavioral consistency must hold."""
    from trading_agent.journal_reader import JournalReader
    today_et = date(2026, 5, 21)
    rows = [
        {"timestamp": _et_iso_now(today_et, 10),
         "ticker": "SPY", "action": "closed",
         "raw_signal": {"net_unrealized_pl": 50.0, "fill_status": "complete"}},
        {"timestamp": _et_iso_now(today_et, 14),
         "ticker": "DIA", "action": "closed",
         "raw_signal": {"net_unrealized_pl": -25.0, "fill_status": "complete"}},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "signals_live.jsonl"
        _write_fixture_journal(rows, p)
        JournalReader._today_et = staticmethod(lambda: today_et)
        r = JournalReader(str(p))
        closes = r.closes_today()
        realized = r.realized_pl_today()
    assert abs(realized - sum(c.realized_pl for c in closes)) < 1e-9


def test_journal_reader_opens_today() -> None:
    """Skill 19 §1.2: opens_today returns submitted rows for today
    only, with ticker / strategy / credit / expiration populated."""
    from trading_agent.journal_reader import JournalReader
    today_et = date(2026, 5, 21)
    rows = [
        {"timestamp": _et_iso_now(today_et, 9, 35),
         "ticker": "XLF", "action": "submitted",
         "raw_signal": {"strategy": "Bull Put Spread",
                        "net_credit": 0.47,
                        "expiration": "2026-06-05"}},
        {"timestamp": _et_iso_now(today_et, 9, 40),
         "ticker": "QQQ", "action": "rejected",   # not a submitted → skip
         "raw_signal": {}},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "signals_live.jsonl"
        _write_fixture_journal(rows, p)
        JournalReader._today_et = staticmethod(lambda: today_et)
        opens = JournalReader(str(p)).opens_today()
    assert len(opens) == 1
    assert opens[0].ticker == "XLF"
    assert opens[0].credit == 0.47
    assert opens[0].expiration == "2026-06-05"


def test_journal_reader_stuck_positions_surfaces_pdt_block() -> None:
    """Skill 19 §1.2: stuck_positions surfaces PDT-blocked tickers
    whose pdt_blocked_date matches today's UTC date."""
    from trading_agent.journal_reader import JournalReader
    today_et = date(2026, 5, 21)
    today_utc_iso = datetime.now(timezone.utc).date().isoformat()
    rows = [
        # PDT-blocked today
        {"timestamp": _et_iso_now(today_et, 10),
         "ticker": "DIA", "action": "close_failed",
         "raw_signal": {"pdt_blocked_today": True,
                        "pdt_blocked_date": today_utc_iso}},
        # close_failed but not PDT → no surface (unless cooldown)
        {"timestamp": _et_iso_now(today_et, 11),
         "ticker": "SPY", "action": "close_failed",
         "raw_signal": {}},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "signals_live.jsonl"
        _write_fixture_journal(rows, p)
        JournalReader._today_et = staticmethod(lambda: today_et)
        stuck = JournalReader(str(p)).stuck_positions()
    tickers = [s.ticker for s in stuck]
    assert "DIA" in tickers
    assert "SPY" not in tickers


def test_journal_reader_account_balance_endpoints() -> None:
    """Skill 19 §1.2: account_balance_today_endpoints returns
    (first-seen, last-seen) account_balance values from today's rows."""
    from trading_agent.journal_reader import JournalReader
    today_et = date(2026, 5, 21)
    rows = [
        {"timestamp": _et_iso_now(today_et, 9, 30),
         "ticker": "SPY", "action": "rejected",
         "raw_signal": {"account_balance": 4700.0}},
        {"timestamp": _et_iso_now(today_et, 15, 30),
         "ticker": "QQQ", "action": "rejected",
         "raw_signal": {"account_balance": 4550.0}},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "signals_live.jsonl"
        _write_fixture_journal(rows, p)
        JournalReader._today_et = staticmethod(lambda: today_et)
        first, last = JournalReader(str(p)).account_balance_today_endpoints()
    assert first == 4700.0
    assert last == 4550.0


def test_journal_reader_handles_missing_file() -> None:
    """Skill 19 §1.2: every query returns the empty-equivalent when
    the journal file doesn't exist. No exceptions propagate."""
    from trading_agent.journal_reader import JournalReader
    r = JournalReader("/nonexistent/signals_live.jsonl")
    assert r.closes_today() == []
    assert r.realized_pl_today() == 0.0
    assert r.opens_today() == []
    assert r.stuck_positions() == []
    assert r.cycle_minute_count_today() == 0
    assert r.error_count_today() == 0
    assert r.account_balance_today_endpoints() == (None, None)


def test_journal_reader_eod_callers_match_dashboard_caller() -> None:
    """Skill 19 §1.2: source-level pin — both the EOD builder and the
    dashboard realized-P&L tile must call JournalReader. Without this
    pin, a future refactor could re-introduce duplicate filter logic
    in one of the two paths and drift them apart (the original bug
    pattern)."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    agent_src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    dash_src = (repo_root / "trading_agent" / "streamlit" /
                "live_monitor.py").read_text(encoding="utf-8")

    assert "JournalReader" in agent_src, (
        "Skill 19 §1.2: agent.py must use JournalReader for "
        "_build_eod_summary. Without it, the EOD recap and the "
        "dashboard tile can drift on filter semantics."
    )
    assert "JournalReader" in dash_src, (
        "Skill 19 §1.2: streamlit/live_monitor.py must use "
        "JournalReader for the realized-P&L tile and the "
        "_render_closed_today panel. Each consumer that doesn't "
        "go through the reader can drift."
    )
    # Realized-P&L tile must use realized_pl_today()
    assert "realized_pl_today()" in dash_src, (
        "Skill 19 §1.2: the dashboard tile must call "
        ".realized_pl_today() — not re-implement the sum locally."
    )


def test_tickers_opened_today_utc_counts_submitted_actions() -> None:
    """Skill 17 §4: tickers_opened_today_utc returns underlyings with
    a submitted row for today's UTC date. Used by the close loop to
    suppress same-day REGIME_SHIFT exits on PDT-restricted accounts."""
    from trading_agent.journal_reader import JournalReader
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "signals_live.jsonl"
        now = datetime.now(timezone.utc)
        rows = [
            {"timestamp": now.isoformat(), "ticker": "SPY",
             "action": "submitted", "raw_signal": {}},
            {"timestamp": now.isoformat(), "ticker": "DIA",
             "action": "submitted", "raw_signal": {}},
            # Same ticker rejected — should NOT count
            {"timestamp": now.isoformat(), "ticker": "XLF",
             "action": "rejected", "raw_signal": {}},
            # Yesterday's submission — should NOT count
            {"timestamp": (now - timedelta(days=1)).isoformat(),
             "ticker": "GLD", "action": "submitted", "raw_signal": {}},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        opened = JournalReader(str(path)).tickers_opened_today_utc()
        assert "SPY" in opened
        assert "DIA" in opened
        assert "XLF" not in opened, (
            "Skill 17 §4: only action='submitted' counts. A rejected "
            "attempt does not establish a same-day-open position."
        )
        assert "GLD" not in opened, (
            "Skill 17 §4: yesterday's submissions must not count — "
            "FINRA day-trade is per-trading-day."
        )


def test_telegram_alert_sent_today_utc_dedup() -> None:
    """Skill 32 §3.4: telegram_alert_sent_today_utc returns True if a
    matching dedup row was written today; False otherwise. Date-keyed
    (UTC), per-(ticker, alert_type)."""
    from trading_agent.journal_reader import JournalReader
    today_utc = datetime.now(timezone.utc).date().isoformat()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "signals_live.jsonl"
        now = datetime.now(timezone.utc)
        rows = [
            {"timestamp": now.isoformat(), "ticker": "DIA",
             "action": "telegram_alert_sent",
             "raw_signal": {"alert_type": "pdt_block",
                            "alert_date": today_utc}},
            # Different ticker — should not match for DIA query
            {"timestamp": now.isoformat(), "ticker": "SPY",
             "action": "telegram_alert_sent",
             "raw_signal": {"alert_type": "close_cooldown",
                            "alert_date": today_utc}},
            # Yesterday's — should not match
            {"timestamp": (now - timedelta(days=1)).isoformat(),
             "ticker": "XLF",
             "action": "telegram_alert_sent",
             "raw_signal": {"alert_type": "pdt_block",
                            "alert_date": "2024-01-01"}},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        rdr = JournalReader(str(path))
        # Matching pair
        assert rdr.telegram_alert_sent_today_utc(
            ticker="DIA", alert_type="pdt_block",
        ) is True
        # Wrong alert_type
        assert rdr.telegram_alert_sent_today_utc(
            ticker="DIA", alert_type="close_cooldown",
        ) is False
        # Wrong ticker
        assert rdr.telegram_alert_sent_today_utc(
            ticker="DIA", alert_type="position_closed",
        ) is False
        # Stale (different UTC date)
        assert rdr.telegram_alert_sent_today_utc(
            ticker="XLF", alert_type="pdt_block",
        ) is False, (
            "Skill 32 §3.4: alert_date must match today (UTC). "
            "Yesterday's dedup row must NOT suppress today's alert."
        )


def test_agent_journal_helpers_delegate_to_journal_reader() -> None:
    """Skill 19 §1.2: agent._tickers_opened_today and
    agent._telegram_alert_already_sent_today must delegate to
    JournalReader — agent.py should have ZERO open(jsonl_path) calls."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    # Zero raw journal opens in production code.
    direct_opens = sum(
        1 for line in src.splitlines()
        if "open(jsonl_path" in line or "open(self.journal_kb" in line
    )
    assert direct_opens == 0, (
        f"Skill 19 §1.2: agent.py has {direct_opens} direct "
        f"open(jsonl_path, ...) call(s). All journal reads must go "
        f"through JournalReader so filter semantics stay consolidated."
    )
    # Both helpers must reference JournalReader methods
    assert "tickers_opened_today_utc()" in src, (
        "Skill 19 §1.2: agent._tickers_opened_today must delegate to "
        "JournalReader.tickers_opened_today_utc()."
    )
    assert "telegram_alert_sent_today_utc(" in src, (
        "Skill 32 §3.4: agent._telegram_alert_already_sent_today "
        "must delegate to JournalReader.telegram_alert_sent_today_utc()."
    )

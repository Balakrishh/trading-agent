"""
Tests for ``trading_agent.backtest.clock``.

Pin the event-stream contract: the right number of intraday bars per
trading day, the right ordering of (day_open, intraday, daily_mark,
day_close), and that the calendar correctly skips weekends/holidays.
"""

from __future__ import annotations

from datetime import date

from trading_agent.backtest.clock import (
    INTRADAY_BAR_MINUTES,
    iter_events,
    trading_days_in_range,
)


# --------------------------------------------------------------------------
# Event ordering & counts on a single trading day
# --------------------------------------------------------------------------

def test_single_day_intraday_event_count():
    # One full trading day at 5m bars: 09:30 → 16:00 = 6.5h × 12/h = 78 bars.
    # Plus 1 day_open + 1 daily_mark + 1 day_close = 81 events.
    # Pick a known trading day: Monday May 5, 2025.
    events = list(iter_events(date(2025, 5, 5), date(2025, 5, 5),
                              intraday=True))
    kinds = [e.kind for e in events]
    assert kinds.count("day_open") == 1
    assert kinds.count("intraday_decision") == 78
    assert kinds.count("daily_mark") == 1
    assert kinds.count("day_close") == 1


def test_event_ordering_on_single_day():
    events = list(iter_events(date(2025, 5, 5), date(2025, 5, 5),
                              intraday=True))
    # First event must be day_open, last must be day_close.
    assert events[0].kind == "day_open"
    assert events[-1].kind == "day_close"
    # day_open at 09:30, daily_mark + day_close at 16:00.
    assert events[0].timestamp.hour == 9 and events[0].timestamp.minute == 30
    assert events[-1].timestamp.hour == 16 and events[-1].timestamp.minute == 0


def test_intraday_off_collapses_to_three_events_per_day():
    events = list(iter_events(date(2025, 5, 5), date(2025, 5, 5),
                              intraday=False))
    kinds = [e.kind for e in events]
    assert kinds == ["day_open", "daily_mark", "day_close"]


# --------------------------------------------------------------------------
# Calendar awareness — weekends / holidays skipped
# --------------------------------------------------------------------------

def test_weekend_produces_no_events():
    # Sat May 3 and Sun May 4 2025 — both non-trading.
    events = list(iter_events(date(2025, 5, 3), date(2025, 5, 4)))
    assert events == []


def test_holiday_skipped():
    # Jul 4 2025 was a Friday holiday.
    days_in_july = trading_days_in_range(date(2025, 7, 1), date(2025, 7, 7))
    assert date(2025, 7, 4) not in days_in_july


# --------------------------------------------------------------------------
# Multi-day spans
# --------------------------------------------------------------------------

def test_three_trading_days_have_three_day_open_events():
    events = list(iter_events(date(2025, 5, 5), date(2025, 5, 7),
                              intraday=False))
    kinds = [e.kind for e in events]
    assert kinds.count("day_open") == 3
    assert kinds.count("daily_mark") == 3
    assert kinds.count("day_close") == 3

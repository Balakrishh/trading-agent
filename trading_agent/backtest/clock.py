"""
clock.py — calendar-aware backtest timeline.

The backtester needs to walk forward through historical time at *the
same cadence the live agent runs* — every 5 minutes during regular
trading hours on every NYSE trading day in the requested range. The
yfinance / Alpaca data we replay against is keyed by exactly these
timestamps, so the clock and the data source line up by construction.

Two concentric loops
--------------------
The runner consumes the clock as a flat iterator of (timestamp, kind)
tuples where ``kind`` is one of:

  * ``"intraday_decision"`` — fires every 5-minute bar during RTH on
    days when the chosen interval is fine-grained enough (5m). The
    PERCEIVE → CLASSIFY → PLAN → RISK → EXECUTE pipeline runs on each.
  * ``"daily_mark"`` — fires once per trading day at the close (16:00
    ET). Re-marks every open position via Black-Scholes with the
    VIX-proxy IV. The runner uses this even when intraday decisions
    aren't possible (range > 30 days back, where 5m bars don't exist).
  * ``"day_open"`` / ``"day_close"`` — bookkeeping bookends the runner
    uses to flush per-day journal lines.

Why this lives in its own module
--------------------------------
``BacktestRunner`` should be agnostic to whether the user picked 5m or
daily granularity. Centralising the timestamp generation here means the
runner just iterates and switches on ``kind`` — no calendar math leaks
into the orchestration layer, and the unit test for "did we fire the
right number of cycles for the SPY July-2025 window?" lives next to the
generator that produces them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterator, Literal, Sequence

from trading_agent.calendar_utils import _valid_days

EventKind = Literal["day_open", "intraday_decision", "daily_mark", "day_close"]


# US/Eastern market hours.  We use *naive* datetimes (no tz) throughout
# the backtest because yfinance's intraday data is delivered in
# US/Eastern naive timestamps when ``auto_adjust=False``; mixing tz-aware
# and naive everywhere just to be "correct" creates more bugs than it
# solves. The clock owns the convention and downstream code never has
# to tz-guard.
RTH_OPEN  = time(9, 30)
RTH_CLOSE = time(16, 0)
INTRADAY_BAR_MINUTES = 5


@dataclass(frozen=True)
class ClockEvent:
    """One discrete event the runner reacts to."""
    timestamp: datetime
    kind: EventKind


def _intraday_bars_for_day(d: date) -> Iterator[datetime]:
    """
    Yield 5-minute bar OPEN timestamps inside the regular-session
    [09:30, 16:00) window. The 16:00 timestamp itself is reserved for
    the ``daily_mark`` event so the runner can re-mark using the close
    print after the last intraday decision has landed.
    """
    cur = datetime.combine(d, RTH_OPEN)
    end = datetime.combine(d, RTH_CLOSE)
    step = timedelta(minutes=INTRADAY_BAR_MINUTES)
    while cur < end:
        yield cur
        cur += step


def iter_events(start: date, end: date,
                *,
                intraday: bool = True) -> Iterator[ClockEvent]:
    """
    Walk the NYSE calendar from ``start`` to ``end`` (inclusive) and
    yield one ``ClockEvent`` at a time.

    ``intraday=False`` collapses each day to ``day_open → daily_mark →
    day_close`` (no per-bar decisions) — the only available cadence
    when the requested window is too far back for yfinance's 5m feed.
    See ``trading_agent/market_data.py:62-72`` for the cadence limits
    that drive the runner's choice between True and False.
    """
    days = _valid_days(start, end)
    for d in days:
        yield ClockEvent(datetime.combine(d, RTH_OPEN), "day_open")
        if intraday:
            for bar_open in _intraday_bars_for_day(d):
                yield ClockEvent(bar_open, "intraday_decision")
        # The mark fires at the official close print so theta/IV
        # accumulated through the session is realised on the equity
        # curve before the day flips.
        yield ClockEvent(datetime.combine(d, RTH_CLOSE), "daily_mark")
        yield ClockEvent(datetime.combine(d, RTH_CLOSE), "day_close")


def trading_days_in_range(start: date, end: date) -> Sequence[date]:
    """Convenience pass-through used by the runner's progress bar."""
    return _valid_days(start, end)


__all__ = [
    "ClockEvent",
    "EventKind",
    "RTH_OPEN",
    "RTH_CLOSE",
    "INTRADAY_BAR_MINUTES",
    "iter_events",
    "trading_days_in_range",
]

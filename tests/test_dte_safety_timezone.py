"""2026-11-24 — DTE safety uses real Eastern time (was a fixed UTC−4)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading_agent.position_monitor import PositionMonitor

check = PositionMonitor._check_dte_safety


@pytest.mark.parametrize("utc, fires", [
    # Winter (EST, UTC−5): Thu 2026-11-25 is the last session before Fri 11-27.
    (datetime(2026, 11, 25, 19, 30, tzinfo=timezone.utc), False),   # 14:30 ET — old code fired here
    (datetime(2026, 11, 25, 20, 29, tzinfo=timezone.utc), False),   # 15:29 ET
    (datetime(2026, 11, 25, 20, 30, tzinfo=timezone.utc), True),    # 15:30 ET
    (datetime(2026, 11, 26, 0, 30, tzinfo=timezone.utc), True),     # 19:30 ET, UTC date already 11-26
])
def test_winter(utc, fires):
    # 2026-11-26 is Thanksgiving: Wednesday 11-25 is the last session before Fri 11-27.
    assert bool(check("2026-11-27", now=utc)) is fires


@pytest.mark.parametrize("utc, fires", [
    # Summer (EDT, UTC−4): Thu 2026-07-16 before Fri 07-17.
    (datetime(2026, 7, 16, 19, 29, tzinfo=timezone.utc), False),    # 15:29 ET
    (datetime(2026, 7, 16, 19, 30, tzinfo=timezone.utc), True),     # 15:30 ET
])
def test_summer(utc, fires):
    assert bool(check("2026-07-17", now=utc)) is fires


def test_not_the_last_day_and_empty():
    assert check("2026-12-18", now=datetime(2026, 11, 25, 21, 0, tzinfo=timezone.utc)) == ""
    assert check("", now=datetime(2026, 11, 25, 21, 0, tzinfo=timezone.utc)) == ""

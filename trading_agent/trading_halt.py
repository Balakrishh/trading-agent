"""trading_halt.py — kill switch + drawdown governor (2026-10-07, backlog §9).

One switch pauses **new entries**; exits, stops and profit-taking keep
running, so a pause never leaves open positions unmanaged (the older 5 %
daily-drawdown breaker exits the whole process, exits included, and stays
as the last line).

Who can pause:

* the operator — ``python -m trading_agent.trading_halt pause --reason "…"``
  (or later a Telegram button);
* the governor — automatically, when equity falls ``halt_daily_loss_pct``
  (2 %) below the day's first reading or ``halt_weekly_loss_pct`` (5 %)
  below the week's first reading.

Only a human resumes: ``python -m trading_agent.trading_halt resume``. A
governor pause is never lifted automatically, even if equity recovers —
the point is that a person looks before the agent adds risk again.

Every change is journaled (``trading_halt_set`` / ``trading_halt_cleared``).
State: ``trade_journal/trading_halt.json`` (atomic temp + rename); it also
holds the day / week equity baselines the governor measures from.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATE_PATH = Path(os.environ.get("TRADING_AGENT_HALT", "trade_journal/trading_halt.json"))


@dataclass
class HaltState:
    paused: bool = False
    reason: str = ""
    set_by: str = ""                 # "operator" | "governor"
    set_at: str = ""
    day: str = ""                    # ET date of day_start_equity
    day_start_equity: float = 0.0
    week: str = ""                   # ISO year-week of week_start_equity
    week_start_equity: float = 0.0


def load(path: Optional[Path] = None) -> HaltState:
    try:
        return HaltState(**json.loads((path or STATE_PATH).read_text()))
    except (OSError, ValueError, TypeError):
        return HaltState()


def save(state: HaltState, path: Optional[Path] = None) -> None:
    p = path or STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(p)


def _keys(now_utc: datetime) -> Tuple[str, str]:
    et = now_utc.astimezone(ET)
    iso = et.isocalendar()
    return et.date().isoformat(), f"{iso[0]}-W{iso[1]:02d}"


def govern(state: HaltState, equity: float, now_utc: datetime,
           daily_pct: float, weekly_pct: float) -> Tuple[HaltState, Optional[str]]:
    """Roll the day / week equity baselines and pause on a breach.
    Returns (state, reason) where ``reason`` is set only when this call
    newly paused. Pure apart from reading the clock argument."""
    day, week = _keys(now_utc)
    if equity > 0:
        if state.day != day or state.day_start_equity <= 0:
            state.day, state.day_start_equity = day, equity
        if state.week != week or state.week_start_equity <= 0:
            state.week, state.week_start_equity = week, equity
    if state.paused or equity <= 0:
        return state, None
    breach = None
    if daily_pct > 0 and state.day_start_equity > 0:
        dd = 1 - equity / state.day_start_equity
        if dd >= daily_pct:
            breach = (f"daily loss {dd:.1%} ≥ {daily_pct:.0%} "
                      f"(${state.day_start_equity:,.0f} → ${equity:,.0f})")
    if breach is None and weekly_pct > 0 and state.week_start_equity > 0:
        dd = 1 - equity / state.week_start_equity
        if dd >= weekly_pct:
            breach = (f"weekly loss {dd:.1%} ≥ {weekly_pct:.0%} "
                      f"(${state.week_start_equity:,.0f} → ${equity:,.0f})")
    if breach:
        state.paused, state.reason, state.set_by = True, breach, "governor"
        state.set_at = now_utc.isoformat()
    return state, breach


def _journal(action: str, state: HaltState) -> None:
    try:
        from trading_agent.journal_kb import JournalKB
        JournalKB("trade_journal", run_mode="live").log_signal(
            ticker="__halt__", action=action, price=0.0, raw_signal=asdict(state),
            exec_status=action, notes=f"{action}: {state.reason}")
    except Exception as exc:                      # noqa: BLE001 — the switch itself must still work
        print(f"warning: journal row not written ({exc})", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pause / resume new entries (exits keep running).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("pause")
    p.add_argument("--reason", required=True)
    r = sub.add_parser("resume")
    r.add_argument("--reason", default="operator resumed")
    a = ap.parse_args(argv)
    st = load()
    if a.cmd == "status":
        print(json.dumps(asdict(st), indent=2))
        return 0
    now = datetime.now(timezone.utc).isoformat()
    if a.cmd == "pause":
        st.paused, st.reason, st.set_by, st.set_at = True, a.reason, "operator", now
        save(st)
        _journal("trading_halt_set", st)
        print(f"Paused new entries: {a.reason}. Exits keep running. Resume with: "
              f"python -m trading_agent.trading_halt resume")
        return 0
    was = st.reason
    st.paused, st.reason, st.set_by, st.set_at = False, a.reason, "operator", now
    save(st)
    _journal("trading_halt_cleared", st)
    print(f"Resumed new entries (was paused: {was or 'not paused'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

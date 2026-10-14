"""liveness.py — page the operator when the agent stops cycling (skill 64).

The agent cannot report its own death, so this runs as a separate
one-shot (systemd timer every 5 min on weekdays, runbook 08). During the
NYSE session, once ``GRACE_AFTER_OPEN_MIN`` has passed since the open, the
newest row in ``signals_live.jsonl`` must be younger than ``STALE_MIN``
and the data server's ``/health`` must answer. Otherwise one error-channel
Telegram alert goes out per outage; a "recovered" note follows when both
are healthy again. State lives in ``trade_journal/liveness_state.json``.

Usage:  python -m trading_agent.liveness          (cron / timer)
        python -m trading_agent.liveness --dry-run (print, never page)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from trading_agent.market_hours import US_MARKET_PROFILE, is_within_market_hours

logger = logging.getLogger(__name__)

STALE_MIN = 10              # no journal row for this long in session → down
GRACE_AFTER_OPEN_MIN = 10   # the first cycle lands ~09:31; allow for a slow start
TAIL_BYTES = 65536          # enough for the last several journal rows
JOURNAL = Path("trade_journal/signals_live.jsonl")
STATE_FILE = Path("trade_journal/liveness_state.json")
HEALTH_URL = "http://127.0.0.1:8765/health"


@dataclass
class LivenessState:
    alerted: bool = False
    since: str = ""          # ISO UTC of the first failed check this outage
    reason: str = ""


def last_activity(path: Path = JOURNAL) -> Optional[datetime]:
    """Timestamp of the newest journal row, or None if there is none."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - TAIL_BYTES))
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            ts = json.loads(line).get("timestamp")
            if ts:
                dt = datetime.fromisoformat(ts)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            continue
    return None


def assess(now: datetime, last: Optional[datetime], in_session: bool,
           server_ok: bool) -> Tuple[str, str]:
    """("idle" | "ok" | "down", reason). Pure — no I/O."""
    tz = US_MARKET_PROFILE.timezone
    now_et = now.astimezone(tz)
    # The profile's window opens at 09:25 (buffer); cycles start at the
    # 09:30 bell, so the grace runs from the bell.
    opened = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if not in_session or now_et < opened + timedelta(minutes=GRACE_AFTER_OPEN_MIN):
        return "idle", ""
    problems = []
    if last is None:
        problems.append("no journal rows at all")
    else:
        age_min = (now - last).total_seconds() / 60
        if age_min > STALE_MIN:
            problems.append(f"no agent cycle for {age_min:.0f} min "
                            f"(last row {last.astimezone(tz):%H:%M} ET)")
    if not server_ok:
        problems.append(f"data server not answering {HEALTH_URL}")
    return ("down", "; ".join(problems)) if problems else ("ok", "")


def server_healthy(url: str = HEALTH_URL) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def load_state(path: Path = STATE_FILE) -> LivenessState:
    try:
        return LivenessState(**json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return LivenessState()


def save_state(state: LivenessState, path: Path = STATE_FILE) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(path)


def step(state: LivenessState, status: str, reason: str,
         now: datetime) -> Tuple[LivenessState, Optional[str]]:
    """Next state and the alert to send ("down" / "recovered" / None).
    One page per outage; a recovery note only after a page."""
    if status == "down" and not state.alerted:
        return LivenessState(True, now.isoformat(), reason), "down"
    if status == "ok" and state.alerted:
        return LivenessState(), "recovered"
    return state, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print, never page")
    args = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except ImportError:                                         # noqa: skill-34-exempt — env may already be set by systemd
        pass

    now = datetime.now(timezone.utc)
    status, reason = assess(now, last_activity(),
                            is_within_market_hours(now), server_healthy())
    state = load_state()
    new_state, alert = step(state, status, reason, now)
    print(f"liveness {now:%Y-%m-%dT%H:%MZ}: {status}"
          + (f" — {reason}" if reason else "") + (f" → {alert}" if alert else ""))
    if args.dry_run or alert is None:
        return 0

    from trading_agent.telegram_notifier import TelegramNotifier
    tg = TelegramNotifier()
    if alert == "down":
        sent = tg.notify_agent_down(reason)
    else:
        sent = tg.notify_agent_recovered(state.since)
    if sent or not tg.is_active:
        save_state(new_state)   # unsent page (Telegram down) retries next run
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

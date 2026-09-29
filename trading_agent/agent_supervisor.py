"""Long-running supervisor for the headless trading agent.

The agent itself (``trading_agent.agent``) is designed to exit cleanly
outside NYSE market hours — that's an intentional invariant so nothing
runs during downtime. This supervisor wraps it in a sleep-and-restart
loop so the operator can start it once (via launchd or `nohup`) and
have it trade every market session automatically:

  09:29 ET  supervisor starts agent → agent sees "outside hours" and
            exits → supervisor sleeps until 09:30
  09:30 ET  supervisor restarts agent → agent runs cycles all session
  16:00 ET  agent exits (`after_hours_shutdown`) → supervisor computes
            next open (16:01 → tomorrow 09:30) and sleeps
  <weekend / holiday>
            supervisor skips non-trading days via
            ``US_MARKET_PROFILE.is_trading_day``.

Read/write invariant: this supervisor NEVER imports the executor or
order-submission primitives — it only shells out to
``python -m trading_agent.agent``. Verified by
``tests/conformance/test_skill_57_supervisor.py``.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Optional


log = logging.getLogger("trading_agent.agent_supervisor")


# Bounds for the sleep-until-next-open computation. Even if the math
# comes back with weeks (a long holiday), the supervisor wakes at least
# once every 12 hours so the operator can inspect a stuck process.
_MIN_SLEEP_SEC = 60
_MAX_SLEEP_SEC = 12 * 60 * 60

# Backoff after an unexpected agent crash. launchd's own ThrottleInterval
# also applies at the process level; this is the in-supervisor throttle
# for consecutive crashes inside one supervisor lifetime.
_CRASH_BACKOFF_SEC = 60


def seconds_until_next_open(now: Optional[datetime] = None) -> int:
    """Return seconds until the next NYSE regular-session open.

    Uses ``US_MARKET_PROFILE`` for holidays / weekend handling. Result
    is clamped to ``[_MIN_SLEEP_SEC, _MAX_SLEEP_SEC]`` so a wildly
    long window (or a bug in the calendar) can never put the
    supervisor to sleep for days.
    """
    from trading_agent.market_profile import US_MARKET_PROFILE      # noqa: PLC0415
    from trading_agent.market_hours import is_within_market_hours   # noqa: PLC0415

    profile = US_MARKET_PROFILE
    tz = profile.timezone
    now = now or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    # If already inside the session, no sleep needed.
    if is_within_market_hours(now, profile):
        return _MIN_SLEEP_SEC

    # Candidate today's open in the profile's timezone.
    today_open = now.replace(
        hour=profile.open_hour, minute=profile.open_minute,
        second=0, microsecond=0,
    )
    candidate = today_open if now < today_open and \
                              profile.is_trading_day(now.date()) \
                            else today_open + timedelta(days=1)

    # Walk forward until we find a trading day.
    for _ in range(14):  # safety cap — 2 weeks of consecutive holidays
        if profile.is_trading_day(candidate.date()) and candidate > now:
            break
        candidate = candidate + timedelta(days=1)

    delta = (candidate - now).total_seconds()
    return max(_MIN_SLEEP_SEC, min(_MAX_SLEEP_SEC, int(delta)))


def _run_agent_once(env: Optional[dict] = None) -> int:
    """Spawn ``python -m trading_agent.agent`` and return its exit code.

    Using a subprocess isolates the supervisor from any lingering state
    the agent's global loggers or signal handlers might leave behind,
    and lets the supervisor stay tiny + testable in isolation.
    """
    # Same interpreter that started the supervisor keeps environment
    # parity; the operator's venv activation carries over automatically.
    cmd = [sys.executable, "-m", "trading_agent.agent"]
    log.info("Starting agent subprocess: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, env=env or os.environ.copy())
    except FileNotFoundError as exc:
        log.error("Cannot spawn agent (%s). Aborting supervisor.", exc)
        return 127

    # Forward SIGTERM to the child so launchd can stop the whole tree
    # cleanly. SIGINT is handled by Popen out of the box.
    def _forward(signum, _frame):
        log.info("Received signal %d — forwarding to agent PID %d",
                 signum, proc.pid)
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    old_term = signal.signal(signal.SIGTERM, _forward)
    try:
        return proc.wait()
    finally:
        signal.signal(signal.SIGTERM, old_term)


def supervise(
    *,
    max_iterations: Optional[int] = None,
    sleep_fn=time.sleep,
    now_fn=None,
) -> int:
    """Run the supervisor loop.

    ``max_iterations`` — safety knob for tests (unbounded when None).
    ``sleep_fn`` — injectable for tests so they don't actually sleep.
    ``now_fn`` — injectable clock for tests.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    iteration = 0
    while True:
        iteration += 1
        if max_iterations is not None and iteration > max_iterations:
            log.info("Reached max_iterations=%d — exiting.", max_iterations)
            return 0

        exit_code = _run_agent_once()
        log.info("Agent exited with code %d (iteration %d)",
                 exit_code, iteration)

        # A non-zero exit means the agent crashed or hit an error that
        # bypassed the graceful shutdown path. Back off, then retry —
        # launchd's ThrottleInterval provides the outer bound.
        if exit_code != 0:
            log.warning("Non-zero exit — backing off %ds before retry.",
                         _CRASH_BACKOFF_SEC)
            sleep_fn(_CRASH_BACKOFF_SEC)
            continue

        # Graceful exit → sleep until next market open, then loop.
        seconds = seconds_until_next_open(now_fn() if now_fn else None)
        log.info("Sleeping %d seconds until next NYSE open.", seconds)
        sleep_fn(seconds)


def main(argv: Optional[list[str]] = None) -> int:
    _ = argv  # no CLI flags today
    return supervise()


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(main())

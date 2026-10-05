# Agent Supervisor — Long-running Wrapper

> **One-line summary:** Wraps `trading_agent.agent` in a sleep-and-restart loop so a single launchd unit keeps the agent trading every session, sleeping cleanly between market opens.
> **Source of truth:** [`trading_agent/agent_supervisor.py`](../../trading_agent/agent_supervisor.py), [`ops/launchd/com.trading-agent.headless.plist`](../../ops/launchd/com.trading-agent.headless.plist).
> **Phase:** 3  •  **Group:** ops
> **Depends on:** `trading_agent.agent` (the process the supervisor runs), `trading_agent.market_hours` + `trading_agent.market_profile` (calendar for sleep math).
> **Consumed by:** the launchd unit + any operator who wants "install once, forget about it" semantics.

---

## 1. Theory & Objective

The trading agent itself exits gracefully outside NYSE market hours (skill 00 §Shutdown, see `after_hours_shutdown` in `shutdown.py`). That's a load-bearing invariant — the agent should not run during downtime, and hot-looping around it (via naive `KeepAlive=true`) would consume power and log noise for no benefit. This supervisor closes the gap between "exits when market closes" and "operator wants it to trade every session without manual restarts" by computing the exact number of seconds until the next open and sleeping until then.

Read/write invariant: the supervisor is a **spawner**, not a trader. It shells out to `python -m trading_agent.agent` and never imports the executor, the order-submission primitives, or `pending_orders/`. AST-verified.

## 2. Mathematical Formula

`seconds_until_next_open(now)`:

```text
if is_within_market_hours(now):           return _MIN_SLEEP_SEC (60s)
today_open  = now.replace(open_hour, open_minute)
candidate   = today_open if (now < today_open AND is_trading_day(now.date()))
              else today_open + 1 day
while not is_trading_day(candidate.date()) or candidate <= now:
    candidate += 1 day    (safety cap = 14 iterations for consecutive holidays)
return clamp(int(candidate - now).total_seconds(),
             _MIN_SLEEP_SEC, _MAX_SLEEP_SEC)
```

Bounds:
- `_MIN_SLEEP_SEC = 60` — the supervisor never sleeps less than a minute to prevent tight spinning.
- `_MAX_SLEEP_SEC = 43200` (12 h) — even a long holiday wakes the supervisor at least twice a day so a stuck process is visible in `ps`.

## 3. Reference Python Implementation

### 3.1 Sleep math

```python
# trading_agent/agent_supervisor.py
def seconds_until_next_open(now: Optional[datetime] = None) -> int:
    """Return seconds until the next NYSE regular-session open."""
```

Uses `US_MARKET_PROFILE.is_trading_day` — the same calendar `is_within_market_hours` uses — so weekends AND holidays skip identically.

### 3.2 Supervise loop

```python
# trading_agent/agent_supervisor.py
def supervise(
    *,
    max_iterations: Optional[int] = None,
    sleep_fn=wall_clock_sleep,
    now_fn=None,
) -> int:
    """Run the supervisor loop.
```

Two exit paths from the child agent:

- **Exit code 0** — graceful (`after_hours_shutdown` or operator SIGTERM). Compute sleep, wait, restart.
- **Non-zero exit** — crash or unhandled exception in the agent. Back off `_CRASH_BACKOFF_SEC` (60 s) then retry. launchd's own `ThrottleInterval` provides the outer bound if the supervisor itself keeps crashing.

`max_iterations`, `sleep_fn`, and `now_fn` are injectable so the loop is testable without actually sleeping.

### 3.3 Signal forwarding

The supervisor installs a `SIGTERM` handler that forwards the signal to the child. When launchd stops the unit, the whole tree (supervisor + agent + any pending Alpaca calls) shuts down cleanly.

### 3.4 launchd unit

`ops/launchd/com.trading-agent.headless.plist` — `RunAtLoad=true`, `KeepAlive=true` (belt-and-suspenders around the supervisor's own loop), `ThrottleInterval=60`. Requires two hand-edits (venv python + repo dir) before installing.

## 4. Edge Cases / Guardrails

- **No executor imports.** The supervisor is a spawner. Conformance test AST-walks `agent_supervisor.py` and rejects any reference to `submit_order`, `place_order`, `OrderExecutor`, or `trading_agent.executor`.
- **Sleep is bounded.** A bug in `is_trading_day` (e.g. an incorrect holiday list) can never put the supervisor to sleep for more than 12 hours.
- **Consecutive-holiday cap.** The while-loop is capped at 14 iterations so a broken calendar library can't infinite-loop.
- **Crash backoff distinct from cadence.** A crashed agent restarts after 60 s, not "next market open" — because a crash inside market hours means the operator wants trading to resume ASAP, not tomorrow.
- **Signal forwarding.** `SIGTERM` from launchd propagates to the child; the agent then runs its own graceful shutdown (skill 00 §Shutdown).
- **`RunAtLoad=true` is safe.** Installing the plist during market close makes the agent try to start, see "outside hours", exit, and the supervisor sleeps until next open. No trades on install.
- **Same interpreter as supervisor.** `_run_agent_once` uses `sys.executable`, so the agent inherits whatever venv the supervisor was launched in. No PATH ambiguity between the launchd process and the trading agent.

- **System sleep (2026-10-05).** On macOS `time.sleep` does not advance while the machine sleeps, so a single 12 h wait stretched by the whole lid-closed time (woke Sat 14:36 instead of 04:05; missed Monday's 09:25 open). `wall_clock_sleep` sleeps in ≤ 300 s slices and re-checks `time.time()`, so a wake is at most 5 minutes late once the Mac is awake. It cannot run while the Mac itself is asleep — keep it awake in market hours (e.g. `pmset` wake schedule or Energy settings).
- **Cycle cadence (documented 2026-10-05).** In-session the supervisor sleeps `in_session_sleep_sec()` = `AGENT_CYCLE_SLEEP_SEC` (default 60, clamped 60–900) after each agent exit, so with ~15 s cycles the agent runs about every 75 s (≈300 cycles/day, measured 2026-09-29…10-02) — not every 5 minutes. Exit debounce of 3 cycles is therefore ≈ 4 min. Set `AGENT_CYCLE_SLEEP_SEC=285` for a ~5-minute cadence.

## 5. Cross-References

- `00_sdlc_and_conventions.md` — the "graceful exit outside market hours" invariant this supervisor honors.
- `trading_agent.agent` — the subprocess this supervisor manages (not itself a skill; the shape is documented in the agent's module docstring).
- `56_daily_journal_reviewer.md` — same launchd-unit pattern; the reviewer plist and the supervisor plist coexist safely (separate labels, separate log paths).
- `ops/launchd/com.trading-agent.headless.plist` — the launchd unit.

---

*Last verified against repo HEAD on 2026-10-05.*

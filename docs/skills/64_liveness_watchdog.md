# Liveness Watchdog — Page When the Agent Stops

> **One-line summary:** a separate 5-minute timer checks, during the NYSE session, that the agent wrote a journal row in the last 10 minutes and that the data server answers; one Telegram page per outage, one "recovered" note after.
> **Source of truth:** [`trading_agent/liveness.py`](../../trading_agent/liveness.py), `deploy/systemd/trading-agent-liveness.{service,timer}`
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `32_telegram_operator_alerts.md`, `47_schwab_data_api.md`, `19_journal_schema.md`
> **Consumed by:** systemd user timer on the Pi (runbook 08)

---

## 1. Theory & Objective

A process cannot report its own death. Before this, a crash, a hung cycle, a failed boot or a missed open went unnoticed until someone looked (backlog §9: "a missed open still goes unnoticed"; 2026-10-13 the Pi was rebooted at 19:25 ET and only a manual check confirmed it came back). The watchdog is deliberately outside the agent: a one-shot run by its own systemd timer, reading only the journal file and the data server's `/health`. It pages once per outage on the Telegram error channel and once more when healthy again, so a long outage is one message, not 78.

## 2. Mathematical Formula

```text
idle   if not in the NYSE session (holidays, weekends, after hours)
       or now < 09:30 ET + GRACE_AFTER_OPEN_MIN (10)
down   if (now − newest signals_live.jsonl timestamp) > STALE_MIN (10 min)
       or there is no journal row at all
       or GET http://127.0.0.1:8765/health is not 200 within 5 s
       or (A8, 2026-10-26) the server answers but GET /ready is 503 —
          Schwab returned no data (token revoked / refresh failing / upstream);
          /ready unreachable counts as ok here (the /health line covers it)
ok     otherwise

alert  = "down"       when status = down and no page is outstanding
       = "recovered"  when status = ok   and a page is outstanding
       = none         otherwise
```

The agent writes rows every ~75 s in session (a `vix_observation` row each cycle at minimum), so 10 minutes is about 8 missed cycles.

## 3. Reference Python Implementation

```python
# trading_agent/liveness.py:66-86
def assess(now: datetime, last: Optional[datetime], in_session: bool,
           server_ok: bool, schwab_ok: bool = True, schwab_detail: str = ""
           ) -> Tuple[str, str]:
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
    elif not schwab_ok:
        problems.append("Schwab not returning data (data server /ready: "
                        f"{schwab_detail or 'not ready'}) — re-login may be needed")
    return ("down", "; ".join(problems)) if problems else ("ok", "")

# liveness.schwab_ready(url=READY_URL) -> (ok, detail): 200 → ok; 503 → the
# server's detail; unreachable → ok (reported by server_healthy). main() calls
# it only in session (it makes one Schwab call through the data server).
# Data server /ready (skill 47) calls provider.auth_probe(): a token must be
# obtainable AND /markets must return a body.
```

```python
# trading_agent/liveness.py:110-118
def step(state: LivenessState, status: str, reason: str,
         now: datetime) -> Tuple[LivenessState, Optional[str]]:
    """Next state and the alert to send ("down" / "recovered" / None).
    One page per outage; a recovery note only after a page."""
    if status == "down" and not state.alerted:
        return LivenessState(True, now.isoformat(), reason), "down"
    if status == "ok" and state.alerted:
        return LivenessState(), "recovered"
    return state, None
```

Install (runbook 08 §7): `systemctl --user enable --now trading-agent-liveness.timer`. Manual check: `myenv/bin/python -m trading_agent.liveness --dry-run`.

## 4. Edge Cases / Guardrails

- **Holidays and weekends** — `is_within_market_hours` uses the NYSE calendar, so the timer's weekday schedule may fire on a holiday and the script stays `idle`.
- **Window buffer** — the market profile's window opens 09:25; the grace is anchored to the 09:30 bell, so checks start at 09:40.
- **One page per outage** — `liveness_state.json` (atomic temp+rename) holds `alerted`/`since`/`reason`; repeated `down` checks send nothing more.
- **Telegram unreachable** — the state is saved only after a successful send, so an unsent page is retried on the next run. With Telegram not configured the state still advances (nothing to retry).
- **Corrupt or missing state / journal** — corrupt state reads as "no outstanding page"; a missing journal is `down` ("no journal rows at all").
- **Agent alive but Schwab dead (A8, 2026-10-26)** — on 2026-10-20 the data server's refresh token was invalid from 10:15 to 20:22 ET while `/health` stayed 200 and the agent kept writing rows, so no page went out. `/ready` used to probe `is_market_open()`, which swallows the auth failure (`_get` returns None) and falls back to Alpaca's clock — it reported "ready" throughout. Now `/ready` uses `auth_probe()` and the watchdog pages on a 503 during the session ("Schwab not returning data … re-login may be needed"); "recovered" follows once it answers again. Outside the session nothing is checked — the Sunday-night re-login reminder (skill 16 §4) covers weekends.
- **Server down** — only the `/health` problem is reported, not a second Schwab line.
- **Read-only** — never touches the agent, orders or `pending_orders/`; writes only its own state file and the log `/tmp/trading-agent.liveness.log`.

## 5. Cross-References

- `32_telegram_operator_alerts.md` — `notify_agent_down` / `notify_agent_recovered` (error channel).
- `47_schwab_data_api.md` — the `/health` and `/ready` routes checked here.
- `62_kill_switch_drawdown_governor.md` — the other operator lever for an unattended agent.

---

*Last verified against repo HEAD on 2026-10-26.*

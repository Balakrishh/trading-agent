# Kill Switch & Drawdown Governor

> **One-line summary:** one switch pauses new entries while exits, stops and profit-taking keep running; the governor flips it automatically at a daily (2 %) or weekly (5 %) equity loss; only a human resumes.
> **Source of truth:** [`trading_agent/trading_halt.py`](../../trading_agent/trading_halt.py)
> **Phase:** 2  •  **Group:** risk / ops
> **Depends on:** `37_position_caps.md`, `58_market_state_playbook.md`, `32_telegram_operator_alerts.md`
> **Consumed by:** `agent.py` (`_trading_halt_result`, start of Stage 2), MCP `get_trading_halt`, the operator CLI

---

## 1. Theory & Objective

An autonomous agent needs a stop the operator can reach in one command, and a rule that stops it without the operator when losses run. The existing 5 % daily-drawdown breaker exits the whole process — the supervisor restarts it, it trips again — so while it is tripped nothing manages the open positions either. The governor is the gentler first line: it pauses only *new entries*, at tighter limits (−2 % on the day, −5 % on the week), and leaves exits running. Recovery does not un-pause it: a person looks before the agent adds risk again. Backlog §9 (autonomy with a human in the loop).

## 2. Mathematical Formula

```text
day_start_equity  = first equity reading of the ET trading day
week_start_equity = first equity reading of the ISO week (ET)
pause if  1 − equity / day_start_equity  ≥ halt_daily_loss_pct   (default 0.02; 0 = off)
      or  1 − equity / week_start_equity ≥ halt_weekly_loss_pct  (default 0.05; 0 = off)
resume only by the operator:  python -m trading_agent.trading_halt resume
```

## 3. Reference Python Implementation

```python
# trading_agent/trading_halt.py:73-100
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
```

```python
# trading_agent/agent.py:2259-2286
    def _trading_halt_result(self, account_balance: float, monitor_results: Dict) -> Optional[Dict]:
        """Kill switch + drawdown governor (backlog §9, trading_halt.py).
        Returns the Stage 2 skip result while new entries are paused, else
        None. Exits already ran in Stage 1 and are never paused here."""
        from trading_agent import trading_halt as th
        try:
            state = th.load()
            state, tripped = th.govern(
                state, account_balance, datetime.now(timezone.utc),
                float(getattr(self.preset, "halt_daily_loss_pct", 0.0) or 0.0),
                float(getattr(self.preset, "halt_weekly_loss_pct", 0.0) or 0.0))
            th.save(state)
        except Exception as exc:  # noqa: skill-34-exempt — a broken halt file must not stop exits; entries continue
            logger.warning("Trading-halt check failed: %s", exc)
            return None
        if tripped:
            logger.critical("DRAWDOWN GOVERNOR — new entries paused: %s", tripped)
            self.journal_kb.log_signal(ticker="__halt__", action="trading_halt_set",
                                       price=0.0, raw_signal=dataclasses.asdict(state))
            try:
                self.telegram.notify_trading_halt(f"Drawdown governor: {tripped}")
            except Exception as exc:  # noqa: skill-34-exempt — alert is best-effort; the pause is already saved
                logger.warning("Halt alert not sent: %s", exc)
        if not state.paused:
            return None
        logger.warning("STAGE 2 SKIPPED — new entries paused (%s, by %s). Exits continue.",
                       state.reason, state.set_by)
        return self._stage2_skip(account_balance, monitor_results, "trading_halt")
```

## 4. Edge Cases / Guardrails

- **Exits never pause.** The check runs at the start of Stage 2; Stage 1 (exits, stops, profit-taking, trailing) has already run that cycle.
- **Operator CLI.** `python -m trading_agent.trading_halt status | pause --reason "…" | resume [--reason "…"]`. Pause and resume are journaled (`__halt__` rows, actions `trading_halt_set` / `trading_halt_cleared`); a governor pause is journaled and sent to the Telegram error channel once, when it trips.
- **No auto-resume.** A governor pause survives equity recovering and new days; the next day's baseline still rolls so the daily limit is measured fresh after a resume.
- **Broken or missing state file.** Treated as not paused (`load()` returns a default) and logged; a corrupt halt file must never stop exits, and new entries keep their other gates.
- **Zero equity reading** (broker hiccup): baselines are not overwritten and no breach is computed.
- **Relation to the 5 % breaker.** Unchanged and still the last line (it exits the process). With defaults the governor trips first.
- **MCP is read-only.** `get_trading_halt` shows the state; pausing / resuming stays with the operator (Telegram buttons are a backlog §9 item).

## 5. Cross-References

- `37_position_caps.md` — the other entry gates (caps, ladder).
- `58_market_state_playbook.md` — market-wide sizing; CAPITULATION also skips Stage 2.
- `61_entry_confirmation.md` — the per-trade entry gates that follow when not paused.
- `48_claude_code_mcp_surface.md` — `get_trading_halt`.

---

*Last verified against repo HEAD on 2026-10-07.*

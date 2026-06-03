# Close-Failure Action + Partial-Fill Cooldown + PDT Suppression

> **One-line summary:** When `executor.close_spread` reports a partial-fill zombie, the agent (a) tags the journal row `action="close_failed"` instead of `"closed"`, (b) **derives** the per-ticker partial-fill streak from journal rows and parks the ticker in a 60-min cooldown after 3 strikes (cross-process safe), (c) suppresses same-day `REGIME_SHIFT` exits on PDT-restricted accounts (<$25K equity), and (d) **reactively** suppresses ALL subsequent close attempts on a ticker once Alpaca has actually returned code `40310100` (pattern day trading protection) earlier the same UTC day — observed via the API response, not speculatively predicted.
> **Source of truth:** [`trading_agent/agent.py:124-220`](../../trading_agent/agent.py), [`trading_agent/agent.py:867-1015`](../../trading_agent/agent.py), [`trading_agent/agent.py:_close_failed_streak_within_window`](../../trading_agent/agent.py), [`trading_agent/streamlit/live_monitor.py:1095-1316`](../../trading_agent/streamlit/live_monitor.py), [`trading_agent/sector_map.py`](../../trading_agent/sector_map.py) *(per-sector cap sibling, added 2026-05-15)*
> **Phase:** 2  •  **Group:** risk
> **Depends on:** `00_sdlc_and_conventions.md` (journal schema + dedup), `13_preset_system_hot_reload.md` (max_risk_pct + budget mismatch warning), `19_journal_schema.md` (close_failed action enum)
> **Consumed by:** `streamlit/live_monitor.py:_render_close_failures_today` (UI banner), Recent Journal Entries panel (downstream of `journal_kb.log_signal`).

---

## 1. Theory & Objective

Three failure modes recur during the close path and the agent treats each one structurally rather than retrying blindly.

**Partial-fill zombie state.** Alpaca evaluates each `DELETE /v2/positions/{symbol}` call independently. When one leg of a multi-leg spread closes successfully and another is rejected (uncovered, PDT, insufficient buying power), the position ends up in a hybrid state — one leg flat, one leg still open on the broker. Naïve retry on the next 5-min cycle just repeats the failure: the broker's reason (PDT lockout, insufficient buying power) doesn't lift on its own. On 2026-05-06 a single SPY position produced 11 close attempts in one session because the cycle kept retrying every 5 minutes.

**Pattern Day Trading (PDT) protection.** FINRA defines a "day trade" as opening AND closing the same security same day. Accounts under $25,000 are limited to 3 day trades in any 5-day rolling window; the 4th triggers a 90-day flag. Alpaca enforces this server-side and rejects close requests with HTTP 403 + code 40310100 ("trade denied due to pattern day trading protection"). For the same-day open + `REGIME_SHIFT` close pattern this is exactly the failure we hit at $5K equity.

**Dashboard-tile dishonesty.** Pre-fix every close attempt — complete OR zombie — was tagged `action="closed"`. The "Closed Today" tile counted both, so an operator watching a single zombie loop saw "11 closed today" and panicked. The fix splits the action into two values so the tile counts only complete fills.

The intent is **fail loud, never silently retry forever, and tell the operator exactly when manual broker intervention is required.**

## 2. Constants and thresholds

```text
PDT_EQUITY_THRESHOLD          = $25,000.0
                                FINRA's PDT cutoff. Sub-threshold accounts
                                must avoid same-day open+close.

PARTIAL_CLOSE_COOLDOWN_THRESHOLD = 3
                                After this many consecutive partial fills
                                on a single underlying, the ticker enters
                                cooldown.

CLOSE_COOLDOWN_MINUTES        = 60
                                Length of the parked window. Auto-closes
                                are suppressed for the underlying until
                                this elapses or an operator manually
                                clears the position on the broker.
```

These are module-level constants in `trading_agent/agent.py` so they're reachable from both the cycle and the test suite. They are **not** preset fields — they're protocol-level safety rules, not strategy knobs.

## 3. Reference Python Implementation

### `trading_agent/agent.py:124-200` — constants + state init

```python
# ── Pattern Day Trading (PDT) threshold ─────────────────────────────────
PDT_EQUITY_THRESHOLD = 25_000.0

# ── Partial-close cooldown ──────────────────────────────────────────────
PARTIAL_CLOSE_COOLDOWN_THRESHOLD = 3
CLOSE_COOLDOWN_MINUTES = 60

class TradingAgent:
    def __init__(self, config: AppConfig):
        ...
        # Cooldown state is derived from the journal on every read
        # (see _close_failed_streak_within_window).  No instance dicts
        # needed.  The pre-2026-05-13 in-memory dicts were reset on
        # every cycle process-restart in production, so the cooldown
        # never engaged — see the 2026-05-13 XLF/GLD post-mortem and
        # § "Edge Cases" below.
```

### `trading_agent/agent.py:867-994` — close loop with PDT + cooldown gates

```python
same_day_tickers = self._tickers_opened_today()
pdt_restricted = account_balance < PDT_EQUITY_THRESHOLD
...
for spread in spreads:
    if spread.exit_signal != ExitSignal.HOLD and self._should_exit_spread(spread):
        # Cooldown guard — derived from journal close_failed rows
        cooldown_left = self._close_cooldown_minutes_remaining(spread.underlying)
        if cooldown_left > 0:
            streak, _ = self._close_failed_streak_within_window(
                spread.underlying)
            logger.warning(
                "[%s] Close cooldown active — %d min remaining "
                "(%d consecutive partial fills). Skipping retry...",
                spread.underlying, cooldown_left, streak)
            continue

        # PDT REGIME_SHIFT suppression
        if (pdt_restricted
                and spread.underlying in same_day_tickers
                and spread.exit_signal == ExitSignal.REGIME_SHIFT):
            logger.warning("[%s] PDT-suppressed REGIME_SHIFT exit ...", ...)
            continue

        # ... close path runs only if both gates pass ...
        result = self.executor.close_spread(spread)
        fill_status = "complete" if result.get("all_closed") else "partial"
        # IMPORTANT: bookkeeping BEFORE journaling so the row captures
        # the resulting cooldown state.
        if fill_status == "complete":
            self._clear_close_cooldown(spread.underlying)
        else:
            self._record_partial_close(spread.underlying)
        self._journal_close_event(
            spread, close_context, leg_results=leg_results,
            fill_status=fill_status, dry_run=False)
```

### `trading_agent/agent.py:1003-1118` — `_journal_close_event` action mapping

```python
def _journal_close_event(self, spread, ctx, leg_results, fill_status, dry_run):
    ...
    is_complete_close = fill_status in ("complete", "dry_run")
    if is_complete_close:
        action = "closed"
        note = f"closed: {ctx['strategy']}, P&L=±$X, {ctx['exit_signal']}"
        exec_status = f"closed_{ctx['exit_signal']}"
    else:
        action = "close_failed"
        failed_legs = [leg["symbol"] for leg in leg_results
                       if leg.get("status") != "closed"]
        note = (f"close_failed: {ctx['strategy']}, "
                f"{ctx['exit_signal']}, fill_status={fill_status} "
                f"failed_legs={','.join(failed_legs)}")
        exec_status = f"close_failed_{ctx['exit_signal']}"

    payload = dict(ctx) | {"leg_close_results": [...],
                           "fill_status": fill_status,
                           "mode": "dry_run" if dry_run else "live"}

    # Surface streak/cooldown on close_failed rows so the dashboard
    # can render a manual-intervention banner without log scraping.
    # Streak is journal-derived: existing close_failed rows in the
    # last 60 min + 1 for the row we're about to write.
    if action == "close_failed":
        existing_streak, _ = self._close_failed_streak_within_window(
            spread.underlying)
        streak = existing_streak + 1
        payload["partial_close_streak"] = streak
        payload["partial_close_threshold"] = PARTIAL_CLOSE_COOLDOWN_THRESHOLD
        if streak >= PARTIAL_CLOSE_COOLDOWN_THRESHOLD:
            deadline = (datetime.now(timezone.utc)
                        + timedelta(minutes=CLOSE_COOLDOWN_MINUTES))
            payload["close_cooldown_until"] = deadline.isoformat()
            payload["close_cooldown_reason"] = (
                f"{streak} consecutive partial fills ≥ threshold "
                f"{PARTIAL_CLOSE_COOLDOWN_THRESHOLD}; auto-close suppressed "
                f"for {CLOSE_COOLDOWN_MINUTES} min — manual broker "
                f"intervention required to clear zombie state.")

    self.journal_kb.log_signal(
        ticker=spread.underlying, action=action,
        price=self._cached_price(spread.underlying),
        exec_status=exec_status, notes=note, raw_signal=payload)
```

### `trading_agent/agent.py:1129-1265` — helper methods

```python
def _tickers_opened_today(self) -> Set[str]:
    """Reads journal directly so the answer survives a Streamlit
    or agent-loop restart. Returns set of underlyings with
    action="submitted" timestamps in today UTC."""
    # ... reads self.journal_kb.jsonl_path, parses each line ...

def _close_failed_streak_within_window(
    self, ticker: str, window_min: int = CLOSE_COOLDOWN_MINUTES,
) -> Tuple[int, Optional[datetime]]:
    """Count consecutive close_failed rows for `ticker` in the last
    `window_min` minutes, since the most recent `closed` row (which
    resets the streak).  Returns (streak, last_failure_ts).

    Journal-derived rather than in-memory — survives process restart
    (which is the failure mode the pre-2026-05-13 design hit in
    production)."""

def _close_cooldown_minutes_remaining(self, ticker: str) -> int:
    """Minutes remaining or 0. Derived from
    _close_failed_streak_within_window:
        if streak ≥ PARTIAL_CLOSE_COOLDOWN_THRESHOLD:
            deadline = last_failure_ts + CLOSE_COOLDOWN_MINUTES
            return max(1, (deadline - now).minutes)
    """

def _record_partial_close(self, ticker: str) -> None:
    """No-op as of 2026-05-13.  Kept for caller-shape stability.
    The actual streak is journal-derived; this method only emits an
    info-level log line for situational awareness."""

def _clear_close_cooldown(self, ticker: str) -> None:
    """No-op as of 2026-05-13.  The `closed` row that triggered the
    call IS the supersede signal — subsequent reads of
    _close_failed_streak_within_window will see it and reset to 0."""
```

### `trading_agent/streamlit/live_monitor.py:1170-1316` — `_render_close_failures_today` panel

```python
def _render_close_failures_today(journal_df):
    failures = journal_df[journal_df["action"] == "close_failed"]
    failures = failures[<= today UTC]
    if failures.empty: return

    cooldown_warnings = []
    for ticker, group in failures.groupby("ticker"):
        latest = group.iloc[0]
        rs = latest.get("raw_signal") or {}
        cooldown_until = parse_iso(rs.get("close_cooldown_until"))
        if cooldown_until and cooldown_until > now_utc:
            cooldown_warnings.append((ticker, cooldown_until))
        ...

    # Auto-expand if any ticker is in active cooldown — manual
    # intervention signal that shouldn't be hidden behind a click.
    with st.expander(label, expanded=bool(cooldown_warnings)):
        for ticker, until in cooldown_warnings:
            st.error(f"🚨 {ticker} is in 60-min cooldown until {until}. ...")
        st.dataframe(...)  # Streak column shows "2/3" or "🚨 cooldown until ..."
```

## 4. Edge Cases / Guardrails

- **Streak is journal-derived, not in-memory.** Pre-2026-05-13 the streak lived in an instance dict (`self._partial_close_count`) — but each cycle subprocess creates a fresh `TradingAgent`, so the dict reset every 5 minutes. The cooldown *never engaged* in production: every partial fill logged "streak 1/3" forever. The 2026-05-12 GLD incident hit 15+ retries; the 2026-05-13 XLF hit 4 before the user noticed and stopped manually. Today's streak is read from the journal on demand — count `close_failed` rows in the last 60 min, reset if a `closed` row appears in between. This works correctly across process restarts because the journal is on disk.

- **Window choice is 60 min = `CLOSE_COOLDOWN_MINUTES`.** Coincides with the cooldown duration: a streak that engages the cooldown will keep engaging it for the full hour because the last `close_failed` is always inside its own 60-min window. If the operator manually closes the position in Alpaca's UI (the prescribed remediation), the cooldown auto-expires on the next read after the deadline because no new `close_failed` rows extend the streak.

- **`_tickers_opened_today` reads the journal, not memory.** Deliberately so — it must work after a restart. Tolerates malformed JSON lines and timestamp parse errors by skipping the offending row (returning empty set on hard read failure). False-empty falls back to "we attempted a close that Alpaca might reject" — which is exactly the failure we already have to handle.

- **PDT suppression is strict — only `REGIME_SHIFT`.** Real-risk exits (`STRIKE_PROXIMITY`, `HARD_STOP`, `DTE_SAFETY`, `PROFIT_TARGET`) still fire because those events are worth the PDT hit *in principle*. But — see next bullet — Alpaca sometimes *blocks* the close outright instead of letting it count as a day trade, so we layer a reactive gate on top of the strict-real-risk policy.

- **Reactive PDT-block suppression (`pdt_blocked_today`, added 2026-05-20).** The strict-real-risk policy above assumes Alpaca will *accept* the close and merely flag it as a day trade. In practice Alpaca often *denies* the trade entirely with code `40310100` ("trade denied due to pattern day trading protection"). Once that happens for a ticker, **retrying same-day is futile** — the next ~78 5-min cycles will produce identical 4×ERROR responses with no state change until UTC midnight when the open is no longer same-day. Pi-diagnostics 2026-05-20: DIA accumulated 18 close-failed events × 4-leg DELETE calls = 72 wasted API requests in 6 hours, with the position still threatened the whole time. The fix is **reactive**: `_journal_close_event` inspects `leg_results` for substring `40310100` OR `"pattern day trading"`, and if matched writes `pdt_blocked_today=True, pdt_blocked_date=<UTC today>` on the close_failed row. Subsequent cycles call `_pdt_blocked_today_tickers()` which reads the journal and returns the set of marked tickers whose `pdt_blocked_date` matches today's UTC date. Any ticker in that set has its close attempts short-circuited with `action="skipped_pdt_blocked"`. State self-clears at UTC midnight (date-keyed). Reactive design means: we never speculatively gate; the first close attempt of any threatened position always proceeds; the gate only engages after the broker has actually told us "no."

- **Cooldown clears on a single complete close.** The ticker's slate is reset the moment Alpaca accepts all legs. We don't require N consecutive successful closes to "trust" the ticker again; one complete fill is enough evidence the broker-side block has lifted.

- **Cooldown auto-expires on read.** `_close_cooldown_minutes_remaining` purges the dict entry the first time it's queried after the deadline. The dict never grows unbounded across an all-day session.

- **`fill_status="dry_run"` maps to `action="closed"`, not `close_failed`.** A dry-run sentinel is a complete-by-design close — there's no broker-side anything to fail. Tested explicitly in `test_close_cooldown.py::test_journal_close_event_dry_run_uses_closed_action`.

- **Dashboard auto-expands the panel only when a cooldown is active.** A pre-cooldown streak (1/3 or 2/3) renders in the table but doesn't force the expander open — that's a heads-up, not an alert. A 🚨 cooldown row IS an alert and the expander opens automatically so the operator can't miss it.

- **Action `close_failed` bypasses the journal dedup gate.** Listed in `_DEDUP_BYPASS_ACTIONS` (`journal_kb.py:95-100`) alongside `submitted`, `closed`, `error`, `warning`. Successive partial-fill rows must remain individually visible so the operator can see the streak progressing toward lockout.

## 5. Cross-References

- `00_sdlc_and_conventions.md` — journal action enumeration; dedup-bypass rules.
- `13_preset_system_hot_reload.md` — `max_risk_pct` mismatch between `.env` and preset (the original 2026-05-06 incident chain that exposed the close-loop fragility).
- `18_order_submission_idempotency.md` — companion skill covering the OPEN side; `client_order_id` UUID + retry. Together skills 17 + 18 describe the agent's full broker-interaction safety story.
- `19_journal_schema.md` — full enumeration of `action` values including `closed`, `close_failed`, `warning`.
- Tests: `tests/test_close_cooldown.py` (12 cases), `tests/test_production_readiness.py` (3 cooldown-surface cases), `tests/test_sector_map.py` (25 cases for the per-sector cap sibling).

## Sibling: per-sector position cap (added 2026-05-15)

Alongside `MAX_POSITIONS_PER_TICKER = 1`, the agent now also enforces `MAX_POSITIONS_PER_SECTOR = 2` (default) via the `trading_agent.sector_map` module. Both caps are computed in the same Stage 1.5 block and union into the same `tickers_with_positions` set that Stage 2 reads. Rationale: a universe like `XLF, KRE, KBE` would otherwise let three financials stack simultaneously even though each respects the per-ticker cap — the per-sector cap prevents that concentration. Sector map source: `trading_agent/sector_map.py`. Dashboard surfaces the sector inline under the ticker name in the guardrail grid and as a new `Sector` column in the Open Positions table.

---

*Last verified against repo HEAD on 2026-06-03 (incl. per-sector cap + 5-ETF universe + ticker-forwarding hotfix in _send_telegram_alert).*

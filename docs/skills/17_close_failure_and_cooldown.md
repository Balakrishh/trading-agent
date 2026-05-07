# Close-Failure Action + Partial-Fill Cooldown + PDT Suppression

> **One-line summary:** When `executor.close_spread` reports a partial-fill zombie, the agent (a) tags the journal row `action="close_failed"` instead of `"closed"`, (b) tracks a per-ticker counter of consecutive partial fills and parks the ticker in a 60-min cooldown after 3 strikes, and (c) suppresses same-day `REGIME_SHIFT` exits on PDT-restricted accounts (<$25K equity).
> **Source of truth:** [`trading_agent/agent.py:124-200`](../../trading_agent/agent.py), [`trading_agent/agent.py:867-998`](../../trading_agent/agent.py), [`trading_agent/agent.py:1003-1265`](../../trading_agent/agent.py), [`trading_agent/streamlit/live_monitor.py:1095-1316`](../../trading_agent/streamlit/live_monitor.py)
> **Phase:** 2  •  **Group:** risk
> **Depends on:** `00_sdlc_and_conventions.md` (journal schema + dedup), `13_preset_system_hot_reload.md` (max_risk_pct + budget mismatch warning)
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
        # Per-ticker counter of consecutive partial-fill close attempts.
        # Cleared on a successful complete close; otherwise survives
        # only as long as the in-process state (in-memory only).
        self._partial_close_count: Dict[str, int] = {}
        self._close_cooldown_until: Dict[str, datetime] = {}
```

### `trading_agent/agent.py:867-994` — close loop with PDT + cooldown gates

```python
same_day_tickers = self._tickers_opened_today()
pdt_restricted = account_balance < PDT_EQUITY_THRESHOLD
...
for spread in spreads:
    if spread.exit_signal != ExitSignal.HOLD and self._should_exit_spread(spread):
        # Cooldown guard
        cooldown_left = self._close_cooldown_minutes_remaining(spread.underlying)
        if cooldown_left > 0:
            logger.warning(
                "[%s] Close cooldown active — %d min remaining "
                "(%d consecutive partial fills). Skipping retry...",
                spread.underlying, cooldown_left,
                self._partial_close_count.get(spread.underlying, 0))
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
    if action == "close_failed":
        streak = self._partial_close_count.get(spread.underlying, 0)
        payload["partial_close_streak"] = streak
        payload["partial_close_threshold"] = PARTIAL_CLOSE_COOLDOWN_THRESHOLD
        cooldown_until = self._close_cooldown_until.get(spread.underlying)
        if cooldown_until is not None:
            payload["close_cooldown_until"] = cooldown_until.isoformat()
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

def _close_cooldown_minutes_remaining(self, ticker: str) -> int:
    """Minutes remaining or 0. Auto-purges expired entries from
    the dict so it doesn't grow unbounded."""

def _record_partial_close(self, ticker: str) -> None:
    """Increment counter; if it reaches THRESHOLD, set
    close_cooldown_until = now + CLOSE_COOLDOWN_MINUTES."""

def _clear_close_cooldown(self, ticker: str) -> None:
    """Purge counter + cooldown_until on successful complete close."""
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

- **Counter survives only in-memory.** A process restart resets `_partial_close_count` and `_close_cooldown_until`. Acceptable trade-off: the journal still holds the historical record of `close_failed` rows; the cooldown rule is a "stop hammering" heuristic, not a permanent state machine. After a restart the agent will retry once, and if the failure recurs the counter rebuilds within ~15 minutes.

- **`_tickers_opened_today` reads the journal, not memory.** Deliberately so — it must work after a restart. Tolerates malformed JSON lines and timestamp parse errors by skipping the offending row (returning empty set on hard read failure). False-empty falls back to "we attempted a close that Alpaca might reject" — which is exactly the failure we already have to handle.

- **PDT suppression is strict — only `REGIME_SHIFT`.** Real-risk exits (`STRIKE_PROXIMITY`, `HARD_STOP`, `DTE_SAFETY`, `PROFIT_TARGET`) still fire because those events are worth the PDT hit. Suppressing a strike-proximity exit at $5K equity to avoid a day-trade flag would let an in-the-money short option ride to expiry.

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
- Tests: `tests/test_close_cooldown.py` (12 cases), `tests/test_production_readiness.py` (3 cooldown-surface cases).

---

*Last verified against repo HEAD on 2026-05-06.*

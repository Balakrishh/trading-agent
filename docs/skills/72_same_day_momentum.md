# Same-Day Momentum Debit

> **One-line summary:** on days whose prior VIX close ≥ 25, buy a 0DTE ≈ATM SPY debit vertical in the direction of the move since 09:35 ET at fixed entry times, and close it the same session (target / stop at 50 % of the debit, forced close 15:25 ET).
> **Source of truth:** [`trading_agent/same_day_momentum.py`](../../trading_agent/same_day_momentum.py), [`trading_agent/agent.py:2319-2437`](../../trading_agent/agent.py), [`trading_agent/position_monitor.py:1060-1086`](../../trading_agent/position_monitor.py)
> **Phase:** 2  •  **Group:** strategy / risk
> **Depends on:** `58_market_state_playbook.md`, `59_debit_spreads_calendars.md`, `62_kill_switch_drawdown_governor.md`, `69_intraday_chain_log.md`
> **Consumed by:** `agent.py` (`_stage2_gates` → `_stage_same_day`), `position_monitor.py` (`_check_same_day_exit`), `session_brief.py` (pre-market line)

---

## 1. Theory & Objective

In a high-volatility sell-off every multi-day playbook is blocked or loses (research_credit_vol.md §7–§10), so the agent sat flat for two weeks. Intraday momentum is a documented effect that is strongest on high-volatility days: the direction of the morning move tends to persist into the afternoon. A same-day debit vertical takes that direction with a defined risk (the debit) and holds nothing overnight, so the overnight-gap risk the market-state gate and the HighIV block protect against does not apply. Evidence (research_credit_vol.md §11): real logged 0DTE SPY quotes, 2026-11-09 … 11-23, momentum +$1,051 / 42 trades vs the same trade against the move −$1,272; two years of hourly SPY bars (model-priced) +0.19 per $ on days with VIX ≥ 25, losing on calm days. Small and clustered samples — a paper trial, reviewed after 20 closes.

## 2. Mathematical Formula

```text
eligible   = same_day_mode == "live" and VIX_close(prior session) ≥ same_day_min_vix
ref        = implied spot of the first cycle at/after 09:35 ET   (median K + C_mid − P_mid)
slot       = entry time t with t ≤ now < t + same_day_entry_window_min, not used today
side       = "call" if spot > ref, "put" if spot < ref, none if equal
long K     = listed 0DTE strike nearest spot;  short K = nearest long ± same_day_width
debit      = ask_long − bid_short (natural);  max_debit = debit;  max_loss = 100 × debit
risk pct   = min(same_day_max_risk_pct, max_loss × max_contracts × 1.10 / equity)
exits      : now ≥ same_day_close_et                → DTE_SAFETY (immediate)
             −P&L_mid ≥ same_day_stop_pct × debit   → STOP_LOSS  (immediate)
             P&L ≥ same_day_target_pct × debit      → PROFIT_TARGET (immediate, no trail)
```

## 3. Reference Python Implementation

```python
# trading_agent/same_day_momentum.py:75-85
def due_slot(now_et: datetime, entry_times: Sequence[str], window_min: int,
             used: Sequence[str]) -> Optional[str]:
    """The entry time whose window [t, t + window_min) contains ``now_et``
    and that has not been used today, else None."""
    t = now_et.time()
    for slot in entry_times:
        start = parse_hhmm(slot)
        end = (datetime.combine(now_et.date(), start) + timedelta(minutes=int(window_min))).time()
        if slot not in used and start <= t < end:
            return slot
    return None
```

```python
# trading_agent/same_day_momentum.py:88-93
def direction(spot: float, ref: Optional[float]) -> Optional[str]:
    """"call" when price is above the 09:35 reference, "put" when below,
    None without a reference or on an exactly flat tape."""
    if not ref or not spot or spot == ref:
        return None
    return "call" if spot > ref else "put"
```

```python
# trading_agent/same_day_momentum.py:201-208
def trade_risk_pct(plan, balance: float, max_risk_pct: float, max_contracts: int) -> float:
    """Per-trade risk fraction that sizes ``plan`` to at most
    ``max_contracts`` (10 % headroom for the live re-quote) and never
    above ``max_risk_pct``."""
    if balance <= 0 or not plan.max_loss:
        return 0.0
    cap = plan.max_loss * max(1, int(max_contracts)) * 1.10 / balance
    return min(float(max_risk_pct), cap)
```

```python
# trading_agent/agent.py:2319-2346
    def _stage2_gates(self, tickers, account_balance: float, buying_power: float,
                      account_type: str, market_open: bool, monitor_results: Dict,
                      tickers_with_positions: Set[str]) -> Optional[Dict]:
        self._same_day_results: List[Dict] = []
        halt = self._trading_halt_result(account_balance, monitor_results)
        if halt is not None:
            return halt
        try:
            res = self._stage_same_day(account_balance, buying_power, account_type,
                                       market_open, monitor_results, tickers_with_positions)
        except Exception as exc:
            logger.exception("Same-day momentum stage failed: %s", exc)
            ...
        finally:
            self._set_trade_risk_pct(self._cycle_risk_pct)
        if res:
            self._same_day_results.append(res)
        skip = self._market_state_skip(tickers, account_balance, monitor_results)
        if skip is not None and self._same_day_results:
            skip["new_trades"] = list(self._same_day_results)
        return skip
```

```python
# trading_agent/position_monitor.py:1060-1086
    def _check_same_day_exit(self, spread: SpreadPosition, debit_position: float):
        from zoneinfo import ZoneInfo
        from trading_agent.same_day_momentum import past_close
        now_et = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
        if past_close(now_et, self.same_day_close_et) or (
                spread.expiration and spread.expiration < now_et.date().isoformat()):
            return (ExitSignal.DTE_SAFETY, ...)
        loss = -spread.net_unrealized_pl
        stop = debit_position * self.same_day_stop_pct
        if loss >= stop > 0:
            return (ExitSignal.STOP_LOSS, ...)
        profit = self._profit_pl(spread)
        target = debit_position * self.same_day_target_pct
        if profit >= target > 0:
            return (ExitSignal.PROFIT_TARGET, ...)
        return (ExitSignal.HOLD, "")
```

Preset fields (`strategy_presets.py`, all defaulted, `same_day_mode="off"`): `same_day_ticker` SPY, `same_day_min_vix` 25, `same_day_entry_times_et` 10:00 / 11:00 / 12:30 / 14:00, `same_day_entry_window_min` 20, `same_day_width` $5, `same_day_target_pct` / `same_day_stop_pct` 0.50, `same_day_close_et` 15:25, `same_day_max_risk_pct` 0.01, `same_day_max_contracts` 1. Summary tag ` • SameDay SPY VIX≥25 @… TP50%/SL50% close 15:25 ≤1c 1.0%`; Streamlit panel "Same-day momentum debit".

## 4. Edge Cases / Guardrails

- **Stale last price after a gap** — the quote feed showed Friday's 689.81 at 09:32 on 2026-11-23 while options priced ~721. Spot is the option-implied median K + C − P, never `get_current_price` (which also falls back to the prior close when the snapshot fails).
- **No reference price** (agent started after 09:35) — the first cycle records it; entries use it from then on. A slot with no move (spot == ref) is skipped and consumed.
- **One attempt per slot** — the slot is marked used before any check, so a rejected or skipped slot never retries in the next cycle; state is a temp + rename JSON sentinel (`trade_journal/same_day_state.json`, env `TRADING_AGENT_SAME_DAY`) keyed by ET date; unreadable or stale → fresh.
- **Prior VIX unavailable** — `prior_vix_close` returns None → the slot is skipped (`skipped_same_day`, "prior VIX close unavailable").
- **Existing SPY position or pending order** — skipped; a submitted same-day order adds SPY to the cycle's dedup set, so the regular pipeline cannot stack an SPY order that cycle.
- **Total risk cap** — skipped when less than one trade's risk (`same_day_max_risk_pct` × equity) is left under `max_total_risk_pct`; the submitted risk counts toward the regular pipeline's budget the same cycle.
- **Market state / HighIV** — not consulted (the stage runs before `_market_state_skip`, outside `_process_ticker`); the kill switch / drawdown governor (skill 62) runs first and still pauses it.
- **Sizing** — `trade_risk_pct` caps the per-trade fraction to `max_contracts` contracts with 10 % headroom; the RiskManager and executor use that fraction (restored to the cycle's value in `finally`). Live debit above the planned natural (`max_debit`) is rejected by the executor's recheck.
- **Wide 0DTE quotes** — each leg must pass the scanner's gate (`max_leg_spread_cents` / `max_leg_spread_pct_mid`), else `leg_spread_wide`.
- **Exits** — target / stop act on the first cycle (no 3-cycle debounce: a 0DTE moves too far in 10 minutes); no profit trail; no regime-shift exit (a call debit is often opened against a bearish daily regime); forced close from 15:25 ET as DTE_SAFETY. Alpaca stops SPY 0DTE opening orders at 15:30 ET and auto-liquidates expiring broad-ETF positions from ~15:45 ET, which leaves several cycles to retry a failed close. The generic DTE safety never fires on the expiry day itself — only this rule closes a 0DTE.
- **Stop cool-down** — a same-day stop does not start the 5-session cool-down for SPY (close rows carry `playbook`; `_stops_from` skips them).
- **Stage failure** — any exception is logged, recorded with the exception monitor, and the cycle carries on with the regular pipeline.

## 5. Cross-References

- `58_market_state_playbook.md` — the gate this stage deliberately sits outside.
- `59_debit_spreads_calendars.md` — debit plan shape, executor debit recheck and close ladder reused here.
- `62_kill_switch_drawdown_governor.md` — the halt that still applies.
- `69_intraday_chain_log.md` — the real 0DTE quotes behind the research (SPY, QQQ, IWM from 2026-11-24).
- `30_profit_target_management.md` — the trail the same-day playbook skips.
- `docs/plans/research_credit_vol.md` §11 — the evidence and its limits.

---

*Last verified against repo HEAD on 2026-11-24.*

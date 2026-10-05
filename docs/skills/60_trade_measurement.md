# Trade Measurement — Entry Fills, Shadow POP, Playbook Scorecard

> **One-line summary:** makes the journal honest enough to judge playbooks: actual multi-leg entry fills are written into the trade plans the monitor reads, every journaled plan carries a realized-volatility POP next to the delta POP, and a per-playbook scorecard turns round trips into win rate, expectancy, return on risk and an advisory size.
> **Source of truth:** [`trading_agent/fill_reconciler.py`](../../trading_agent/fill_reconciler.py), [`trading_agent/shadow_pop.py`](../../trading_agent/shadow_pop.py), [`trading_agent/playbook_scorecard.py`](../../trading_agent/playbook_scorecard.py)
> **Phase:** 2  •  **Group:** architecture / risk
> **Depends on:** `58_market_state_playbook.md` (playbook tag), `59_debit_spreads_calendars.md` (debit sign), `40_long_term_options_evaluator.md` (Wheel fill recording), `48_claude_code_mcp_surface.md`
> **Consumed by:** `agent.py` (`_reconcile_entry_fills`, `_shadow_pop_fields`), MCP `get_playbook_scorecard`, the `journal-analyst` subagent

---

## 1. Theory & Objective

Backlog §6.6 says whether a playbook earns its place is decided by its track record after 20–30 trades, not by its model EV. That needs three honest inputs. (1) **Entry fills**: spread orders are fire-and-forget, so the monitor's profit target and stops used the *estimated* net (SPY IC: plan 0.49, fill 0.48); `fill_reconciler` writes the broker's fill into the plan once the order fills, and invalidates runs cancelled unfilled. (2) **A second probability**: the scorers read |Δ| as P(ITM), i.e. implied volatility; `shadow_pop` logs a realized-volatility lognormal POP at the breakeven beside it, with no trading effect, so weeks of paper data can say which predicts better (§3, §6.8). (3) **Round trips by playbook**: `playbook_scorecard` pairs opens and closes and reports per playbook. Sizing stays an operator decision: the verdict and suggested risk are advisory.

## 2. Mathematical Formula

```text
Fill sign      net_fill = −|filled_avg_price| if planned net < 0 (debit) else +|filled_avg_price|
               Alpaca reports mleg debits as positive amounts (GLD/IWM/QQQ/SPY 2026-10-05)
               debit plan: max_loss = −net_fill × 100;  credit: (width − net_fill) × 100

Shadow POP     σ_RV = stdev(log returns, last 20) × √252
               P(S_T > L) = N( (ln(S/L) − σ²t/2) / (σ√t) ),  t = DTE/365
               credit:  breakevens  put K − credit,  call K + credit;  POP_rv = P(>low) − P(>high)
               debit:   call K_long + debit (above), put K_long − debit (below)
               pop_delta: credit 1 − Σ|Δshort|;  debit |Δ| interpolated at the breakeven

Scorecard      per playbook:  win_rate = wins / n,  expectancy = Σ P&L / n
               return_on_risk = Σ P&L / Σ (max_loss × contracts)   (trades with a known contract count)
               entry_slippage = filled − estimated net (per share)
               verdict (n ≥ 20, risk known):  RoR ≥ +5 % → 3 %;  ≥ 0 → 2 %;  < 0 → 1 %;  ≤ −10 % → disable
```

## 3. Reference Python Implementation

```python
# trading_agent/fill_reconciler.py:36-39
def signed_fill(planned_net: float, filled_avg_price: float) -> float:
    """Fill per share with the plan's sign: negative for a debit plan."""
    fill = abs(float(filled_avg_price))
    return -fill if float(planned_net) < 0 else fill
```

```python
# trading_agent/fill_reconciler.py:75-102
def reconcile_fills(plan_dir: str, *, get_order: Callable[[str], Any],
                    record_fill: Callable[[str, str, float], bool],
                    mark_unfilled: Callable[[str, str], bool],
                    since_days: int = 3) -> Dict[str, int]:
    """Process :func:`pending_runs`; returns counts by outcome."""
    counts = {"recorded": 0, "unfilled": 0, "working": 0, "unknown": 0}
    for run in pending_runs(plan_dir, since_days=since_days):
        try:
            order = get_order(run["order_id"])
        except Exception as exc:                  # noqa: BLE001 — retry next cycle
            logger.warning("Fill lookup for %s failed: %s", run["order_id"], exc)
            order = None
        if order is None:
            counts["unknown"] += 1
            continue
        status = getattr(getattr(order, "status", None), "value", str(getattr(order, "status", "")))
        avg = getattr(order, "filled_avg_price", None)
        if status == "filled" and avg not in (None, ""):
            fill = signed_fill(run["net_credit"], float(avg))
            if record_fill(run["plan_path"], run["run_id"], fill):
                counts["recorded"] += 1
                logger.info("Recorded entry fill %.2f (plan %.2f) for run %s",
                            fill, run["net_credit"], run["run_id"])
        elif status in _TERMINAL_UNFILLED and float(getattr(order, "filled_qty", 0) or 0) == 0:
            if mark_unfilled(run["plan_path"], run["run_id"]):
                counts["unfilled"] += 1
        else:
            counts["working"] += 1
```

```python
# trading_agent/shadow_pop.py:34-43
def prob_above(spot: float, level: float, dte: int, sigma: float) -> float:
    """P(S_T > level), lognormal, zero drift."""
    if level <= 0:
        return 1.0
    t = max(dte, 1) / 365.0
    vt = sigma * math.sqrt(t)
    if vt <= 0:
        return 1.0 if spot > level else 0.0
    d = (math.log(spot / level) - 0.5 * vt * vt) / vt
    return 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))
```

```python
# trading_agent/playbook_scorecard.py:127-141
def _verdict(s: PlaybookStats, min_trades: int) -> Tuple[str, Optional[float]]:
    if s.trades < min_trades:
        return f"collecting ({s.trades}/{min_trades} trades)", None
    if s.return_on_risk is None or s.risk_known_trades < min_trades:
        return (f"{'negative' if s.expectancy < 0 else 'positive'} expectancy — "
                f"risk known for only {s.risk_known_trades}/{s.trades} trades, "
                f"no sizing suggestion"), None
    ror = s.return_on_risk
    if s.expectancy < 0 and ror <= -0.10:
        return "disable suggested — losing ≥ 10 % of risk per trade", 0.0
    if s.expectancy < 0:
        return "shrink — negative expectancy", 0.01
    if ror >= 0.05:
        return "size up — positive expectancy, ≥ 5 % return on risk", 0.03
    return "keep — positive expectancy", 0.02
```

## 4. Edge Cases / Guardrails

- **Order still working** — left for the next cycle; only runs from the last 3 days with no `estimated_net_credit` are looked up, so a recorded fill is never re-fetched.
- **Cancelled / expired / rejected with nothing filled** — the run is marked `valid=False` so it cannot claim the legs of a later live position (the 2026-09-30 VZ shadowing bug, now for spreads too).
- **Broker lookup fails** — counted `unknown`, retried next cycle; reconciliation never raises into the cycle.
- **Dry-run** — reconciliation is skipped (no broker orders).
- **Shadow POP inputs missing** — `shadow_pop_available=False`, `pop_rv=None`; `pop_delta` is still logged. Calendars (two expiries) get `None` for every field.
- **Legacy journal rows without a contract count** (spread `submitted` rows before 2026-10-05) — risk is *unknown*, not one contract: the first live run read −704 % return on risk for a 2-contract bear call. Such trades still count for win rate and expectancy, but no sizing verdict is given until ≥ `min_trades` trades have known risk. Spread entries now journal `contracts` (the executor result's `qty`).
- **Pre-tag opens** — rows before the playbook tag (Phase 3) fall back to the playbook their strategy implies (`_STRATEGY_PLAYBOOK`).
- **Advisory only** — `suggested_risk_pct` is never applied; the MCP tool is read-only and the `journal-analyst` subagent reports it.

## 5. Cross-References

- `58_market_state_playbook.md` — the `playbook` tag on every signal row.
- `59_debit_spreads_calendars.md` — debit sign convention and exits.
- `40_long_term_options_evaluator.md` — Wheel single-leg fills recorded at submission (`_record_fill_credit`).
- `48_claude_code_mcp_surface.md` — `get_playbook_scorecard`.
- `56_daily_journal_reviewer.md` — the evening review can cite the scorecard.

---

*Last verified against repo HEAD on 2026-10-05.*

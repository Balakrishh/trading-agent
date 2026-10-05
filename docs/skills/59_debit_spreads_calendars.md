# Debit Spreads, Calendars & the Bounce Bull Put

> **One-line summary:** low-volatility and oversold playbooks from the skill-58 table — call / put debit spreads for trends, long calendars for sideways tapes, and a bull put sold only after an oversold downtrend stabilises — scored in `decision_engine.py`, planned by `StrategyPlanner`, gated by market state, and managed with their own exits.
> **Source of truth:** [`trading_agent/debit_policy.py`](../../trading_agent/debit_policy.py), [`trading_agent/decision_engine.py`](../../trading_agent/decision_engine.py)
> **Phase:** 2  •  **Group:** strategy
> **Depends on:** `58_market_state_playbook.md`, `03_credit_to_width_floor.md`, `14_adaptive_vs_static_scan_modes.md`, `29_per_leg_liquidity_gate.md`, `13_preset_system_hot_reload.md`
> **Consumed by:** `strategy.py` (routing + planners), `risk_manager.py` (debit cap check), `executor.py` (sizing, live recheck, order sign), `position_monitor.py` (exits, leg inference), `market_state.py` (gates)

---

## 1. Theory & Objective

Week 1 opened no spreads: in a low-volatility, oversold tape the credit scorer rejected ~8,400 grid points for no positive EV, because selling cheap premium is correctly unattractive. Backlog §6.3–6.5 adds the other side of the table. In a trend with low volatility, **buy** a vertical (call debit up, put debit down). In a sideways tape with low volatility, buy a **calendar** (sell the near-dated ATM option, buy the same strike further out). After an oversold sell-off with elevated volatility, sell a **bull put only once price stabilises**, below the recent low. The delta-probability model cannot see a directional or range thesis, so debit structures are not required to show positive model EV. Instead the debit is capped at model value × (1 + `debit_max_overpay`) and must offer reward/risk ≥ `debit_min_reward_risk`. Whether the thesis actually pays is measured by the §6.6 scorecard. Credit plans keep priority: for debit and calendar playbooks the credit plan runs first and the debit structure is the fallback, so the agent never trades less than before. The bounce and wait playbooks **replace** the credit plan, so no bear calls are sold into RSI < 30.

## 2. Mathematical Formula

```text
Vertical debit (call: buy |Δ|≈0.60, sell |Δ|≈0.30 above; put: mirror below)
  debit     = long_ask − short_bid                 (fill_model natural; mid: mid − mid + 0.02)
  fair      = width × (|Δlong| + |Δshort|) / 2     (payoff linear between strikes, |Δ| = P(ITM))
  accept    0 < debit < width,  debit ≤ fair × (1 + overpay),  (width − debit)/debit ≥ min_RR
  POP       = |Δlong| + (|Δshort| − |Δlong|) × debit/width     (|Δ| at breakeven)
  max_debit = fair × (1 + overpay)                 → SpreadPlan.max_debit

Calendar (sell near K, buy far K, K = strike nearest spot)
  at the near expiry  V(S) = BS(S, K, far−near days, σ_far) − intrinsic_near(S)
  S ~ lognormal(spot, σ_near, near days), zero drift, 81-point grid z ∈ [−4, 4]
  model = E[V],  max profit = max V − debit,  POP = P(V > debit)
  accept  debit ≤ model × (1 + overpay),  (max V − debit)/debit ≥ min_RR
  (flat vol ⇒ model = far − near prices: no-arbitrage check; σ_near > σ_far lowers the cost of the edge)

Bounce bull put (bearish, RSI < 30, vol rank ≥ 30)
  stabilised  ⇔  price > max(close over last N sessions)       N = bounce_lookback_days
  short strike < min(low over last 2N sessions); otherwise the ordinary bull-put scorer

Sign convention: net_credit = −debit, max_loss = debit × 100, qty = ⌊max_risk_pct × equity / (debit × 100)⌋

Exits (position-scale; debit_pos = debit × 100 × contracts)
  stop     loss ≥ debit_stop_loss_pct × debit_pos
  target   verticals: profit ≥ debit_profit_target_pct × (width − debit) × 100 × contracts
           calendars: profit ≥ calendar_profit_target_pct × debit_pos
  then DTE safety on the (near) expiry, then regime shift (call debit ↔ bullish,
  put debit ↔ bearish, calendar ↔ sideways; bounce bull put never regime-closed)
```

## 3. Reference Python Implementation

```python
# trading_agent/chain_scanner.py:283-291
def debit_spread_fair_value(width: float, long_delta: float,
                            short_delta: float) -> float:
    """Model value (per share) of a vertical debit spread at expiry.

    With |Δ| read as the probability of finishing in the money, the payoff
    rises linearly from 0 at the long strike to ``width`` at the short
    strike, so E[payoff] ≈ width × (|Δlong| + |Δshort|) / 2. Single source
    for the scorer, RiskManager and the executor's live recheck."""
    return max(0.0, width) * (abs(long_delta) + abs(short_delta)) / 2.0
```

```python
# trading_agent/decision_engine.py:529-554
def _score_debit_spread_with_reason(*, debit: float, width: float,
                                    long_delta: float, short_delta: float,
                                    dte: int, max_overpay: float,
                                    min_reward_risk: float) -> Dict[str, Any]:
    """Score a vertical debit spread. Accepted when 0 < debit < width,
    debit ≤ fair × (1 + max_overpay) and (width − debit) / debit ≥
    min_reward_risk. POP = |Δ| interpolated at the breakeven."""
    if dte <= 0:
        return {"status": "rejected", "reason": DEBIT_REJECT_DTE_NON_POSITIVE}
    if debit <= 0:
        return {"status": "rejected", "reason": DEBIT_REJECT_NON_POSITIVE}
    if debit >= width:
        return {"status": "rejected", "reason": DEBIT_REJECT_GE_WIDTH}
    fair = debit_spread_fair_value(width, long_delta, short_delta)
    ceiling = debit_spread_ceiling(width, long_delta, short_delta, max_overpay)
    rr = (width - debit) / debit
    out = {"fair": fair, "ceiling": ceiling, "rr": rr}
    if debit > ceiling:
        return {**out, "status": "rejected", "reason": DEBIT_REJECT_ABOVE_FAIR}
    if rr < min_reward_risk:
        return {**out, "status": "rejected", "reason": DEBIT_REJECT_REWARD_RISK}
    dl, ds = abs(long_delta), abs(short_delta)
    pop = dl + (ds - dl) * (debit / width)
    ev = (fair - debit) / debit
    return {**out, "status": "accepted", "pop": pop, "ev": ev,
            "annualized": ev * 365.0 / dte, "max_profit": width - debit}
```

```python
# trading_agent/decision_engine.py:653-693
def _score_calendar_with_reason(*, debit: float, spot: float, strike: float,
                                option_type: str, near_dte: int, far_dte: int,
                                near_iv: float, far_iv: float,
                                max_overpay: float,
                                min_reward_risk: float) -> Dict[str, Any]:
    """Score a long calendar (sell near, buy far, same strike).

    At the near expiry the position is worth V(S) = BS(far leg, S, far −
    near days, far IV) − intrinsic(near leg). S is lognormal with the near
    leg's IV (zero drift). model value = E[V]; max profit = max V − debit;
    POP = P(V > debit). Accepted when debit ≤ model × (1 + max_overpay) and
    (max V − debit) / debit ≥ min_reward_risk. Selling richer near-dated
    vol than the far leg's is the edge the model can see."""
    if near_dte <= 0 or far_dte <= near_dte:
        return {"status": "rejected", "reason": DEBIT_REJECT_DTE_NON_POSITIVE}
    if debit <= 0:
        return {"status": "rejected", "reason": DEBIT_REJECT_NON_POSITIVE}
    if near_iv <= 0 or far_iv <= 0:
        return {"status": "rejected", "reason": CAL_REJECT_IV_MISSING}
    t1 = near_dte / 365.0
    t_rem = (far_dte - near_dte) / 365.0
    vt = near_iv * math.sqrt(t1)
    values = []
    for z in _Z_GRID:
        s_t = spot * math.exp(-0.5 * vt * vt + vt * z)
        intrinsic = (max(0.0, s_t - strike) if option_type == "call"
                     else max(0.0, strike - s_t))
        values.append(_bs_price(s_t, strike, t_rem, far_iv, option_type) - intrinsic)
    model = sum(w * v for w, v in zip(_Z_W, values))
    peak = max(values)
    ceiling = model * (1.0 + max_overpay)
    rr = (peak - debit) / debit
    out = {"fair": model, "ceiling": ceiling, "rr": rr}
    if debit > ceiling:
        return {**out, "status": "rejected", "reason": DEBIT_REJECT_ABOVE_FAIR}
    if rr < min_reward_risk:
        return {**out, "status": "rejected", "reason": DEBIT_REJECT_REWARD_RISK}
    pop = sum(w for w, v in zip(_Z_W, values) if v > debit)
    ev = (model - debit) / debit
    return {**out, "status": "accepted", "pop": pop, "ev": ev,
            "annualized": ev * 365.0 / near_dte, "max_profit": peak - debit}
```

```python
# trading_agent/strategy.py:333-357
        # --- Priority 4: Normal regime mapping ---
        # Skill 59 + the skill-58 playbook table:
        #   * oversold downtrend (bounce / wait playbooks) — the playbook
        #     REPLACES the credit plan: no bear calls sold into RSI < 30.
        #   * low volatility (call / put debit, calendar) — the credit
        #     plan goes first (positive model EV is the stronger signal);
        #     the debit structure is the fallback when it finds nothing.
        #     When both fail the credit plan is returned (stable journal
        #     strategy names) with the fallback's reason appended.
        playbook = self._playbook_name(analysis)
        if playbook in ("bounce_bull_put", "wait_for_stabilization"):
            replacement = self._plan_playbook(ticker, analysis, playbook)
            if replacement is not None:
                return replacement
        credit_plan = self._plan_credit_by_regime(ticker, analysis)
        if credit_plan.valid:
            return credit_plan
        fallback = self._plan_playbook(ticker, analysis, playbook)
        if fallback is None:
            return credit_plan
        if fallback.valid:
            return fallback
        credit_plan.rejection_reason = (
            f"{credit_plan.rejection_reason}; fallback {fallback.rejection_reason}")
        return credit_plan
```

```python
# trading_agent/position_monitor.py:753-789
    def _check_debit_exit(self, spread: SpreadPosition,
                          current_regimes: Dict[str, Regime]):
        """Stop at ``debit_stop_loss_pct`` of the debit; profit target at
        ``debit_profit_target_pct`` of max profit (verticals) or
        ``calendar_profit_target_pct`` of the debit (calendars); DTE
        safety on the (near) expiry; regime shift against the thesis."""
        contracts = max(1, spread.contracts_open)
        debit_position = -spread.original_credit * 100 * contracts
        if debit_position <= 0:
            return (ExitSignal.HOLD, "Debit: no recorded debit — holding")
        loss = -spread.net_unrealized_pl
        stop = debit_position * self.debit_stop_loss_pct
        if loss >= stop > 0:
            return (ExitSignal.STOP_LOSS,
                    f"Debit: loss ${loss:.2f} ≥ {self.debit_stop_loss_pct:.0%} of "
                    f"debit ${debit_position:.2f}")
        if spread.strategy_name in DEBIT_VERTICALS:
            max_profit = (spread.spread_width + spread.original_credit) * 100 * contracts
            target = max_profit * self.debit_profit_target_pct
            label = f"{self.debit_profit_target_pct:.0%} of max profit ${max_profit:.2f}"
        else:
            target = debit_position * self.calendar_profit_target_pct
            label = f"{self.calendar_profit_target_pct:.0%} of debit ${debit_position:.2f}"
        profit_pl = self._profit_pl(spread)
        if profit_pl >= target > 0:
            return (ExitSignal.PROFIT_TARGET,
                    f"Debit: profit ${profit_pl:.2f} ({self.profit_target_basis}) ≥ {label}")
        dte_signal = self._check_dte_safety(spread.expiration)
        if dte_signal:
            return (ExitSignal.DTE_SAFETY, dte_signal)
        expected = STRATEGY_REGIME_MAP.get(spread.strategy_name)
        current = current_regimes.get(spread.underlying)
        if expected and current is not None and current != expected:
            return (ExitSignal.REGIME_SHIFT,
                    f"Regime shifted to {current.value} but holding "
                    f"{spread.strategy_name} (expects {expected.value})")
        return (ExitSignal.HOLD, "")
```

```python
# trading_agent/executor.py:365-383
    def _recheck_live_debit(self, plan: SpreadPlan, live_net: float,
                            account_balance: float) -> Tuple[bool, str]:
        """Skill 59: a debit plan's live debit must stay within the
        scorer's ``max_debit`` cap and its max loss (the debit) within
        ``max_risk_pct × equity``. ``live_net`` is the signed net credit."""
        debit = -live_net
        cap = plan.max_debit
        if debit <= 0:
            return (False, f"live_debit_risk: live net {live_net:.2f} is not a debit")
        if cap is None or debit > cap:
            return (False, f"live_debit_risk: debit ${debit:.2f} > cap "
                           f"{'n/a' if cap is None else f'${cap:.2f}'} "
                           f"(planned ${-plan.net_credit:.2f})")
        max_allowed = account_balance * self.max_risk_pct
        if debit * 100 > max_allowed:
            return (False, f"live_debit_risk: max_loss ${debit * 100:.2f} > "
                           f"{self.max_risk_pct*100:.0f}% × ${account_balance:,.2f} "
                           f"(=${max_allowed:.2f})")
        return (True, "")
```

## 4. Edge Cases / Guardrails

- **Credit plan valid** — it wins; the debit/calendar fallback is not even planned. Both invalid → the credit plan is returned (stable journal strategy names) with `; fallback <name>: no acceptable candidate (<reasons>)` appended to its reason.
- **Overpay** — natural fills sit above mid, so `debit_max_overpay = 0` rejects nearly everything; default 0.05. The best near miss (debit, fair, cap, reward/risk) lands in the scan diagnostics.
- **Missing IV** — a calendar leg without `iv` is rejected (`calendar_iv_missing`); no IV is invented.
- **Far expiry not listed** — the calendar tries near + gap, then ±7 days; none listed → `no far-dated chain listed`.
- **Live drift** — the executor re-quotes at natural; a live debit above `max_debit`, or a max loss above `max_risk_pct × equity`, aborts (`live_debit_risk`). No extra tick is paid past the natural price.
- **Order sign** — Alpaca mleg `limit_price` is positive for a debit. A net-credit close of a debit structure with a positive `filled_avg_price` is re-signed so `realized_pl_from_close` stays correct.
- **RiskManager** — the C/W floor and sold-|Δ| cap do not apply to debit plans (the short leg is a hedge, or ATM by design); the check is `0 < debit ≤ max_debit`. Max loss vs account uses `max_loss` = debit × 100.
- **Strike proximity** — never applied to debit structures (a call debit wants price through the short strike). Defensive rolls only fire on that signal, so they never touch debit positions.
- **Leg inference** — a 2-leg same-type position with a net debit is inferred as a call / put debit spread; a short and a long at the same strike and type with different expiries is inferred as a calendar (the per-expiry buckets would otherwise split it into "Naked Short" + an orphan long).
- **Market state** (skill 58) — call debits are bullish (NORMAL / RECOVERY), put debits bearish (also DEFENSIVE), calendars neutral (not DEFENSIVE); the bounce bull put is allowed in CAUTION because its own trigger is the stabilisation evidence.
- **No history** — the bounce planner reports `price history unavailable` and does not trade.
- **Backtester** — `decide_debit_spread` / `decide_calendar` are pure and importable, but the backtester does not yet replay debit structures (synthetic option pricing could not validate them anyway); the §6.6 paper scorecard is the evidence.
- **Toggles** — `debit_spreads_enabled`, `calendar_enabled`, `bounce_bull_put_enabled` (all default on, paper). Off → that playbook never plans; the bounce/wait replacements also turn off, restoring the bear call.

## 5. Cross-References

- `58_market_state_playbook.md` — `playbook_for` routes here; gates per state.
- `03_credit_to_width_floor.md` — the credit-side floor these structures replace.
- `29_per_leg_liquidity_gate.md` — same per-leg width gate on every leg.
- `30_profit_target_management.md` — `profit_target_basis` (natural) applies to debit targets too.
- `13_preset_system_hot_reload.md` — the 15 `debit_*` / `calendar_*` / `bounce_*` fields.

---

*Last verified against repo HEAD on 2026-10-05.*

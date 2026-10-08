# Profit-target management

> **One-line summary:** Close any open spread once unrealized P&L reaches `profit_target_pct × initial_credit` — the industry-standard "manage winners early" rule that captures most theta while avoiding late-cycle gamma risk. Per-preset tunable so each risk profile can pick its own turnover/per-trade-$ trade-off.
> **Source of truth:** [`trading_agent/strategy_presets.py:PresetConfig`](../../trading_agent/strategy_presets.py), [`trading_agent/position_monitor.py:601-609`](../../trading_agent/position_monitor.py), [`trading_agent/backtest/runner.py:258-271`](../../trading_agent/backtest/runner.py).
> **Phase:** 1  •  **Group:** risk
> **Depends on:** `13_preset_system_hot_reload.md` (where the knob lives), `15_backtest_live_parity.md` (the live + backtest exits must share this value).
> **Consumed by:** `position_monitor.evaluate(...)` (live exit signal generation), `backtest.runner._handle_intraday_decision()` (backtest parity), `streamlit/live_monitor.render_strategy_profile_panel()` (operator surface).

---

## 1. Theory & Objective

A credit spread's expected value is front-loaded: theta decay is largest in the first half of the position's life, and gamma risk concentrates in the final 7–10 DTE. Holding to expiration earns the last few dollars of credit but exposes the position to a regime in which small underlying moves can flip a 60% winner into a 40% loser quickly.

The industry-standard rule from Tastytrade's "manage winners at 50%" studies and most professional credit-spread playbooks is to close once unrealized P&L reaches 50% of the initial credit collected. This trades the final 50% of potential profit for two compounding benefits:

1. **Capital turnover.** A 21-DTE position that hits 50% in ~7 days frees its buying power for the next trade — over a 21-day window the same capital deploys 2–3× instead of once.
2. **Risk reduction.** Closing before the last 7-10 DTE skips the gamma-heavy tail. Annualised return per unit of risk improves even if per-trade return drops.

The rule is parameterised because the optimum depends on the universe and the rest of the preset:
- **Aggressive presets** (short DTE, high credit, frequent fills) benefit from a *lower* target (0.40) because they need fast capital recycling.
- **Conservative presets** (long DTE, far OTM, infrequent fills) benefit from a *higher* target (0.60) because each trade is rarer and worth more $.

## 2. Mathematical Formula

```text
exit when  unrealized_pl ≥ profit_target_pct × initial_credit

where
  unrealized_pl     = sum over legs of (current_mark − fill_price) × side × 100
                      [dollars, positive when in profit]
  initial_credit    = net credit received at open, in dollars per spread
  profit_target_pct ∈ [0.20, 0.85]  — fraction of initial credit at which to close

Note: the comparison uses `>= threshold > 0` so an unprofitable position
(net debit at open, edge cases) never satisfies the trigger.
```

Preset defaults:

```text
  conservative   profit_target_pct = 0.60   # ride winners further
  balanced       profit_target_pct = 0.50   # industry-standard
  aggressive     profit_target_pct = 0.40   # recycle capital faster
```

## 3. Reference Python Implementation

### 3.1 The exit predicate (live)

```python
# trading_agent/position_monitor.py:950-959
# --- 3. Profit target: 50% of credit captured ---
_, _, profit_threshold = self.profit_basis(spread)
profit_pl = self._profit_pl(spread)
if profit_pl >= profit_threshold > 0:
    return (
        ExitSignal.PROFIT_TARGET,
        f"Profit ${profit_pl:.2f} ({self.profit_target_basis}) ≥ "
        f"{self.profit_target_pct*100:.0f}% of credit "
        f"${credit_position:.2f} ({contracts}×${credit_per_contract:.2f})"
    )
```

### 3.1b Trailing profit (2026-10-05, `profit_trail.py`)

`PositionMonitor.profit_basis(spread)` is the single source for each structure's yardstick and target (credit / debit vertical / calendar / wheel). `TradingAgent._apply_profit_trail` runs right after `evaluate()`; `profit_trail.evaluate` arms at the target and then:

```python
def evaluate(state: TrailState, profit: float, *, basis: float, target: float,
             params: TrailParams, dte: Optional[int]) -> Tuple[TrailState, str, str]:
    """Advance ``state`` with this cycle's ``profit`` ($, position scale).
    Returns (state, decision, reason). Pure — no I/O."""
    state.last_pl = round(profit, 2)
    if not state.armed:
        if target <= 0 or profit < target:
            return state, HOLD_UNARMED, ""
        state.armed = True
        state.basis, state.target = basis, target
        state.armed_at = datetime.now(timezone.utc).isoformat()
        state.peak_pl = profit
    state.peak_pl = max(state.peak_pl, profit)
    ceiling = params.ceiling(state.kind, basis)
    if ceiling > 0 and profit >= ceiling:
        return state, CLOSE_CEILING, (f"trail ceiling: profit ${profit:.2f} ≥ "
                                      f"${ceiling:.2f}")
    lock = max(params.floor(state.kind, basis), state.peak_pl * (1.0 - params.giveback_pct))
    if profit <= lock:
        return state, CLOSE_GIVEBACK, (f"trail giveback: profit ${profit:.2f} ≤ lock "
                                       f"${lock:.2f} (peak ${state.peak_pl:.2f})")
    if dte is not None and dte <= params.max_hold_dte:
        return state, CLOSE_TIME, f"trail time stop: {dte} DTE ≤ {params.max_hold_dte}"
    return state, HOLD_ARMED, (f"trail armed: profit ${profit:.2f}, peak "
                               f"${state.peak_pl:.2f}, lock ${lock:.2f}, ceiling ${ceiling:.2f}")
```

| Mode (`profit_trail_mode`) | Effect |
|---|---|
| `off` | close at the target (legacy) |
| `shadow` (default) | close at the target; armed winners keep being priced after the close until the trail would have closed → journal `profit_trail_shadow` with `actual_exit_pl`, `trail_exit_pl`, `trail_minus_actual` |
| `live` | the trail decides: target → HOLD (armed); ceiling / giveback / time stop → PROFIT_TARGET (still through the 3-cycle debounce) |

Defaults: giveback 25 % of the peak; credit floor 40 % and ceiling 75 % of the credit; debit ceiling 90 % of max profit (the debit *target*, where the trail arms, is 50 % of the debit paid by default — skill 59 `debit_profit_target_basis`); calendar ceiling 40 % of the debit; time stop at ≤ 7 DTE. Wheel legs are excluded.


### 3.2 PresetConfig field

```python
# trading_agent/strategy_presets.py:PresetConfig
profit_target_pct:       float = 0.50
```

### 3.3 Agent-side wiring

```python
# trading_agent/agent.py — PositionMonitor instantiation
self.position_monitor: PositionsPort = PositionMonitor(
    api_key=config.alpaca.api_key,
    secret_key=config.alpaca.secret_key,
    base_url=config.alpaca.base_url,
    profit_target_pct=self.preset.profit_target_pct,
)
```

### 3.4 Backtester parity (skill 15)

```python
# trading_agent/backtest/runner.py — _handle_intraday_decision
signal, reason = pos.evaluate_exit(
    t=t, spot=spot, current_regime=current_regime,
    profit_target_pct=getattr(
        self.preset, "profit_target_pct", 0.50
    ),
    hard_stop_multiplier=3.0,
    strike_proximity_pct=0.01,
)
```

## 4. Edge Cases / Guardrails

- **Net-debit positions** — the `>= threshold > 0` chained comparison short-circuits when the threshold itself isn't positive, so a position that opened at a net debit (unusual but possible on weird fills) never triggers PROFIT_TARGET on a "smaller loss" misread.
- **PDT same-day-open** — on sub-$25K accounts, a same-day open + close trips FINRA's pattern-day-trading rule. PROFIT_TARGET still fires (banking gains is worth the day-trade flag), but REGIME_SHIFT exits are suppressed on same-day-open tickers. See `agent.py:1022-1032`.
- **Partial-close cooldown** — three consecutive close-fill failures within the cooldown window park further auto-closes for 60 minutes (skill 17). PROFIT_TARGET retries get suppressed during the cooldown to avoid pinging Alpaca every cycle.
- **Overlay validation** — overlays in `STRATEGY_PRESET.json` are bounded to `[0.10, 0.95]` at load time. Out-of-range values log a warning and fall back to the preset default rather than crashing the cycle.
- **Backtester parity** — both the live `position_monitor` and the backtest's `_handle_intraday_decision` read from the same preset field; a divergence (e.g., backtester hardcodes 0.50, live uses 0.40) breaks skill 15's parity invariant and would silently produce backtest results that overstate aggressive-preset performance.

- **Profit target judged at the cost to close (2026-10-05, §6.1).** With `fill_model="natural"`, `PositionMonitor._profit_pl` uses `SpreadPosition.net_natural_pl` — each leg valued at its natural closing price (short → ask, long → bid) — so the 50 % target fires only when it can actually be captured. Falls back to the mid valuation when any leg lacks a quote. Stops and the hard stop keep using the mid valuation so bid/ask noise does not trigger them.

- **Trailing profit (2026-10-05).** Stops always win — the trail only rewrites HOLD / PROFIT_TARGET. Profit is the natural-price P&L (cost to close now), and a trail close still needs 3 consecutive votes, so one bad quote cannot end a winner. The trail watches intraday only: an overnight gap through the lock closes at the next open, possibly below it (bounded by the ceiling and the 7-DTE stop). State lives in `trade_journal/profit_trail.json` (atomic) because the agent restarts every cycle; positions that close without ever arming are dropped. A ghost leg with no usable quote is retried next cycle; a ghost past expiry records its last P&L.

## 5. Cross-References

- `13_preset_system_hot_reload.md` — `profit_target_pct` is hot-reloadable like the other preset knobs; changes apply on the next cycle without a restart.
- `15_backtest_live_parity.md` — backtester and live must read the same field; the `getattr(..., 0.50)` fallback exists so a backtest run against an old preset file doesn't crash.
- `17_close_failure_and_cooldown.md` — close-failure cooldown intercepts even PROFIT_TARGET-triggered closes; the cooldown wins.

---

*Last verified against repo HEAD on 2026-10-08.*

# Adaptive spread width

> **One-line summary:** Pick a wing distance via `max(SPREAD_WIDTH_floor, 3 × grid, 0.025 × spot)`, then snap UP to the strike grid. Preset overrides bypass the formula in favor of `pct_of_spot` or `fixed_dollar`.
> **Source of truth:** [`trading_agent/strategy.py:653-685`](../../trading_agent/strategy.py)
> **Phase:** 1  •  **Group:** strategy
> **Depends on:** [02 Strike snapping to grid](02_strike_snapping.md).
> **Consumed by:** `strategy._plan_bull_put`, `strategy._plan_bear_call`, `strategy._find_bought_strike`.

---

## 1. Theory & Objective

The width of a credit spread controls two things at once: **max loss** (= width − credit) and **collateral** (= width × multiplier). Picking a width that scales with the underlying preserves these characteristics across radically different price levels. A flat $5 wing is too narrow on SPY ($400) and too wide on a $30 ETF.

The formula `max(SPREAD_WIDTH, 3 × grid, 2.5 % × spot)` enforces three independent floors:

1. **`SPREAD_WIDTH = 5.0`** — a hard dollar floor. Below this, commissions eat too much of the credit on any retail brokerage.
2. **`3 × grid`** — at least three strikes wide. A two-strike-wide wing on a thin chain often has no liquid long leg; three strikes gives the scanner a real choice.
3. **`0.025 × spot`** — proportional to underlying. 2.5 % of spot puts the wing roughly at the same standard-deviation distance across symbols, which keeps risk-per-position in a consistent neighborhood.

Whichever floor binds wins. After that, `_strike_grid_step()` finds the actual listed strike grid (often $1 or $5 for ETFs, $2.50 for some) and the snapper rounds UP (skill 02) to a tradeable width.

The **preset override** path was added later (see skill 13). Two modes:

- `pct_of_spot` — width = `value × spot` (e.g. `value = 0.015` → 1.5 % of spot). Preferred for cross-symbol consistency.
- `fixed_dollar` — width = `value` flat (e.g. `value = 5.0` → $5). Preferred for traders who think in absolute risk dollars.

When a preset specifies neither, the legacy formula above runs unchanged. Backward compatibility is preserved by the `else` branch.

## 2. Mathematical Formula

```text
                          ┌─ pct_of_spot:    candidate = max(grid, value × spot)
candidate = preset?       ├─ fixed_dollar:   candidate = max(grid, value)
                          └─ none (legacy):  candidate = max(SPREAD_WIDTH, 3 × grid, 0.025 × spot)

snapped = grid × max(1, round(candidate / grid + 0.4999))   ← skill 02

where
  SPREAD_WIDTH ∈ ℝ⁺   — legacy dollar floor; constant 5.0
  grid         ∈ ℝ⁺   — actual strike step from the chain
  spot         ∈ ℝ⁺   — sold-leg strike, used as spot proxy
  value        ∈ ℝ⁺   — preset width param
  snapped      ∈ ℝ⁺   — final width in dollars
```

Why **sold-leg strike** as the spot proxy and not the actual spot? Because for a 0.20-delta short put the sold strike is roughly 1.5–2 σ below spot — close enough for width sizing, and it requires no extra fetch for the underlying mark.

## 3. Reference Python Implementation

```python
# trading_agent/strategy.py:121
SPREAD_WIDTH = 5.0
```

```python
# trading_agent/strategy.py:653-685
def _pick_spread_width(self, contracts: List[Dict],
                       sold_strike: float) -> float:
    """
    Compute the spread width.

    Two paths:
      * **Preset override** — when the active preset specifies a
        ``width_mode`` and ``width_value`` (set in __init__), that
        policy takes precedence. ``pct_of_spot`` uses ``width_value
        × sold_strike``; ``fixed_dollar`` uses ``width_value`` raw.
        Either is then snapped UP to the strike grid.
      * **Legacy adaptive formula** — when no override is supplied,
        take ``max(SPREAD_WIDTH, 3 × strike_grid_step, 2.5% × spot
        proxy)`` and snap UP to the grid. This is the original
        behavior and remains the back-compat default.

    The sold-leg strike is the spot proxy (within ~2 σ of spot for a
    0.20-delta short put — close enough for width sizing).
    """
    grid = self._strike_grid_step(contracts)
    spot_proxy = sold_strike

    if self._width_mode == "pct_of_spot" and self._width_value is not None:
        candidate = max(grid, self._width_value * spot_proxy)
    elif self._width_mode == "fixed_dollar" and self._width_value is not None:
        candidate = max(grid, float(self._width_value))
    else:
        # Legacy adaptive width.
        candidate = max(self.SPREAD_WIDTH, 3 * grid, 0.025 * spot_proxy)

    # Snap UP to the strike grid so a real strike sits at this distance.
    snapped = grid * max(1, int(round(candidate / grid + 0.4999)))
    return float(snapped)
```

## 4. Edge Cases / Guardrails

- **Empty chain** — `_strike_grid_step(contracts)` raises if `contracts` is empty. The caller (`_plan_bull_put`) catches this and returns a `SpreadPlan` with `kind=KIND_NO_TRADE` and a reason string. No crash.
- **Bizarre grid (e.g. $0.01 grid on a high-priced equity)** — `3 × grid` floor becomes too small to dominate; the `0.025 × spot` floor binds instead. Behavior remains sane.
- **Preset says `pct_of_spot` with `value = 0.0`** — `max(grid, 0)` = `grid` → wing is one strike step. The C/W floor (skill 03) usually rejects such a thin wing because credit is too small.
- **Preset says `fixed_dollar` with `value < SPREAD_WIDTH`** — preset wins, so legacy floor doesn't apply. Intentional: the user explicitly asked for a smaller wing.
- **`width_mode` is some other string** — falls through `if/elif` to the legacy `else` branch. Defensive; the preset loader validates `width_mode` upstream.
- **Grid jumps mid-day** — width is recomputed each plan call, so a strike-grid change between cycles is automatically picked up.

## 5. Cross-References

- [02 Strike snapping to grid](02_strike_snapping.md) — the `+ 0.4999` snapper used at the end.
- [03 Credit-to-Width floor](03_credit_to_width_floor.md) — the resulting width interacts with C/W: too-wide wings make `C / W` shrink and trip the floor.
- [13 Preset system & hot-reload](13_preset_system_hot_reload.md) — where `width_mode` and `width_value` are set.
- [14 Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md) — adaptive scan sweeps `width_grid_pct` instead of using this single-point picker.

## 6. 2026-05-15 Grid retune

The three named presets carry distinct grid shapes optimized for their risk posture. All three (except Conservative) include the narrow `0.005` (0.5% of spot) entry so high-spot tickers (SPY, QQQ, DIA, IWM) become budget-eligible without raising `max_risk_pct`. Whether they actually clear C/W on any given day depends on the underlying's IV regime — surfaced in the guardrail grid as a colour-coded `IV NN` annotation under each ticker.

| Preset | `delta_grid` | `dte_grid` | `width_grid_pct` | `edge_buffer` | `min_pop` |
|---|---|---|---|---|---|
| Conservative | 0.15–0.30 | 14–45 | 1.0%–2.5% | 0.15 | 0.65 |
| Balanced | 0.20–0.40 | 7–45 | 0.5%–2.5% | 0.08 | 0.55 |
| Aggressive | 0.25–0.45 | 7–30 | 0.5%–3.0% | 0.05 | 0.50 |

Conservative deliberately omits the 0.5% width because it shouldn't be trading SPY-style spreads. Aggressive extends to 0.45 delta + 3.0% width for premium-rich tickers but keeps DTE short to maximise theta velocity.

The 0.5% width does NOT lower the C/W floor — it only adds candidate shapes to the scanner's sweep. The agent will still reject narrow-width SPY candidates on low-IV days because C/W stays below `|Δ| × (1 + edge_buffer)`. This is correct self-regulating behavior; trades fire when IVRank rises enough to support the C/W math.

## 7. IVRank Goldilocks zone — the FULL gating story

Credit-spread profitability isn't a "more IV = better" monotonic — it's a Goldilocks band gated from BOTH ends:

```
IVRank:    0 ────── 25 ────── 50 ────── 75 ────── 95 ────── 100
            ←─ too thin ─→ ←──── tradable zone ────→ ←─ too hot ─→
            C/W floor                                defense_first
            rejects                                  filter blocks
```

| Zone | IVRank | Agent behavior | Dashboard annotation |
|---|---|---|---|
| Too thin | <25 | C/W floor rejects (premium can't clear `|Δ|×1.10`) | Grey `IV NN` |
| Borderline | 25–50 | May fire on shorter widths / higher delta | Amber `IV NN` |
| Sweet spot | 50–95 | Trades fire most readily | **Green** `IV NN` |
| Too hot | >95 | `defense_first` filter blocks all new entries (`agent.py:_process_ticker`, log line `HighIV: IV rank N > 95th pct — extreme volatility, blocking all new entries`) | **Red** `IV NN` |

The "too hot" defense exists because IVRank >95 typically means the market is pricing in a known catalyst (earnings, FOMC, geopolitical shock). The fat premium is COMPENSATION for that risk, not free money — selling vol into the spike has poor risk-adjusted returns historically. The 2026-05-15 XLE incident (IVRank 97.2 → defense_first skipped despite excellent C/W) is the canonical example.

Both filters work in concert with this skill's width-grid retune: the 0.5% width opens narrow-spread shapes the scanner can consider, the C/W floor gates the bottom of the IV range, and `defense_first` gates the top. The dashboard's IVRank annotation reads green only in the 50–95 zone — anywhere else the operator should expect the agent to skip the ticker, and that's the correct call.

---

*Last verified against repo HEAD on 2026-05-20 (grid retune + IVRank UI + Goldilocks zone documentation + PDT DTE cap).*

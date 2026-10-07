# Broken-Wing Butterfly Scoring

> **One-line summary:** Asymmetric-wing extension of Iron Butterfly (skill 45). Same 4-leg scaffold — both shorts sit at ATM, both long wings on the far side — but the long put and long call sit at *different* distances from center. Direction of the position (bullish vs bearish) is inferred from which wing is wider, so the operator's directional view becomes a structural knob instead of a strategy dispatch decision. Ships in scorer-only form; live-cycle integration gated behind `PresetConfig.broken_wing_butterfly_enabled=False` until backtest signs off.
> **Source of truth:** [`trading_agent/chain_scanner.py:_score_broken_wing_butterfly`](../../trading_agent/chain_scanner.py), [`trading_agent/chain_scanner.py:_pop_from_bwb_structure`](../../trading_agent/chain_scanner.py), [`trading_agent/strategy_presets.py:PresetConfig`](../../trading_agent/strategy_presets.py) (the `broken_wing_butterfly_*` fields).
> **Phase:** 2  •  **Group:** strategy
> **Depends on:** `45_iron_butterfly.md` (this skill is the asymmetric-wing extension), `01_pop_from_delta.md` (vertical POP heritage), `05_ev_per_dollar_risked.md` (EV normalization pattern).
> **Consumed by:** future `ChainScanner._scan_broken_wing_butterfly` + `decision_engine.decide_broken_wing_butterfly` — both are Phase 2.5, backtest-gated.

---

## 1. Theory & Objective

Iron Butterfly (skill 45) is a directionally-neutral trade — both wings are equidistant from the ATM center strike, so the risk-reward is symmetric on both sides. Broken-Wing Butterfly (BWB) breaks that symmetry deliberately. By moving one long wing further out than the other, you get:

- **Higher net credit** than the symmetric IB at the same DTE. The wider-wing side sells a further-OTM long option that's cheaper, so more credit stays in your pocket.
- **Asymmetric risk profile.** The wider-wing side has a bigger max loss but a lower probability of ever being hit (the underlying has further to travel). The narrower-wing side has a smaller but more probable max loss. The trade is a bet that the underlying stays near ATM OR drifts *toward the narrower wing*.
- **Direction becomes structural, not dispatch-time.** In the current agent, "should I trade bull put or bear call?" is a strategy-selection decision made upstream by the regime classifier. With BWB the same architectural machinery expresses direction as an asymmetry knob — put wing wider = bullish bias, call wing wider = bearish bias. A single BWB scorer covers both directions with different inputs.

The regime BWB is best suited for is "moderately directional + range-bound + IV elevated." The agent's Bull Put / Bear Call already covers "strongly directional" (via short-strike delta far OTM) and Iron Condor / Iron Butterfly cover "range-bound + neutral." BWB fills the "directional + range-bound" gap — expect the underlying to stay near current price with a small drift, want higher credit than an IC, willing to accept an asymmetric worst-case.

The scorer ships as a pure function alongside the IB scorer. Live-cycle integration (the analogue of Phase 1.5 for BWB) is next-session work — call it Phase 2.5 — gated on backtest sign-off.

## 2. Mathematical Formula

### 2.1 POP formula — average-wing normalization

```text
POP_BWB = min(1.0, 4·C / (W_put + W_call))

where
  C       = net credit per share (dollars)
  W_put   = distance from center to lower long put (dollars)
  W_call  = distance from center to upper long call (dollars)
```

The formula reduces to the IB POP formula (`2C/W`) when `W_put = W_call = W`. Its geometric intuition: the profit zone is [K−C, K+C] regardless of the wings' asymmetry; POP normalizes the profit-zone width (2C) by the average wing width `(W_put + W_call)/2`. Same rank-order as the more expensive lognormal integral for typical BWB structures (wing ratios 1.2–3.0).

### 2.2 EV formula — equal-probability side split

```text
per-share EV:
  max_loss_put   = W_put  − C
  max_loss_call  = W_call − C
  EV = POP·C − ((1 − POP)/2)·(max_loss_put + max_loss_call)
     = POP·C − ((1 − POP)/2)·(W_put + W_call − 2·C)

per-dollar-risked:
  EV/$risked = EV / max(max_loss_put, max_loss_call)
```

The equal-probability side split (`(1 − POP)/2` on each side) is a simplification — in reality the wider wing has a lower probability of being hit. At the ranking level (candidate ordering by score) the approximation is close enough for scanner use; the absolute EV number is within 5–10% of the more expensive Black-Scholes derivation for typical BWB parameters.

The normalizer `max(max_loss_put, max_loss_call)` — dividing by the WORST-case loss — makes EV/$risked directly comparable to the vertical and IB scorer outputs. A candidate ranked "10% EV per $ risked" costs 10¢ in expectation for every $1 that could be lost at the worst-case wing.

### 2.3 Direction inference

```text
if W_put > W_call:  direction = "bullish"    (bigger loss to the downside)
if W_call > W_put:  direction = "bearish"    (bigger loss to the upside)
if W_put == W_call: reject — this is IB, use skill 45
```

Direction is inferred, not passed in. The scorer's caller — the future `decide_broken_wing_butterfly` orchestrator — will construct BWB candidates by pairing every `put_wing_pct` with every different `call_wing_pct` from the preset grid, so both directions naturally arise.

### 2.4 Gate ordering

```text
1. dte ≤ 0                                 → BWB_REJECT_DTE_NON_POSITIVE_BWB
2. put_wing ≤ 0 OR call_wing ≤ 0           → BWB_REJECT_WING_TOO_NARROW_BWB
3. put_wing == call_wing                   → BWB_REJECT_WINGS_EQUAL
4. wing_ratio ∉ [1.20, 3.00]               → BWB_REJECT_WING_RATIO_OUT_OF_BAND
5. credit ≤ 0                              → BWB_REJECT_CREDIT_NON_POSITIVE_BWB
6. credit ≥ min(put_wing, call_wing)       → BWB_REJECT_CREDIT_GE_MIN_WING
7. |Δ_short_call| off ATM by > tolerance   → BWB_REJECT_NOT_ATM_SHORTS_BWB
8. |Δ_short_put|  off ATM by > tolerance   → BWB_REJECT_NOT_ATM_SHORTS_BWB
9. POP_BWB < min_pop                       → BWB_REJECT_POP_BELOW_MIN_BWB
10. EV ≤ 0                                 → BWB_REJECT_EV_NON_POSITIVE_BWB
```

The wing-ratio band `[BWB_WING_RATIO_MIN, BWB_WING_RATIO_MAX] = [1.20, 3.00]` excludes two failure modes. Below 1.20 the structure is close enough to symmetric that it's better modeled as IB (avoids duplicate work in the scanner). Above 3.00 one wing is essentially degenerate — the narrower wing is so tight that the loss on that side happens on tiny underlying moves; the risk profile stops behaving like a butterfly.

## 3. Reference Python Implementation

### 3.1 POP helper

```python
# trading_agent/chain_scanner.py
def _pop_from_bwb_structure(
    credit: float, put_wing: float, call_wing: float,
) -> float:
    if credit <= 0 or put_wing <= 0 or call_wing <= 0:
        return 0.0
    if credit >= min(put_wing, call_wing):
        return 0.0
    total_wing = put_wing + call_wing
    return min(1.0, (4.0 * credit) / total_wing)
```

### 3.2 EV helper

```python
# trading_agent/chain_scanner.py
def _ev_per_dollar_risked_bwb(
    credit: float, put_wing: float, call_wing: float,
) -> Optional[float]:
```

Returns None on degenerate inputs (matches the IB helper's contract).

### 3.3 Scorer + verbose sibling

```python
# trading_agent/chain_scanner.py
def _score_broken_wing_butterfly(
    *,
    credit: float,
    put_wing: float,
    call_wing: float,
    short_call_delta: float,
    short_put_delta: float,
    dte: int,
    min_pop: float,
) -> Optional[Tuple[str, float, float, float, float]]:
```

Accept returns `(direction, pop, cw_ratio, ev_per_$risked, annualized)` — one extra field vs. IB's tuple to expose the inferred direction. Rejects return None; the `_with_reason` sibling returns the reject-taxonomy dict.

### 3.4 PresetConfig knobs

```python
# trading_agent/strategy_presets.py — PresetConfig additions
broken_wing_butterfly_enabled:     bool  = False
broken_wing_butterfly_min_pop:     float = 0.40
broken_wing_butterfly_dte_grid:    Tuple[int, ...] = (21, 30, 45)
broken_wing_butterfly_wing_width_pct: Tuple[float, ...] = (0.020, 0.030, 0.040, 0.060)
```

`enabled=False` — same safety posture as skill 45. Grid entries are ascending so the orchestrator (Phase 2.5) can build asymmetric pairs by sampling `wing_grid[i], wing_grid[j]` for `i ≠ j`.

## 4. Edge Cases / Guardrails

- **Reuses IB ATM-delta tolerance.** Both shorts must have `|Δ|` within `IB_ATM_DELTA_TOLERANCE = 0.10` of 0.5. BWB and IB use the same tolerance because the ATM detection is structurally identical — the wings differ, not the shorts.
- **Symmetric-wing rejection.** When `W_put == W_call` exactly (or within floating-point noise, 1e-6), the scorer rejects with `BWB_REJECT_WINGS_EQUAL`. The Phase 2.5 orchestrator skips symmetric pairs at grid-construction time; the reject is defensive for hand-constructed test inputs.
- **Wing ratio band `[1.20, 3.00]`.** Below 1.20 → indistinguishable from IB (routing to the IB scorer is better). Above 3.00 → the narrower wing is degenerate, POP + EV numbers become numerically unstable, and the risk profile is essentially a vertical on one side + an unpaired long option on the other. Both bounds pinned by conformance tests.
- **Credit-vs-min-wing gate.** `credit ≥ min(put_wing, call_wing)` produces negative max_loss on one side (structurally impossible for a "loss" side of a butterfly). Reject with `BWB_REJECT_CREDIT_GE_MIN_WING`. This gate fires before the ATM check because the math is definitive regardless of delta.
- **Direction is a STRING output, not a strategy dispatch.** The scorer returns `"bullish"` or `"bearish"` in the tuple's first slot. Consumers use it to route candidates into direction-appropriate downstream systems (journal tags, Telegram body sections, dashboard color coding) but the scorer itself is direction-agnostic.
- **EV per-dollar-risked normalizes by WORST-case max_loss, not average.** This is a conservative choice — the average would look better in ranking but the worst-case is what the operator actually pays if the trade goes wrong. Matches the risk-management posture of the vertical and IB scorers.
- **Contract-count scaling reuses skill 44.** When BWB positions land in `position_monitor._check_exit`, the same `contracts_open` scaling that applies to verticals and IB applies to BWB transparently. Nothing in skill 44 assumes symmetric wings — the `original_credit` and `max_loss` fields on `SpreadPosition` carry per-contract dollar values and are multiplied by contract count in the exit-threshold math.
- **Live-cycle wiring is deliberately not in this session.** Same posture as Phase 1 of skill 45: ship the scorer and the skill doc, defer the orchestrator and strategy dispatch. Phase 2.5 will add `decide_broken_wing_butterfly` alongside `decide_iron_butterfly`, and only after that will `PresetConfig.broken_wing_butterfly_enabled` be flippable per-preset.
- **No CI-invariant scanner rule for BWB either.** Same reasoning as IB (skill 45 §4): the "no scan-time invariant" decision is because BWB's structural gate is EV > 0, not a formula that can be statically checked with an AST walker. When Phase 4 (calendar spreads) lands, the CI scanner will need per-strategy invariant support and BWB will get retrofitted at that point.

## 5. Cross-References

- `45_iron_butterfly.md` — the symmetric-wing sibling this skill extends. Same 4-leg scaffold, same ATM detection, same EV normalization pattern.
- `01_pop_from_delta.md` — the vertical POP formula this skill's average-wing normalization is the BWB analogue of.
- `05_ev_per_dollar_risked.md` — the EV-per-$risked formulation reused with BWB's worst-case max_loss.
- `44_position_monitor_scaling.md` — contract-count scaling + post-fill grace that BWB positions inherit transparently.
- `docs/plans/additional_strategies_plan.md` — the phased plan this skill is Phase 2 of. Phase 2.5 (BWB decide() orchestrator) is next; Phase 4 (calendars) requires multi-expiration architecture and is a bigger lift.

---

*Last verified against repo HEAD on 2026-10-07.*

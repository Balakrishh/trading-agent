# Iron Butterfly Scoring

> **One-line summary:** Structurally distinct 4-leg credit strategy — both shorts sit ATM at the same strike K, both long wings equidistant at K ± W. Collects roughly 2× the credit of an Iron Condor at the same DTE (because the shorts are ATM, not OTM) at the cost of a narrower profit zone [K−C, K+C] instead of the IC's wider [K_put_short, K_call_short] range. Ships as a pure-function scorer alongside the vertical scorer in `chain_scanner.py`; live wiring gated behind `PresetConfig.iron_butterfly_enabled=False` until backtest sign-off.
> **Source of truth:** [`trading_agent/chain_scanner.py:_score_iron_butterfly`](../../trading_agent/chain_scanner.py), [`trading_agent/chain_scanner.py:_pop_from_ib_structure`](../../trading_agent/chain_scanner.py), [`trading_agent/strategy_presets.py:PresetConfig`](../../trading_agent/strategy_presets.py) (the `iron_butterfly_*` fields).
> **Phase:** 2  •  **Group:** strategy
> **Depends on:** `01_pop_from_delta.md` (vertical POP formula — this skill's POP formula is the IB analogue), `03_credit_to_width_floor.md` (the invariant that applies to verticals — IB has its own scoring gate instead), `05_ev_per_dollar_risked.md` (EV formulation reused with IB-specific max_loss).
> **Consumed by:** future `ChainScanner._scan_iron_butterfly` (next phase, backtest-gated).

---

## 1. Theory & Objective

The credit-spread agent's three current strategies — Bull Put, Bear Call, Iron Condor — all price off the same vertical breakeven: at fair value, C/W ≈ |Δ_short|. When the market underprices verticals (thin premium regime, current 2026-07 environment) all three sit out simultaneously because they're the same trade with different sign conventions. That's a *regime-fit* gap: the agent has no structural coverage for range-bound markets where implied vol is elevated but the market isn't paying enough on OTM verticals to clear the breakeven.

Iron Butterfly (IB) fills that gap. Both short strikes sit at the same ATM strike, so credit collected is roughly 2× the IC equivalent at the same DTE. The trade-off is a much narrower profit zone: an IC wins if the underlying stays inside [K_put_short, K_call_short] (typically ±5-8% at open); IB wins only inside [K−C, K+C] (typically ±1-3% of ATM). IB is the right trade when you expect **pin-precision range-bound behavior + IV mean-reversion**: elevated IV rank on entry, expectation that the underlying will finish near where it started, and enough DTE for theta decay to shrink the mid.

Structurally identical to Iron Condor (4 legs, same order semantics), so almost every piece of existing infrastructure — leg pricing helpers, journal event shapes, position monitor exit gates (per skill 44 the same contract-count scaling now applies to IB) — reuses without modification. The one meaningful architectural difference is the **breakeven invariant**: the C/W-floor formula `|Δ_short| × (1 + edge_buffer)` was derived assuming the short strike sits OTM at some |Δ|. IB shorts are at |Δ|≈0.5 which would make the vertical formula compute a floor of ≈0.5 — meaningless. IB has its own scoring gate (POP ≥ `iron_butterfly_min_pop`, EV > 0) that plays the equivalent role.

The skill ships in **backtest-gated mode**: the scorer is live, unit-tested, callable, and can be exercised by a manual backtest run — but `PresetConfig.iron_butterfly_enabled` defaults to False so no live cycle will ever attempt to plan an IB until a follow-on session enables it after backtest validation. This is the SDD equivalent of a feature flag.

## 2. Mathematical Formula

### 2.1 POP formula — profit-zone-to-wing ratio

```text
POP_IB = min(1.0, (2 · C) / W)

where
  C = credit received per share (dollars)
  W = wing width per share (dollars)

Geometric intuition: at expiration the position profits iff the
underlying finishes in [K − C, K + C] — a window of width 2C. The
outer boundaries at K ± W define the max-loss threshold. POP_IB is
the fraction of the wing width covered by the profit zone.
```

This is the pragmatic approximation used across TastyLive playbooks and matches the rank-order of the lognormal-integral BS-derived POP for typical IB structures (2-5% of spot wing width, 21-45 DTE). The absolute number can differ by 3-5 percentage points but the *ordering* of candidates by POP is preserved, which is what matters for the scanner selecting the highest-scoring grid point.

### 2.2 EV formula

```text
per-share:
  max_profit = C
  max_loss   = W − C
  EV        = POP_IB · max_profit − (1 − POP_IB) · max_loss

per-dollar-risked:
  EV/$risked = EV / max_loss

annualized_score = EV/$risked × (365 / DTE)
```

Same shape as the vertical helper (`_ev_per_dollar_risked`) — the scanner's ranking function is portable across strategies. The `annualized_score` is the sort key for candidate selection.

### 2.3 Gate ordering

Rejection checks fire in a fixed order so the journal histogram has a stable reject-reason distribution:

```text
1. dte ≤ 0                                 → IB_REJECT_DTE_NON_POSITIVE_IB
2. wing_width ≤ 0                          → IB_REJECT_WING_TOO_NARROW
3. credit ≤ 0                              → IB_REJECT_CREDIT_NON_POSITIVE_IB
4. credit ≥ wing_width                     → IB_REJECT_CREDIT_GE_WING
5. call_wing_width ≠ put_wing_width        → IB_REJECT_WINGS_ASYMMETRIC
6. ||Δ_short_call| − 0.5| > tolerance      → IB_REJECT_NOT_ATM_SHORTS
7. ||Δ_short_put|  − 0.5| > tolerance      → IB_REJECT_NOT_ATM_SHORTS
8. POP_IB < min_pop                        → IB_REJECT_POP_BELOW_MIN_IB
9. EV ≤ 0                                  → IB_REJECT_EV_NON_POSITIVE_IB
```

`IB_ATM_DELTA_TOLERANCE = 0.10` — both shorts must have |Δ| ∈ [0.40, 0.60] to qualify as "ATM enough" for an Iron Butterfly. Outside that band the structure becomes a broken-wing butterfly (skill 46, next phase) or an off-ATM IC variant. The gate blocks accidental construction of degenerate structures under the IB label.

## 3. Reference Python Implementation

### 3.1 POP helper

```python
# trading_agent/chain_scanner.py
def _pop_from_ib_structure(credit: float, wing_width: float) -> float:
    """Skill 45 §2.1 — POP for an Iron Butterfly from credit + wing width."""
    if credit <= 0 or wing_width <= 0 or credit >= wing_width:
        return 0.0
    return min(1.0, (2.0 * credit) / wing_width)
```

### 3.2 EV helper

```python
# trading_agent/chain_scanner.py
def _ev_per_dollar_risked_ib(
    credit: float, wing_width: float,
) -> Optional[float]:
    if credit <= 0 or wing_width <= 0 or credit >= wing_width:
        return None
    pop = _pop_from_ib_structure(credit, wing_width)
    max_loss = wing_width - credit
    ev = pop * credit - (1.0 - pop) * max_loss
    return ev / max_loss
```

### 3.3 Scorer + verbose reject sibling

```python
# trading_agent/chain_scanner.py
def _score_iron_butterfly(
    *,
    credit: float,
    wing_width: float,
    short_call_delta: float,
    short_put_delta: float,
    dte: int,
    min_pop: float,
) -> Optional[Tuple[float, float, float, float]]:
```

Same output-shape contract as the vertical scorer's siblings — return None on reject, tuple on accept, with the reject-reason string returned by the `_with_reason` verbose variant. Consumer code that already handles vertical candidates can treat IB candidates identically at the tuple level; only the reject taxonomy is IB-specific.

### 3.4 Orchestrator — `decide_iron_butterfly()`

Phase 1.5 wire-up: the IB analogue of the vertical `decide()`. Sweeps `(DTE × wing_width)` (not the vertical `Δ × DTE × width`), ranks candidates, returns the top-N.

```python
# trading_agent/decision_engine.py
def decide_iron_butterfly(
    inp: DecisionInput, *, max_candidates: int = 5,
) -> IronButterflyDecisionOutput:
```

Per-slice algorithm:

1. Infer ATM strike from `|Δ|≈0.5` via `ChainScanner._infer_spot_proxy`.
2. Find short call + short put nearest ATM using `_find_closest_delta` (call/put-typed).
3. For each `wing_width_pct` in the grid, snap raw wing to strike step, locate long call/put at `ATM ± wing`.
4. Compute per-share credit = `mid(short_call) + mid(short_put) − mid(long_call) − mid(long_put)`.
5. Score via `_score_iron_butterfly_with_reason`; on accept, package as `IronButterflyCandidate`.
6. Sort accepted candidates by `annualized_score desc`, truncate to `max_candidates`.

Returns `IronButterflyDecisionOutput(candidates, diagnostics)` — same shape as `DecisionOutput` so backtester / journal / dashboard code can dispatch uniformly on the `strategy` field.

### 3.5 `IronButterflyCandidate` dataclass

```python
# trading_agent/chain_scanner.py
@dataclass
class IronButterflyCandidate:
    strategy:            str = "iron_butterfly"
    expiration:          str = ""
    dte:                 int = 0
    center_strike:       float = 0.0
    wing_width:          float = 0.0
```

Full field set includes the four leg strikes + symbols, both short deltas, credit / max_profit / max_loss, POP, C/W ratio, EV/$risked, annualized_score, and width_pct. `.to_journal_dict()` rounds every float to 4 decimal places for compact journal rows.

### 3.6 PresetConfig knobs

```python
# trading_agent/strategy_presets.py — PresetConfig additions
iron_butterfly_enabled:            bool  = False
iron_butterfly_min_pop:            float = 0.40
iron_butterfly_dte_grid:           Tuple[int, ...] = (21, 30, 45)
iron_butterfly_wing_width_pct:     Tuple[float, ...] = (0.020, 0.030, 0.040)
```

`enabled=False` is deliberate — the scorer is live and testable but no live cycle will attempt to plan an IB until this flag flips. That flip is a separate commit (backtest-gated).

## 4. Edge Cases / Guardrails

- **Both shorts must be ATM within tolerance 0.10.** `IB_ATM_DELTA_TOLERANCE = 0.10` means each short's |Δ| must be in [0.40, 0.60]. A ticker's real ATM strike drifts from spot as the chain quantizes (nearest strike step), so a strict Δ=0.50 check would reject nearly every real chain. The tolerance is loose enough to accept the actual ATM strike on any strike-step but tight enough to reject anything that looks like a broken-wing structure. Pinned by `test_atm_tolerance_rejects_just_beyond`.
- **Call and put wing widths must match exactly.** The IB scorer accepts a symmetric-only structure. Any asymmetry (call wing 5, put wing 6) is a broken-wing butterfly — different scoring math, different risk profile, different POP formula. Broken-wing lives in skill 46. Rejecting asymmetric chains here prevents accidental construction of BWB under the IB label. Tolerance is 1e-6 to allow floating-point noise but reject any meaningful asymmetry.
- **Wing width vs credit — the C ≥ W gate.** A degenerate case where the credit collected ≥ wing width means the position is net debit or has no max-loss. Neither is a valid IB. Reject with `IB_REJECT_CREDIT_GE_WING`. This gate fires before the ATM check because degenerate math is definitive regardless of delta.
- **POP formula caps at 1.0.** For extreme parameters (say credit=0.6, wing=1.0) the raw `2C/W = 1.2` would exceed 1.0 which is meaningless as a probability. The formula caps at 1.0 mathematically and the scorer accepts the cap without a special-case reject. In practice this only fires on hand-constructed test inputs; real chains don't produce these ratios because market fair-value keeps 2C/W < 1 unless the position is degenerate.
- **Signed put delta.** Puts have Δ ∈ [-1, 0]. The ATM check uses `abs(short_put_delta)` so both `-0.50` and `+0.50` are recognized as ATM. Pinned by `test_signed_delta_handled_correctly_for_puts`.
- **Contract-count scaling (skill 44 interaction).** When the position monitor evaluates an open IB position, the hard-stop / stop-loss / profit-target thresholds all scale by `contracts_open`. The IB's per-contract economics (`max_profit = credit`, `max_loss = wing_width − credit`) plug into the same `_check_exit` formulas as verticals with no changes needed. Skill 44's fix applies transparently.
- **Post-fill grace (skill 44 §2 interaction).** The 60-second post-fill grace on `_check_exit` applies to IB positions the same way it applies to verticals. IB legs are frequently wider bid-ask than vertical legs (ATM options are the highest-vega, so mid moves fastest post-fill), so the grace period is arguably MORE important for IB than for verticals.
- **No CI-invariant scanner rule for IB yet.** The vertical invariant is `|Δ_short| × (1 + edge_buffer)`. IB has no analogous scan-time invariant because the equivalent breakeven condition (`2C/W = 1`) is degenerate (implies zero max-loss). The gate here is the EV > 0 check, which is a structural guarantee at the scoring-function level rather than an AST-walker check at the source-file level. When Phase 2 (broken-wing) or Phase 4 (calendar) land, the CI scanner will need an extension to check per-strategy scoring invariants — that's tracked in the additional-strategies plan doc's cross-cutting decision #3.
- **Live-cycle wiring is future work.** This skill's Phase 1 delivers the scorer, the reject taxonomy, the preset knobs, and the tests. The chain-scanner orchestration (finding the ATM strike, sweeping the wing-width grid, packaging accepted candidates as SpreadCandidate objects) is deliberately NOT in this session — it's the follow-on, alongside the backtester enablement. The pattern matches how skill 40's long-term evaluator shipped as scorer + design doc first, orchestrator second.

## 5. Live Dispatch Wiring (added 2026-07-05)

`Strategy._plan_iron_butterfly` is the strategy-planner's IB entry point. Wired into the sideways-regime branch of `Strategy.plan_trade` — when `preset.iron_butterfly_enabled` is True, the planner tries IB first; if `decide_iron_butterfly` returns no positive-EV candidate the planner falls back to `_plan_iron_condor` so the SIDEWAYS branch is never left empty.

```python
# trading_agent/strategy.py — sideways-regime dispatch
if getattr(self.preset, "iron_butterfly_enabled", False):
    ib_plan = self._plan_iron_butterfly(ticker, analysis, expiration)
    if ib_plan.valid:
        return ib_plan
    # else fall through to IC
return self._plan_iron_condor(ticker, analysis, expiration)
```

The IB planner fetches put+call chains for the expiration, tags each contract with its `type`, merges into a single `ChainSlice`, delegates to `decide_iron_butterfly`, and converts the winning `IronButterflyCandidate` into a `SpreadPlan` with 4 legs (short put, long put, short call, long call) in the exact shape the executor expects.

**Two ways to enable per-preset.**

*Dashboard (recommended)*: open the Streamlit **Live Monitoring** tab → **Strategy Profile** panel → **Custom** preset → scroll to *Iron Butterfly* section → tick "Enable Iron Butterfly on sideways-regime tickers" → click Save. Preset hot-reloads on the next 5-minute cycle. The min-POP slider and DTE / wing-width grids in the same section let you tune without touching JSON.

*Raw JSON edit*: alternatively edit `STRATEGY_PRESET.json` directly:

```json
{
  "profile": "custom",
  "custom": {
    ...
    "iron_butterfly_enabled": true
  }
}
```

Hot-reloads on the next cycle (skill 13). Set back to `false` to disable without restarting.

## 6. Cross-References

- `01_pop_from_delta.md` — vertical POP formula this skill's §2.1 is the IB analogue of.
- `03_credit_to_width_floor.md` — the C/W floor invariant that applies to verticals but NOT to IB; explains why the IB scorer has its own gate rather than reusing the invariant.
- `05_ev_per_dollar_risked.md` — the EV-per-$risked formulation reused with IB-specific `max_loss = wing_width − credit`.
- `44_position_monitor_scaling.md` — the contract-count scaling + post-fill grace period that applies to IB open positions.
- `docs/plans/additional_strategies_plan.md` — the phased plan this skill is Phase 1 of. Phase 2 (broken-wing butterfly) reuses this skill's leg scaffold with asymmetric wings.

---

*Last verified against repo HEAD on 2026-07-06.*

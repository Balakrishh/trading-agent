# Defensive roll

> **One-line summary:** When `STRIKE_PROXIMITY` would otherwise close a threatened spread at a loss, evaluate a six-predicate gate; if all predicates pass, close the threatened spread and open a new further-OTM spread in a single atomic operation that collects a fresh credit to offset the close debit.
> **Source of truth:** [`trading_agent/defensive_roll_evaluator.py`](../../trading_agent/defensive_roll_evaluator.py), [`trading_agent/executor.py:roll_position_defensive`](../../trading_agent/executor.py), [`trading_agent/agent.py:_maybe_defensive_roll`](../../trading_agent/agent.py).
> **Phase:** 1  •  **Group:** risk
> **Depends on:** `03_credit_to_width_floor.md` (the floor compliance predicate uses the same `|Δshort|×(1+edge_buffer)` formula), `13_preset_system_hot_reload.md` (six new tunables live here), `17_close_failure_and_cooldown.md` (the partial-close cooldown wins over a roll attempt), `30_profit_target_management.md` (companion exit-management feature).
> **Consumed by:** `agent.py` Stage-1 close loop — between the PDT-suppression check and the regular close path.

---

## 1. Theory & Objective

A credit spread's `STRIKE_PROXIMITY` exit (short strike within 1% of spot) fires at the worst possible moment economically: the position is deep in the loss zone, and the close executes at the mark, locking in 70–90% of max loss. The defensive-roll alternative offers a structural improvement: instead of just closing, roll the spread to a further-OTM strike — possibly at a later expiry — collecting a fresh credit that offsets some or all of the close debit.

When all six predicates pass, the roll captures three benefits over a plain close:

1. **Loss reduction.** Net dollar damage is `debit_to_close − new_credit_collected`. If new credit covers the close debit, the position is net-flat; if it exceeds, the roll is *credit-positive*.
2. **Re-positioned exposure.** The new short strike is again at the preset's normal `max_delta` target, restoring the original risk profile rather than holding a position whose delta has been pulled toward 1.0 by spot's approach.
3. **Time bought.** A roll to a later expiry buys additional theta runway. The trade-off is more gamma exposure; the `min_dte_for_roll` predicate keeps that bounded.

This is the industry-standard "rolling for defense" pattern from Tastytrade and most professional credit-spread playbooks. The six-predicate gate exists because not every threatened position is rollable — a roll attempt that fails the credit-positive or floor predicates would compound the loss rather than reduce it.

**Why six predicates, not fewer.** A naive "roll whenever threatened" rule fires on positions that can't be saved (no viable OTM strike has positive credit), at expiries too close to manage gamma (DTE < 5), and into candidates that don't clear the same floor we require at open (skill 03 invariant). Each predicate corresponds to a specific failure mode observed in credit-spread trading playbooks; removing any one re-introduces that failure mode.

## 2. Mathematical Formula

Six predicates evaluated in fixed order — any failure produces a named outcome that the journal records.

```text
P1  TRIGGER BAND       proximity_pct ∈ [roll_trigger_min_pct, roll_trigger_max_pct]
                       where proximity_pct = |spot − short_strike| / spot
                       defaults: 0.005 ≤ p ≤ 0.015 (0.5% – 1.5%)

P2  DTE VIABILITY      dte ≥ min_dte_for_roll          (default 5)

P6  BUDGET             current_roll_count < max_defensive_rolls_per_position
                       (default 1)

P5  SAME DIRECTION     norm(strategy_name) == norm(new_strategy_name)
                       where norm(s) = s.lower().replace(" ", "_").strip()

P3  CREDIT POSITIVE    new_projected_credit > debit_to_close

P4  FLOOR COMPLIANCE   new_cw_ratio ≥ |new_short_delta| × (1 + edge_buffer)
                       — same formula as skill 03's open-time floor
```

Predicate order matters because the journal records *the first* failure — so trigger-band rejections (most common) don't shadow downstream rejections that need different debugging.

Per-preset defaults:

```text
  conservative   defensive_roll_enabled = False  (far OTM; rare proximity)
  balanced       defensive_roll_enabled = True   trigger=[0.5%, 1.5%]
  aggressive     defensive_roll_enabled = True   trigger=[0.5%, 1.2%]  (earlier)
```

## 3. Reference Python Implementation

### 3.1 RollDecision enum

```python
# trading_agent/defensive_roll_evaluator.py
import enum
# class RollDecision(str, enum.Enum):
#     Inherits from str so values round-trip cleanly through the
#     journal's JSON serialisation.
#     Members:
ROLL = "roll"
SKIP_NOT_IN_TRIGGER_BAND = "skip_not_in_trigger_band"
SKIP_DTE_TOO_SHORT = "skip_dte_too_short"
SKIP_CREDIT_NEGATIVE = "skip_credit_negative"
SKIP_BELOW_CW_FLOOR = "skip_below_cw_floor"
SKIP_DIFFERENT_STRATEGY = "skip_different_strategy"
SKIP_BUDGET_EXHAUSTED = "skip_budget_exhausted"
```

### 3.2 evaluate_defensive_roll signature

```python
# trading_agent/defensive_roll_evaluator.py
def evaluate_defensive_roll(inp: RollEvalInputs) -> RollDecision:
    """Return RollDecision describing the six-predicate gate."""
```

### 3.3 PresetConfig fields

```python
# trading_agent/strategy_presets.py:PresetConfig
defensive_roll_enabled:            bool  = False
roll_trigger_min_pct:              float = 0.005
roll_trigger_max_pct:              float = 0.015
min_dte_for_roll:                  int   = 5
max_defensive_rolls_per_position:  int   = 1
```

### 3.4 Atomic executor method

```python
# trading_agent/executor.py:roll_position_defensive
def roll_position_defensive(self, spread, new_verdict: "RiskVerdict") -> Dict:
    """Atomic defensive roll: close + open in one operation."""
    close_result = self.close_spread(spread)
    if not close_result.get("all_closed"):
        return {"status": "roll_close_failed",
                "close_result": close_result, "open_result": None}
    try:
        open_result = self.execute(new_verdict)
    except Exception as exc:
        # CRITICAL: close filled, open crashed → position FLAT
        return {"status": "roll_open_failed", ...}
    # else: return "roll_completed" or "roll_dry_run" or "roll_open_failed"
```

### 3.5 Agent-side dispatcher

```python
# trading_agent/agent.py — Stage 1 close loop, after PDT suppression
if (spread.exit_signal == ExitSignal.STRIKE_PROXIMITY
        and self.preset.defensive_roll_enabled):
    roll_outcome = self._maybe_defensive_roll(
        spread, account_balance,
    )
    if roll_outcome is not None:
        closed.append(roll_outcome)
        continue
# Falls through to the normal close path otherwise.
```

## 4. Edge Cases / Guardrails

- **Close fills, open crashes.** Position is *flat*, not doubled — the close-then-open ordering guarantees no naked or doubled-up window. A `CRITICAL` log + `defensive_roll_open_failed` journal action alert the operator. Recovery is manual (re-open via dashboard).
- **Close partial fill.** The executor refuses to open the replacement on top of a stuck-partial close. Next cycle's `STRIKE_PROXIMITY` re-evaluation re-attempts the close; the partial-close-cooldown (skill 17) intercepts after 3 consecutive partial fills.
- **Regime flip mid-defense.** If the underlying breaks through what was sideways into bearish, the planner returns a Bear Call instead of the original Bull Put. The `SAME_DIRECTION` predicate rejects, and the spread falls through to the normal close. This is intentional — rolling into a different strategy is a regime change, not a defense.
- **PDT interaction.** A defensive roll on a same-day-opened position trips PDT *twice* on sub-$25K accounts (one for the close, one for the open). The MVP does not yet suppress this; running defensive rolls on sub-$25K accounts can hit the PDT limit faster than expected. Follow-up: gate `defensive_roll_enabled` behind the PDT-equity check.
- **Floor compliance uses the same formula as open.** `|Δshort|×(1+edge_buffer)` is the skill-03 invariant — a roll that lands below the open-time floor is a bug that the predicate prevents. Equality at the floor passes (matches `risk_manager.py:119`'s `>=` behaviour).
- **Roll count tracking is MVP-stub.** `current_roll_count=0` is passed unconditionally. The `max_defensive_rolls_per_position=1` cap still bounds total rolls in steady state because each successful roll opens a *new* position with a fresh trade plan ID and no roll history. Follow-up: journal-based count of `defensive_roll_completed` rows linked to a position identity.
- **Trigger band asymmetric for Aggressive.** Aggressive tightens `roll_trigger_max_pct` to 0.012 — it opens nearer-ATM so by the time spot reaches the 1.5% band it's already deeply tested. Earlier intervention preserves more reaction time before gamma blows up.
- **Per-leg liquidity gate still applies.** The new candidate's chain has the same skill-29 leg-spread floor — a roll into an illiquid leg is rejected at risk-evaluate time, falling through to the normal close.

## 5. Cross-References

- `03_credit_to_width_floor.md` — `|Δshort|×(1+edge_buffer)` floor reused by predicate P4. A roll must clear the same floor as a fresh open; no defensive-discount.
- `13_preset_system_hot_reload.md` — the six new fields are hot-reloadable like the rest of `PresetConfig`; changes apply on the next cycle without a restart.
- `17_close_failure_and_cooldown.md` — the partial-close cooldown intercepts the close half of a roll attempt; the cooldown wins over the roll.
- `29_per_leg_liquidity_gate.md` — the new candidate's legs run through the same liquidity gate as an organic open.
- `30_profit_target_management.md` — companion exit-management feature; together with PROFIT_TARGET, the close-and-redeploy / roll-for-defense pattern covers both winners and losers.

---

*Last verified against repo HEAD on 2026-05-22.*

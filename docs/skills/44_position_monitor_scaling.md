# Position-Monitor Scaling — Contract Count + Post-Fill Grace

> **One-line summary:** Two structural invariants in `position_monitor._check_exit` that together prevented the SPY 2026-07-02 phantom hard-stop incident (paper-account -$837 max loss on a defined-risk spread with per-contract max loss of $69). First: all three exit thresholds (hard_stop, stop_loss, profit_target) MUST scale by the number of contracts open, because `net_unrealized_pl` is a position-scale total while `original_credit` and `max_loss` are per-contract economics. Second: exit-signal evaluation MUST skip for the first N seconds after the trade plan's `opened_at` timestamp, because the Alpaca spread quote can lag reality immediately post-fill and produce phantom unrealized losses.
> **Source of truth:** [`trading_agent/position_monitor.py:_check_exit`](../../trading_agent/position_monitor.py) (the scaled thresholds + grace gate), [`trading_agent/position_monitor.py:SpreadPosition`](../../trading_agent/position_monitor.py) (the two new dataclass fields).
> **Phase:** 1  •  **Group:** risk
> **Depends on:** `06_stale_spread_risk_gate.md` (companion — that skill handles stale bid-ask on scan-time; this skill handles stale mark on monitor-time), `30_profit_target_management.md` (the same scaling invariant applies to the profit-target threshold, previously silent), `18_order_submission_idempotency.md` (the `opened_at` timestamp is written by the executor's submit path).
> **Consumed by:** `agent._stage_monitor` (every cycle), the close-event collaborators (skill 35).

---

## 1. Theory & Objective

The exit logic in `PositionMonitor._check_exit` compares the position's unrealized P&L against three thresholds — hard-stop at `3× credit`, stop-loss at `50% of max_loss`, profit-target at `50% of credit`. Pre-fix, each threshold was computed using the PER-CONTRACT credit/max_loss from the trade plan, while `net_unrealized_pl` summed every leg's unrealized P&L across all contracts. On a single-contract position the units match by accident. On a multi-contract position they diverge: a 12-contract position trips the hard-stop line at 1/12 of the intended loss because the threshold ($93) is checked against a total that's 12× larger than the per-contract expectation.

The SPY 2026-07-02 incident concretised the risk: a 12-contract Bear Call Spread opened at $0.31/share credit ($31/contract, $372 position credit) tripped hard-stop **four seconds after submission** with a reported unrealized loss of $162 — a value below the intended per-contract threshold but far above the buggy per-contract-formula-vs-position-scale-unrealized comparison. The close was then executed at $837 realized loss (12× per-contract max loss of $69). Two overlapping bugs caused it: contract-count scaling (this skill) and phantom marks on the immediate post-fill cycle (this skill's second invariant).

The fix is small and testable. `SpreadPosition` gains two fields — `contracts_open` and `opened_at`. `_check_exit` scales all three thresholds by `contracts_open` and gates evaluation on a fresh `now_utc - opened_at ≥ post_fill_grace_seconds` check. Both invariants are pinned by conformance tests that use the exact numerics from the 2026-07-02 incident.

## 2. Mathematical Formula

```text
Post-fix, per cycle, for each SpreadPosition s:

  # 0. Post-fill grace gate (skill 44 §4)
  if s.opened_at is set AND (now_utc - parse(s.opened_at)) < grace_seconds:
      return HOLD, "Post-fill grace"

  # 1. Position-scale economics — the critical invariant
  contracts        = max(1, s.contracts_open)
  credit_position  = s.original_credit * 100 * contracts    # dollars for full position
  max_loss_position = s.max_loss * contracts                # dollars for full position

  # 2. Threshold checks (all scaled by contracts)
  hard_stop_threshold  = credit_position  * hard_stop_multiplier      # 3× default
  stop_loss_threshold  = max_loss_position * stop_loss_pct            # 50% default
  profit_target        = credit_position  * profit_target_pct         # 50% default

  loss   = -s.net_unrealized_pl              # positive when losing
  profit =  s.net_unrealized_pl              # positive when winning

  if loss   ≥ hard_stop_threshold  > 0:  return HARD_STOP
  if loss   ≥ stop_loss_threshold  > 0:  return STOP_LOSS
  if profit ≥ profit_target        > 0:  return PROFIT_TARGET
```

The pre-fix code omitted the `× contracts` factor on the three threshold lines. The 2026-07-02 incident's specific numbers demonstrate the failure: 12 contracts × $31 = $372 position credit → correct `hard_stop_threshold` = $1116. The buggy code computed $93 and tripped when `loss` crossed $93, which happens on per-contract mark drift of $8 — a routine number for a tail-strike option's bid-ask spread.

## 3. Reference Python Implementation

### 3.1 `SpreadPosition` new fields

```python
# trading_agent/position_monitor.py
contracts_open: int = 1
opened_at: str = ""
```

`contracts_open` derived from `min(|leg.qty| for leg in spread.legs)` at construction time (guards against partial fills where one leg is at a lower qty than the other). `opened_at` populated from the trade plan's `timestamp` field (`str(tp.get("timestamp", "") or plan_outer_ts)`). Empty string means "no known submit time" — the grace gate is skipped rather than blocking indefinitely.

### 3.2 `_check_exit` — grace gate

```python
# trading_agent/position_monitor.py
if spread.opened_at:
    try:
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        open_ts = _dt.fromisoformat(
            spread.opened_at.replace("Z", "+00:00")
        )
        age = (_dt.now(_tz.utc) - open_ts).total_seconds()
        if 0 <= age < self.post_fill_grace_seconds:
            return (
                ExitSignal.HOLD,
                f"Post-fill grace ({age:.0f}s < "
                f"{self.post_fill_grace_seconds}s)"
            )
    except (ValueError, TypeError):
        pass   # malformed timestamp → fall through
```

### 3.3 `_check_exit` — position-scale thresholds

```python
# trading_agent/position_monitor.py
contracts = max(1, spread.contracts_open)
credit_per_contract = spread.original_credit * 100
credit_position = credit_per_contract * contracts
max_loss_position = spread.max_loss * contracts

hard_stop_threshold = credit_position * self.hard_stop_multiplier
loss = -spread.net_unrealized_pl
if loss >= hard_stop_threshold > 0:
    return (
        ExitSignal.HARD_STOP,
        f"Loss ${loss:.2f} ≥ {self.hard_stop_multiplier:.0f}× credit "
        f"${credit_position:.2f} ({contracts}×${credit_per_contract:.2f}) "
        f"threshold=${hard_stop_threshold:.2f}"
    )
```

The reason string now includes the contract-count breakdown so an operator reading the journal can immediately see whether the threshold was applied correctly.

### 3.4 `PositionMonitor.__init__` — grace-seconds knob

```python
# trading_agent/position_monitor.py
post_fill_grace_seconds: int = 60
```

Default 60 seconds. Set to 0 in tests to exercise the exit paths directly. Not currently threaded to `PresetConfig` — the number is a defensive floor, not a preset tunable.

### `trading_agent/position_monitor.py` — mid re-mark (added 2026-09-29)

```python
def remark_positions_at_mid(positions: List[PositionSnapshot],
                            quotes: Dict[str, Dict]) -> List[PositionSnapshot]:
```

```python
        mid = (bid + ask) / 2
        pl = round((mid - p.avg_entry_price) * p.qty * 100, 2)
```

Called by `agent.py` right after `fetch_open_positions()` and before `group_into_spreads()`, with quotes from `data_provider.fetch_option_quotes`. Every exit threshold (hard stop, stop loss, profit target) therefore evaluates mid-based P&L. `PositionSnapshot.mark_source` records `"mid"` or `"broker"`.

## 4. Edge Cases / Guardrails

- **Empty `opened_at` — grace gate skipped, not blocking.** Inferred spreads (broker-side positions with no matching trade plan) and legacy positions from before this fix have no known submit time. The gate returns nothing rather than blocking indefinitely — the exit paths engage immediately, matching pre-fix behavior for those spreads.
- **Malformed `opened_at` — falls through, doesn't crash.** A `ValueError` from `fromisoformat` (unusual: hand-edited plan files, schema drift) is caught. The exit paths still evaluate; the operator sees the grace gate silently skip rather than the monitor crashing mid-cycle.
- **`contracts_open` defaults to 1 for backward compatibility.** Any existing test or dashboard code that constructs `SpreadPosition` without the new field still works. New field is on the dataclass with `= 1` default; the derivation logic only fires in the plan-matched construction path, which is the only path that has multi-contract positions.
- **Partial-fill contract count.** `contracts_open` is derived as `min(|leg.qty|)` across legs so a partial fill shows the *smaller* side. The position's actual exposure is bounded by the smaller side; using max here would over-count exposure and understate the correct threshold.
- **Zero-qty leg guard.** The min-across-legs computation filters out any `leg.qty == 0` legs to avoid `min([0, 12]) == 0` degenerate cases. If ALL legs are zero-qty (shouldn't happen; would mean the position is closed), `contracts_open` falls back to 1 — the exit gates would then compare per-contract economics against zero unrealized P&L and return HOLD.
- **Grace period is per-position, not per-monitor.** A monitor scanning 5 positions in one cycle applies the grace gate independently to each. A brand-new SPY spread and a 3-day-old QQQ spread both see the grace check; only the SPY spread is inside the window.
- **Grace period respects future timestamps.** `age = now - opened_at` can go negative if a plan's timestamp is somehow in the future (clock skew on the pi, hand-edited plan). The gate uses `0 <= age < grace_seconds`, so a negative age skips the gate rather than treating a future timestamp as an eternal grace period.
- **Interaction with skill 30 (profit-target management).** Skill 30 documents the 50%-of-credit profit target with per-preset overlays. Post-fix, the profit-target threshold ALSO scales by contracts — a multi-contract position now correctly hits profit-target at 50% of the total credit, not the per-contract credit. Existing skill-30 conformance tests use single-contract positions and continue to pass (single contract → position economics equal per-contract economics).
- **Not a `PresetConfig` field yet.** `post_fill_grace_seconds` lives as a `PositionMonitor.__init__` kwarg with a default of 60s. If a preset ever wants to tune this, add `post_fill_grace_seconds: int = 60` to `PresetConfig` (skill 13) and thread it through `agent.py:PositionMonitor` construction. Not needed today; 60s is a defensive floor rather than a strategy tunable.
- **Pre-existing conformance tests still pass.** Skill 30 (`test_skill_30_profit_target_management.py`) and skill 17 (`test_skill_17_close_failure_and_cooldown.py`) both exercise `_check_exit` with single-contract positions. Contracts=1 makes `contracts_open × per_contract == per_contract`, so the pre-fix numeric expectations still hold.

- **Stale broker marks (2026-09-29 SPY IC)** — Alpaca's `current_price` is often the last trade. Summed over 4 legs × 16 contracts, a flat condor (≈ −$8 at mid) showed −$160, and with wide quotes the natural-price view was −$336. Re-marking at mid removes that noise from stop decisions.
- **No usable quote** — missing symbol, bid ≤ 0, or crossed (ask < bid): the leg keeps the broker mark (`mark_source="broker"`). Mixed marks within one spread are allowed; better than dropping the leg.
- **Quote RPC fails** — the whole re-mark is skipped with a WARNING; broker marks are used for that cycle.
- **Wheel legs bypass spread exits (2026-09-29).** `SpreadPosition.strategy_name ∈ WHEEL_STRATEGIES` routes to `_check_wheel_exit` (skill 40 §2.9): profit target at 50 % of `original_credit × 100 × contracts_open`, CSP `DELTA_STOP` from the append-only `short_delta` field, otherwise HOLD. `ExitSignal.DELTA_STOP` is debounced (not in `IMMEDIATE_EXIT_SIGNALS`).

- **`natural_unrealized_pl` (2026-10-05).** `remark_positions_at_mid` also records each leg's P&L at its natural closing price; `group_into_spreads` / inference sum it into `net_natural_pl` (None if any leg is unquoted). Used only for the profit target (skill 30 §4).

## 5. Cross-References

- `06_stale_spread_risk_gate.md` — companion invariant on the scan side (rejects wide bid-ask legs at scan time); this skill's grace period is the monitor-side symmetric protection against stale marks at evaluation time.
- `30_profit_target_management.md` — the profit-target threshold in `_check_exit` shares the scaling fix; that skill's docs remain accurate at the per-share level but its math is now correctly applied to multi-contract positions.
- `18_order_submission_idempotency.md` — the `opened_at` timestamp is populated from the trade plan's `timestamp` field written at submit time by the executor.
- `35_close_event_collaborators.md` — the close-event journal writer downstream of a hard-stop exit signal; the fix here ensures those hard-stop journal rows fire for the right reasons.

---

*Last verified against repo HEAD on 2026-10-06.*

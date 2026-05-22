# Per-leg liquidity gate

> **One-line summary:** Reject any candidate where a single option leg's bid-ask is wider than the looser of two thresholds — catches structurally illiquid chains (typically GLD, high-spot ETFs) before they produce a position that opens at -50%+ of credit due to bid-ask drag.
> **Source of truth:** [`trading_agent/chain_scanner.py:_leg_spread_too_wide`](../../trading_agent/chain_scanner.py), [`trading_agent/decision_engine.py`](../../trading_agent/decision_engine.py) call-site.
> **Phase:** 1  •  **Group:** strategy
> **Depends on:** `03_credit_to_width_floor.md` (next gate in the same chain), `06_stale_spread_risk_gate.md` (related but underlying-level, not option-leg-level), `13_preset_system_hot_reload.md` (where the two threshold knobs live).
> **Consumed by:** `decision_engine.decide()` — fires after the `_find_short` / `_find_strike` resolution but before `_quote_credit`, so the credit/EV math never runs on illiquid candidates.

---

## 1. Theory & Objective

Credit-spread strategies are scored on theoretical credit (mid-mid), but the broker marks the position at **worst-case bid/ask** — shorts at the ask (cost to buy back), longs at the bid (cost to sell). When per-leg bid-ask spreads are wide, the gap between the credit you collect and the cost to close immediately can eat **50%+ of the credit on day one**, even at fill. Theta and IV mean-reversion eventually erase this, but only over multiple days, and the dashboard P&L during that wait looks like a real loss.

The 2026-05-15 GLD trade is the canonical case. The agent filled an Iron Condor at $1.95 credit per spread; the same chain's worst-case round-trip cost was $3.10 → -$115/spread day-1 mark, persistent through the next 90 minutes despite GLD being flat. The C/W floor (0.39 > 0.30 min) passed because it doesn't see bid-ask spreads — only credit and width.

This gate fills that hole. It looks at each leg's bid-ask directly and rejects candidates where *any leg* has a structurally wide quote, before any of the scoring math runs.

**Why AND-of-thresholds, not OR.** A pure absolute cap (e.g. 15¢) would reject XLF's penny options (5¢ bid-ask = 100% of $0.075 mid — still tradable in absolute terms). A pure relative cap (e.g. 5% of mid) would reject SPY's $4 options with a 10¢ spread (2.5% — fine). Requiring **both** to be exceeded means only structurally bad legs (wide in dollars AND wide proportionally) get rejected — the GLD pattern.

## 2. Mathematical Formula

```text
For each leg in (short_contract, long_contract):
    spread = ask − bid
    mid    = (ask + bid) / 2

    if bid ≤ 0  or  ask ≤ 0  or  mid ≤ 0:
        ignore — caller has a separate "no usable quote" reject

    pct_of_mid = spread / mid

    REJECT if  spread > max_leg_spread_cents
         AND  pct_of_mid > max_leg_spread_pct_mid

where
  max_leg_spread_cents    ∈ ℝ⁺   — absolute dollar cap (default 0.15 = 15¢)
  max_leg_spread_pct_mid  ∈ [0,1] — relative cap as fraction of mid (default 0.05 = 5%)
```

If either leg of the spread fails the gate, the whole spread is rejected with `REJECT_LEG_SPREAD_WIDE`.

## 3. Reference Python Implementation

### 3.1 The gate predicate

```python
# trading_agent/chain_scanner.py:_leg_spread_too_wide
def _leg_spread_too_wide(bid: float, ask: float,
                         max_cents: float, max_pct_mid: float) -> bool:
    if bid <= 0 or ask <= 0:
        return False
    spread = ask - bid
    mid = (ask + bid) / 2.0
    if mid <= 0:
        return False
    pct_of_mid = spread / mid
    return spread > max_cents and pct_of_mid > max_pct_mid
```

### 3.2 Call site in `decision_engine.decide()`

```python
# trading_agent/decision_engine.py — inserted before _quote_credit
if _leg_spread_too_wide(
    float(short_contract["bid"]), float(short_contract["ask"]),
    max_leg_spread_cents, max_leg_spread_pct_mid,
) or _leg_spread_too_wide(
    float(long_contract["bid"]), float(long_contract["ask"]),
    max_leg_spread_cents, max_leg_spread_pct_mid,
):
    diag.record(REJECT_LEG_SPREAD_WIDE)
    continue
```

### 3.3 Threshold config in `PresetConfig`

```python
# trading_agent/strategy_presets.py
max_leg_spread_cents:    float = 0.15
max_leg_spread_pct_mid:  float = 0.05
```

Both fields are appended at the end of `PresetConfig` (frozen dataclass, default values per CLAUDE.md soft-rule for append-only fields). The Streamlit Strategy-Profile panel exposes both as sliders.

## 4. Edge Cases / Guardrails

- **GLD 2026-05-15 (the worked example).** Short put 408 bid=5.15, ask=5.50 → 35¢ spread (6.6% of $5.33 mid). Both gates exceeded — leg rejected. Repeat for the long put 403 (30¢, 7.6%) and long call 432 (25¢, 6.1%). The short call 428 (25¢, 4.8% of mid) actually *passes* the relative cap by a hair — but since the spread is rejected if ANY leg fails, the long call's failure tanks the whole candidate.

- **XLF penny options.** Bid 0.05 / ask 0.10 → 5¢ spread (100% of $0.075 mid). Relative cap fails but absolute passes (5¢ ≤ 15¢). AND-of-thresholds means it's not rejected. Validates that XLF stays tradable.

- **SPY tight spreads.** Bid 4.00 / ask 4.10 → 10¢ (2.5% of $4.05 mid). Both caps pass. Validates the gate doesn't accidentally fire on the most liquid name.

- **Missing quote (bid or ask ≤ 0).** Gate returns False (does not fire). The caller's separate `_find_short` zero-bid filter handles this — the gate is silent so it doesn't double-fire on what's really a different failure mode.

- **Inverted market (bid > ask, crossed quote).** Mathematically `spread < 0`, which cannot exceed `max_cents > 0`. Gate doesn't fire. The caller's normal data-validity logic should handle inverted quotes elsewhere — this gate stays narrowly focused on liquidity, not data sanity.

- **Defaults are permissive.** 15¢ + 5% catches GLD-style wide spreads but lets normal index ETF and large-cap option chains through. Tune via the preset slider for tighter or looser policy.

- **Interaction with `_quote_credit`.** This gate runs *before* `_quote_credit`. So credit estimation only happens on candidates with usable quotes — no wasted compute, and `_quote_credit`'s zero-quote fallback path never fires due to a legitimate liquidity issue.

- **Interaction with the C/W floor.** The C/W floor (skill 03) is necessary but not sufficient. The 2026-05-15 GLD candidate passed C/W (0.39 > 0.30) and was still a bad trade. The leg-liquidity gate is the second guardrail that the C/W floor alone misses.

## 5. Cross-References

- `03_credit_to_width_floor.md` — paired risk gate, fires later in the same chain on different signal.
- `05_ev_per_dollar_risked.md` — EV computation that runs only after this gate passes.
- `06_stale_spread_risk_gate.md` — similar idea (reject illiquid trades) but operates on the *underlying's* bid-ask, not the option legs. The two are complementary.
- `13_preset_system_hot_reload.md` — where the two threshold knobs live in `PresetConfig`.
- **Runbook 06** §6 — when the user lands on the dashboard with a wide-spread surprise, the remediation step "preset-tighten the leg-liquidity gate" points to this skill.

---

*Last verified against repo HEAD on 2026-05-22.*

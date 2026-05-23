# Backtester slippage + commissions

> **One-line summary:** `BacktestRunner` exposes `slippage_ticks_per_leg` and `commission_per_leg` kwargs so the operator can model real-world execution gaps. Each spread fills at BS mid ± slippage on both legs; commission is per-leg per-contract on both sides. Defaults preserve legacy behavior (no slippage, $0.65/leg).
> **Source of truth:** [`trading_agent/backtest/runner.py`](../../trading_agent/backtest/runner.py), [`trading_agent/backtest/account.py`](../../trading_agent/backtest/account.py), [`trading_agent/backtest/cycle.py`](../../trading_agent/backtest/cycle.py).
> **Phase:** 2  •  **Group:** backtest
> **Depends on:** `15_backtest_live_parity.md` (the parity surface this widens).
> **Consumed by:** `scripts/compare_backtest_vs_live.py` (slippage sweeps).

---

## 1. Theory & Objective

Pre-2026-05-23 the backtester filled every spread at BS-mid with no execution gap. Real fills land at mid ± 1-3 ticks per leg, so the backtest is systematically over-optimistic by 5-15% per round-trip — exactly the magnitude of the unexplained gap between paper trading and a real-money equivalent.

Two knobs close most of that gap:

1. **Slippage** — extra dollars paid per share on entry (credit reduced) and exit (debit increased). Symmetric: same value on both sides. Expressed in ticks so the operator can sweep `0 / 1 / 2 / 3` and see the equity-curve impact.

2. **Commission** — already modeled at $0.65/leg before this commit, but the constant was hardcoded. Now overridable per run so you can A/B free brokers vs $0.50/contract vs $1.00/contract tiers.

Default values (`slippage_ticks_per_leg=0`, `commission_per_leg=$0.65`) preserve legacy results.

## 2. Mathematical Formula

For one spread of `qty` contracts with `slippage_ticks_per_leg = k` and `commission_per_leg = c`:

```text
effective_credit_open   = max(0, BS_mid_credit - slippage_per_share)
effective_debit_close   = BS_mid_debit + slippage_per_share

where:
  slippage_per_share = k × TICK_SIZE × 2   (2 legs per credit spread)
  TICK_SIZE = $0.05

cash_at_open = +effective_credit_open × qty × 100 - 2c × qty
cash_at_close = -effective_debit_close × qty × 100 - 2c × qty

realised_pnl  = (effective_credit_open - effective_debit_close) × qty × 100 - 4c × qty
```

For an Iron Condor (which the live agent submits as ONE 4-leg order; the backtester models as TWO independent SimPositions for bull_put + bear_call), commission lands as 4c × qty on each side = 8c × qty round-trip. Slippage applies to each spread independently.

## 3. Reference Python Implementation

### 3.1 SimAccount apply_open / apply_close

```python
# trading_agent/backtest/account.py
def apply_open(self, *, credit_per_share: float, qty: int,
               spread_width: float,
               slippage_per_share: float = 0.0,
               commission_per_leg: float = COMMISSION_PER_LEG) -> float:
    ...
    effective_credit = max(0.0, credit_per_share - slippage_per_share)
    credit_dollars = effective_credit * qty * 100.0
    commission = commission_per_leg * 2 * qty
    self.cash += credit_dollars - commission
    self.open_market_value -= credit_dollars
```

### 3.2 BacktestRunner constructor

```python
# trading_agent/backtest/runner.py
                 slippage_ticks_per_leg: int = 0,
                 commission_per_leg: Optional[float] = None):
    ...
        self.slippage_ticks_per_leg = int(slippage_ticks_per_leg)
        self.slippage_per_share = (
            self.slippage_ticks_per_leg * self.TICK_SIZE * 2
        )
```

### 3.3 Threading through `run_one_cycle`

The cycle module accepts the kwargs and passes them to `account.apply_open`. The runner passes them to `account.apply_close` directly. Symmetric on both sides.

## 4. Edge Cases / Guardrails

- **Symmetric accounting.** `apply_close` mirrors `apply_open`'s slippage handling — the SAME slippage value is subtracted from the open's `credit_per_share` AND added to the close's `closing_debit_per_share`. If only one side applied slippage, equity would drift by exactly `slippage × qty × 100` per round-trip. This is hard-pinned by `test_skill_38_slippage_is_symmetric_across_open_close`.
- **Slippage clamps to ≥0.** `max(0, credit - slip)` prevents an absurd-large slippage from booking negative credit (which would crash the realised P&L math). At sensible slippage values (1-3 ticks) this clamp never engages.
- **Commission default is `None` at the runner level.** `None` means "use the module constant ($0.65)". Passing `0.0` explicitly models a free broker. Passing a positive number overrides.
- **TICK_SIZE is $0.05 (post-pennypilot).** Sub-$3 options trade on $0.01 ticks, but credit spreads in this strategy always price above that. If the strategy ever opens 0DTE sub-$3 spreads, this constant needs adjusting.
- **Iron Condor commission is correct.** An IC is two SimPositions (bull_put + bear_call), each pays its own 2-leg commission per side = 4 legs per side = 8 legs round-trip = `8 × commission_per_leg × qty`. Matches what a real broker would charge.
- **Slippage applies to net credit, not per leg.** Although the parameter is named `slippage_ticks_per_leg`, the runner pre-multiplies by 2 (two legs per credit spread) before passing to `apply_open`. So `slippage_per_share` in the SimAccount is per-spread, not per-leg. This keeps the apply_* signatures clean.

## 5. Cross-References

- `15_backtest_live_parity.md` — the parity contract this commit widens.
- `scripts/compare_backtest_vs_live.py` — the diff harness that consumes these knobs to attribute the live-vs-backtest gap.
- `39_volatility_skew.md` (planned) — the next backtester improvement, addressing the OTM-mispricing gap.

---

*Last verified against repo HEAD on 2026-05-23.*

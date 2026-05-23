# Backtester volatility skew

> **One-line summary:** Parametric IV-skew adjustment applied at synthetic-chain build time so OTM puts price MORE expensively than ATM and OTM calls price slightly more. Closes the second-biggest gap between backtest and live: flat-vol BS systematically underprices the protective long leg of bull put spreads and the wings of Iron Condors.
> **Source of truth:** [`trading_agent/backtest/skew_model.py`](../../trading_agent/backtest/skew_model.py), [`trading_agent/backtest/synthetic_chain.py`](../../trading_agent/backtest/synthetic_chain.py), [`trading_agent/backtest/cycle.py`](../../trading_agent/backtest/cycle.py), [`trading_agent/backtest/runner.py`](../../trading_agent/backtest/runner.py).
> **Phase:** 2  •  **Group:** backtest
> **Depends on:** `38_backtest_slippage.md` (the parity surface widening continues).
> **Consumed by:** `scripts/compare_backtest_vs_live.py` (skew sweeps).

---

## 1. Theory & Objective

Black-Scholes assumes a single σ across all strikes — flat vol. Real equity markets show pronounced **skew**: OTM puts trade at HIGHER IV than ATM (operators bid up protective puts), and OTM calls usually at slightly LOWER or similar IV. For index ETFs (SPY/QQQ/IWM/DIA/XLF/XLE) the typical shape is:

| Strike | Delta | IV ratio to ATM |
|---|---|---|
| 10% OTM put | ~0.10 | 1.18-1.25× |
| 5% OTM put | ~0.25 | 1.05-1.10× |
| ATM | ~0.50 | 1.00× |
| 5% OTM call | ~0.25 | 1.02-1.05× |
| 10% OTM call | ~0.10 | 1.05-1.15× |

Single-stock skew is steeper on the put side (often 1.7-2.0× at far OTM).

Pre-this commit the backtester used flat-vol BS, producing two systematic biases:

1. **Bull put spread credits look too big.** The protective long-put leg is FURTHER OTM than the short. With real skew, the long costs MORE (it's at a higher IV). Net credit shrinks. Backtest didn't see this — credits were 5-15% too generous depending on width and delta.

2. **Iron Condor wing prices are mispriced** — call wings approximately right, put wings too cheap on the long-leg side. Net IC credit slightly overstated.

The skew model is a constructor-injected `SkewModel` dataclass that the synthetic-chain builder applies per-strike. The default `SkewModel()` is flat (preserves legacy behavior); operators opt in to skewed pricing via `INDEX_ETF_SKEW`, `SINGLE_STOCK_SKEW`, or a custom calibration.

## 2. Mathematical Formula

For a strike `K` at spot `S` with ATM σ `σ_ATM`:

```text
m = K/S - 1                                  (linear moneyness)

σ(K) = σ_ATM × (1 + α × max(0, -m) + β × max(0, +m))

where:
  α (put_skew)   — slope on the OTM-put side. ~0.6 for index ETFs.
  β (call_skew)  — slope on the OTM-call side. ~0.15 for index ETFs.
```

Behaviour:
- `m < 0` (OTM put): `σ = σ_ATM × (1 + α × |m|)`
- `m = 0` (ATM): `σ = σ_ATM`
- `m > 0` (OTM call): `σ = σ_ATM × (1 + β × m)`

Output clipped to `[0.01, 5.0]` so adversarial inputs can't crash the BS pricer.

## 3. Reference Python Implementation

### 3.1 SkewModel

```python
# trading_agent/backtest/skew_model.py
@dataclass(frozen=True)
class SkewModel:
    put_skew: float = 0.0
    call_skew: float = 0.0

    def sigma_for_strike(self, strike: float, spot: float,
                         atm_sigma: float) -> float:
        ...
```

### 3.2 Built-in presets

```python
FLAT_SKEW = SkewModel(put_skew=0.0, call_skew=0.0)
INDEX_ETF_SKEW = SkewModel(put_skew=0.6, call_skew=0.15)
SINGLE_STOCK_SKEW = SkewModel(put_skew=1.0, call_skew=0.25)
```

### 3.3 Chain integration

```python
# trading_agent/backtest/synthetic_chain.py
                      r: float = 0.0,
                      skew_model=None) -> ChainSlice:
    ...
        if skew_model is not None:
            sigma_k = skew_model.sigma_for_strike(
                strike=float(k), spot=spot, atm_sigma=sigma_annual,
            )
        else:
            sigma_k = sigma_annual
        delta = bs_delta(spot, k, t_years, sigma_k, r=r,
                         option_type=opt_type)
        mid = bs_price(spot, k, t_years, sigma_k, r=r,
                       option_type=opt_type)
```

### 3.4 Runner integration

```python
# trading_agent/backtest/runner.py
        self.skew_model = skew_model
        ...
                    skew_model=self.skew_model,
```

The runner stores the skew model and threads it into every
`run_one_cycle` invocation alongside `slippage_per_share` and
`commission_per_leg`, so all three execution-model knobs travel
through the same path.

## 4. Edge Cases / Guardrails

- **Default is flat.** `skew_model=None` (or explicit `FLAT_SKEW`) preserves the pre-2026-05-23 behavior so existing backtests don't silently shift. Operators opt in to non-flat skew.
- **Per-strike σ is used for BOTH delta and price.** This is the matching constraint that keeps Δ-targeting consistent with mid pricing — if delta were computed at ATM σ but price at skewed σ, the engine would pick strikes whose actual delta doesn't match the target.
- **Mid-life re-marks DON'T currently honor skew.** `SimPosition.remark` still uses the entry σ scaled by VIX, not skew-adjusted per current spot. This is a known residual gap — for accurate close-side P&L on positions that move ITM, the re-mark would need its own skew application. Tracked as a follow-up.
- **VIX-proxy + skew are independent.** VIX-proxy scales σ_ATM with realized vol; skew lifts σ_K relative to σ_ATM. They compose: σ_K(t) = σ_ATM(t) × skew(K, spot(t)).
- **Single-stock skew is steeper.** `SINGLE_STOCK_SKEW` has `put_skew=1.0` vs the ETF `0.6`. Use only if the strategy expands to single-name underlyings; the current universe is all ETFs.
- **Calibration is operator-driven.** No live-options-chain pulls happen in the backtester (yfinance options coverage is uneven historically). Operators sweep `put_skew` until backtest credits match observed live credits for the same strikes.

## 5. Cross-References

- `38_backtest_slippage.md` — the first parity-widening commit; this is its successor.
- `15_backtest_live_parity.md` — the overarching parity surface.
- `scripts/compare_backtest_vs_live.py` — operator harness that sweeps both knobs together.

---

*Last verified against repo HEAD on 2026-05-23.*

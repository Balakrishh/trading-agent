# POP calibration study — realized-vol vs implied vs outcomes (backlog B2)

*Run 2026-10-13 on the Pi: `python scripts/research/calibrate_pop_rv.py` (Yahoo daily closes 2019-01-01 → 2026-10-13, 12 watchlist tickers; vol indices ^VIX, ^VXN, ^GVZ, ^VXEEM; ^RVX unavailable). One-sided short strikes at z = 0.5 / 0.75 / 1.0 σ (≈ 31 / 23 / 16 Δ), 14 / 21 / 30 DTE, held to expiry. ~419k overlapping samples — counts overstate independent evidence.*

## Results

**1. `pop_rv` is well calibrated where the agent trades.** Pooled, mean `pop_rv` 78.0 % vs 77.8 % realized; the dense buckets miss by ≤ 3 points (0.6–0.7: +2.8, 0.7–0.8: −1.0, 0.8–0.9: −0.7). It is overconfident above 0.9 (93.9 % predicted, 88.8 % realized — fat tails). Brier scores are almost identical to the implied model (0.1548 vs 0.1540 on vol-index tickers): realized vol is *as good* a predictor as implied vol, not clearly better.

**2. Selling puts had an edge; selling calls did not — in both trends.**

| Side × trend (pooled) | pop_rv | implied (vol-index tickers) | realized (same rows) |
|---|---|---|---|
| short put, up-trend | 77.6 % | 76.2 % | **87.4 %** |
| short put, down-trend | 76.6 % | 76.0 % | **85.4 %** |
| short call, up-trend | 79.1 % | 77.6 % | 75.5 % |
| short call, down-trend | 78.5 % | 77.7 % | **72.0 %** |

Put strikes survived ~9–11 points more often than implied odds said; call strikes ~2–6 points *less*, including in down-trends, where the agent's playbook sells bear call spreads. The 50-day trend filter did not protect the call side. Part of the put edge is 2019–2026's upward drift (the model assumes zero drift); the rest is the variance risk premium. Skew makes it starker: real OTM-put implied vols sit above the ATM index used here (true implied POP lower → more put edge), OTM-call vols below it (less call edge).

**3. By horizon** the put-side gap grows with DTE (+2 to +7 points at 14 → 30 days); the call-side shortfall grows too (−2 to −9 points).

## What it means for the plan

- **B3 (two-model EV) is supported for the put side only.** Bull put spreads, cash-secured puts and the put wing of condors have historical edge beyond implied odds; a gate that requires positive EV under `pop_rv` *capped at 0.90* (the overconfident bucket) is reasonable there.
- **Bear call spreads look structurally negative** in this sample. Candidate rule change (operator decision): in bearish regimes prefer put debit spreads to bear call spreads, or require a larger edge buffer on calls.
- **Iron condors** carry a losing call wing; skewing condors put-heavy (wider/further call wing) follows from the same data.
- Caveats: one bull-dominated period (2020 and 2022 included); outcomes are hold-to-expiry survival, not P&L with the agent's exits; overlapping samples; no transaction costs. The live shadow book (B1) is the out-of-sample check.

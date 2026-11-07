# Selling and buying volatility on real option quotes (2026-11-07)

Question (operator, 2026-11-06): "when volatility is high people make a
lot of money — can the agent make same-day trades on volatility? Fewer
trades don't give more value." Two studies on the real end-of-day
chains (`~/data/options_chains`: SPY / IWM 2008–2025, QQQ 2011–2025;
see `research_debit_stress.md` §2 for the source).

Script: `scripts/research/credit_chain_backtest.py` (§2). The 1-day
study (§1) was a one-off in-session run with the same data. "Vol rank"
is the live RV20 percentile over the last ~200 sessions (the regime's
`iv_rank`); P&L is per $1 risked (width − credit for credit spreads,
the premium for long options), after paying the real bid / ask.

## 1. One-day trades (enter at the close, settle at the next close)

Next-session expiry, entry at natural, settled at intrinsic on the next
close. 1,960 sessions, mostly 2016–2025 (daily expiries).

| 1-day trade | rank < 50 | 50–80 | 80–95 | ≥ 95 |
|---|---|---|---|---|
| Long ATM straddle (buy vol) | 39 % / −0.050 | 36 % / −0.105 | 39 % / −0.073 | 44 % / −0.015 |
| Short iron condor, 10Δ, $5 wings | 86 % / +0.005 | 87 % / +0.012 | 87 % / +0.006 | 81 % / −0.019 |
| Short put spread, 10Δ, $5 | 94 % / +0.001 | 95 % / +0.012 | 92 % / −0.003 | 94 % / +0.004 |

Buying volatility loses in every regime — 1-day options already price
big moves. Selling it wins most days and earns ≈ 0 after the bid / ask;
every variant has full-loss days. Not tested: true intraday trades (the
data is end-of-day) — hence the intraday chain log (skill 69).

## 2. Credit spreads held 1–3 weeks, by volatility regime

Entries every session (overlapping trades — per-trade averages, not an
account curve). Shapes: bull put spreads and iron condors at 7 / 14 /
21 DTE, short |Δ| 0.15 / 0.25 (condors 0.15 / 0.20), width 2 % (one
1 %) of spot; legs pass the live liquidity gate. Entry fill halfway
mid → natural; closes halfway. Exit sets: **agent** (profit target 50 %
of credit on mid, stop at 50 % of max loss, hard stop 3× credit, strike
proximity 1 % on the daily close, DTE safety), **no-prox** (agent
without the proximity rule), **tp-only** (target, else to expiry),
**2x-stop** (target, stop at 2× credit). ~10,000 entries per shape
across the three ETFs.

Avg P&L per $ risked (win rate) by vol rank at entry:

| Shape, exits | < 50 | 50–80 | 80–95 | ≥ 95 | all | after a spike, vol receding |
|---|---|---|---|---|---|---|
| put 21d 0.25Δ, agent | −0.002 (64 %) | +0.020 (72 %) | +0.005 | +0.012 | +0.006 (68 %) | +0.002 |
| put 21d 0.25Δ, no-prox | +0.007 (86 %) | +0.034 (89 %) | +0.002 | +0.008 | +0.014 (86 %) | −0.022 |
| put 14d 0.25Δ, no-prox | +0.014 | +0.031 | −0.001 | −0.005 | +0.015 (86 %) | −0.020 |
| put 21d 0.15Δ, agent | +0.006 (85 %) | +0.016 | −0.004 | −0.002 | +0.007 (86 %) | −0.005 |
| put 7d 0.15Δ, agent | +0.001 | +0.008 | −0.010 | −0.002 | +0.001 (80 %) | −0.001 |
| condor 21d 0.15Δ, agent | −0.011 | −0.021 | −0.042 | −0.056 | −0.021 (57 %) | −0.048 |
| condor 21d 0.20Δ, agent | −0.020 | −0.026 | −0.051 | −0.069 | −0.029 (40 %) | −0.050 |

Month-block bootstrap, 95 % intervals:
put 21d 0.25Δ agent +0.006 [−0.004, +0.015]; no-prox +0.014 [−0.004,
+0.031]; at rank ≥ 80 +0.008 [−0.013, +0.028]; at rank 50–80 no-prox
+0.034 [+0.009, +0.057]. No-prox − agent: put 14d 0.25Δ +0.009
[+0.001, +0.018], put 21d 0.25Δ +0.008 [−0.004, +0.018]. Condor 21d
0.15Δ −0.021 [−0.031, −0.011].

What it shows:

- **Selling after a volatility spike does not pay.** Rank ≥ 80 and the
  "spike, now receding" entries are no better than calm markets, and
  worse for condors. 2008, 2018 and 2020 lose heavily at high rank
  (−0.05 to −0.30 per $). The HighIV gate (no entries above rank 95)
  costs nothing and should stay.
- **The best regime is moderately elevated volatility (rank 50–80)**:
  +0.02 to +0.035 per $ on 0.25Δ put spreads — the only interval that
  excludes zero. Chosen after looking at four buckets, so treat it as a
  hypothesis, not a proven edge.
- **Iron condors lose in every regime and every exit set** (the call
  side gives back what the put side earns: index ETFs drift up and
  call skew is cheap). The agent's delta-sum floor already blocks
  almost all of them; keep it that way.
- **Strike proximity costs ~0.008 per $ on 0.25Δ put spreads** and
  closes 31–38 % of them, turning 86 % winners into 66 %. Measured on
  daily closes only; live, the rule fires on intraday touches, so the
  real cost is probably higher. Candidate change (not built): skip the
  1 % proximity exit for credit verticals and rely on the max-loss stop.
- **The live C/W floor (credit / width ≥ |Δshort|) selects the worst
  trades**: the ~2 % of entries that pass lose −0.03 to −0.04 per $
  (condors −0.21 to −0.29). Rich put credit relative to delta is the
  market pricing crash risk, not free money. This explains why the
  agent rarely sells premium, and it is the right outcome.
- **Edge size**: the best put-spread variants make +0.01 to +0.02 per $
  risked. At 2 % of equity per trade that is about $5–10 a trade on
  this account — real, but small, and it disappears in 2008-type years.

## 3. Recommendations

1. **No same-day or after-spike volatility trades.** Neither buying
   (straddles) nor selling (condors, put spreads) shows an edge after
   real bid / ask; the HighIV gate stays.
2. **Collect intraday data first** — the intraday chain log (skill 69)
   is on from 2026-11-09 for SPY 0–1 DTE. Re-test same-day ideas after
   4–6 weeks of snapshots.
3. **Worth a follow-up study** (not built): 0.25Δ bull put spreads
   without the 1 % strike-proximity exit, gated to vol rank 50–80, as a
   second premium-selling playbook next to the call debits. Needs a
   test that replays the agent's own scan (scoring, C/W floor) before
   it goes on paper.
4. **More trades only help with a per-trade edge.** At ≈ 0 per trade,
   more trades add bid / ask cost and tail risk, not profit.

## 4. Caveats

End-of-day marks: stops and proximity are judged on closes, so
intraday gaps are missed (flatters stops). Overlapping daily entries
inflate n; the month-block bootstrap is the honest interval. Three
index ETFs only. Fills halfway mid → natural; Alpaca paper fills on
0DTE are more generous than real ones.

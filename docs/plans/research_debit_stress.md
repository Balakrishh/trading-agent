# Stress tests and a real-chain check of the debit-spread rules (2026-11-06)

Scripts: `scripts/research/debit_stress_study.py` (the Black–Scholes model
of `debit_exit_study.py` under harsher assumptions) and
`scripts/research/debit_chain_backtest.py` (the agent's selection and exit
rules on **real end-of-day option quotes**). "Old" = the rules before
2026-11-06 (stop 50, trail arms 50 / gives back 25 on natural, no trend /
volatility filters); "new" = the rules live from 2026-11-06 (call debits
stop 80, put debits arm at 40, exits judged at mid, 200-day trend and
volatility-regime filters).

## 1. The model under stress (Black–Scholes, 16 ETFs, 2008–2026)

**Costs** (all entries, avg P&L per $1 risked):

| Costs | old | new |
|---|---|---|
| IV/RV 1.0, fills 3 % | −0.029 | +0.071 |
| IV/RV 1.1 (base) | −0.030 | +0.063 |
| IV/RV 1.2 | −0.032 | +0.054 |
| IV/RV 1.3 | −0.030 | +0.047 |
| IV/RV 1.4 | −0.034 | +0.033 |
| fills 5 % each way | −0.065 | +0.021 |
| entry 3 %, profit exits 2 %, stops 5 % (realistic, mid-exit ladder) | −0.027 | +0.070 |
| fixed $0.05 / share per side | −0.163 | −0.111 |

A fixed cost in cents is what kills it: with $0.05 a side, spreads under
$1.00 lose (−0.28 per $ under $0.50) while spreads over $1.50 stay at
+0.15. Measured in-session half-spreads (scan near-misses, 2026-11-02 →
11-06): TLT $0.025, IWM $0.11, XLU $0.09 (25 % of a $0.42 mid), GLD
$0.275, DIA $0.275, QQQ $0.31. The fair-price check (natural ≤ mid ×
1.05) caps the entry cost; exits pay again (XLU 10-29 lost $0.085 a share
to the close, 15 % of its $0.56 debit).

**Periods** (new rules): positive in 2010–11 (+0.167), 2020 (+0.139),
2013 / 2017 bulls (+0.26 / +0.29), 2023–24 (+0.087); negative in the
2008–09 crisis (−0.268), 2015–16 (−0.463), 2018 Q4 (−0.503) and 2022
(−0.182) — worse than the old rules there, because the wider call-debit
stop gives more back when a bull trend breaks. Neither a market gate
(SPY above SMA50 / SMA200, SPY vol rank) nor tightening the stop on a
vol spike reduced those losses: crashes start from calm markets and gap
through stops.

**Rolling 3-year windows**: negative in 4 of 16 (old: 13 of 16).

**Parameter noise** (every threshold ±25 % at once, 200 draws): average
+0.021 (5th pct) / +0.056 (median) / +0.103 (95th); positive in 200 / 200
and better than the old rules in 200 / 200.

**Month-block bootstrap**: new +0.063, 95 % interval [−0.004, +0.131],
positive in 97 % of draws (old −0.030, [−0.074, +0.016]).

**Account** (2 trades a week, 1.5 % of equity at risk each, 3,000
bootstrapped years): new rules with realistic costs — median +4.2 %, 5th
pct −16 %, chance of a losing year 38 %, max drawdown 22 % (95th pct),
losing streak 6 (median) / 10 (95th). Old rules: median −6 %, 70 %.

## 2. Real option quotes (SPY, QQQ, IWM)

Data: end-of-day chains with bid, ask and delta per contract (SPY and IWM
2008–2025, QQQ 2011–2025), published as release `data-v1` of
lambdaclass/options_portfolio_backtester (MIT repo; the data is offered
for research use — its upstream, philippdubach/options-data, went offline
in 2026). Kept outside the repo. The backtest copies the agent's
selection (0.50Δ long, 1–2.5 % width grid, leg gate, natural ≤ mid ×
1.05, reward/risk ≥ 1, cheapest natural) and marks each day on the two
contracts' real quotes. Entry fill mid + ½ of the half-spread (the
ladder), closes halfway mid → natural, unless noted.

| Variant | SPY + QQQ n / win / avg | 2008–16 | 2017–25 | IWM n / win / avg |
|---|---|---|---|---|
| old, entry at natural | 345 / 44 % / −0.058 | −0.208 | +0.053 | 191 / 41 % / −0.097 |
| old + entry ladder | 345 / 44 % / −0.047 | −0.199 | +0.065 | 191 / 41 % / −0.099 |
| new exits, no filters | 345 / 55 % / −0.022 | −0.167 | +0.085 | 191 / 51 % / −0.099 |
| old exits + filters | 239 / 46 % / −0.013 | −0.236 | +0.193 | 113 / 38 % / −0.169 |
| **new (live)** | **239 / 58 % / +0.023** | −0.187 | **+0.217** | 113 / 48 % / **−0.190** |
| new, closes at natural | 239 / 58 % / −0.003 | −0.215 | +0.193 | 113 / 48 % / −0.217 |
| new, entry at natural | 239 / 58 % / +0.011 | −0.210 | +0.216 | 113 / 48 % / −0.193 |

By ticker (new vs old at natural): SPY +0.07 vs −0.060 (61 % vs 43 % win),
QQQ −0.03 vs −0.056, IWM −0.190 vs −0.097.

What the real quotes confirm:

- **The new exits raise the win rate on every ticker** — +11 points on
  SPY + QQQ, +10 on IWM — and the money on SPY (+0.13 per $).
- **Implied runs ~1.1× realized** (IWM ATM IV ÷ next-21-day RV: calls
  1.08, puts 1.16), as the model assumed.
- **Execution is worth ~0.025 per $**: closing at natural instead of
  halfway turns SPY + QQQ from +0.023 to −0.003.
- **2008–2016 lost under every rule**; 2017–2025 made money. The edge is
  regime-dependent, as the model's periods table showed.

What they correct:

- **IWM loses under every rule, and the filters make it worse**
  (−0.169 / −0.190). The model also had IWM negative (−0.042). IWM debit
  spreads should not trade.
- **Put debits lose on real quotes**: −0.13 (SPY + QQQ) and −0.07 (IWM)
  per $ under the old rules; the filters leave 3 and 6 trades, all
  losers. The model's +0.050 for filtered put debits is not borne out —
  though 9 trades decide nothing. On these three index ETFs the new rules
  are effectively call debits only.
- The model overstated the level (+0.063); on real SPY + QQQ quotes it
  is +0.023 per $, all of it from 2017 onward.

## 3. Libraries and data

| Tool | What it would add | Verdict |
|---|---|---|
| **Real historical chains** (`data-v1`, SPY / QQQ / IWM) | Real bid / ask / delta instead of Black–Scholes and assumed costs | **Used** (§2). The single biggest improvement to the research. No free source found for GLD, TLT, XLU or the sector ETFs; paid: iVolatility DataShop (pay per use), historicaloptiondata.com, EODHD |
| lambdaclass/options_portfolio_backtester (MIT, Rust core) | Contract-level options backtests, tear sheets | Not needed — the agent's rules are bespoke; the 300-line `debit_chain_backtest.py` replays them exactly |
| optopsy 2.x | 38 preset options strategies on chain data; needs an EODHD key for data | Same — useful for generic strategies, not for replaying this agent |
| quantstats | Sharpe / drawdown / monthly tear sheets from a returns series | Worth adding later for the paper account's daily equity (a weekly report) — reporting only |
| arch (GARCH) | Volatility forecasts | Low value: the vol-regime filter already ranks realized vol, and the edge is in the trend / regime, not the vol estimate |
| py_vollib(_vectorized), QuantLib | Fast IV / Greeks | No need — the agent's Black–Scholes and Schwab's Greeks cover it |
| vectorbt, backtrader, zipline-reloaded | Fast / event-driven equity backtests | No options-chain handling; our studies run in minutes |

## 4. Recommendations

1. **Stop trading IWM debit spreads** (call and put): negative under every
   rule in both the model and the real quotes. *Built 2026-11-06*
   (`debit_excluded_tickers`).
2. **Treat put debits as unproven**: negative on real SPY / QQQ / IWM
   quotes under the old rules; the filters already leave very few. Keep
   them on paper with the filters, but review after ~10 closes, or turn
   them off for index ETFs. *Done 2026-11-06 for SPY and QQQ*
   (`put_debit_excluded_tickers` on the paper preset).
3. **Extend the mid-first ladder to stop closes** (today: halfway, then
   natural): execution is worth ~0.025 per $ on real quotes. *Built
   2026-11-06* (mid → halfway → natural, then the per-leg fallback).
4. **Keep the new exits and filters** for call debits on SPY / QQQ —
   the one place the real quotes show an edge, from 2017 on.
5. **Size for a thin edge**: even with realistic costs the model gives a
   38 % chance of a losing year at 1.5 % risk per trade and 2 trades a
   week; the 2 % per-trade cap (1 % in CAUTION) should not go up.

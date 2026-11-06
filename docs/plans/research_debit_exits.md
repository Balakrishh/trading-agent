# Debit-spread exits and entry filters — research note (2026-11-06)

Script: `scripts/research/debit_exit_study.py` (daily closes from yfinance,
2008 → 2026-11-06, 16 watchlist ETFs, cached with `--cache DIR`). It asks
why the paper account's debit spreads lose (3 winners in 12 closes since
2026-10-14) and what would raise the win rate without lowering the
profit per dollar risked.

## Model

Same as `research_debit_entry.md`: Black–Scholes with volatility =
max(20-day realized, 8 %) × 1.10; an entry every 5th session per ticker
when the trend agrees (bear: close < SMA50 and SMA50 falling over 10
sessions; bull mirror) **and the live pullback filter passes**; 30-day
vertical, long leg 0.50Δ, short leg 2 % of spot further out; entry at
value × 1.03, skipped when reward/risk < 1; marked daily at value × 0.97.
Exits reproduce the live rules: stop at `stop` × debit lost; the trail
arms at the target, closes when profit falls to peak × (1 − giveback),
at the ceiling (0.90 × max profit) or, once armed, at ≤ 7 days left;
otherwise expiry. P&L is per $1 of debit; "1st / 2nd" are the halves
(split 2017-10-23).

Limits: daily marks (the agent checks every ~95 s), no put skew, no
intraday gaps through stops. Compare variants; the absolute level is
not a forecast.

## 1. What the paper account shows

12 closed debit trades (excursion rows, 2026-10-14 → 11-05):

| Group | Trades | Note |
|---|---|---|
| Up ≥ 30 % of the debit, then stopped | TLT 10-26 (+$100 → −$155), XLU (+$100 → −$210) | the pattern the operator described |
| Up < 20 %, then stopped | AMZN, TLT 10-22, IWM | |
| Never in profit | QQQ ×2, TLT 10-15, IWM calendar | direction wrong from the start |
| Winners | GLD +$190, SPY calendar +$149, SPY put debit +$131 | |

Several recorded lows exceed the debit (AMZN −$1,275 on $460, IWM −$838
on $246, GLD −$555 on $265) — all at 09:25 ET, pre-open quotes (§5).

## 2. Exit rules

Live entries (no new filter), 2 % wide:

| Variant | Bull n / win / avg | Bear n / win / avg | All avg (1st / 2nd) |
|---|---|---|---|
| live: stop 50, trail arms 50 / gives back 25 | 3,294 / 45 % / −0.005 | 1,248 / 38 % / −0.095 | −0.030 (−0.034 / −0.026) |
| bear arms at 30 %, gives back 40 %, floor 10 % | — | 1,248 / 44 % / −0.092 | −0.029 |
| bear arms at 40 %, gives back 25 % | — | 1,248 / 42 % / −0.088 | −0.028 |
| bull stop 80 % | 3,294 / 56 % / +0.037 | — | +0.001 |
| both | 3,294 / 56 % / +0.037 | 1,248 / 42 % / −0.088 | +0.003 (+0.011 / −0.005) |

- **A wider stop helps bullish spreads** in both halves (stop 50 → 60 →
  70 → 80 → none: +0.014 → +0.023 → +0.042 → +0.068 → +0.106 on the
  filtered set of §4, 1-year vol lookback). Bullish pullbacks in uptrends usually resume.
  80 % keeps a stop for gaps and trend breaks the model cannot see.
- **Holding bearish spreads longer hurts** (unfiltered: stop 70 −0.117,
  no stop −0.173 vs −0.094 at 50).
- **Waiting rules did not beat a plain stop**: "sell on the bounce after
  −30 %" −0.120, "defer the stop while ≥ 15 days left" −0.096 (bear 2 %).
- **Locking profit earlier raises the win rate but not the money**
  once the bad bear entries are filtered out (§4): arm 30 % / give 40 %
  wins 53 % but earns +0.030; arm 40 % / give 25 % earns +0.050–0.056.
  A floor of 10 % never binds (the lock is already ≥ 40 % × 75 %).
- A fixed +30 % target wins most often (51–56 %) and earns the least.

## 3. Width — a selection effect, not an edge

Earlier studies found 1 %-wide bear spreads lost about half as much as
2 %. On the **same entries** (where both widths pass reward/risk ≥ 1):

| | 1 % | 2 % |
|---|---|---|
| bear (632) | −0.054 | **−0.042** |
| bull (3,294) | −0.000 | −0.005 |

The narrow spread "won" because reward/risk ≥ 1 rejects more 1 %-wide
bear spreads, and the rejected half was the worst (−0.145 vs −0.041).
**Not built.** Fit to budget already narrows widths when the cap needs it.

## 4. Entry filters

Bear entries by condition (2 % wide, unfiltered; exits of the first plan — bull stop 70, bear arm 30 / give 40):

| Condition | n | win | avg | 1st | 2nd |
|---|---|---|---|---|---|
| all | 1,248 | 44 % | −0.092 | −0.109 | −0.077 |
| ticker below SMA200 | 701 | 47 % | −0.056 | −0.087 | −0.032 |
| ticker above SMA200 | 547 | 41 % | −0.138 | −0.134 | −0.142 |
| RV20 low tercile (< 15 %) | 416 | 49 % | −0.027 | −0.027 | −0.026 |
| RV20 high tercile (≥ 20 %) | 416 | 42 % | −0.131 | −0.177 | −0.093 |
| SPY below SMA50 | 590 | 41 % | −0.120 | −0.169 | −0.073 |

Bull: above SMA200 +0.022 vs below −0.040; RV low +0.046 vs high −0.019.

Volatility regime = today's RV20 percentile among the last 200 sessions'
(every 5th) — what the regime classifier already reports as `iv_rank`.

| Variant (all entries) | n | win | avg | 1st | 2nd |
|---|---|---|---|---|---|
| live exits, no filter | 4,542 | 43 % | −0.030 | −0.034 | −0.026 |
| live exits + trend200 | 3,652 | 45 % | −0.013 | −0.023 | −0.002 |
| live exits + trend200 + vol regime | 2,871 | 46 % | +0.013 | −0.007 | +0.035 |
| **plan exits + trend200 + vol regime (built)** | **2,871** | **56 %** | **+0.063** | **+0.054** | **+0.074** |
| … bull | 2,587 | 57 % | +0.065 | +0.055 | +0.076 |
| … bear | 284 | 52 % | +0.050 | +0.040 | +0.058 |
| … bear vol cap 33 instead of 50 | 192 | 54 % | +0.097 | +0.103 | +0.093 |
| … bear vol cap 67 | 386 | 49 % | +0.003 | +0.018 | −0.009 |
| … no bull vol cap | 2,951 | 56 % | +0.045 | +0.059 | +0.030 |
| … + bear only when SPY < SMA50 | 83 | 58 % | +0.132 | +0.305 | +0.029 |

Caps chosen from a smooth curve, not its best point: bear 50 (33 is
higher but keeps a third fewer trades), bull 80. The SPY-below-SMA50 gate
is unstable between halves and was **not built**. A 1-year lookback
scored slightly higher (+0.067) but needs more history than the agent
fetches (~220 bars); the 200-bar version matches the live `iv_rank`.

By ticker (1-year-lookback run): 14 of 16 positive with the plan vs 5
live; win rate higher on every ticker; IWM (−0.042) and GDX (−0.015) stay
negative. By year: bullish win rate higher in all 18 years; bad years
stay bad (2015 −0.417, 2022 −0.247, 2026 −0.161 bull).

## 5. Bugs found on the way

- **Pre-open marks**: the agent runs from 09:25 ET (session buffer);
  excursion lows were recorded there. The trail could arm on a fake
  pre-open peak and close a winner at 09:45 on "giveback". Fixed: the
  trail and excursion skip cycles before `no_exit_before_et`, and a
  debit mark outside [−debit, max profit] (5 % + $5 slack) is ignored —
  the monitor holds instead of stopping.
- **Immediate exits before the bell**: `HARD_STOP`, `STRIKE_PROXIMITY`,
  `DTE_SAFETY`, `EARNINGS_EXIT` bypassed the 09:45 window, so an
  earnings or DTE close could go out at 09:25 on pre-open quotes.
  Fixed: nothing closes before 09:30; the calendar facts (DTE, earnings)
  wait for 09:45; the mark emergencies stay immediate after the bell.
- **Calendar far leg on a holiday**: near + 28 days landed on 2026-12-25
  and 2027-01-01 (weeklies expire Thursday) — 1,322 empty-chain warnings
  in a week and DIA fell back to a 21-day gap. Fixed: snapped to the
  week's last trading day.
- Noise: the Schwab chain diagnostic dump logged at WARNING on every
  cycle (the agent is a fresh process) — now DEBUG. `MAX_RISK_PCT` in
  `.env` (0.05) differs from the preset (0.02); the preset wins, the
  warning repeats every cycle — set `.env` to 0.02 to silence it.

## 6. What this means now

On 2026-11-06 most watchlist tickers had a realized-vol rank of 92–100.
With the filters live, only a GLD put debit qualified; new debit
spreads will be rare until volatility settles. That is the intended
effect — the simulation's worst bucket is exactly bear debits bought in
high volatility (−0.13 per $).

---

*Run 2026-11-06 on the Pi. Re-run with ~20 more closed paper trades and
compare the excursion rows (`max_pl` at mid vs `max_natural_pl`).*

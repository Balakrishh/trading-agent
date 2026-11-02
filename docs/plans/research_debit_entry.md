# Cheaper debit structures and trend confirmation — research note (2026-11-01)

Script: `scripts/research/debit_entry_study.py` (daily closes from yfinance,
2008 → 2026-10-30, 16 watchlist ETFs). It answers two questions raised by
the week of 2026-10-26 (`docs/plans/journal_learnings.md`):

1. **Fit to budget.** Under CAUTION the per-trade cap halves to ~1 % of
   equity, and 1,140 valid plans that week were rejected for size (SPY
   calendar $669, QQQ put debit $586, DIA call debit $505, GLD put debit
   $400–465). Can a cheaper debit vertical keep the 0.50-delta spread's
   edge?
2. **Trend confirmation.** Four of the seven debit-spread losers in October
   were bearish TLT / XLU entries while price sat above its 20-day average
   and rising. Would requiring price on the trade's side of the 20-day SMA,
   with 10-day momentum agreeing, beat the live pullback filter?

## Model

Black–Scholes with volatility = max(20-day realized, 8 %) × 1.10 (implied
over realized — skill 68 measured 1.1–1.3 on these ETFs). An entry every 5th
session per ticker when the trend agrees (bear: close < SMA50 and SMA50
falling over 10 sessions; bull mirror). 30-day vertical, long strike at the
target |Δ|, short strike `width` × spot further out. Entry at model value ×
1.03, skipped when reward/risk < 1.0 (`debit_min_reward_risk`). Marked daily;
exit at the agent's rules (stop at −50 % of the debit, target at +50 %, else
expiry) at value × 0.97. P&L is per $1 of debit. The comparison is between
variants — the absolute level depends on the 1.10 implied markup and the
fill haircuts.

## 1. Cheaper structures

All entries in a trend (no filter), P&L per $1 risked; "1st / 2nd" are the
two halves of the sample.

| Direction | Variant | n | Win | Avg | 1st | 2nd | Debit, % of spot |
|---|---|---|---|---|---|---|---|
| bear | 0.50Δ, 2 % wide | 2,304 | 38 % | −0.117 | −0.119 | −0.116 | 0.91 % |
| bear | 0.50Δ, 1 % | 1,202 | 41 % | −0.087 | −0.064 | −0.106 | 0.48 % |
| bear | 0.50Δ, 0.75 % | 762 | 42 % | −0.078 | −0.060 | −0.093 | 0.37 % |
| bear | 0.50Δ, 0.5 % | 275 | 44 % | −0.066 | −0.015 | −0.106 | 0.25 % |
| bear | 0.40Δ, 2 % | 3,450 | 35 % | −0.136 | −0.146 | −0.127 | 0.76 % |
| bear | 0.35Δ, 2 % | 3,538 | 34 % | −0.142 | −0.146 | −0.138 | 0.66 % |
| bull | 0.50Δ, 2 % | 7,592 | 48 % | −0.023 | −0.035 | −0.012 | 0.81 % |
| bull | 0.50Δ, 1 % | 7,592 | 49 % | −0.016 | −0.027 | −0.005 | 0.45 % |
| bull | 0.50Δ, 0.5 % | 7,592 | 50 % | −0.011 | −0.024 | +0.001 | 0.23 % |
| bull | 0.40Δ, 2 % | 7,592 | 44 % | −0.051 | −0.067 | −0.036 | 0.62 % |
| bull | 0.35Δ, 2 % | 7,592 | 43 % | −0.063 | −0.079 | −0.048 | 0.53 % |

(Bear n shrinks with width because more narrow put spreads fail the
reward/risk floor: put skew is absent from the model, so this is
conservative.)

- **Moving the long leg out of the money is worse** in both directions and
  both halves (−0.02 to −0.04 per $ vs 0.50Δ). Not built.
- **Narrowing the width at 0.50Δ is no worse — slightly better** in every
  row and half except bear 0.5 % in the second half (n = 275). Built as
  **fit to budget**: when no grid width fits the cycle's cap, try narrower
  widths, widest that fits first.
- The model's fill haircut is a % of value; real bid/ask is in cents, so a
  narrow spread on a high-priced ETF pays proportionally more. The live
  fair-price check (natural ≤ mid × 1.05) and the reward/risk floor still
  apply to every narrower candidate. Read-only check on 2026-10-30 closing
  quotes: QQQ Dec-4 puts are on $5 strikes with 25–45 ¢ legs, so the 5-wide
  is 16 % and the 10-wide 8.5 % over mid — only the 20-wide passes. Fit to
  budget will not rescue QQQ; it can on $1-strike chains.
- Every variant loses on average: with implied over realized, buying
  premium has a negative expectation unless the direction call adds edge.
  Bear entries lose 4–5× more than bull ones. That matches the paper
  account (debit verticals 2 wins of 9 since 2026-10-08, −$1,174).

## 2. Trend confirmation

P&L per $1, 0.50Δ:

| Direction | Filter | Width | n | Win | Avg | 1st | 2nd |
|---|---|---|---|---|---|---|---|
| bear | none | 2 % | 2,304 | 38 % | −0.117 | −0.119 | −0.116 |
| bear | pullback (live) | 2 % | 1,248 | 39 % | −0.107 | −0.116 | −0.099 |
| bear | confirm | 2 % | 1,530 | 38 % | −0.117 | −0.099 | −0.131 |
| bear | pullback + confirm | 2 % | 492 | 41 % | −0.089 | −0.037 | −0.129 |
| bear | pullback (live) | 1 % | 632 | 43 % | −0.060 | −0.035 | −0.082 |
| bear | pullback + confirm | 1 % | 257 | 42 % | −0.078 | −0.023 | −0.128 |
| bull | pullback (live) | 2 % | 3,292 | 47 % | −0.031 | −0.032 | −0.031 |
| bull | pullback + confirm | 2 % | 1,172 | 46 % | −0.040 | −0.038 | −0.043 |

"confirm" = bear: close < SMA20 and 10-day return < 0 (bull mirror).

- Confirmation alone does nothing (bear −0.117 = none).
- On top of the live pullback filter it helps in the first half and hurts
  in the second (bear −0.129 vs −0.099 / −0.082), and it hurts bulls in both.
  Not robust. **Not built**; the live pullback filter (since 2026-10-27)
  stays the trend check.

## 3. Earnings exit (no study)

AMZN 255/265 call debit (opened 2026-10-06, before the earnings guard)
held through the 2026-10-29 report: −16 % gap, −$403 of $460. The entry
guard (skill 65) keeps new trades' expiries before a report, so this only
catches positions opened before a date was known or moved. Holding a
defined-risk spread through a report is a bet on the gap with no model
edge, so the rule closes such positions in the last session before it.
Built as `earnings_exit_enabled` (live on paper).

---

*Run 2026-11-01 on the Pi. Re-run after ~20 more closed debit trades to
compare with the paper results.*

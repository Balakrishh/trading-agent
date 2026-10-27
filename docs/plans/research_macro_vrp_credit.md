# Macro events, implied vs realized, credit stress — research note (2026-10-27)

Script: `scripts/research/macro_vrp_credit_study.py` (daily closes from
yfinance; event dates from `trading_agent/macro_calendar.py`). Skill 68
documents the rules built from it. All three went **live on the paper
account** on 2026-10-27 (operator's call: learn on paper, not in shadow),
with every plan journalling its macro / IV-RV context so the thresholds
can be re-tested on outcomes.

## 1. Macro event days

Event dates: FOMC decisions 2015 → 2027 from the Federal Reserve's
calendar pages (unscheduled / notation votes excluded); CPI and the jobs
report (NFP) 2025 → 2026 from FRED's public release calendar — BLS blocks
scripted requests and FRED's calendar lists nothing earlier, so the CPI /
NFP sample is ~20 events each.

Ratio of event-day to other-day moves, close-to-close (so CPI / NFP's
08:30 gap and FOMC's 14:00 move are inside the event day):

| | FOMC 2015→ (n=93) r² | CPI 2025→ median \|r\| / r² ex-Apr-2025 | NFP 2025→ median \|r\| / r² ex-Apr-2025 |
|---|---|---|---|
| SPY | 0.96 | 1.07 / 0.60 | **1.62 / 1.22** |
| QQQ | 1.10 | 1.29 / 0.69 | **1.80 / 1.63** |
| IWM | 1.14 | 1.29 / 0.95 | **1.42 / 1.46** |
| XLK | 1.04 | 1.60 / 0.73 | **1.54 / 1.92** |
| SMH | 1.11 | 1.61 / 0.80 | **1.89 / 2.00** |
| TLT | 0.90 | 1.10 / 1.07 | 1.18 / 1.20 |
| GLD | 1.26 | 0.49 / 0.61 | 1.04 / 0.77 |
| XLU | 0.86 | 0.65 / 0.76 | 0.83 / 0.79 |

April 2025 is excluded from the variance column because the tariff crash
(SPY −6 % on Apr 4, a jobs-report day) dominates a 20-event sample.

Reading: the jobs report moves equity ETFs 1.4–1.9× a normal day; CPI's
median move is a little larger but its variance is not; FOMC days are not
bigger close-to-close (the 14:00 swing mostly reverses by the close);
bonds, gold and utilities barely react. One 1.5–2× day is about
2 / (2 + 20) ≈ 9 % of a 21-session trade's variance — small — but
2 / (2 + 4) ≈ 33 % of a 5-session trade's.

**Rule:** no new short-premium equity-ETF trade of ≤ 14 DTE whose life
holds a pending NFP release (`macro_event_kinds = ("NFP",)`,
`macro_event_max_dte = 14`). CPI / FOMC can be switched on in the preset.

## 2. Implied vs realized (variance risk premium)

SPY 2007 → 2026, VIX as the 30-day implied vol, by VIX / trailing 20-day
realized vol:

| IV / RV20 | n | P(next-21 RV < IV) | P(\|21-day move\| < 1σ) | mean IV − RV21 |
|---|---|---|---|---|
| < 0.90 | 400 | 79.2 % | 81.8 % | +0.037 |
| 0.90–1.00 | 374 | 81.0 % | **77.0 %** | +0.030 |
| 1.00–1.10 | 476 | 84.0 % | 84.2 % | +0.032 |
| 1.10–1.25 | 882 | 83.6 % | 81.0 % | +0.033 |
| 1.25–1.50 | 1118 | 84.0 % | 82.7 % | +0.036 |
| 1.50–2.00 | 1257 | 83.9 % | **87.6 %** | +0.040 |
| ≥ 2.00 | 437 | 77.6 % | **88.1 %** | +0.038 |

Implied beats realized ~80 % of the time in every bucket (the premium is
persistent), but the move stays inside the priced 1σ range more often
when implied is rich relative to recent realized: 77–82 % under 1.0,
87–88 % over 1.5.

**Rule:** a short-premium trade needs the ticker's 30-day ATM IV
(yesterday's vol-surface row) ≥ 1.0 × its 20-day realized vol
(`vrp_min_ratio`). Missing IV or RV never blocks.

## 3. Credit stress

HYG / IEF (high yield vs 7–10-year Treasuries, history since 2007; FRED's
ICE HY OAS series only serves 3 years), % under its 50-day average,
dividend-adjusted closes, → SPY's next 21 sessions:

| | n | fwd 21 return | mean max drawdown | P(drawdown ≥ 5 %) |
|---|---|---|---|---|
| all days | 4698 | +1.05 % | −2.83 % | 17.5 % |
| stress < 1 % | 3725 | +1.02 % | −2.37 % | 13.8 % |
| stress ≥ 1 % | 973 | +1.19 % | −4.59 % | 31.8 % |
| **stress ≥ 2 %** | 615 | +1.47 % | **−5.22 %** | **36.9 %** |
| stress ≥ 5 % | 191 | +1.64 % | −7.47 % | 45.5 % |
| stress ≥ 2 % and SPY > SMA-50 | 56 | −1.64 % | −5.16 % | **42.9 %** |
| stress < 1 % and SPY > SMA-50 | 3112 | +0.94 % | −2.22 % | 12.1 % |

Stress predicts bigger drawdowns, not lower returns (rebounds follow) —
a volatility signal — and it carries information the trend rule lacks
(the SPY > SMA-50 rows, few independent episodes though).

**Rule:** credit stress ≥ 2 % is a CAUTION reason in the market state
(half size, no new bull puts / call debits / CSPs); CAUTION is not left
until it is back under 1 %. Live HYG has to be read from
**dividend-adjusted** closes: unadjusted, each monthly ex-date reads as
~0.4 % of stress (1.42 % vs 0.78 % on 2026-10-27).

## 4. Dealer gamma (record only)

No free history, so `gamma_exposure.py` records SPY / QQQ / IWM net GEX,
flip level and call / put walls daily at 16:05 ET from 2026-10-27. Judge
after 2–3 months: next-day range vs `regime` (long / short gamma) and
distance to the flip.

## Limits

Close-to-close data; overlapping 21-session windows (not independent);
VIX history is SPY only (other ETFs use the same floor on their own chain
IV live); CPI / NFP sample is short (2025 →) and includes a tariff shock.

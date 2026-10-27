# Macro Events, Implied vs Realized, Credit Stress, Dealer Gamma

> **One-line summary:** market-wide inputs for the ETF book — no short-premium trade of ≤ 14 DTE across a pending jobs report, none when implied vol is under realized, credit stress as a CAUTION reason, and a daily dealer-gamma record.
> **Source of truth:** [`trading_agent/macro_vol_gates.py`](../../trading_agent/macro_vol_gates.py), [`trading_agent/macro_calendar.py`](../../trading_agent/macro_calendar.py), [`trading_agent/market_state.py:97-104, 238-245, 395-415`](../../trading_agent/market_state.py), [`trading_agent/gamma_exposure.py`](../../trading_agent/gamma_exposure.py), [`trading_agent/agent.py:2375-2445`](../../trading_agent/agent.py)
> **Phase:** 2  •  **Group:** risk / data_quality
> **Depends on:** `58_market_state_playbook.md`, `66_calibrated_probability_model.md` (vol surface), `65_earnings_expiry_and_trend_filters.md`, `47_schwab_data_api.md`
> **Consumed by:** `agent.py` — entry gate chain and journal fields; `market_state.py` — CAUTION rule; data server `/macro-events`, `/credit-stress`, `/vol`, `/gamma`, `/market-context`

---

## 1. Theory & Objective

Sixteen of the eighteen watchlist tickers are ETFs, so company research (skill 67) says little about them; what moves them is the market. Before 2026-10-27 the agent read SPY's trend, breadth, VIX and VIX term structure (skill 58), the regime and an IV-rank playbook, and a realized-vol POP in shadow (skill 66). This skill adds the three market inputs a backtest supported (`docs/plans/research_macro_vrp_credit.md`) and records a fourth:

1. **Macro events.** Jobs-report days move equity ETFs 1.4–1.9×; a 5-session short-premium trade spanning one carries ~⅓ of its variance on that day. CPI and FOMC days were not bigger close-to-close, so they are off by default.
2. **Implied vs realized.** Short premium pays when implied vol exceeds what happens. With IV under trailing realized vol, SPY's 21-day move stayed inside the priced 1σ 77–82 % of the time vs 87–88 % with IV ≥ 1.5× realized.
3. **Credit stress.** HYG/IEF ≥ 2 % under its 50-day average: SPY's chance of a ≥ 5 % drawdown in 21 sessions 37 % vs 14 % — and 43 % even when SPY was above its 50-day.
4. **Dealer gamma**, record-only: no free history, so the agent collects its own.

The operator chose to run 1–3 **live on the paper account** from day one instead of a shadow period; every journalled plan carries the context fields so the thresholds can be re-tested on outcomes.

## 2. Mathematical Formula

```text
Short premium = Bull Put, Bear Call, Iron Condor, Iron Butterfly, Mean Reversion, Bounce Bull Put
(debit spreads and calendars are long premium — never gated here)

Macro guard   block ⇔ short premium ∧ ticker ∉ NON_EQUITY_TICKERS
                      ∧ DTE ≤ macro_event_max_dte (14)
                      ∧ ∃ release of macro_event_kinds (NFP) in [now, expiration]
                        not yet released (today's 08:30 release is past by the 09:45 entry window)

VRP floor     ratio = IV_ATM,30d / RV_20d        IV: latest vol_surface row ≤ 4 days old
              block ⇔ short premium ∧ ratio < vrp_min_ratio (1.0); ratio None → no block

Credit stress s = (1 − R_t / mean(R_{t−49..t})) × 100,   R = HYG / IEF (dividend-adjusted)
              CAUTION reason ⇔ s ≥ 2.0;  hysteresis: CAUTION held while s ≥ 1.0

Dealer gamma  GEX_c = Γ × OI × 100 × S² × 0.01      (+ calls, − puts; $ delta per 1 % move)
              net = Σ GEX over expiries ≤ 35 days;  flip = S′ where Σ GEX(S′) changes sign
              (Γ recomputed with Black–Scholes at S′, each contract's own IV)
```

## 3. Reference Python Implementation

```python
# trading_agent/macro_vol_gates.py:56-83
def pending_events(today, expiration, now_et, kinds=mc.KINDS, events=None):
    out = []
    for e in mc.events_between(today, expiration, events, kinds):
        if e.day == today and now_et.time() >= e.release_et:
            continue
        out.append(e)
    return out


def macro_event_failure(ticker, strategy, expiration, now_et, kinds, max_dte, events=None):
    if not is_short_premium(strategy) or ticker in NON_EQUITY_TICKERS or max_dte <= 0:
        return None
    today = now_et.date()
    dte = (expiration - today).days
    if dte > max_dte:
        return None
    hits = pending_events(today, expiration, now_et, kinds, events)
    if not hits:
        return None
    return (f"macro_event_guard ({', '.join(e.label() for e in hits)} inside a {dte}-day "
            f"{strategy}; short premium ≤ {max_dte} DTE waits until after the release)")
```

```python
# trading_agent/macro_vol_gates.py:139-147
def vrp_failure(strategy, iv, rv, min_ratio):
    if not is_short_premium(strategy) or min_ratio <= 0:
        return None
    ratio = iv_rv_ratio(iv, rv)
    if ratio is None or ratio >= min_ratio:
        return None
    return (f"vrp_below_min (30d ATM IV {iv:.1%} / 20d realized {rv:.1%} = {ratio:.2f} "
            f"< {min_ratio:g}: premium too cheap to sell)")
```

```python
# trading_agent/agent.py:2408-2427 — in the entry gate chain after _direction_risk_block
def _macro_vol_block(self, ticker, plan):
    from trading_agent import macro_vol_gates as mvg
    if not mvg.is_short_premium(plan.strategy_name):
        return None
    now = self._now_et()
    exp = self._plan_expiry(plan)                    # latest leg expiry (calendar far leg)
    if exp is not None and getattr(self.preset, "macro_event_guard_enabled", False):
        reason = mvg.macro_event_failure(ticker, plan.strategy_name, exp, now,
                                         tuple(self.preset.macro_event_kinds), int(self.preset.macro_event_max_dte))
        if reason:
            return reason
    min_ratio = float(getattr(self.preset, "vrp_min_ratio", 0.0) or 0.0)
    if min_ratio > 0:
        return mvg.vrp_failure(plan.strategy_name, mvg.latest_atm_iv(ticker, now.date()),
                               self._realized_vol_20d(ticker), min_ratio)
    return None
# _macro_vol_fields(plan) → raw: macro_next_event, macro_events_in_trade, iv_atm_30d,
#                                iv_rv_ratio, vrp_available — on every plan with legs
```

```python
# trading_agent/market_state.py — credit stress (input + rule)
caution_credit_stress: float = 2.0          # MarketStateConfig
caution_exit_credit_stress: float = 1.0

def yfinance_adjusted_history(symbol):      # dividend-adjusted; HYG pays monthly
    return yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=True)

def credit_stress_pct(ratio, n=CREDIT_SMA):
    if len(ratio) < n or not ratio[-1]:
        return None
    avg = sum(ratio[-n:]) / n
    return round((1 - ratio[-1] / avg) * 100, 3) if avg > 0 else None

# classify_market_state: CAUTION list
if cs is not None and cs >= cfg.caution_credit_stress:
    caution.append(f"credit stress: HYG/IEF {cs:.1f}% under its 50-day (≥ {cfg.caution_credit_stress:g}%)")
```

```python
# trading_agent/gamma_exposure.py:61-71
def contract_gex(c, spot):
    g, oi = float(c.get("gamma") or 0), float(c.get("open_interest") or 0)
    if g <= 0 or oi <= 0 or spot <= 0:
        return None
    sign = 1.0 if str(c.get("type", "")).lower() == "call" else -1.0
    return sign * g * oi * 100 * spot * spot * 0.01
```

Preset fields (`strategy_presets.py`, after the earnings guard): `macro_event_guard_enabled=True`, `macro_event_kinds=("NFP",)`, `macro_event_max_dte=14`, `vrp_min_ratio=1.0`; summary-line token `Macro guard NFP ≤14d • IV/RV ≥1`; Streamlit Strategy Profile → "Macro events & implied vs realized".

Calendar: `macro_calendar.py` holds FOMC 2015 → 2027 and CPI / NFP 2025 → 2026; `--refresh` (run weekly by the gamma timer) merges new dates from the Fed and FRED calendar pages into `trade_journal/macro_events.json`.

Timer: `trading-agent-gamma.timer`, 16:05 ET weekdays → `trade_journal/gamma_exposure.jsonl`.

## 4. Edge Cases / Guardrails

- **Missing IV or RV never blocks.** No vol-surface row in 4 days (timer failed, new ticker) or no price history → `iv_rv_ratio` None, `vrp_available` False, trade allowed. A 0.0 IV is treated as missing, never as "cheap".
- **IV is a day old.** The floor reads yesterday's 15:50 ET vol-surface 30-day ATM IV, not the live chain — no extra Schwab calls per plan; IV/RV moves slowly. A Friday row still serves Monday (4-day limit covers a holiday).
- **Release already out.** A CPI / NFP at 08:30 is in the price by the 09:45 entry window, so a same-day release does not block; an FOMC day blocks until 14:00 only if FOMC is enabled.
- **Calendars and far legs.** `_plan_expiry` uses the latest leg expiry, so a calendar's far leg would count — but calendars are long premium and exempt anyway.
- **Bonds, metals, commodities exempt** from the macro guard (`NON_EQUITY_TICKERS`): TLT / GLD / XLU barely move on releases.
- **Calendar runs out.** CPI / NFP dates are only published about a year ahead. `coverage_warnings` (logged by the gamma run, shown by `/macro-events`) names a kind whose last date is < 45 days away; `refresh()` keeps the stored dates when a source fails. FRED lists nothing before 2025 (BLS blocks scripts), so the CPI / NFP history is short.
- **Unknown kinds in the preset** (`"GDP"`) are dropped with a warning by `_coerce_overrides`.
- **Credit stress must use adjusted closes.** The agent's own history is unadjusted (to match live quotes); HYG's monthly ex-dividend drop read as 0.4 % of stress. `compute_inputs(..., fetch_adjusted=yfinance_adjusted_history)`; the data server uses the same function. A failed fetch leaves `credit_stress` None — skipped, never 0.
- **Credit hysteresis.** Entering CAUTION at 2 % and leaving under 1 % keeps a hovering ratio from flipping the state every cycle.
- **Gamma needs open interest.** `/chain` contracts carry `open_interest` from 2026-10-27 (`market_data_schwab._normalize_contract`, None when Schwab omits it). A data server still running older code returns contracts without it; the recorder logs "data server needs a restart" and records nothing instead of a zero.
- **Naive GEX convention.** Assumes dealers are long calls / short puts; wrong for a market where customers sell calls heavily. It is a record for testing, not a trading input.
- **Flip may not exist** within ±8 % of spot (all-call or all-put books) → `flip` None.

## 5. Cross-References

- `58_market_state_playbook.md` — the state machine credit stress now feeds (CAUTION rule + hysteresis).
- `66_calibrated_probability_model.md` — `vol_surface.py` rows that supply the 30-day ATM IV.
- `65_earnings_expiry_and_trend_filters.md` — the single-stock earnings guard this mirrors for ETFs.
- `47_schwab_data_api.md` — `/macro-events`, `/credit-stress`, `/vol/{ticker}`, `/gamma/{ticker}`, `/market-context`; `/chain` `open_interest`.
- `48_claude_code_mcp_surface.md` — `get_macro_events`, `get_credit_stress`, `get_vol`, `get_gamma`, `get_market_context`.
- `docs/plans/research_macro_vrp_credit.md` — the backtest tables.

---

*Last verified against repo HEAD on 2026-10-27.*

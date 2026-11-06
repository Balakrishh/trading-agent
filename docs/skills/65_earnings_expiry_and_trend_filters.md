# Earnings-Aware Expiries and Trend Entry Filters

> **One-line summary:** single-stock trades expire before the next earnings report (the planner picks the latest weekly that fits and only waits out the report when none is far enough away), and directional entries are checked for relative strength vs SPY and for a pullback to the 20/50-day SMAs — recorded in shadow by default.
> **Source of truth:** [`trading_agent/strategy.py`](../../trading_agent/strategy.py), [`trading_agent/chain_scanner.py`](../../trading_agent/chain_scanner.py), [`trading_agent/earnings_calendar.py`](../../trading_agent/earnings_calendar.py), [`trading_agent/entry_filters.py`](../../trading_agent/entry_filters.py), [`trading_agent/agent.py`](../../trading_agent/agent.py), [`trading_agent/calendar_utils.py`](../../trading_agent/calendar_utils.py)
> **Phase:** 3  •  **Group:** risk / strategy
> **Depends on:** `13_strategy_presets.md`, `58_market_state_playbook.md`, `59_debit_spreads_calendars.md`, `61_entry_confirmation.md`
> **Consumed by:** `StrategyPlanner.plan`, `ChainScanner.scan`, `TradingAgent._process_ticker` gate chain, journal rows (`trend_filters`, `earnings_date`)

---

## 1. Theory & Objective

**Earnings.** A report moves a single stock far more than its normal daily range, and option prices collapse right after it. Holding a defined-risk spread through one is a coin flip the strategy never priced. Blocking a ticker for the whole quarter would remove it entirely (reports come every ~90 days and a 30-DTE trade spans one about a third of the time), so the rule is about **expiry choice**, not the ticker: pick the latest weekly expiry that ends at least `earnings_buffer_days` before the report. Only when no such expiry is `earnings_min_dte` days or more away — about a week each quarter — does the ticker wait until after the report. ETFs have no earnings and are never affected. On 2026-10-15 AMZN (report 10-29) and META (10-21) were on the watchlist with nothing stopping a 30-DTE trade across the report; the open AMZN 255/265 call debit (Nov 6) holds through one.

**Relative strength.** The old `leadership_zscore` compares the last 5-minute bar with an anchor ETF — intraday noise. Traders mean multi-week outperformance: the ticker's 20- and 60-day returns minus SPY's, ranked across the watchlist. Bullish trades go to leaders, bearish trades to laggards.

**200-day trend and volatility regime for debit verticals (2026-11-06).** A 2008–2026 simulation of the live exits on 16 ETFs (`docs/plans/research_debit_exits.md`) found that the paper account's losing pattern — bearish debits bought on pullbacks in long-term uptrends, during volatile markets (TLT, XLU) — is the simulation's worst bucket too: bear debits above the 200-day SMA lost −0.138 per $ risked vs −0.056 below it, and bear debits in the top volatility tercile −0.131 vs −0.027 in the bottom. Call debits mirror it (above the 200-day +0.022 vs −0.040). Two checks for call / put debit spreads only — credit spreads sell the volatility and are not filtered.

**Pullback.** The regime classifier looks at price vs the 200-day SMA and the 50-day *slope* only. A bullish entry after a run-up is the worst-priced entry in an uptrend: the QQQ call debit opened 2026-10-05 after a rally was stopped on the first dip (−$420). The filter asks for price above the 50-day SMA (the trend's support holds) and within `pullback_max_atr` × ATR(14) of the 20-day SMA (not extended); bearish is the mirror.

## 2. Mathematical Formula

```text
cap        = next_earnings_date − earnings_buffer_days            (single stocks only)
expiry     = chosen expiry               if expiry ≤ cap
           = latest weekly ≤ cap         if its DTE ≥ earnings_min_dte
           = no trade until after the report otherwise
every leg (calendar far leg included) must expire ≤ cap

RS_score   = mean over n ∈ {20, 60} of (close_t/close_(t−n) − 1) − (SPY_t/SPY_(t−n) − 1)
RS_pct     = percentile rank of RS_score across the watchlist ∈ [0, 1]
bullish ok ⇔ RS_pct ≥ rs_min_percentile          bearish ok ⇔ RS_pct ≤ 1 − rs_min_percentile

stretch    = (price − SMA20) / ATR14   (bullish)      (SMA20 − price) / ATR14   (bearish)
bullish ok ⇔ price > SMA50 and stretch ≤ pullback_max_atr
bearish ok ⇔ price < SMA50 and stretch ≤ pullback_max_atr
```

Directional = Bull Put, Call Debit (bullish); Bear Call, Put Debit (bearish). Condors, butterflies, calendars, the mean-reversion spread and the bounce bull put (counter-trend by design, its own levels) are not filtered.

## 3. Reference Python Implementation

```python
# trading_agent/strategy.py:308-344
    def _set_earnings_cap(self, ticker: str) -> None:
        """Latest allowed expiration = next earnings − buffer days; None
        for ETFs, unknown dates, or when the guard is off."""
        self._expiry_cap = self._earnings_date = None
        self.earnings_ticker = ticker
        if (self.earnings_lookup is None
                or not getattr(self.preset, "earnings_guard_enabled", True)):
            return
        try:
            d = self.earnings_lookup(ticker)
        except Exception as exc:  # noqa: skill-34-exempt — unknown earnings date → no cap, logged
            logger.warning("[%s] earnings lookup failed: %s", ticker, exc)
            return
        if d is None:
            return
        buffer = int(getattr(self.preset, "earnings_buffer_days", 1))
        self._earnings_date = d
        self._expiry_cap = d - timedelta(days=buffer)

    def _cap_leaves_no_expiry(self) -> bool:
        return latest_weekly_on_or_before(
            datetime.now().date(), self._expiry_cap, self._earnings_min_dte()) is None

    def _capped(self, expiration: date) -> date:
        """``expiration`` if it ends before earnings, else the latest weekly
        that does (≥ earnings_min_dte out); unchanged when none fits —
        ``_enforce_earnings_cap`` then rejects the plan."""
        if self._expiry_cap is None or expiration <= self._expiry_cap:
            return expiration
        alt = latest_weekly_on_or_before(datetime.now().date(), self._expiry_cap,
                                         self._earnings_min_dte())
        if alt is not None:
            logger.info("Expiration %s spans earnings on %s — using %s",
                        expiration, self._earnings_date, alt)
            return alt
        return expiration

```

`plan()` calls `_set_earnings_cap`, hands the cap to the scanner, returns a no-trade plan when `_cap_leaves_no_expiry()`, and runs every finished plan through `_enforce_earnings_cap`, which rejects one whose latest leg (from the OCC symbol) or `expiration` is past the cap. `_expiration_near` (debit, calendar near leg) and `_pick_expiration` (verticals, condors, butterflies, mean reversion, bounce) pass their pick through `_capped`; `_plan_calendar` skips far legs past the cap.

```python
# trading_agent/chain_scanner.py:986-993
            dte_lo = max(1, int(target_dte) - self.dte_window_days)
            dte_hi = int(target_dte) + self.dte_window_days
            if cap is not None:
                # Skill 65: never past the earnings cap; a grid point whose
                # whole window lies beyond it is dropped.
                dte_hi = min(dte_hi, (cap - today).days)
                if dte_hi < dte_lo:
                    continue
```

```python
# trading_agent/entry_filters.py:126-161
def pullback_verdict(direction: str, price: float, closes: Sequence[float],
                     highs: Sequence[float], lows: Sequence[float],
                     max_atr: float, mode: str) -> FilterVerdict:
    s20, s50, a = sma(closes, 20), sma(closes, 50), atr(highs, lows, closes)
    m = {"sma20": None if s20 is None else round(s20, 2),
         "sma50": None if s50 is None else round(s50, 2),
         "atr14": None if a is None else round(a, 2)}
    if s20 is None or s50 is None or not a or not price:
        return FilterVerdict("pullback", mode, None, "not enough daily bars", m)
    stretch = (price - s20) / a if direction == "bull" else (s20 - price) / a
    m["stretch_atr"] = round(stretch, 2)
    if direction == "bull":
        if price <= s50:
            return FilterVerdict("pullback", mode, False,
                                 f"price {price:.2f} below 50-day SMA {s50:.2f} — support broken", m)
        ok = stretch <= max_atr
        side = "above"
    else:
        if price >= s50:
            return FilterVerdict("pullback", mode, False,
                                 f"price {price:.2f} above 50-day SMA {s50:.2f} — resistance broken", m)
        ok = stretch <= max_atr
        side = "below"
    if stretch < 0:
        other = "below" if direction == "bull" else "above"
        reason = f"price {-stretch:.1f} ATR {other} the 20-day SMA {s20:.2f} — pullback"
    else:
        reason = (f"price {stretch:.1f} ATR {side} the 20-day SMA {s20:.2f}"
                  + ("" if ok else f" — extended (max {max_atr:g})"))
    return FilterVerdict("pullback", mode, ok, reason, m)


@dataclass(frozen=True)
class TrendEvaluation:
    direction: str
    verdicts: List[FilterVerdict]
```

```python
# trading_agent/entry_filters.py:163-175
    @property
    def block(self) -> Optional[str]:
        """Reason for the first failed check in live mode, else None."""
        for v in self.verdicts:
            if v.mode == "live" and v.ok is False:
                return f"trend: {v.name} — {v.reason}"
        return None

    def to_journal_dict(self) -> Dict:
        return {"direction": self.direction,
                "blocked": self.block is not None,
                "checks": [asdict(v) for v in self.verdicts]}

```

```python
# trading_agent/agent.py:2756-2761
            block = (market_state.gate_failure(
                         self._market_state, plan.strategy_name,
                         [l.option_type for l in plan.legs if l.action == "sell"])
                     or self._ladder_block(ticker, plan.expiration)
                     or self._trend_filter_block(ticker, plan, analysis)
                     or self._entry_confirmation_block(ticker, plan))
```

Inputs: `_begin_entry_gates` computes `_relative_strength_map(watchlist)` (SPY, the benchmark, excluded from the ranking) once per cycle from the daily history the cycle already pre-fetched; `_trend_filter_block` adds the ticker's daily bars and stores `TrendEvaluation.to_journal_dict()` for `_log_signal`, which writes it as `raw_signal.trend_filters` together with `earnings_date` / `earnings_expiry_cap` (`StrategyPlanner.earnings_fields`). Earnings dates come from `earnings_calendar.next_earnings_date` — yfinance, cached in `trade_journal/earnings_dates.json` for 12 h (the agent is a fresh process each cycle).

Preset (`PresetConfig`, dashboard "Earnings & trend entry filters"): `earnings_guard_enabled` (True), `earnings_buffer_days` (1), `earnings_min_dte` (7), `rs_filter_mode` / `pullback_filter_mode` (`"shadow"` | `"live"` | `"off"`, default shadow), `rs_min_percentile` (0.50), `pullback_max_atr` (1.0). Summary: `Earnings exp ≤ report−1d (≥7d) • RS shadow ≥p50% • Pullback shadow ≤1ATR`.

### Earnings exit for open positions (2026-11-01)

```python
# trading_agent/position_monitor.py — in _check_exit, after the Wheel branch,
# before every P&L rule
earnings = self._check_earnings_exit(spread)
if earnings is not None:
    return earnings

def _check_earnings_exit(self, spread, today=None):
    if not self.earnings_exit_enabled or self.earnings_lookup is None or self.earnings_exit_sessions <= 0:
        return None
    report = self.earnings_lookup(spread.underlying)        # exceptions → None, logged
    if report is None:
        return None
    last = self._last_expiry(spread)                         # latest leg (calendar far leg)
    if last is None or last < report:
        return None
    n = sessions_before(today or _today_et(), report)        # NYSE sessions in [today, report)
    if n == 0 or n > self.earnings_exit_sessions:
        return None
    return (ExitSignal.EARNINGS_EXIT, f"Earnings {report}: position runs to {last} — closing {n} session(s) before the report")
```

### 200-day trend and volatility regime (2026-11-06)

```python
# trading_agent/entry_filters.py — evaluate(..., sma200=analysis.sma_200, iv_rank=analysis.iv_rank)
if strategy_name in (CALL_DEBIT_STRATEGY, PUT_DEBIT_STRATEGY):
    trend200_verdict(direction, price, sma200, preset.trend200_filter_mode)
        # bull: price > SMA200, bear: price < SMA200; no SMA → ok None
    vol_regime_verdict(direction, iv_rank, cap, preset.vol_regime_filter_mode)
        # iv_rank ≤ vol_regime_bull_max_rank (80) / vol_regime_bear_max_rank (50); none → ok None
```

`iv_rank` is the regime classifier's realized-vol percentile (`RegimeClassifier._compute_iv_rank`: today's 20-day RV against every 5th 20-day window of the fetched ~220 bars, 0–100) — the same number the HighIV block (> 95) uses, so there is one source. `sma_200` is the regime's. Verdicts land in `raw_signal.trend_filters.checks` with names `trend200` / `vol_regime`; live blocks read `risk: trend: trend200 — price 735.29 above the 200-day SMA 730.38 — bearish debit needs below` / `risk: trend: vol_regime — realized-vol rank 92 — bearish debit needs ≤ 50 (calmer market)`. Preset: `trend200_filter_mode`, `vol_regime_filter_mode` (default "shadow"; "live" on the paper preset from 2026-11-06), `vol_regime_bull_max_rank` 80, `vol_regime_bear_max_rank` 50; dashboard "200-day trend filter (debit spreads)", "Volatility-regime filter (debit spreads)" and the two caps; summary ` • Trend200 live • VolRegime live bull≤80/bear≤50`.

`ExitSignal.EARNINGS_EXIT` is in `IMMEDIATE_EXIT_SIGNALS` (a calendar fact — no mark to debounce; since 2026-11-06 it is also in `CALENDAR_EXIT_SIGNALS`, so it waits for the 09:45 exit window — skill 30 §4). `TradingAgent.__init__` passes `earnings_exit_enabled` / `earnings_exit_sessions` and sets `position_monitor.earnings_lookup = next_earnings_date`. Preset: `earnings_exit_enabled` (True), `earnings_exit_sessions` (1); dashboard "Close before earnings"; summary ` • Earnings exit 1 sess before`. `calendar_utils.sessions_before(today, day)`.

## 4. Edge Cases / Guardrails

- **Unknown earnings date** (yfinance down, rate-limited, no date listed) — no cap; the failed lookup is cached for 12 h so a failing source is not hammered. ETFs (`_NON_ISSUER_TICKERS`) are never looked up.
- **Report already passed** — a cached date before today is refetched; a fetched date in the past counts as unknown.
- **Weekly after the cap** — `latest_weekly_on_or_before` walks back week by week to the last trading day ≤ cap (Thanksgiving: Thursday cap → the previous Friday).
- **No expiry fits** — strategy name `""` (a deliberate no-trade, like `wait_for_stabilization`) with reason "Earnings on D (in N d): … no new X trade until after it"; no chain is fetched.
- **Calendar** — the far leg is ~28 days past the near leg, so a calendar is skipped whenever the report falls between; reason "no far leg ends before earnings on D".
- **Scanner** — a grid point whose whole DTE window lies past the cap is dropped; the rest are clamped to the cap. A cap shorter than the grid's shortest window leaves no expirations (the plan-level no-trade check normally fires first).
- **Open positions spanning a report (2026-11-01).** The entry guard applies to new plans; an open position whose latest leg expires on or after the next report is closed by `EARNINGS_EXIT` during the last `earnings_exit_sessions` sessions before it (AMZN 255/265, opened 10-06 before the guard, held through 10-29: −16 % gap, −$403 of $460). Report time is unknown (BMO / AMC), so the report day itself never triggers — an after-close report on day D is exited on D − 1, a day early. The exit runs before the stop / target rules, so a position at its stop closes as `earnings_exit`; the stop cool-down (skill 37) does not count it. ETFs (no date), unknown dates, a failed lookup and weekends (0 sessions) never exit. Wheel legs are exempt — assignment is part of the plan.
- **Trend filters, missing data** — no SPY history, no ticker history, or fewer than 51 daily bars → verdict `ok: null`, never blocks.
- **Shadow vs live** — shadow writes the verdicts (`blocked: false`) and lets the entry continue; live blocks with `risk: trend: <filter> — <reason>`. The filters run after the market-state and ladder gates and before entry confirmation, so a filtered plan does not advance the confirmation streak.
- **One ticker / ties** — a single ranked ticker is percentile 0.5; equal scores share the mean rank.
- **SPY itself (changed 2026-11-02)** — its score vs itself is 0 by construction, so its percentile only said how many watchlist names beat the index that day, not anything about SPY. On 2026-11-02 that blocked a valid SPY put debit in all 233 cycles ("percentile 59% (strong vs SPY, +0.0%)"). `_relative_strength_map` now leaves SPY out of the ranking and marks it `{"score": None, "percentile": None, "benchmark": True}`; `rs_verdict(..., benchmark=True)` returns `ok: null` with reason "SPY is the benchmark — not applicable" (never blocks). The pullback filter still applies to SPY.

- **200-day trend / volatility regime (2026-11-06).** Debit verticals only: Bull Put / Bear Call credit spreads, condors, butterflies, calendars and the bounce are not filtered by either. Missing data never blocks: no 200-day SMA (fewer than 200 bars, NaN or 0) → `ok: null`; `iv_rank` missing → `ok: null` (a real 0 passes). The caps sit on a smooth part of the curve, not its best point: bear cap 33 scored +0.097 per $ but kept a third fewer trades; 67 fell to +0.003. A 1-year RV lookback scored slightly higher than the classifier's ~200 bars but needs more history than the agent fetches. With the filters live on 2026-11-06 closes, most tickers had a vol rank of 92–100 and only a GLD put debit qualified: in a volatile market the agent opens few or no debit spreads — the intended effect (bear debits in high vol lost −0.13 per $). Rejected ideas: the SPY-below-SMA50 gate for bear trades (unstable between halves: +0.305 / +0.029) and narrower widths (a selection effect — on the same entries 2 % beat 1 % for bears).
- **Filters live on the paper account (2026-10-27).** `rs_filter_mode` and `pullback_filter_mode` = "live" in the Pi's custom preset (backlog §14). Shadow evidence: relative strength 0 failures in 28 verdicts; pullback failed on two trades that were opened — XLU 10-16 (back above its 50-day) and GLD 10-22 (2.1 ATR below its 20-day) — both under water on 10-27. Small sample; the defaults stay "shadow".

## 5. Cross-References

- `59_debit_spreads_calendars.md` — debit and calendar planners whose expiries are capped.
- `61_entry_confirmation.md` — the gate chain the trend filters join.
- `58_market_state_playbook.md` — breadth over the watchlist grows with the sector ETFs added 2026-10-15.
- `30_profit_target_management.md` — the other shadow-first rollout (trailing profit).

---

*Last verified against repo HEAD on 2026-11-06.*

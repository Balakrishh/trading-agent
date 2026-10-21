# Calibrated Probability Model (walk-forward) and Daily Vol Surface

> **One-line summary:** Estimates the odds behind every strategy (finish beyond a strike, touch a level before expiry) from the conditional distribution of the standardized move, by horizon, asset group, direction, volatility regime and market regime (term structure and ^SKEW tested and dropped), fitted walk-forward so only information known at the decision time is used; plus a daily per-ticker skew / term-structure snapshot to add skew once enough history exists.
> **Source of truth:** [`trading_agent/prob_model.py`](../../trading_agent/prob_model.py), [`trading_agent/vol_surface.py`](../../trading_agent/vol_surface.py), [`scripts/research/calibrate_prob_model.py`](../../scripts/research/calibrate_prob_model.py), [`deploy/systemd/trading-agent-vol-surface.*`](../../deploy/systemd/)
> **Phase:** 2  •  **Group:** strategy / data_quality
> **Depends on:** `01_pop_from_delta.md`, `58_market_risk_state.md`, `60_measurement_shadow_pop_scorecard.md`, `16_market_data_provider_routing.md`, `47_schwab_data_server.md`
> **Consumed by:** research only (phase 1); phase 3 will journal `pop_cal` beside `pop_delta` / `pop_rv` in shadow.

---

## 1. Theory & Objective

The scorers read win probability from option delta (implied odds), and `shadow_pop` adds `pop_rv` (zero-drift lognormal on 20-day realized vol). The 2026-10-13 study (`docs/plans/research_pop_calibration.md`) found both biased in a stable way: short puts survived ~10 points more often than implied, short calls 2–6 points less. A single correction would ignore that the bias depends on the situation; one table per strategy × direction × DTE × regime would be too sparse (7 years × 18 tickers gives a few dozen independent paths per fine cell).

So the model estimates one object — the distribution of the **standardized move** to expiry (and its lowest low / highest high on the way) — conditioned on the situation, with thin cells shrunk toward coarser ones. Every strategy's probabilities are read off it at that strategy's own strikes. Fitting is **walk-forward**: each month is predicted by a model trained only on samples whose outcome was known before the month began.

Phases: **1** research script + report (this skill); **2** daily vol-surface collector (this skill); **3** shadow integration (`pop_cal` journaled, weekly refit) — not built; **4** live use in the EV gate per strategy after evidence.

## 2. Mathematical Formula

```text
x      = ln(S_T / S_0) / (σ_rv · √(h/365))          standardized terminal move
x_min  = ln(min Low(0,T] / S_0) / (σ_rv √(h/365))    lowest low on the way
x_max  = ln(max High(0,T] / S_0) / (σ_rv √(h/365))   highest high on the way

σ_rv   = 20-day close-to-close realized vol on the decision day (annualized)
h      ∈ {7, 14, 21, 30, 45} calendar days;  T = first close on or after day h

Strike K in the same units:   z_K = ln(K / S_0) / (σ_rv √(h/365))
  short put survives      P(x > z_K)        short call survives   P(x < z_K)
  debit vertical pays     P(x beyond z_breakeven)
  condor inside           P(z_put < x < z_call)
  stop / touch risk       P(x_min ≤ z)  /  P(x_max > z)

Hierarchy over ordered dims D = (horizon, asset, direction, vol, market)   [chosen; term, skew tested]:
  level k cell = values of D[:k];  F_raw = weighted empirical CDF on Z_GRID (−3 … 3 by 0.1)
  F_k = (n_eff · F_raw + K · F_{k−1}(parent)) / (n_eff + K),   F_0 = pooled CDF
  n_eff = Σ w,  w = 1 / (trading days in the window)   (h overlapping daily samples share one path)
  K = 100 (chosen walk-forward over 10 / 30 / 100)

Walk-forward: for test month M, train on {samples : outcome date < first day of M}.
Isotonic recalibration of each event family: PAV map fitted on earlier months'
out-of-sample predictions whose outcome was known before M.

Baselines (same event, same units):
  pop_rv     P(x > z) = 1 − Φ(z + ½σ_rv√t)                         (zero drift)
  implied    P(x > z) = 1 − Φ(z·σ_rv/σ_iv + ½σ_iv√t)               (σ_iv = ^VIX / ^VXN / ^GVZ)
  touch      2 × terminal probability (reflection, zero drift)
```

Buckets (all trailing, known on the decision day): vol = trailing-252-day percentile of σ_rv (low < ⅓ < mid < ⅔ < high); term = VIX/VIX3M (contango < 0.90 ≤ flat < 1.00 ≤ inverted); skew = trailing-252-day percentile of ^SKEW; market = `classify_market_state` → normal / caution / stress (DEFENSIVE, CAPITULATION) / recovery; direction = `_determine_regime` → bull / bear / side / mr; asset = broad (SPY QQQ IWM DIA EEM) / sector (XL*, SMH) / alt (TLT GLD GDX) / single.

## 3. Reference Python Implementation

### 3.1 Model core (`prob_model.py`)

```python
# trading_agent/prob_model.py
def fit(enc: Encoded, mask: Optional[np.ndarray] = None, k: float = DEFAULT_K,
        fit_through: str = "") -> ProbModel:
    """Fit the shrunk hierarchy on the rows in ``mask`` (all when None)."""
```

`Encoded` pre-encodes the samples once (bin index of x / x_min / x_max on `Z_GRID`, cell code and parent cell per level) so each walk-forward fold is a handful of `np.bincount` calls. `ProbModel.prob_above / prob_below / prob_touch_down / prob_touch_up(values, z)` answer for one situation (`values` = {dim: label}) from the deepest cell present; `to_dict` / `from_dict` serialize it (with `fit_through`, the last outcome date used).

```python
# trading_agent/prob_model.py
def walk_forward(enc: Encoded, decision: np.ndarray, outcome: np.ndarray,
                 test_from: np.datetime64, k: float = DEFAULT_K,
                 on_fold: Optional[Callable[[np.datetime64, ProbModel, np.ndarray], None]] = None,
                 refit_months: int = 1) -> int:
```

### 3.2 Research driver (`scripts/research/calibrate_prob_model.py`)

Downloads adjusted daily OHLC (2009 →) for the 18-ticker watchlist and ^VIX, ^VIX3M, ^SKEW, ^VXN, ^GVZ; rebuilds direction with `RegimeClassifier._determine_regime` (SMA-50/200, 5-day SMA-50 slope, 20-day 2σ width and 3σ touch) and the market state with `classify_market_state` (SPY SMA-20/50/200 and RSI via `market_state._rsi`, VIX, VIX3M, breadth over `breadth_universe(watchlist)`, hysteresis day to day). Single stocks drop windows containing an earnings date (Yahoo, 2012 →), matching the live earnings rule (skill 65). Variants: horizon only (drift / fat-tail correction, no situation), core (horizon, asset, direction, vol, market), core+term, core+term+skew, core K = 10 / 100 and core+term K = 100, all refitted monthly. The variant is chosen on 2013–2019 test months and confirmed on 2020 →. Report: `docs/plans/research_prob_model.md`; `--save PATH` writes a model fitted on all data.

```python
# scripts/research/calibrate_prob_model.py
def run_variant(s: pd.DataFrame, dims: Tuple[str, ...], k: float, refit: int,
                with_iso: bool = True) -> pd.DataFrame:
```

### 3.3 Daily vol surface (`vol_surface.py`, phase 2)

```python
# trading_agent/vol_surface.py
def summarize_expiry(calls: List[Dict[str, Any]], puts: List[Dict[str, Any]]
                     ) -> Optional[Dict[str, Any]]:
```

For every ticker in `TICKERS`, at expiries near 7 / 30 / 60 days (`candidate_expirations`: the weekly nearest the target, then neighbouring weeks): ATM IV (mean of the call and put nearest |Δ| 0.50), 25-delta put and call IV, `rr25` = call25 − put25, `fly25`, `put_skew` = put25 − ATM, ATM bid-ask %, and the term ratios 7/30 and 30/60. One row per ticker per day appended to `trade_journal/vol_surface.jsonl` (`TRADING_AGENT_VOL_SURFACE`); a ticker already recorded that day is skipped. Chains come from the data server's `/chain` (skill 47) through `mcp.tools.market._data_server_get` — **never a Schwab session of its own** (one token holder, skill 16 §4; AST-tested). `trading-agent-vol-surface.timer` runs it at 15:50 ET on weekdays (before the close, so quotes are live); on a non-trading day it exits without recording. ~0.3 s between calls; 18 tickers take about 2½ minutes.

## 4. Edge Cases / Guardrails

- **Result of the 2026-10-16 run** (`docs/plans/research_prob_model.md`; 362k samples, 166 monthly folds 2013-01 → 2026-10). Chosen: **core, K = 100** — log loss 0.5677 / 0.5735 (2013–19 / 2020 →) vs `pop_rv` 0.5836 / 0.5860 and horizon-only 0.5714 / 0.5774: about three quarters of the gain is the unconditional drift / fat-tail correction, a quarter the situation. VIX/VIX3M and ^SKEW made it slightly worse (thinner cells, no signal) and were dropped. Better than `pop_rv` on all 11 events (90 % intervals below zero); better than implied odds on puts, debits and touches, slightly worse on call survival and condors — implied vol carries call-side information (skew) the realized-vol model lacks, which is what the per-ticker vol surface is for.

- **No look-ahead.** Every feature is computed from data up to the decision day's close (rolling windows, trailing ranks, the day's VIX / VIX3M / SKEW close); training rows need `outcome < test month start`. `tests/test_prob_model.py::test_walk_forward_trains_only_on_known_outcomes` pins it.
- **Overlap.** Daily samples with an h-day horizon share most of their path; weights are 1 / (trading days in the window) so `n_eff` approximates independent paths, and report intervals resample whole months.
- **Thin and unseen cells.** Shrinkage with K pulls a cell toward its parent in proportion to how little data it has; a cell never seen in training returns its parent's estimate (`ProbModel._lookup`).
- **Grid clamp.** CDFs live on z ∈ [−3, 3]; beyond it the edge value is used (a 3σ strike is rarely traded).
- **Expiry events, not trade P&L.** The model predicts where the underlying finishes or travels, not P&L after the agent's stops, targets and trailing; the touch probabilities speak to stop risk. Live shadow (phase 3) is the check.
- **One era.** 2010–2026 is mostly a rising market; drift appears as put-side edge. The "horizon only" variant shows how much of the gain is drift alone.
- **Skew.** ^SKEW is S&P 500 only; per-ticker skew has no free history, hence the phase-2 collector. Add `put_skew` / `rr25` buckets as a dimension only after ≥ 3–6 months of rows, tested on that period alone.
- **Holiday-week expiries.** The neighbouring weeks tried after the first pick are each week's own last trading day (`calendar_utils.last_trading_day_of_week`), not the first pick ± 7 days. On 2026-10-16 the 60-day pick was Thu Dec 24 (Christmas Friday); ± 7 days gave Thu Dec 17 instead of the Dec 18 monthly, so the 9 names without weeklies that far out (GDX, META, AMZN, XLK, XLV, XLY, XLI, SMH, XLU) recorded no 60-day IV that day.
- **After-close quotes.** Until 2026-10-20 the timer ran at 16:05, after the close; the less liquid sector ETFs then quote wide (`atm_spread_pct` 0.77–1.02 for XLV, XLY, XLI on 10-16; 0.85–1.60 for XLE, XLV, XLY, XLI, XLU on 10-20, when XLU's 30-day ATM IV read 0.50 vs 0.16 the session before). Since 2026-10-20 it runs at 15:50 ET. Rows before that: filter on `atm_spread_pct` (≤ 0.25) before using them.
- **Token shared with the agent.** The collector reads through the data server, whose Schwab session shares `~/.schwab_tokens.json` with the agent. On 2026-10-20 the data server still held the refresh token from before the previous evening's login and every chain came back empty (0 of 18 recorded at 16:05; rerun at 20:24 after a data-server restart). `SchwabOAuth` now re-reads the cache when another process has written it (skill 16 §4).
- **Vol surface quality.** Contracts without IV or delta are ignored; an expiry missing either side is skipped and the next candidate week is tried; `atm_spread_pct` flags wide quotes (XLU's ATM puts were 0.79 / 1.20 on 2026-10-15). Single-stock rows spanning earnings show it as a low 7/30 ratio (AMZN 0.77 on 2026-10-15, report 2026-10-29).

## 5. Cross-References

- `01_pop_from_delta.md` — the delta POP the model is compared against.
- `60_measurement_shadow_pop_scorecard.md` — `pop_rv`, the realized-vol baseline (`shadow_pop.prob_above`).
- `58_market_risk_state.md` — the market-regime classifier replayed historically.
- `65_earnings_expiry_and_trend_filters.md` — why earnings windows are excluded for single stocks.
- `47_schwab_data_server.md` — the `/chain` endpoint the collector reads.
- `docs/plans/research_pop_calibration.md` — the 2026-10-13 unconditional study this extends.

---

*Last verified against repo HEAD on 2026-10-21.*

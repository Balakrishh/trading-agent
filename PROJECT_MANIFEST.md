# Trading Agent — Project Manifest

A concise, citation-backed handoff document for cross-LLM context transfer. Every claim below is grounded in `file:line` references so a new collaborator can verify against the source instead of trusting prose.

Last verified: 2026-05-05 against repo state at HEAD.

---

## 1. Core Strategy Logic

### 1a. DTE targeting — per-strategy, dashboard-configurable, preset-driven
**There is no single `TARGET_DTE`.** DTE is one of three values controlled per-strategy via the preset system.

**Built-in presets** (`trading_agent/strategy_presets.py:142-191`):

| Preset | Vertical DTE | Iron Condor DTE | Mean Reversion DTE | Notes |
|---|---|---|---|---|
| Conservative | 35 | 45 | 21 | ~85 % POP, far-OTM |
| **Balanced (default)** | **21** | **35** | **14** | ~75 % POP, baseline |
| Aggressive | 10 | 21 | 7 | ~65 % POP, near-ATM |

Plus a **Custom slot** (`strategy_presets.py:218-232`) that starts from Balanced and applies user overrides.

**Streamlit controls** (`trading_agent/streamlit/live_monitor.py:808-820`)
- `dte_vertical` slider, range `5–60`
- `dte_iron_condor` slider, range `7–60`
- `dte_mean_reversion` slider, range `3–30`
- `dte_window_days` slider, range `1–14` — sets the ± tolerance around each target DTE
- Adaptive scan mode adds a `dte_grid` text input (`live_monitor.py:868-872`); default `(7, 14, 21, 30)` from `PresetConfig.dte_grid`.

**Persistence + hot reload** (`strategy_presets.py:1-17, 235+`)
- Active preset is serialized to `STRATEGY_PRESET.json` at the repo root.
- The agent subprocess re-reads this file at the start of every 5-minute cycle — changes from the dashboard apply on the next tick without restart.
- Missing / malformed file → fall back to `BALANCED` (`strategy_presets.py:199`).

**Ranges per strategy** are derived from `target_dte ± dte_window_days` (`strategy_presets.py:99-112`):
```python
dte_range_vertical       = (max(1, dte_vertical       - dte_window_days), dte_vertical       + dte_window_days)
dte_range_iron_condor    = (max(1, dte_iron_condor    - dte_window_days), dte_iron_condor    + dte_window_days)
dte_range_mean_reversion = (max(1, dte_mean_reversion - dte_window_days), dte_mean_reversion + dte_window_days)
```

**Legacy fallback only** — `trading_agent/strategy.py:122-123` defines `TARGET_DTE = 35` and `DTE_RANGE = (28, 45)` as **class-level defaults** that are used **only** when the constructor's `dte_vertical`/`dte_iron_condor`/`dte_mean_reversion` kwargs are `None` (`strategy.py:177-179`). The Streamlit dashboard always passes per-strategy values explicitly, so these class constants are not what production runs against.

If you see "50 DTE" or any single-DTE claim in older notes, that's **stale**. The live system is per-strategy and per-preset.

### 1b. Adaptive spread width formula — also preset-driven
**Floor formula** at `trading_agent/strategy.py:653-685` (`_pick_spread_width()`):

```
candidate = max(SPREAD_WIDTH, 3 × strike_grid_step, 0.025 × spot_proxy)
snapped   = grid * max(1, int(round(candidate / grid + 0.4999)))
```

- `SPREAD_WIDTH = 5.0` is the legacy floor (`strategy.py:121`).
- The `3 × grid` term keeps the wing at least three strikes wide on illiquid grids.
- The `0.025 × spot_proxy` term scales width to **2.5 % of underlying price** so a $400 SPY contract gets a $10 wing while a $40 stock gets a $1 wing.
- The final snap rounds **up** to the next grid increment via the `+ 0.4999` bias.

**Preset overrides the constants.** The dashboard exposes a width-mode radio (`pct_of_spot` vs `fixed_dollar`) at `live_monitor.py:825-847` plus a value slider:
- `pct_of_spot`: 0.5 % – 5.0 % of spot (Conservative 2.5 %, Balanced 1.5 %)
- `fixed_dollar`: $1 – $25 (Aggressive default $5)

In **adaptive scan mode**, `width_grid_pct` (default `(0.010, 0.015, 0.020, 0.025)`) is swept as part of the `(DTE × Δ × width)` cross-product, scored by `EV_per_$risked = (POP×C/W − (1−POP)×(1−C/W)) / (1−C/W)`, and the highest-EV candidate that clears the `edge_buffer` floor wins (`strategy_presets.py:43-93`).

### 1c. Relative Strength Bias (a.k.a. Leadership z-score)
Defined in `trading_agent/regime.py`, consumed in two trade-routing places.

**Definition** (`trading_agent/regime.py:138-176`)
- `RegimeAnalysis.leadership_anchor: str` — the sibling ticker we compare against (sector ETF for single names, broad-market ETF for sectors).
- `RegimeAnalysis.leadership_zscore: float` — Z-scored 5-min return differential of `(ticker - anchor)` over `LEADERSHIP_WINDOW_BARS` bars.
- `RegimeAnalysis.leadership_signal_available: bool` — **added 2026-05-02**. Distinguishes "RPC failed / no data" from "real 0 σ reading." Without this flag the UI can't tell `+0.000` (genuine) from `+0.000` (default).
- Anchor map: `LEADERSHIP_ANCHORS` at `regime.py:36-88`. 15 ETFs + 25 large-cap single names. Helper `leadership_anchor_for(ticker)` (`regime.py:91-110`) falls back to SPY for unmapped tickers.

**Bias threshold:** `|z| > 1.5 σ`.

**Consumers:**
- `trading_agent/strategy.py:268-278` — when regime is SIDEWAYS, leadership z > 1.5 σ routes to a Bull Put.
- `trading_agent/thesis_builder.py:38-65` — fires Bull Put justification text when z > 1.5.
- `trading_agent/streamlit/watchlist_ui.py` — read-only display only; no trade routing influence.

### 1d. Strategy dispatch
Three "kind" labels in `strategy.py:127-129`:
- `KIND_VERTICAL = "vertical"` (Bull Put / Bear Call)
- `KIND_IRON_CONDOR = "iron_condor"`
- `KIND_MEAN_REVERSION = "mean_reversion"`

Full regime → strategy table in `README.md:82-89`.

### 1e. Regime classifier (single source of truth)
- `trading_agent/regime.py:200+` — `RegimeClassifier._determine_regime(...)` is the **only** rule. The watchlist's multi-timeframe wrapper at `trading_agent/multi_tf_regime.py:_classify_intraday` reuses this exact rule on intraday bars; it does not implement a shadow scorer.
- `BOLLINGER_NARROW_THRESHOLD = 0.04` (`regime.py:199`) — Bollinger width below 4 % is classified SIDEWAYS.
- `VIX_INHIBIT_ZSCORE = 2.0` (`regime.py:117`) — VIX 5-min Δ z > 2 σ inhibits new bullish-premium openings.

### 1f. Architectural invariants (enforced by `scripts/checks/scan_invariant_check.py`)
1. **Single C/W floor formula** `|Δ|×(1+edge_buffer)` exists in `chain_scanner.py`, `risk_manager.py`, and `executor.py`.
2. **Single scoring source** — scoring primitives only in `chain_scanner.py` and `decision_engine.py`. No shadow scorers in any other module.
3. **Backtester wires through `decision_engine.decide()`** — `streamlit/backtest_ui.py` must contain at least one `decide(` call. Live and backtest paths share one decision function. See skill 15.

A new LLM should run `python scripts/checks/scan_invariant_check.py` after any structural change.

### 1g. Strategy Profile preset system — the central control surface
`trading_agent/strategy_presets.py` bundles the four knobs that meaningfully change credit-spread economics into a single `PresetConfig` dataclass: `max_delta`, per-strategy DTE (vertical / iron condor / mean reversion), spread-width policy, C/W floor, max account risk %, directional bias, and the adaptive scan grids.

**Three built-in profiles + Custom** (`strategy_presets.py:142-197`):
- `CONSERVATIVE` — `max_delta=0.15`, vertical 35d / IC 45d / MR 21d, 2.5 % spot width, C/W ≥ 0.20, 1 % account risk
- `BALANCED` (default) — `max_delta=0.25`, vertical 21d / IC 35d / MR 14d, 1.5 % spot width, C/W ≥ 0.30, 2 % account risk
- `AGGRESSIVE` — `max_delta=0.35`, vertical 10d / IC 21d / MR 7d, $5 fixed width, C/W ≥ 0.40, 3 % account risk
- `Custom` — starts from Balanced, applies user overrides from the dashboard

**Two scan modes** (`strategy_presets.py:44-51`):
- `static` — single `(Δ, DTE, width)` point from scalar fields, C/W gated by `min_credit_ratio`. Original behaviour.
- `adaptive` — sweep grid `dte_grid × delta_grid × width_grid_pct`, score each by `EV_per_$risked`, pick the highest-scoring candidate that clears `edge_buffer` (default 10 % over breakeven), or **sit out** if none do.

**Persistence:** `STRATEGY_PRESET.json` at the repo root. Hot-reloaded by the agent subprocess every 5-min cycle (`strategy_presets.py:8-12`).

**Dashboard panel:** `live_monitor.py:780+` "Strategy Profile" expander.

**Why a new LLM should care:** when changing strategy behavior, change the **preset**, not class-level constants in `strategy.py`. The constants are legacy fallbacks. The preset is the live control surface and the only thing the dashboard, the agent, and the backtester all read from.

---

## 2. Hardware & Environment Profile

### 2a. Hardware (recommendation, not enforcement)
The runtime imposes no hardware lock. Profile guidance lives in `setup_intelligence.sh`:
- `setup_intelligence.sh:23` — "tuned for Apple Silicon Pro/Max with 64-128 GB RAM"
- `setup_intelligence.sh:172` — `PROFILE=high-mem` is for "64-128 GB unified-memory Macs, RTX 6000, etc."
- A single comment in `trading_agent/sentiment_verifier.py:23` references the user's "M5 Pro Max" — informational only.

**Treat 128 GB RAM as a profile target, not a hard prerequisite.** The live trading loop, regime classifier, and Streamlit UI run on commodity hardware. Memory pressure shows up only when running large local LLMs alongside the agent.

### 2b. LLM backend
- **Default: Ollama**, configured via `trading_agent/llm_client.py:30,42` (`provider="ollama"`).
- **Dispatch** at `llm_client.py:73-110` supports `ollama / openai / anthropic / lmstudio`.
- The sentiment verifier (`trading_agent/sentiment_verifier.py:252,332`) supports the same set with an Anthropic path when `VERIFIER_PROVIDER=anthropic`.
- **MLX is NOT integrated.** If a downstream LLM was told "OLLAMA/MLX backend," correct that assumption — MLX would be a new integration, not an existing one.

### 2c. Network access requirements
- Alpaca paper-api: `https://paper-api.alpaca.markets/v2` (`market_data.py:111`)
- Alpaca data: `https://data.alpaca.markets/v2` (`market_data.py:110`)
- yfinance for historical daily bars and ^VIX
- Optional: Reddit (PRAW), Twitter (Tweepy) for sentiment, configured via env vars

---

## 3. Tech Stack & Integration

### 3a. Python
- Runtime: **Python 3.10.12** (verified by `python --version` in repo root).
- CI matrix tests **3.11 and 3.12** (`.github/workflows/ci.yml:53`).
- No explicit `python_requires` floor in `requirements.txt`. New code should be 3.10+ compatible (no `match` statements with patterns added in 3.11+, no `tomllib`).

### 3b. Top-level dependencies (`requirements.txt`)
- Market data: `yfinance`, `alpaca-py`, `pandas_market_calendars`
- Numerics: `pandas`, `numpy`, `scipy`
- Config: `python-dotenv`, `requests`
- Testing: `pytest`, `pytest-mock`
- UI: `streamlit`, `plotly`, `watchdog`
- LLM (optional): `anthropic`
- Sentiment (optional): `praw`, `tweepy`

### 3c. Alpaca configuration
- **Stocks feed default: `iex`** (`market_data.py:435`, env var `ALPACA_STOCKS_FEED`).
- **Options feed default: `indicative`** (`market_data.py:646`, env var `ALPACA_OPTIONS_FEED`).
- Paper trading is the default; live URLs would need to be overridden via `alpaca_base_url`.
- IEX feed limitation: thin volume on sector ETFs (XLF, XLK, XLY, XLE, XLV, XLP, XLC, XLB, XLU, XLRE) means `get_5min_return_series` often returns `None` for these on weekends or after the open-bar-skip cut. Setting `ALPACA_STOCKS_FEED=sip` requires a paid SIP subscription.

### 3c-bis. Multi-provider market-data routing (added 2026-05-05)
The market-data plane is now hexagonal — three providers live behind the
same `MarketDataPort` Protocol and the surface chooses one via env var.
Alpaca is still the **execution broker** in every config; only the data
plane is swappable.

- `MarketDataProvider` (`market_data.py`) — Alpaca; default.
- `SchwabMarketDataProvider` (`market_data_schwab.py`) — real-time
  options + Greeks via Schwab Trader API (OAuth 2.0, 30-min access /
  7-day refresh tokens, `python -m trading_agent.schwab_oauth login`
  for the one-time consent flow).
- `YahooMarketDataProvider` (`market_data_yahoo.py`) — yfinance only;
  options return `None`, live bid/ask returns `None`.  Suitable for
  Watchlist + future Backtest surfaces.

`build_market_data_provider(...)` in `market_data_factory.py` walks
`MARKET_DATA_PROVIDER_<SURFACE>` → `MARKET_DATA_PROVIDER` → `alpaca`.
Surfaces wired through the factory: agent live cycle (`agent.py:156`,
`surface="live"`), Watchlist tab (`watchlist_ui.py:_get_data_provider`,
`surface="watchlist"`), and the dashboard's market-open badge
(`live_monitor.py:_is_market_open_cached`, `surface="live"`).  The
`_fetch_account_cached` callsite stays Alpaca-direct intentionally —
account state has no equivalent shape on Schwab/Yahoo.

See [`docs/skills/16_market_data_provider_routing.md`](docs/skills/16_market_data_provider_routing.md)
for OCC symbol translation, IV-percent normalization, and the auth
failure modes the adapter surfaces.

### 3d. Streamlit UI (4 tabs in `trading_agent/streamlit/app.py:52-73`)
1. **Live Monitoring** — `live_monitor.py`
2. **Backtesting** — `backtest_ui.py` (rewritten 2026-05-04; ~475 lines, down from ~4,057. All trading logic now lives in `trading_agent/backtest/` and reuses live primitives.)
3. **LLM Extension** — `llm_extension.py`
4. **Watchlist** — `watchlist_ui.py` (read-only analyst view, no trade routing)

The Watchlist tab is **architecturally isolated**: it does not import `decision_engine`, `chain_scanner`, `executor`, or `risk_manager`, so changes there cannot affect trade decisions.

### 3e. Backtest package — `trading_agent/backtest/` (new 2026-05-04)
A small replay harness that **wires through live primitives** instead of reimplementing them. See [`docs/skills/15_backtest_live_parity.md`](docs/skills/15_backtest_live_parity.md).

- `clock.py` — calendar-aware iterator (NYSE trading days × intraday bar times). Hybrid cadence: intraday 5-min when window ≤ ~30 days, daily otherwise (yfinance 5-min API limit).
- `historical_port.py` — `HistoricalPort` wraps `MarketDataProvider`/yfinance with a hard cursor; reading past `now_t` raises `LookaheadError`.
- `synthetic_chain.py` — builds a `ChainSlice` (the dict shape `decide()` expects) from `(spot, σ-proxy, preset's strike grid)`.
- `account.py` — `SimAccount` cash + open-market-value ledger; commission $0.65/leg.
- `sim_position.py` — open-spread bookkeeping with **VIX-proxy IV scaling** for re-marks: `σ_t = σ_entry × (vix_t / vix_entry)`. Exit logic delegates to `PositionMonitor._check_exit`.
- `cycle.py` — `run_one_cycle` runs PERCEIVE → CLASSIFY → PLAN → RISK → EXECUTE; calls live `decide()`, `RiskManager`, `calculate_position_qty`.
- `runner.py` — `BacktestRunner` drives the clock, emits a `BacktestResult` (equity curve + closed trades).

---

## 4. Resolved Technical Debt

A new LLM should **not** re-litigate any of these. Each fix has a regression test.

### 4a. Lead-z showed `+0.000` for non-ETF tickers (fixed 2026-05-02)
**Symptom:** JPM, TSLA, AAPL etc. always showed Lead-z = +0.000.
**Cause:** `LEADERSHIP_ANCHORS` only had ETFs. Non-ETF tickers fell through to `""` (empty anchor).
**Fix:** Extended `LEADERSHIP_ANCHORS` to 25 large-cap single names mapped to sector ETFs; added `leadership_anchor_for()` helper with SPY fallback (`regime.py:91-110`).
**Regression test:** `verify_consolidated_patch.py` cases `anchors_extended_with_jpm_xlf`, `anchor_for_unknown_ticker_falls_back_to_spy`.

### 4b. Lead-z couldn't distinguish "RPC failed" from "real 0 σ" (fixed 2026-05-02)
**Symptom:** After 4a, Lead-z still showed +0.000 for some tickers because the RPC was returning `None` (Alpaca IEX feed had < 2 bars after `OPEN_BAR_SKIP`, weekend, degenerate stdev).
**Cause:** `leadership_zscore` defaulted to `0.0`, indistinguishable from a real near-zero reading.
**Fix:** Added `leadership_signal_available: bool` field to `RegimeAnalysis`. Watchlist UI now renders `— (no data vs XLF)` when False, plus a top-of-table source-health banner that classifies the failure mode (systemic vs sector-only).
**Regression test:** `verify_consolidated_patch.py` cases `classify_intraday_signal_unavailable_when_rpc_returns_none`, `lead_z_health_classifies_*` (5 cases).

### 4c. Intraday `RegimeAnalysis` left macro fields at zero defaults (fixed 2026-05-02)
**Symptom:** Watchlist intraday rows showed VIX-z = 0, Lead-z = 0, IV rank = 0 even when the daily row had real values.
**Cause:** `multi_tf_regime._classify_intraday` only computed SMA/RSI/BB; the macro RPCs lived in `RegimeClassifier.classify()` only.
**Fix:** Populated `vix_zscore`, `leadership_zscore`, `iv_rank`, `inter_market_inhibit_bullish` in `_classify_intraday` (`multi_tf_regime.py:289-321`), wrapped each RPC in try/except so a single failure doesn't blank the row.

### 4d. ADX dtype crash on `replace(0, pd.NA)` (fixed)
**Symptom:** `adx_strength()` raised `pandas.errors.DataError: No numeric types to aggregate`.
**Cause:** `replace(0, pd.NA)` coerced the float Series to object dtype, breaking `.ewm().mean()`.
**Fix:** Use `np.nan` instead of `pd.NA`, force-cast with `pd.to_numeric(..., errors="coerce")` (`multi_tf_regime.py:404-412`).
**Regression test:** `verify_adx_dtype_fix.py` (4 cases).

### 4e. Wilder smoothing for ADX (fixed)
**Symptom:** ADX values disagreed with TradingView / pandas-ta reference implementations.
**Fix:** Switched to Wilder smoothing (`alpha = 1/window`, `adjust=False`) for ATR, +DI, -DI, and ADX (`multi_tf_regime.py:400-415`).

### 4f. Trend-conflict diagnostic (added 2026-05-02)
**Need:** No way to flag "long-term uptrend, short-term rolling over" patterns without changing the regime label (which would alter strategy routing and break invariants).
**Fix:** Added `sma_200_slope: float` and `trend_conflict: bool` to `RegimeAnalysis`. UI renders ⚠ next to a Bullish/Bearish label when slopes disagree. Strategy code does not read these, so the label and routing are unchanged.

### 4g. Stale-data warning (added 2026-05-02)
**Need:** Markets-closed weekends showed stale Lead-z / VIX-z without any indication.
**Fix:** Added `last_bar_ts: Optional[datetime]` to `RegimeAnalysis`. Watchlist UI computes age vs wall clock, renders ⏰ banner when stale > 30 min.
**Note:** This is "stale **data**" — different from "stale **spread**" (4h below).

### 4h. Stale spread (separate; pre-existing, not new)
`trading_agent/risk_manager.py:39,166-189` — soft-pass when relative quoted spread > `stale_spread_pct` (1 % default). This is a pre-existing risk gate, **not** a recently-fixed bug. Don't conflate it with stale-data detection.

### 4i. Alpaca 403 (clarification, not a bug)
`market_data.py:427` comment: when no `feed=` parameter is sent, Alpaca silently returns 403 for non-IEX subscribers. We always pass an explicit feed parameter, so 403 should never occur in practice. If it does, check that `ALPACA_STOCKS_FEED` is set to a feed your account has access to.

### 4j. Logging volume pass (completed)
Hot-path logs in `position_monitor.py:162-163` and `market_data.py:185-495` were demoted from INFO to DEBUG to keep the live monitor scrollable.

---

## Regression test suite

Six standalone harnesses at `/sessions/determined-eager-goodall/verify_*.py` (run without pytest, important because the sandbox doesn't have all deps):

1. `verify_consolidated_patch.py` — 28 cases covering 4a–4g and the lead-z health banner
2. `verify_adx_dtype_fix.py` — 4 cases for 4d
3. `verify_intraday_bars.py` — 10 cases for `MarketDataProvider.fetch_intraday_bars`
4. `verify_multi_tf_regime.py` — 10 cases for the multi-TF wrapper
5. `verify_watchlist_chart.py` — 11 cases for chart indicator math
6. `verify_watchlist_store.py` — 17 cases for watchlist persistence schema

Plus the pytest suite at `tests/` (not run in the sandbox; requires CI).

Plus the architectural invariant scan: `scripts/checks/scan_invariant_check.py`.

---

## Cross-LLM handoff prompt (paste verbatim)

> I'm working on a trading agent that opens credit-spread positions on US ETFs and large-cap single names through Alpaca's paper-trading API. Read `PROJECT_MANIFEST.md` at the repo root for the full architecture; key facts you should know before suggesting any change:
>
> - **DTE is per-strategy and preset-driven**, not a single global. The `Strategy Profile` panel in the Streamlit dashboard sets `dte_vertical`, `dte_iron_condor`, `dte_mean_reversion` independently (Balanced default: 21 / 35 / 14). Persisted to `STRATEGY_PRESET.json`, hot-reloaded each 5-min cycle. The constants `TARGET_DTE = 35` / `DTE_RANGE = (28, 45)` in `strategy.py` are **legacy fallbacks only** — production never uses them. If you change strategy behavior, change the **preset** (`strategy_presets.py`), not the class-level constants.
> - **Three architectural invariants** (1) single credit-width-floor formula `|Δshort|×(1+edge_buffer)` in `chain_scanner.py` + `risk_manager.py` + `executor.py`, (2) single scoring source (`_score_candidate*` / `_quote_credit` defined only in `chain_scanner.py` and `decision_engine.py`), (3) backtester wires through `decision_engine.decide()` — `streamlit/backtest_ui.py` must contain a literal `decide(` call. Verify with `python scripts/checks/scan_invariant_check.py` after any change.
> - **Backtester is a thin replay shim.** Live trading logic lives in `decision_engine`, `risk_manager`, `executor.calculate_position_qty`, `position_monitor`. The backtest package (`trading_agent/backtest/`) only simulates the *inputs* (cursor-bound historical port, synthetic chain, sim account/position) and threads them through those live primitives. Re-marks use **VIX-proxy IV scaling**: `σ_t = σ_entry × (vix_t / vix_entry)`. See skill 15.
> - **Watchlist tab is architecturally isolated** — it cannot affect trade decisions. Changes there are safe.
> - **Regime classifier `_determine_regime` is the single source of truth** — `multi_tf_regime._classify_intraday` reuses it; never write a shadow scorer.
> - **`RegimeAnalysis` fields are append-only with safe defaults.** Changing existing field types breaks `strategy.py` and `thesis_builder.py`. Add new fields with defaults; don't change `leadership_zscore` from `float` to `Optional[float]`.
> - **Lead-z renders `— (no data vs X)` when `leadership_signal_available=False`** — that's intentional, not a bug. Indicates Alpaca IEX feed had insufficient 5-min bars (weekend, off-hours, sector-ETF anchor on free feed).
> - **Default LLM backend is Ollama**; Anthropic / OpenAI / LM Studio are alternatives. MLX is not integrated.
> - **Six `verify_*.py` harnesses** at `/sessions/determined-eager-goodall/verify_*.py` — these are the regression suite the sandbox runs (no pytest deps required). Run them after any change to `regime.py`, `multi_tf_regime.py`, or `streamlit/watchlist_ui.py`.
> - **Market-data plane is hexagonal**: three providers (Alpaca / Schwab / Yahoo) sit behind the `MarketDataPort` Protocol, dispatched per-surface by `build_market_data_provider(surface=...)` in `trading_agent/market_data_factory.py`. Env-var resolution: `MARKET_DATA_PROVIDER_<SURFACE>` → `MARKET_DATA_PROVIDER` → `alpaca`. Alpaca is always the execution broker; only the data plane is swappable. Schwab uses OAuth 2.0 (30-min access tokens, 7-day refresh tokens, `python -m trading_agent.schwab_oauth login` to bootstrap). When changing market-data behavior, prefer extending an existing adapter over a new branch in agent code; the agent core only sees the port. Skill 16 is the canonical reference.
>
> Before proposing any change, grep the active invariant guards (`scan_invariant_check.py`, `run_journal_split_check.py`, `run_scan_diagnostics_check.py`) under `scripts/checks/` to understand what's protected. The two parity smoke scripts (`run_unified_backtest_check.py`, `run_live_vs_backtest_parity_check.py`) were retired 2026-05-04 with the backtest rewrite — parity is now structural via the `trading_agent/backtest/` package wiring through `decide()` directly, enforced by AST invariant #2/#3.

---

*This manifest reflects repo state at 2026-05-05. If you're reading it more than a week later, re-verify the citations — strategy parameters and dataclass fields move quickly.*

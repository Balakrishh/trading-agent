# Autonomous Options Credit Spread Trading Agent

An autonomous trading agent that generates daily income through high-probability, risk-defined options credit spreads. Primary goal: **capital preservation** — every trade has a known, capped maximum loss.

## Contents

- [Quickstart](#quickstart)
- [Architecture Overview](#architecture-overview)
- [Strategy Selection](#strategy-selection)
- [Adaptive Chain Scanner](#adaptive-chain-scanner)
- [Risk Management Guardrails](#risk-management-guardrails)
- [Live ↔ Backtest Unified Decision Engine](#live--backtest-unified-decision-engine)
- [Intelligence & Sentiment Layers (Optional)](#intelligence--sentiment-layers-optional)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Multi-provider market data](#multi-provider-market-data)
- [Setup & Configuration](#setup--configuration)
- [Project Structure](#project-structure)
- [Signal Journal Format](#signal-journal-format)
- [Spec-Driven Development (SDD)](#spec-driven-development-sdd)

---

## Quickstart

```bash
pip install -r requirements.txt

# minimum .env
echo "ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
TICKERS=SPY,QQQ
DRY_RUN=true" > .env

# run a single cycle (paper / dry-run)
python -m trading_agent.agent --dry-run

# launch the dashboard (Live + Backtest + LLM tabs)
streamlit run trading_agent/streamlit/app.py

# continuous: the launchd supervisor restarts the agent after every cycle
# (skill 57) — a cycle ≈ every 75 s in market hours (AGENT_CYCLE_SLEEP_SEC=60)
python -m trading_agent.agent_supervisor
```

After-hours behaviour: exits cleanly before 9:25 AM ET, after 4:05 PM ET, and on weekends. Override with `FORCE_MARKET_OPEN=true` for paper testing.

---

## Architecture Overview

Each cycle runs two stages sequentially: monitor existing positions first, then evaluate new opportunities per ticker.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       AGENT CYCLE  (agent.py)                       │
│                                                                     │
│  STAGE 1 — Monitor Open Positions                                   │
│    Position Monitor → Exit Signal Check → Order Tracker             │
│      (credit: 50 % profit at natural, 3× hard stop, strike prox.;   │
│       debit: 50 % of max profit / 50 % of debit stop; DTE safety)   │
│                                                                     │
│  STAGE 2 — Open New Positions  (per ticker)                         │
│    I·Perceive → II·Classify → III·Plan → IV·Risk → V·LLM → VI·Exec  │
│    yfinance     SMA / RSI     ChainScanner   8 guardrails           │
│    Alpaca       Bollinger     decision_      buying-power           │
│    snapshots    VIX-z gate    engine.decide  daily-DD breaker       │
│                                                                     │
│    Sentiment Pipeline runs concurrently (Tier 0/1/2 gating) and     │
│    delivers a VerifiedSentimentReport to Phase V (LLM Analyst).     │
│    Future is cancelled if Phase III/IV reject the trade.            │
└─────────────────────────────────────────────────────────────────────┘
```

> An interactive HTML diagram lives at `architecture_diagram.html`.

**SMA-50 slope units.** `MarketDataProvider.sma_slope()` returns the 5-day average **dollar change per day** of the SMA. Downstream consumers only read the sign; logs and the LLM prompt annotate `$/day` so the magnitude isn't mistaken for a percentage.

---

## Strategy Selection

*Current as of 2026-10-05 (backlog phases 1–5, skills 58–60).* Every cycle the agent first rates the **whole market**, then plans each ticker through a strict priority order, then gates the plan.

**0 · Market risk state (skill 58).** SPY vs its 20/50/200-day averages, SPY RSI, VIX, VIX/VIX3M and equity-ETF breadth → `NORMAL / CAUTION / DEFENSIVE / CAPITULATION / RECOVERY`. It sets a size multiplier (1.0 / 0.5 / 0.25 / 0, 0.5 in RECOVERY) and which strategy families may open; CAPITULATION skips all new entries. Exits are never gated.

**Per-ticker plan, first match wins:**

| Priority | Condition | Plan |
|---|---|---|
| **1** | **Mean reversion** — price touches a 3-σ Bollinger band | Mean-Reversion Spread (bear call above / bull put below) |
| 2 | **VIX inhibit** — `vix_z > +2 σ` and regime Bullish / Sideways | Bear Call (demoted) |
| 3 | **Leadership bias** — `leadership_z > +1.5 σ`, Bullish / Sideways | Bull Put |
| 4 | **Oversold downtrend** — Bearish, RSI < 30 | vol rank ≥ 30: **Bounce Bull Put** once price reclaims its 5-day high close (short strike below the 10-day low), else no trade; vol rank < 30: no trade (*wait for stabilisation*) — never a bear call into RSI < 30 |
| 5 | **Regime credit plan** | Bullish → Bull Put · Bearish → Bear Call · Sideways → Iron Butterfly (opt-in) then Iron Condor |
| 6 | **Low-volatility fallback** — the credit plan found no positive-EV candidate and vol rank < 30 | Bullish → **Call Debit Spread** · Bearish → **Put Debit Spread** · Sideways → **Calendar Spread** |

The playbook table behind rows 4–6 (trend × volatility × RSI → playbook) lives in `market_state.playbook_for`; every journal row carries `playbook` and `playbook_implemented`.

**Then the gates, in order:** RiskManager (below) → market-state strategy gate (e.g. CAUTION blocks new bull puts / call debits / CSPs) → ladder gate → total-risk sizing. A rejected plan is journaled with its reason (`risk: <first failed check>`).

**Credit vs debit economics.**
* Credit structures need positive EV with `|Δ|` as P(ITM): `C/W ≥ |Δshort| × (1 + edge_buffer)` (one formula in three files, CI-enforced).
* Debit structures (skill 59) are not required to show model EV — their edge is the trend / range thesis. Instead the debit must be ≤ **market mid × (1 + `debit_max_overpay`)** (5 %) and reward/risk ≥ 1.0. Verticals buy the |Δ|≈0.50 leg and sell one `width_grid_pct` step out (smallest debit first); calendars sell the near (≈21 DTE) and buy the far (+28 d) strike nearest spot, payoff shape from Black-Scholes rescaled to the mid.
* All pricing uses `fill_model = "natural"` (sell at the bid, buy at the ask) — the paper account filled only at natural.

**Wheel (operator-driven, skill 40).** Cash-secured puts / covered calls are screened by MCP `wheel_screen` (fundamentals, earnings, per-leg quote gate, no CSP below the 200-day average, one ticker per sector, CSP pause in CAUTION / DEFENSIVE / CAPITULATION), staged with `/propose`, and submitted by `executor_promote` (mid → halfway → bid). The agent monitors them (50 % profit, |Δ| ≥ 0.45 stop; assignment accepted) and reconciles expiries.

**Z-scored leadership bias.** `LEADERSHIP_ANCHORS` in `regime.py` maps each ticker to a sibling benchmark (`SPY → QQQ`, sector ETFs → SPY, …). The latest 5-min return differential is z-scored against its own ~20-bar distribution; `z > 1.5 σ` in a Bullish/Sideways regime picks Bull Put.

**VIX inter-market gate.** `^VIX` (yfinance) z-scored over ~20 5-min bars; `vix_z > +2 σ` sets `inter_market_inhibit_bullish` and demotes Bull Put / Iron Condor → Bear Call for that cycle. Mean reversion bypasses it.

**DTE and width.** Per-strategy and per-preset (skill 13): verticals / condors / mean reversion / debit (30) / calendar near (21); the adaptive scanner sweeps `dte_grid × delta_grid × width_grid_pct`. There is no single `TARGET_DTE`.

---

## Adaptive Chain Scanner

`chain_scanner.py` replaces the legacy "single point in chain space" planner with a scored sweep. For every `(DTE, target Δshort, width)` tuple in the configured grid the scanner fetches the relevant put/call chain, picks the contract closest to target Δ, picks the protective leg `width × spot` strikes away (snapped to grid), prices the spread off NBBO mids, and scores it.

**Credit pricing — `_quote_credit`.** With the default `fill_model = "natural"` (backlog §6.1): `short_bid − long_ask` — what the paper account actually fills at. With `"mid"`: `short_mid − long_mid − fill_haircut` ($0.02), conservative side when a quote is missing. Debits use the mirror, `_quote_debit` (natural: `long_ask − short_bid`).

**Score formula:**

```
POP         ≈ 1 − |Δshort|
C/W         = credit / width
EV/$risked  = (POP × C/W − (1 − POP) × (1 − C/W)) / (1 − C/W)
annualized  = EV/$risked × (365 / DTE)
```

**Hard filters before scoring:** `POP ≥ min_pop` (default 0.55), `C/W ≥ |Δshort| × (1 + edge_buffer)` — the **edge floor** (default `edge_buffer = 0.10` → 10 % over breakeven), positive net credit, both legs quoting non-zero bid/ask.

The scanner returns `[]` when nothing passes — the agent treats this as `skipped: no edge` and journals it. Annualized score breaks ties.

**Single-source-of-truth invariant.** The same `cw_floor = |Δshort| × (1 + edge_buffer)` formula appears in `chain_scanner.py`, `risk_manager.py`, and `executor.py`. `scan_invariant_check.py` AST-walks all three sites every release; the static `MIN_CREDIT_RATIO=0.33` floor is now only used when adaptive mode is disabled.

---

## Risk Management Guardrails

Every trade must pass **all eight checks** before execution:

| # | Check | Rule |
|---|---|---|
| 1 | Plan Validity | Strategy planner found valid strikes and contracts |
| 2 | Credit-to-Width | **Adaptive**: `C/W ≥ |Δshort| × (1 + edge_buffer)`. **Static**: `C/W ≥ min_credit_ratio`. **Debit plans** (skill 59): `0 < debit ≤ max_debit` (mid × (1 + overpay)) instead |
| 3 | Sold Delta | `≤ max_delta` — credit plans only (a debit spread's sold leg is a hedge; a calendar's is ATM by design) |
| 4 | Max Loss | `≤ max_risk_pct × equity` (preset; 2 % live) × market-state size multiplier, capped by the remaining total-risk budget |
| 5 | Account Type | Must be `paper` |
| 6 | Market Hours | Market must be open |
| 7 | Underlying Liquidity | `bid/ask spread < max(LIQUIDITY_MAX_SPREAD, LIQUIDITY_BPS_OF_MID × mid)`; stale quotes (`spread/mid > STALE_SPREAD_PCT`) soft-pass with a WARNING |
| 8 | Buying Power | `available BP ≥ (1 − MAX_BUYING_POWER_PCT) × equity` |

`Max Loss = (Width − Credit) × 100` for credit structures and `Debit × 100` for debit structures. The sentiment pipeline is advisory only — it can tighten constraints, never loosen them.

**Portfolio caps (skill 37).**
* **Per ticker:** up to `max_positions_per_ticker` (2) — a second position only on a later day, with an expiration ≥ `ladder_min_gap_days` (7) from the open one, and only while the open one is at or above breakeven (add to winners only). Pending orders block the ticker.
* **Per sector:** 2 (`sector_map.py`); counts pending orders and every submission earlier in the same cycle.
* **Total open risk:** Σ max loss of open defined-risk positions ≤ `max_total_risk_pct` (10 %) of equity; each trade is sized into what remains; Wheel legs excluded (their collateral has its own 40 %-of-equity rule).
* **Market state:** size multiplier and allowed families (above).

**Daily Drawdown Circuit Breaker.** Equity drop > `DAILY_DRAWDOWN_LIMIT` (default 5 %) from the day's open → log + `os._exit(1)`.

**Liquidation Mode.** Available BP > `MAX_BUYING_POWER_PCT` (default 80 %) → Stage 2 skipped; Stage 1 continues.

**Capital Retainment Guards.** Macro Guard (skips Bull Put when `price < SMA-200`); High-IV Block (skips ALL new entries when realized-vol IV rank > 95th percentile); RSI gate (no bear call at RSI ≤ 30, no iron condor while RSI is outside [35, 65)).

**Exits.**
* *Credit spreads:* profit target 50 % of credit judged at the natural cost to close; `HARD_STOP` at 3× credit; 50 %-of-max-loss stop; `STRIKE_PROXIMITY` (within 1 % of a short strike, optional defensive roll); `DTE_SAFETY` (15:30 ET on the last trading day before expiry); regime shift.
* *Debit spreads / calendars (skill 59):* stop at 50 % of the debit; target 50 % of max profit (verticals) or 25 % of the debit (calendars); DTE safety on the near expiry; regime exit only on a real trend change (verticals: the opposite trend; calendars: any bullish or bearish trend, never a one-cycle mean-reversion reading); calendars also close when price drifts ≥ 3 % from the strike.
* *Wheel:* 50 % of credit; CSP |Δ| ≥ 0.45 stop; otherwise assignment is accepted.
* Closes go out as one multi-leg order (improved, then natural price); realized P&L is computed from the actual fills.

**Position Exit Debouncing.** Non-immediate exit signals require **3 consecutive cycles** (≈ 4 min at the ~75 s cadence). Bypassed by `HARD_STOP`, `STRIKE_PROXIMITY` and `DTE_SAFETY`.

**Live Quote Refresh at Execution.** The executor re-quotes both legs right before submission and re-validates the economics-bearing checks (credit ratio or debit cap, max loss) against the live price; debit orders go at natural, credit orders one tick inside.

**Measurement (skill 60).** Entry fills are written back into the trade plans the monitor reads; every journaled plan logs a realized-volatility POP beside the delta POP (shadow, no trading effect); MCP `get_playbook_scorecard` reports win rate / expectancy / return on risk per playbook with an advisory size after 20 trades.

---

## Live ↔ Backtest Unified Decision Engine

The previous biggest source of drift was that the live agent ran a `(DTE × Δ × width)` adaptive sweep through `ChainScanner.scan()` while the backtester ran a homegrown σ-distance heuristic with three separate credit-pricing modes — different strike picker, different credit pricing, different EV/POP logic, no shared code. Any change on one side could silently diverge the other.

The **May 2026 backtester rewrite** (`trading_agent/backtest/`) collapses this entirely. The backtester no longer owns *any* trading logic; it is a thin replay shim that reuses the live primitives:

```
chain_scanner.py            ← pure helpers (_quote_credit, _score_candidate_with_reason)
        │
        ▼
decision_engine.decide()    ← pure scoring entrypoint (no I/O)
        │
        ├──── ChainScanner.scan()                  (live)
        └──── trading_agent.backtest.run_one_cycle (backtest)
```

`decision_engine.decide(DecisionInput) -> DecisionOutput` is a pure function with no I/O, no calendar lookups, no broker calls. It takes `ChainSlice`s in (one expiration's `{strike, delta, bid, ask, symbol}` dicts plus the DTE), runs the full `(Δ × width)` sweep, and returns ranked `SpreadCandidate`s plus a `ScanDiagnostics` block. Live (`ChainScanner.scan`) and backtest (`run_one_cycle`) both delegate to it — neither owns the scoring logic.

**Backtest package layout** (`trading_agent/backtest/`, full reference in [`docs/skills/15_backtest_live_parity.md`](docs/skills/15_backtest_live_parity.md)):

| Module | Role |
|---|---|
| `clock.py` | Calendar-aware iterator (NYSE trading days × intraday bar times). Hybrid cadence — intraday 5-min when window ≤ ~30 days, daily otherwise. |
| `historical_port.py` | `HistoricalPort` wraps `MarketDataProvider`/yfinance with a hard cursor; reading past `now_t` raises `LookaheadError`. |
| `synthetic_chain.py` | Builds a `ChainSlice` (the dict shape `decide()` expects) from `(spot, σ-proxy, preset's strike grid)`. |
| `account.py` | `SimAccount` cash + open-market-value ledger; commission $0.65/leg. |
| `sim_position.py` | Open-spread bookkeeping with **VIX-proxy IV scaling** for re-marks: `σ_t = σ_entry × (vix_t / vix_entry)`. Exit logic delegates to `PositionMonitor._check_exit`. |
| `cycle.py` | `run_one_cycle` — single PERCEIVE → CLASSIFY → PLAN → RISK → EXECUTE step calling live `decide()` / `RiskManager` / `calculate_position_qty`. |
| `runner.py` | `BacktestRunner` drives the clock, emits a `BacktestResult` (equity curve + closed trades). |
| `streamlit/backtest_ui.py` | ~475-line UI shim (down from ~4,057 pre-rewrite). Settings → run → render. |

**Drift-prevention enforcement** lives in three places:

1. **`scan_invariant_check.py`** — AST walker (CI). Asserts (a) the `|Δshort|×(1+edge_buffer)` C/W floor formula appears in `chain_scanner.py`, `risk_manager.py`, and `executor.py`; (b) no module *outside* `chain_scanner` and `decision_engine` defines `_score_candidate`, `_score_candidate_with_reason`, or `_quote_credit`; (c) `streamlit/backtest_ui.py` contains a literal `decide(` call.
2. **Cursor-bound port.** `HistoricalPort` raises `LookaheadError` if any code path tries to read a bar past `now_t`. Removes a class of "future leakage" bugs by construction.
3. **Separate journal files.** Live writes `trade_journal/signals_live.jsonl`; backtest writes `trade_journal/signals_backtest.jsonl`. The LLM analyst corpus and the live-monitor diagnostics panel deliberately read only the live file so synthetic backtest counterfactuals can't bias guardrail recommendations. `JournalKB.__init__` accepts `run_mode={"live","backtest"}`; an unknown value raises `ValueError`.

**Smoke checks** (run in CI; see `scripts/checks/README.md`):

```bash
python3 scripts/checks/scan_invariant_check.py                # AST invariants
python3 scripts/checks/scan_skill_quotes_match.py             # SDD §3 quotes vs source
python3 scripts/checks/scan_skill_freshness.py                # SDD footer date vs git-mtime
python3 scripts/checks/build_traceability.py                  # SDD coverage matrix refresh
python3 scripts/checks/run_scan_diagnostics_check.py          # ChainScanner + decide() integration
python3 scripts/checks/run_journal_split_check.py             # JournalKB run_mode split
pytest tests/conformance/                                     # SDD documented-behavior tests
```

All must pass before any change to scoring, pricing, or floor logic ships. The three SDD checks (`scan_skill_quotes_match`, `scan_skill_freshness`, `build_traceability`) catch spec-vs-code drift at CI time — see [Spec-Driven Development](#spec-driven-development-sdd) below. (The legacy `run_unified_backtest_check.py` and `run_live_vs_backtest_parity_check.py` were retired with the May 2026 backtest rewrite — live ↔ backtest parity is now structural since the backtest package wires through `decide()` directly.)

### Backtest Pricing Model

Per-bar credit comes from a Black-Scholes synthetic chain (`bs_price` in `trading_agent/backtest/black_scholes.py`, no scipy dependency) reconstructed from the historical spot, the preset's strike grid, and an IV proxy seeded from realised σ at entry. Daily re-marks scale that IV by the VIX ratio `vix_t / vix_entry`. This is honest within a single regime; for multi-regime backtests the intraday cadence (≤ 30 days, real 5-min spot bars) gives the most faithful exits.

**Known residual drift sources (track here so they don't get rediscovered as bugs):**

1. **Bid/ask spread modelling** — synthetic chain treats mid = bid = ask. Live fills against bid/ask, so backtest credit slightly over-estimates real credit. Compensated partially by the per-leg commission ($0.65) charged on both open and close.
2. **Mean-Reversion priority** — Priority 1 of live `plan()` is not yet wired through the backtest run-loop. Vertical (Bull Put / Bear Call) and Iron Condor coverage is full.
3. **Greeks from BS, not from chain** — listed `delta` is the BS delta at the synthetic IV, not the broker-reported Greek. Near-no-op when synthetic IV ≈ implied; can pick neighbouring strikes when they diverge.

---

## Intelligence & Sentiment Layers (Optional)

Both layers are off by default. Each degrades gracefully if its dependencies aren't installed.

### Core Intelligence Layer

`Trade Executes → Journal Entry → LLM Post-Trade Analysis → Lessons → KB → Better Decisions Next Cycle → (after 20+ trades) Fine-Tune Local Model`

| Component | File | Role |
|---|---|---|
| LLM Client | `llm_client.py` | OpenAI-compatible — Ollama, LM Studio, Claude API |
| Trade Journal | `trade_journal.py` | Full lifecycle per trade: entry, execution, exit, P&L, lessons |
| Knowledge Base / RAG | `knowledge_base.py` | File-based vector store via `nomic-embed-text`; cosine search; no external DB |
| LLM Analyst | `llm_analyst.py` | Returns approve/modify/skip — **advisory only**, can't override the risk manager |
| Fine-Tuning Pipeline | `fine_tuning.py` | Exports Chat JSONL / Alpaca / DPO formats once 20+ closed trades exist |

### Multi-Source Sentiment Pipeline

A `SentimentPipeline` facade (`sentiment_pipeline.py`) runs concurrently in a background thread during every cycle and delivers a `VerifiedSentimentReport` to the LLM Analyst at Phase V. Three tiers of gating prevent redundant local-LLM calls within the per-cycle budget (a cycle runs about every 75 s).

| Tier | Gate | When it fires | LLM calls |
|---|---|---|---|
| **0** | Earnings calendar short-circuit | yfinance reports a scheduled earnings date within `EARNINGS_CALENDAR_LOOKAHEAD_DAYS` (default 7) | None — deterministic `event_risk=1.0` |
| **1** | Content-hash cache | SHA-1 fingerprint over normalised evidence matches a previously-produced `VerifiedSentimentReport` | None — replays cached verified report |
| **2** | Full chain | Evidence changed — runs NewsAggregator → FinGPT specialist → reasoning verifier | FinGPT + verifier |

The cache only ever holds **post-verifier** results, so Tier 1 never weakens the no-hallucination guarantee.

**Source authority weights** (overridable via `NEWS_SOURCE_WEIGHTS_JSON`):

| Source | Weight | Auth | Captures |
|---|---|---|---|
| SEC EDGAR 8-K / 10-Q | 1.00 | None (free REST) | Material events, earnings, insider changes |
| Federal Reserve RSS | 0.95 | None | FOMC statements, rate decisions |
| Yahoo Finance | 0.70 | None | General financial news |
| Twitter / X cashtag | 0.50 | Bearer token | Breaking news, retail momentum |
| Reddit r/options, r/stocks | 0.45 | PRAW | Options-specific sentiment |
| Reddit r/wallstreetbets | 0.35 | PRAW | High-noise retail sentiment |

**FinGPT specialist** — local Ollama model returns a `SentimentReport` with `sentiment_score`, `event_risk`, `confidence`, `recommendation`, `key_themes`, `reasoning`.

**Reasoning verifier** — independent stronger model (`qwq:32b` / `deepseek-r1:32b` locally, or Claude via `VERIFIER_PROVIDER=anthropic`) cross-checks every claim in FinGPT's reasoning against raw evidence. Outputs `verified_sentiment_score`, `verified_event_risk`, `hallucination_flags`, `agreement_score`, `evidence_mapping`. Falls back to passing through the original report unchanged when unavailable.

**Lifecycle.** The pipeline owns its own single-worker `ThreadPoolExecutor` and is used as a context manager from `agent._run_cycle`:

```python
with SentimentPipeline.from_config(cfg.intelligence) as pipeline:
    for ticker in tickers:
        fut = pipeline.submit(ticker, regime, price, rsi, iv_rank, strategy)
        # ... Phase III + IV
        if not (plan.valid and verdict.approved):
            fut.cancel()             # don't waste a local LLM call
        sentiment = fut.result() if fut else None
        # ... Phase V + VI
# pool.shutdown(wait=True) here — every Future belongs to exactly one cycle
```

---

## Streamlit Dashboard

```bash
pip install "streamlit>=1.42.0" "plotly>=6.0.0" "watchdog>=3.0,<5"
streamlit run trading_agent/streamlit/app.py
```

| Tab | Features |
|---|---|
| **📡 Live Monitoring** | Agent Start / Stop / Dry-Run · cycle PID · equity · **Daily P&L (realized + unrealized)** · regime badge · **cycle-staleness beacon** · open positions · **Closed Today** + **Close Failures Today** panels · equity curve · 8-guardrail status · agent log · Strategy Profile applier · journal expander |
| **📊 Backtesting** | Date range · multi-ticker · timeframe (1Day / 5Min) · Live Quote Refresh · **Unified Decision Engine** toggle (preset selector) · simulated P&L · per-regime bars · equity + drawdown · trade log · CSV / JSON / Journal export |
| **🤖 LLM Extension** | Chat with local Ollama model (RAG over `signals_live.jsonl`) · Optimize Strategy → one-click `.env` update |
| **📊 Watchlist** | Persistent ticker watchlist (`knowledge_base/watchlist.json`) · multi-timeframe regime table (1d / 4h / 1h / 15m / 5m) · ADX strength badge · VIX-z macro strip · 4-row Plotly chart (Price + overlays / Volume / Oscillators / Trend) with 6 timeframes (5m → 1d) and indicator toggles. **Read-only — never imports `decision_engine`, `chain_scanner`, `executor`, or `risk_manager`.** |

**Refresh model.** Refresh is event-driven via `watchdog`. A single per-process `Observer` (wrapped in `@st.cache_resource`) watches `trade_journal/` and `trade_plans/`; loaders cache by `(version, mtime, size)` so unrelated reruns hit the cache and only real journal writes invalidate. Default tick is `LIVE_MONITOR_REFRESH_SECS=3` (was 30). Kill switches: `WATCHDOG_DISABLE=1` (fall back to mtime polling) and `WATCHDOG_FORCE_POLLING=1` (NFS / network mounts where inotify is unavailable).

**Broker-state gating.** Alpaca account / positions / clock fetches in the Live Monitor are TTL-cached (`BROKER_STATE_TTL_SECS=30`) and **only run when the agent loop is started** — opening Streamlit alone makes zero broker calls. A manual `↻ Refresh broker state` button is shown when the loop is stopped, for ad-hoc inspection.

### Operator surfaces (added 2026-05-06)

Three pre-live-trading visibility additions worth knowing about:

**Daily P&L tile.** The metric row's "Unrealized P&L" became "Daily P&L" — sum of unrealized from open positions PLUS realized from today's `action="closed"` rows. The split is rendered as the delta caption (`realized $+80 / unrealized $-20`). After the first close of the day the headline number now reflects true daily performance instead of just open-position drift.

**Cycle-staleness beacon.** Sits below the metric row. When the loop is running but the latest journal timestamp is more than 5 minutes old, an orange warning appears; over 10 minutes, a red error directing the operator to logs and the Stop/Start orphan-sweep path. Suppressed when the loop is stopped (the operator already knows nothing's running).

**Closed Today + Close Failures Today.** Two collapsible panels below Open Positions. **Closed Today** counts only complete fills (all legs accepted); **Close Failures Today** filters to `action="close_failed"` rows where Alpaca rejected one or more legs (PDT, uncovered, insufficient buying power) and the position is still open on the broker. The Failures panel auto-expands when any ticker is in a 60-min cooldown and renders a 🚨 manual-intervention banner with the cooldown deadline. Pre-cooldown rows show progress as `2/3` in the Streak column.

See [`docs/skills/17_close_failure_and_cooldown.md`](docs/skills/17_close_failure_and_cooldown.md), [`docs/skills/18_order_submission_idempotency.md`](docs/skills/18_order_submission_idempotency.md), and [`docs/skills/19_journal_schema.md`](docs/skills/19_journal_schema.md) for the underlying mechanics.

### Strategy Profile

The Live Monitoring tab has a Strategy Profile expander that controls the four knobs that meaningfully change credit-spread economics: short-leg |Δ|, DTE per strategy, spread-width policy, C/W floor. Picking a profile + a directional bias and clicking **Apply** writes `STRATEGY_PRESET.json`; the agent subprocess re-reads it at the start of every cycle.

| Profile | Δ-short | DTE (Vert / IC / MR) | Width | C/W floor | Risk/trade | Approx. POP |
|---|---|---|---|---|---|---|
| Conservative | 0.15 | 35 / 45 / 21 | 2.5 % × spot | 0.20 | 1 % | ~85 % |
| **Balanced** (default) | 0.25 | 21 / 35 / 14 | 1.5 % × spot | 0.30 | 2 % | ~75 % |
| Aggressive | 0.35 | 10 / 21 / 7  | $5 fixed     | 0.40 | 3 % | ~65 % |
| Custom | sliders | sliders | sliders | sliders | sliders | — |

**Directional-bias filter.** Restricts which classifier outputs the agent will trade. Fires immediately after Phase II so disallowed regimes short-circuit before sentiment / chain fetch. Mean-reversion is always allowed (the 3-σ touch override is a fear-spike signal, not a directional view).

**Where the preset is applied.** `agent.__init__` calls `load_active_preset()` and forwards the knobs into `StrategyPlanner` + `RiskManager`; `strategy._pick_expiration(kind=...)` honours the per-strategy DTE override; `strategy._pick_spread_width` honours `width_mode`; `agent._process_ticker` calls `regime_is_allowed(regime, bias)` after classify and writes `action="skipped_bias"` to the live journal on rejection.

`STRATEGY_PRESET.json` is written atomically (temp + rename). Missing / malformed JSON falls back to **Balanced** (logged) so a fresh install is always operational without touching the dashboard.

### Backtest Quote Model

After the May 2026 rewrite (skill 15) the backtester no longer hits Alpaca for an "is the live quote still good?" refresh — every per-bar credit comes from the Black-Scholes synthetic chain (`trading_agent/backtest/synthetic_chain.py`) reconstructed from the historical spot, the preset's strike grid, and an IV proxy seeded from realised σ at entry. Re-marks scale that IV by the VIX ratio. This eliminates the previous `_SNAPSHOT_FRESH_DAYS` heuristic and the "is this entry too old to refresh?" gating logic; a backtest that reaches the same `(spot, σ, VIX)` triple twice deterministically produces the same credit twice.

Regression: `tests/test_backtest/test_black_scholes.py`, `tests/test_backtest/test_synthetic_chain.py`, `tests/test_backtest/test_sim_position.py`.

### Watchlist Tab

A read-only analyst surface for multi-timeframe regime monitoring. Add tickers via the input row, persist them across restarts (`knowledge_base/watchlist.json`, atomic temp+rename writes), and view each ticker's regime / ADX / IV-rank across five timeframes simultaneously.

**Multi-timeframe regime parity.** `multi_tf_regime.classify_multi_tf` reuses `RegimeClassifier._determine_regime` — the same pure rule the live agent uses on daily bars — fed intraday bars at `1d / 4h / 1h / 15m / 5m`. There is no fork of regime logic. SMA windows scale per timeframe (`(50, 200)` for 1d, `(20, 50)` for intraday). The "TF agree" column is the share of timeframes whose trend matches each ticker's longest interval — 100% means fully aligned across the stack.

**Hybrid intraday data path.** `MarketDataProvider.fetch_intraday_bars(ticker, interval)` pulls history from yfinance (chart depth: 5m/15m/30m capped at 60 days; 4h synthesised via `df.resample("4h")` from 60m) and overlays the right-most live tick from the Alpaca snapshot when available. Cached for 60s by `(ticker, interval)` so a Streamlit rerun within the refresh window is free.

**Chart panel.** A 4-row Plotly subplot stack (Price · Volume · Oscillators · Trend) with collapsible rows. Indicators include SMA-50/200, Bollinger Bands, ATR bands, full Ichimoku Kinkō Hyō (Tenkan / Kijun / Cloud), RSI(14), Stoch RSI, MACD(12,26,9), and ADX(14). All indicators are pure pandas/numpy in `watchlist_chart.py` — no `pandas-ta`, no `TA-Lib`, no `numba` (the latter has no Python 3.13+ wheels yet, which broke an earlier prototype). The ADX line on the chart and the strength badge in the table use the same `_adx_series` math so they cannot drift.

**Architectural safety.** `watchlist_ui.py` and `watchlist_chart.py` import only `market_data`, `multi_tf_regime`, `regime`, `watchlist_store`, plus `streamlit` / `plotly`. They explicitly **do not import** `decision_engine`, `chain_scanner`, `executor`, or `risk_manager` — the watchlist is a display surface and cannot influence trade decisions even by accident.

**Refresh model.** `@st.cache_data(ttl=WATCHLIST_REFRESH_SECS)` (default 60s) keyed on `(ticker, intervals_tuple, refresh_token)`. The `↻ Refresh` button bumps the token for immediate invalidation; otherwise the cache self-expires every minute so yfinance doesn't get rate-limited.

Regression: `tests/test_market_data.py::TestFetchIntradayBars`, `tests/test_multi_tf_regime.py`, `tests/test_watchlist_store.py`, `tests/test_watchlist_chart.py`.

---

## Multi-provider market data

The agent supports three market-data providers behind the `MarketDataPort` protocol, dispatched per-surface via env vars. Alpaca remains the execution broker regardless — only the **data plane** is swappable.

| Provider | Options + Greeks | Real-time NBBO | Notes |
|---|---|---|---|
| `alpaca` (default) | yes | only on paid OPRA tier (`indicative` is 15-min delayed on free) | Same creds the executor uses |
| `schwab` | yes (real-time) | yes — free for Schwab brokerage holders | OAuth 2.0; tokens rotate every 30 min, re-auth weekly |
| `yahoo` | **no** (returns `None`) | **no** | yfinance — fine for charts/regime, unsupported for the live trading surface |

**Per-surface routing.** The factory walks `MARKET_DATA_PROVIDER_<SURFACE>` → `MARKET_DATA_PROVIDER` → `alpaca`. Recognised surfaces: `LIVE` (the agent's cycle), `WATCHLIST` (the dashboard's chart + regime tab and market-open badge), `BACKTEST` (reserved). Example mixed config:

```bash
MARKET_DATA_PROVIDER=alpaca               # global default
MARKET_DATA_PROVIDER_LIVE=schwab          # agent trades on real-time Schwab quotes
MARKET_DATA_PROVIDER_WATCHLIST=alpaca     # dashboard reads Alpaca
MARKET_DATA_PROVIDER_BACKTEST=yahoo       # backtester reads yfinance
```

The watchlist tab's resolved provider is logged at startup (`MarketData factory: surface='watchlist' → provider=alpaca`).

### Schwab one-time setup

Only required if any surface is set to `schwab`.

1. Register an app at [developer.schwab.com](https://developer.schwab.com/) → "Add a new app", select "Individual Developer", set the redirect URI to `https://127.0.0.1:8182`, choose API Product = **Trader API**. Wait for the status to change to "Ready For Use".
2. Add the credentials to `.env`:
   ```
   MARKET_DATA_PROVIDER_LIVE=schwab
   SCHWAB_CLIENT_ID=<your app key>
   SCHWAB_CLIENT_SECRET=<your app secret>
   SCHWAB_REDIRECT_URI=https://127.0.0.1:8182
   ```
3. Run the one-time authorization-code exchange:
   ```bash
   python -m trading_agent.schwab_oauth login
   ```
   The CLI prints a Schwab login URL — open it in a browser logged into your Schwab account, click Approve. Schwab redirects to `https://127.0.0.1:8182/?code=…&session=…`. Your browser will show a "connection refused" page (expected — nothing's running on port 8182), but the URL bar contains the authorization code. Paste the full URL back into the CLI prompt; it extracts the code, exchanges it for tokens, and persists to `~/.schwab_tokens.json`.
4. Verify with `python -m trading_agent.schwab_oauth status`.

**Refresh-token lifetime is 7 days, absolute.** The 30-min access token refreshes silently inside the agent loop, but every 7 days you need to re-run `login`. The agent will log a clear WARNING with the exact CLI command when this happens.

**Why a manual login at all?** OAuth's authorization-code flow requires the brokerage account holder to physically click "Approve" once. This is regulatory (data-access consent) and can't be bypassed by storing your username/password — that would violate Schwab's TOS and their API rules.

### Symbol-format note

Schwab option symbols are **space-padded** (`"AMZN  220617C03170000"`); the rest of the agent uses **compact OCC** (`"AMZN220617C03170000"`). Translation happens at the adapter boundary, so the agent always speaks compact OCC. If you ever see a padded symbol in `signals_live.jsonl` or a trade plan, that's a leak worth filing.

See [skill 16](docs/skills/16_market_data_provider_routing.md) for the deeper architectural reference.

---

## Setup & Configuration

### 1. Install

```bash
pip install -r requirements.txt

# optional sentiment-pipeline deps (each degrades gracefully if absent)
pip install praw>=7.7.0        # Reddit
pip install tweepy>=4.14.0     # Twitter / X
pip install anthropic>=0.40.0  # Claude verifier
```

### 2. Configure `.env`

A minimal `.env` is shown in [Quickstart](#quickstart). The full reference follows.

#### Core trading

| Variable | Default | Description |
|---|---|---|
| `TICKERS` | `SPY,QQQ` | Comma-separated underlyings |
| `DRY_RUN` | `true` | Log plans; don't submit orders |
| `MODE` | `dry_run` | `live` or `dry_run` |
| `MAX_RISK_PCT` | `0.02` | Max loss per trade as fraction of equity |
| `MIN_CREDIT_RATIO` | `0.33` | Minimum credit / spread width (static mode only) |
| `MAX_DELTA` | `0.20` | Max abs delta of sold strike |
| `EDGE_BUFFER` | `0.10` | Adaptive C/W margin over breakeven; required `C/W = |Δshort| × (1 + edge_buffer)` |
| `SCAN_MODE` | `adaptive` | `adaptive` activates the scanner + Δ-aware C/W floor; `static` falls back to legacy planner |
| `DAILY_DRAWDOWN_LIMIT` | `0.05` | Kill process if equity drops > N % in one day |
| `MAX_BUYING_POWER_PCT` | `0.80` | Enter Liquidation Mode at > N % BP used |
| `LIQUIDITY_MAX_SPREAD` | `0.05` | Absolute floor of underlying bid/ask gate ($) |
| `LIQUIDITY_BPS_OF_MID` | `0.0005` | Slope of bid/ask gate (5 bps × mid). Effective threshold = `max(LIQUIDITY_MAX_SPREAD, LIQUIDITY_BPS_OF_MID × mid)` |
| `STALE_SPREAD_PCT` | `0.01` | Stale-quote threshold; soft-passes with WARNING |
| `FORCE_MARKET_OPEN` | `false` | Bypass market-hours check (paper testing) |
| `ALPACA_STOCKS_FEED` | `iex` | `iex` (free) or `sip` (paid SIP) |
| `ALPACA_OPTIONS_FEED` | `indicative` | `indicative` (free, 15-min delayed) or `opra` (paid real-time) |
| `MARKET_DATA_PROVIDER` | `alpaca` | Global default provider — `alpaca`, `schwab`, or `yahoo`. See "Multi-provider market data" below. |
| `MARKET_DATA_PROVIDER_LIVE` | _(inherits)_ | Per-surface override for the trading agent loop |
| `MARKET_DATA_PROVIDER_WATCHLIST` | _(inherits)_ | Per-surface override for the Watchlist tab + market-open badge |
| `MARKET_DATA_PROVIDER_BACKTEST` | _(inherits)_ | Reserved — backtester currently uses its own historical port |
| `SCHWAB_CLIENT_ID` / `_SECRET` / `_REDIRECT_URI` | _(empty)_ | Required when any surface is set to `schwab`. See setup walkthrough. |
| `SCHWAB_TOKEN_PATH` | `~/.schwab_tokens.json` | Where the OAuth helper persists access + refresh tokens |
| `LOG_MAX_BYTES` | `10485760` | Per-file log rotation threshold (10 MB) |
| `LOG_BACKUP_COUNT` | `7` | Rollover files retained |

#### Streamlit dashboard

| Variable | Default | Description |
|---|---|---|
| `LIVE_MONITOR_REFRESH_SECS` | `3` | Live Monitor fragment auto-refresh tick (was 30 pre-watchdog) |
| `BROKER_STATE_TTL_SECS` | `30` | TTL on cached Alpaca account/positions/clock fetches |
| `WATCHLIST_REFRESH_SECS` | `60` | TTL on per-ticker multi-timeframe classification cache (Watchlist tab) |
| `WATCHDOG_DISABLE` | `0` | Set to `1` to disable the journal `Observer` (cache keys go to mtime+size only) |
| `WATCHDOG_FORCE_POLLING` | `0` | Set to `1` for `PollingObserver` on NFS / network mounts where inotify is unavailable |

#### Core Intelligence (analyst)

| Variable | Default | Description |
|---|---|---|
| `LLM_ENABLED` | `false` | Master switch for the LLM intelligence layer |
| `LLM_PROVIDER` | `ollama` | `ollama`, `lmstudio`, `openai`, `anthropic` |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API endpoint |
| `LLM_MODEL` | `mistral` | Primary reasoning model |
| `LLM_EMBEDDING_MODEL` | `nomic-embed-text` | Embeddings model for RAG |
| `LLM_TEMPERATURE` | `0.3` | Analyst sampling temperature |
| `LLM_MAX_TOKENS` | `2048` | Analyst response cap |
| `LLM_TIMEOUT` | `60` | Analyst HTTP timeout (s) |
| `TRADE_JOURNAL_DIR` | `trade_journal` | Trade lifecycle logs |
| `KNOWLEDGE_BASE_DIR` | `knowledge_base` | RAG vector store |

All three LLM callers (analyst, FinGPT, verifier) share the same `make_llm_client(role, cfg)` factory so their parameters live in one place.

#### Sentiment pipeline

| Variable | Default | Description |
|---|---|---|
| `FINGPT_ENABLED` | `false` | Enable sentiment pipeline |
| `FINGPT_MODEL` | `qwen2.5-trading` | Ollama model for FinGPT analysis |
| `FINGPT_NEWS_LIMIT` | `10` | Max headlines from yfinance fallback |
| `FINGPT_CACHE_TTL` | `300` | FinGPT in-process cache TTL (s) |
| `FINGPT_TEMPERATURE` | `0.1` | Keep deterministic |
| `FINGPT_MAX_TOKENS` | `512` | Short JSON cap |
| `FINGPT_TIMEOUT` | `45` | HTTP timeout (s) |
| `NEWS_SOURCES` | `yahoo,sec_edgar,fed_rss` | Comma-separated source keys |
| `NEWS_LOOKBACK_HOURS` | `24` | How far back to fetch news |
| `NEWS_MAX_ITEMS_PER_SOURCE` | `20` | Items per source per cycle |
| `NEWS_CACHE_TTL` | `240` | Per-`(ticker, source)` cache TTL (s) |
| `NEWS_SOURCE_WEIGHTS_JSON` | _(empty)_ | JSON object overriding `DEFAULT_SOURCE_WEIGHTS` |
| `REDDIT_CLIENT_ID` / `_SECRET` | _(empty)_ | PRAW credentials — enables all Reddit sources |
| `REDDIT_USER_AGENT` | `TradingAgent/1.0` | PRAW user agent |
| `TWITTER_BEARER_TOKEN` | _(empty)_ | Twitter API v2 Bearer token |
| `VERIFIER_ENABLED` | `false` | Enable reasoning-model verification |
| `VERIFIER_PROVIDER` | `ollama` | `ollama` (local) or `anthropic` (cloud) |
| `VERIFIER_MODEL` | `qwq:32b` | Verifier model |
| `VERIFIER_API_KEY` | _(empty)_ | Anthropic key when `VERIFIER_PROVIDER=anthropic` |
| `VERIFIER_TEMPERATURE` | `0.15` | Low but non-zero — reasoning models benefit |
| `VERIFIER_MAX_TOKENS` | `2048` | Response cap |
| `VERIFIER_TIMEOUT` | `90` | HTTP timeout (s) — reasoning is slower |
| `EARNINGS_CALENDAR_ENABLED` | `true` | Tier-0 short-circuit |
| `EARNINGS_CALENDAR_LOOKAHEAD_DAYS` | `7` | Tier-0 firing window |
| `EARNINGS_CALENDAR_REFRESH_HOURS` | `12` | Per-ticker earnings cache freshness |
| `SENTIMENT_HASH_CACHE_SIZE` | `32` | Tier-1 LRU cap; TTL auto-scales to `max(NEWS_CACHE_TTL, FINGPT_CACHE_TTL)` |

### 3. Run

```bash
python -m trading_agent.agent              # paper trading
python -m trading_agent.agent --dry-run    # log plans, no orders
python -m trading_agent.agent --env /path/to/.env

# tests
python run_tests.py                        # full repo suite
pytest tests/ -v                           # equivalent
```

After-hours: exits cleanly before 9:25 AM ET, after 4:05 PM ET, and on weekends. Override with `FORCE_MARKET_OPEN=true`.

---

## Data Sources

| Source | Purpose | Auth |
|---|---|---|
| Yahoo Finance | Regime detection (SMA / RSI / BB), backtest history, `^VIX` | None |
| Alpaca Market Data | Real-time snapshots, option chains, Greeks | API key |
| Alpaca Paper API | Order execution, account equity, market clock | API key |
| SEC EDGAR | 8-K / 10-Q (sentiment pipeline) | None |
| Federal Reserve RSS | FOMC statements (sentiment pipeline) | None |
| Reddit | r/wsb, r/stocks, r/options (sentiment pipeline) | PRAW |
| Twitter / X | Cashtag stream (sentiment pipeline) | Bearer token |

---

## Project Structure

```
trading-agent/
├── .env                              # API keys + config (not committed)
├── requirements.txt
├── README.md
├── architecture_diagram.html
├── setup_intelligence.sh             # Ollama setup helper
├── run_tests.py
│
├── trading_agent/
│   ├── agent.py                      # Orchestrator: two-stage cycle, timeout guard, sentiment pipeline
│   ├── config.py                     # AppConfig + IntelligenceConfig
│   ├── ports.py                      # Hexagonal protocols: MarketDataPort, BrokerPort, SentimentReadout
│   ├── market_profile.py             # MarketProfile (TZ, session bounds, trading-day oracle)
│   ├── logger_setup.py
│   │
│   │   # ── Core Phases ──
│   ├── market_data.py                # Phase I — yfinance + Alpaca (TTL cache, parallel, split timeouts, fetch_intraday_bars)
│   ├── regime.py                     # Phase II — SMA / RSI / Bollinger / VIX-z
│   ├── multi_tf_regime.py            # Multi-timeframe regime wrapper (reuses _determine_regime, no shadow scorer)
│   ├── strategy.py                   # Phase III — strike selection, nearest-Friday DTE
│   ├── chain_scanner.py              # Phase III — adaptive (Δ × DTE × width) sweep
│   ├── decision_engine.py            # Pure scoring entrypoint shared by live + backtest
│   ├── calendar_utils.py             # NYSE trading-day oracle (lazy lru_cache)
│   ├── strategy_presets.py           # Conservative / Balanced / Aggressive presets
│   ├── risk_manager.py               # Phase IV — 8-guardrail validator
│   ├── executor.py                   # Phase VI — mleg execution + HTML report
│   ├── trade_plan_report.py
│   ├── watchlist_store.py            # Persistent JSON watchlist (atomic writes, RLock for nested CRUD)
│   │
│   │   # ── Position Management ──
│   ├── position_monitor.py
│   ├── order_tracker.py
│   │
│   │   # ── Backtesting ──
│   ├── backtest/
│   │   ├── black_scholes.py         # Pure stdlib BS pricing + Greeks (no scipy)
│   │   ├── clock.py                 # NYSE-calendar iterator (intraday vs daily cadence)
│   │   ├── historical_port.py       # Cursor-bound MarketDataProvider wrapper (LookaheadError)
│   │   ├── synthetic_chain.py       # Builds ChainSlice from (spot, σ, strike grid)
│   │   ├── account.py               # SimAccount cash + open-market-value ledger
│   │   ├── sim_position.py          # Open-spread bookkeeping + VIX-proxy IV scaling
│   │   ├── cycle.py                 # run_one_cycle: PERCEIVE → CLASSIFY → PLAN → RISK → EXECUTE
│   │   └── runner.py                # BacktestRunner — drives the clock, emits BacktestResult
│   │
│   │   # ── Core Intelligence ──
│   ├── journal_kb.py                 # Always-on signal logger (live | backtest split)
│   ├── trade_journal.py              # Full-lifecycle trade logging
│   ├── knowledge_base.py             # File-based RAG vector store
│   ├── llm_client.py                 # OpenAI-compatible client + make_llm_client(role) factory
│   ├── llm_analyst.py                # Pre/post-trade LLM analysis
│   ├── fine_tuning.py                # Training data export (JSONL / Alpaca / DPO)
│   │
│   │   # ── Sentiment Pipeline ──
│   ├── sentiment_pipeline.py         # Tier-0/1/2 facade, cycle-scoped pool
│   ├── earnings_calendar.py          # Tier-0
│   ├── sentiment_cache.py            # Tier-1
│   ├── news_aggregator.py            # Tier-2 — NewsItem + NewsAggregator
│   ├── fingpt_analyser.py            # Tier-2 — FinGPT specialist
│   ├── sentiment_verifier.py         # Tier-2 — Reasoning verifier
│   │
│   └── streamlit/
│       ├── app.py                    # 4-tab dashboard entrypoint
│       ├── live_monitor.py           # Live tab — broker-gated, watchdog-driven refresh
│       ├── backtest_ui.py            # Backtest tab — thin shim around trading_agent.backtest.BacktestRunner
│       ├── llm_extension.py          # LLM tab — RAG over signals_live.jsonl
│       ├── watchlist_ui.py           # Watchlist tab — multi-tf regime table + macro strip
│       ├── watchlist_chart.py        # Watchlist tab — 4-row Plotly chart, pure-pandas indicators
│       ├── file_watcher.py           # Per-process Observer + version counters
│       └── components.py
│
├── trade_journal/                    # Auto-created
│   ├── trades/
│   ├── index.json
│   ├── stats.json
│   ├── signals_live.jsonl            # Always-on live-mode signal log
│   ├── signals_backtest.jsonl        # Backtest-mode signal log (deliberately separate)
│   └── signals_*.md                  # Human-readable mirrors
│
├── trade_plans/                      # Per-ticker persistent trade-plan files
├── knowledge_base/                   # RAG vector store (LLM layer) + watchlist.json (Watchlist tab)
└── logs/
```

---

## Signal Journal Format

Live cycles append one JSON object per line to `trade_journal/signals_live.jsonl`; backtests append to `trade_journal/signals_backtest.jsonl`. The two files use identical schema. When the sentiment pipeline is active each record also carries `fingpt_sentiment`, `fingpt_event_risk`, `fingpt_recommendation`, `fingpt_agreement`, `fingpt_hallucination_flags`, and `fingpt_verified_by`.

**Action values:** `dry_run`, `submitted`, `rejected`, `closed`, `close_failed`, `dry_run_close`, `warning`, `skipped_by_llm`, `skipped_existing`, `skipped_liquidation_mode`, `skipped_bias`, `skipped_rsi_gate`, `skipped_defense_first`, `skipped`, `error`, `cycle_timeout`, `daily_drawdown_circuit_breaker`. Full schema reference: [`docs/skills/19_journal_schema.md`](docs/skills/19_journal_schema.md).

**Material actions bypass the dedup gate** — `submitted`, `closed`, `close_failed`, `error`, `warning`, `dry_run`, `dry_run_close` — so successive cycles never silently suppress them. Rejection-spam actions (`rejected`, `skipped_*`) get per-ticker dedup with periodic heartbeat rows (every 12 cycles by default).

**`close_failed`** (added 2026-05-06) tags a partial-fill close where one or more legs were rejected by Alpaca (PDT, uncovered, insufficient buying power); the position is **still open on the broker**. After 3 consecutive partial fills on a single ticker the agent enters a 60-min cooldown and the dashboard's Close Failures Today panel renders a 🚨 manual-intervention banner. See [`docs/skills/17_close_failure_and_cooldown.md`](docs/skills/17_close_failure_and_cooldown.md).

**`warning`** (added 2026-05-06) is emitted by `JournalKB.log_warning(source=...)` when a vendor retry budget is exhausted — order submission timed out N times, Schwab OAuth refresh failed N times, position fetch failed N times. Carries the `client_order_id` (when applicable) so an operator can search the broker UI to confirm whether any attempt landed.

### `scan_results` block (adaptive scanner only)

When the active preset is in adaptive mode, every `plan()` invocation that runs the chain scanner also writes a `raw_signal.scan_results` block — the single source of truth for *why* the scanner picked / rejected each ticker (populated even when zero candidates pass).

```jsonc
"scan_results": {
  "scan_mode":        "adaptive",
  "side":             "bull_put",
  "edge_buffer":      0.10,
  "min_pop":          0.55,
  "candidates_total": 0,
  "selected_index":   -1,
  "top_k":            [],
  "diagnostics": {
    "grid_points_total":    16,
    "grid_points_priced":   12,
    "expirations_resolved": 4,
    "rejects_by_reason":    {"cw_below_floor": 11, "no_long_contract": 1},
    "best_near_miss": {
      "expiration":   "2026-05-15",
      "dte":          14,
      "short_strike": 590.0,
      "long_strike":  585.0,
      "short_delta":  -0.20,
      "credit":       0.95,
      "width":        5.0,
      "cw_ratio":     0.19,
      "cw_floor":     0.22,
      "pop":          0.80,
      "ev":          -0.04
    }
  }
}
```

**How to read it.** When `candidates_total == 0` the answer to *"why didn't the scanner trade?"* is in two fields: `rejects_by_reason` (which filter dominated) and `best_near_miss` (how close the chain came). If `cw_ratio` is close to `cw_floor`, one click of `EDGE_BUFFER` toward zero unblocks the trade; if they're far apart, the chain isn't paying enough — wait or skip.

**Reject-reason taxonomy** (stable string keys):

| Key | Meaning |
|---|---|
| `no_chain` | `fetch_option_chain()` returned empty for that expiration |
| `no_short_contract` | No contract matches the target Δ |
| `no_long_contract` | No protective strike at requested width (grid too sparse) |
| `non_positive_width` | Snapped width came out as 0 or negative |
| `dte_non_positive` | Resolved expiration is today or earlier |
| `pop_below_min` | `1 − |Δshort| < min_pop` (Δ-target grid too aggressive) |
| `credit_non_positive` | `bid_short − ask_long ≤ 0` |
| `credit_ge_width` | Credit ≥ width — would be a debit, not a credit |
| `cw_below_floor` | `C/W < |Δ| × (1 + edge_buffer)` — most common in thin-premium regimes |

---

## Spec-Driven Development (SDD)

This project follows a Spec-Driven Development workflow: every architectural concept is documented as an atomic "skill" file before the corresponding code is shipped, and CI verifies the spec and code never drift apart. Five tools — three CI gates and two workflow helpers — enforce this mechanically.

**Where the specs live.** Each concept is one file under `docs/skills/NN_short_name.md` (currently 22 skills). The format is fixed: theory → math → reference Python (quoted verbatim from source) → edge cases → cross-references → footer dated to the last verification. `CLAUDE.md` documents the SDLC contract; `CONTRIBUTING.md` operationalises it into commands.

**The five CI gates** (`.github/workflows/ci.yml` runs all of them on every push):

| Gate | What it asserts |
|---|---|
| `scan_invariant_check.py` | AST-level architectural invariants (C/W formula parity, no shadow scorers, backtest wires through `decide()`) |
| `scan_skill_quotes_match.py` | Every skill's §3 first code-line still appears verbatim in the cited source. Catches signature drift. |
| `scan_skill_freshness.py` | Every skill's footer date ≥ the git-mtime of every cited source file. Catches forgotten footer re-stamps. |
| `build_traceability.py` + `git diff --exit-code` | The skill ↔ source ↔ test coverage matrix at `docs/traceability.md` is in sync with the repo. |
| `pytest tests/conformance/` | Documented math/behavior is asserted against the live implementation for skills 03, 04, 17, 29 (back-fill in progress). |

**The two workflow helpers** (run on demand, not gates):

| Script | Use case |
|---|---|
| `scripts/specify_new_feature.py NAME` | Scaffolds `docs/skills/NN_NAME.md` + `tests/conformance/test_skill_NN_NAME.py` from the template. Forces spec-first (the conformance test fails until the spec is authored). |
| `scripts/checks/check_skill_updates_for_diff.py` | Pre-commit helper. Reads the current diff, prints a checklist of which skills cite the changed source files. Wire into `.git/hooks/pre-commit` to enforce locally. |

**The coverage map.** `docs/traceability.md` is auto-generated and committed. It surfaces orphan source files (paths with no skill) and orphan skills (skills with no conformance test) — useful both for back-fill prioritisation and for sanity-checking that new code didn't escape the spec layer.

**The principle.** Specs are the source of truth. When the code says one thing and the spec says another, the question is "did the spec change?" — not "is the spec wrong?". This inversion is what makes the whole pipeline work, and what catches the kind of subtle drift that produced the 2026-05-15 GLD dashboard incident (see [skill 28's §4](docs/skills/28_position_monitor_spread_grouping.md) for that worked example).

For the rationale, the war story, and the full toolchain in one place, see `CONTRIBUTING.md` Step 5.

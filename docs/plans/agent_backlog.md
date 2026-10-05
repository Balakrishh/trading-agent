# Agent backlog

Open work for the trading agent, in priority order within each group. Each code item follows the SDLC in `docs/skills/00_sdlc_and_conventions.md`: skill doc first, tests for every §4 edge case, `pytest tests/` and `scripts/checks/*` green.

*Created 2026-09-30.*

---

## 0. This week — observe before building (2026-09-30 → 2026-10-02)

The agent runs unchanged all week on the paper account. The goal is to learn from the journal, not to add features.

**Scheduled (set up 2026-09-30):**

| When | What | Output | Lives in |
|---|---|---|---|
| Mon–Fri 16:15 ET | Daily reviewer (skill 56), `--no-telegram` | `daily_reviews/YYYY-MM-DD.json`, digest in `/tmp/trading-agent.daily-reviewer.out.log` | launchd `com.trading-agent.daily-reviewer` (permanent) |
| Mon–Fri 16:22 ET | Claude daily journal review (MCP tools, read-only) | Dated section in `docs/plans/journal_learnings.md` | Claude Code session job — stops if the session closes; expires 2026-10-07 |
| Fri 2026-10-02 16:47 ET | Claude weekly review via `journal-analyst` | "Week 1 results" in `journal_learnings.md`; §0 ticked; next-build recommendation | Claude Code session job (one-shot) |

- [x] Fix: `daily_reviewer --date` reviewed the current day instead of the requested one (`JournalReader(as_of=...)`, 2026-09-30).
- [ ] The daily reviewer's LLM step needs Ollama running at `LLM_BASE_URL` (not running on 2026-09-30, so reviews contain stats only). Start Ollama, or accept stats-only reviews since the 16:22 Claude review covers the analysis.
- [ ] Add Telegram tokens, then drop `--no-telegram` from the installed plist to get the digest on your phone.

- [x] **Every morning:** `/portfolio` — open positions, fills, alerts, `cycle_minute_count_today` > 0 after 09:45 ET.
- [x] **Every evening:** `/review` (skill 56) — reject-reason histogram, exits, errors.
- [x] **Friday:** weekly review with the `journal-analyst` subagent. Record the answers below.
  - Trades opened / closed, win rate, realized P&L.
  - Top 5 reject reasons, and whether each is correct behaviour or a bug.
  - Slippage: fill price vs mid at entry and exit.
  - Any exit that fired on noise (closed and the position would have recovered).
  - Wheel: screen results at 09:45 ET each day; did any pick survive the earnings gate?
  - Open SPY iron condor (exp 2026-10-23): outcome and which rule acted.
- [x] Decide the next build from what the week shows, not from this list alone.

---

## 1. Correction readiness — market risk-state overlay

Motivation: the agent judges each ticker alone. In a broad 10–20 % correction, single tickers can still read "sideways" and receive bull puts or condors, and the Wheel keeps selling puts into a falling market.

- [x] **Risk-state classifier (new skill).** Each cycle, rate the market Normal / Caution / Defensive / Capitulation / Recovery from:
  - SPY vs 50- and 200-day averages
  - VIX level and z-score
  - Breadth: share of watchlist ETFs above their 50-day average
  - VIX / VIX3M term structure (needs the data source below)
- [x] **Gates by state:** position-size multiplier (1.0 / 0.5 / 0.25 / 0), allowed strategies per state (Defensive = bear calls only), journal the state every cycle, show it in `/portfolio`.
- [x] **Wheel pause:** `wheel_screen` refuses new cash-secured puts in Caution and Defensive; covered calls unaffected. Resume in Recovery at half size.
- [x] **VIX3M data source** for term structure (Schwab `$VIX3M.X` or equivalent).
- [x] **Hedge suggestion in `/portfolio`:** in Defensive, propose an SPY put spread sized to beta-weighted holdings; staged through `/propose`, never automatic.
- [x] **Validate thresholds** on 2020, 2022 and 2025 SPY/VIX history. The backtester's synthetic option pricing cannot validate the option P&L itself.

## 2. Wheel improvements

- [x] Bid/ask width gate in `_score_cash_secured_put` / covered calls, reusing the spread scanner's `max_leg_spread_pct_mid` (5 %) and `max_leg_spread_cents`. Found 2026-09-30 pre-market: BMY $60P 0.40/1.11, VZ $44P 0.32/0.85 — mid-based yields of 20 %+ on untradeable quotes.
- [x] `wheel_screen` treats a JSON-array string (`'["VZ"]'`, as some MCP clients send lists) as one ticker → `missing:fundamentals`. Parse JSON-looking strings before the comma split. Found 2026-09-30 during `/propose VZ`.
- [x] Spread entries have the same estimate-vs-fill gap (SPY IC: plan 0.49, fill 0.48). Spread orders are fire-and-forget, so record the fill when the order tracker (skill 26/`order_tracker.py`) sees it filled, reusing `OrderExecutor._record_fill_credit`. *(Done 2026-10-05 — `fill_reconciler.py`, skill 60.)*
- [x] 200-day trend filter: no cash-secured puts on a stock below its 200-day average. *(Done 2026-10-05 — fails closed when unknown; skill 40.)*
- [x] One pick per sector (banks and telecom cluster today). *(Done 2026-10-05 — `csp_max_per_sector`, skill 40.)*
- [x] Wire `csp_*` / `cc_*` tunables into `PresetConfig` (skill 40 §3.3): `to_summary_line()`, Streamlit Strategy-Profile panel, `agent.py`. *(Done 2026-10-05.)*
- [x] Fundamentals: map `short_int_to_float` to `None` when Schwab returns 0.0.

## 3. Strategy research

- [x] Shadow mode: log a realized-volatility probability of profit next to the delta-based one on every candidate, no trading effect. Compare after 4+ weeks of paper data. *(Done 2026-10-05 — `pop_delta` / `pop_rv` / `rv_20d` on every journaled plan; skill 60.)*
- [ ] Quality + momentum stock sleeve with ATR trailing stops (after the Wheel has a track record).

## 4. Hygiene

- [x] **`/triage` must use the agent's valuation, not Alpaca's indicative feed.** 2026-09-30 10:15 ET: triage priced the SPY IC legs from Alpaca's free `indicative` options snapshots and reported −$392 ("$16 from stop loss, close now"); the agent's Schwab-mid re-mark showed −$112 to −$136 and HOLD. Fix: add option-symbol support to MCP `get_quote` (data server `/quotes`), or expose the monitor's per-position mid P&L via a read-only tool, and update skill 50 so triage never mixes quote sources.
- [x] **Correct the cycle interval in docs.** Logs show a monitor cycle roughly every 75 s (10:21:48, 10:23:05, 10:24:21 …; supervisor restarts the agent ~60 s after each cycle), not every 5 minutes. The 3-cycle exit debounce is therefore ~4 min, not 15. Fix the Trading Day Flow artifact, skills 44 / 57 / PROJECT_MANIFEST, and decide whether the cadence is intended (API load, journal volume, debounce length).
- [x] **Subagent `tools:` lists use bare MCP names** (`list_recent_trades` …), so `journal-analyst` (and likely `scanner-runner`, `risk-reviewer`) start with zero tools and Claude Code refuses to spawn them (found 2026-10-02 in the weekly review). Use `mcp__trading-agent__<name>`; check `scan_subagent_allowlists.py` accepts the prefixed form.
- [x] *(2026-10-05: no longer reproducible — 20/20 pass outside market hours in a clean worktree and with the local `.env`; watch CI on evening pushes.)* `tests/test_after_hours_shutdown.py` SIGKILLs pytest (exit 137) when run after market close — **CI risk for evening pushes**. `test_after_close_calls_graceful_exit_0` patches `_is_within_market_hours` and `graceful_exit`, so `run_cycle` continues past the mocked exit; suspects: `_maybe_send_eod_summary()` (agent.py:727, real clock) or the real `TradingAgent` picking up the local `.env`. Reproduce after 16:05 ET, in a clean worktree with `env -i`, to see whether CI is affected at all.
- [x] CI red since 2026-09-29 (fixed 2026-10-01): skill-freshness failures (footers stamped before commit / in local time vs CI's UTC), stale traceability matrix, missing fastapi/uvicorn/httpx2 in requirements, and four tests stale since the 29bb020 cache-off default. Before pushing: commit, then run `TZ=UTC python scripts/checks/scan_skill_freshness.py` and the CI steps in a clean worktree.
- [x] "(no reason recorded)" rejects were valid plans vetoed by RiskManager (reason only in `checks_failed`); the writer now records `risk: <first failed check>` and the reader falls back to `checks_failed` (2026-10-05).
- [x] The "23 pre-existing test failures" are local-environment only: a clean worktree with `env -i` (no `.env`, UTC) gives 0 failures (2026-10-01). Run local checks that way before trusting a failure count.
- [ ] Delete merged branches: `fix/iron-condor-wing-width`, `fix/mcp-dataserver-env-key`, `feat/wheel-lifecycle`.

## 5. Operator actions (not code)

- [ ] Add Telegram bot tokens to `.env` so outages and errors page you.
- [x] Reconcile the July XLE iron condor P&L from the old paper account's order history. *(2026-10-05: history unreachable (account replaced; strikes and size not in the journal, trade plan rotated) → `position_reconciled` row with P&L unknown via `trading_agent/journal_reconcile.py`; if you recover the P&L from the old account, append the real close with `--realized-pl`.)*
- [ ] Refresh the June holdings paste in the dashboard.
- [ ] Confirm the rotated data-server token is the only valid one.

## 6. Full playbook — trade every market condition (plan agreed 2026-10-01; start Friday)

**Why:** two days of paper trading opened 0 new spreads (≈3,100 "no positive-EV" rejects). Causes: (1) the agent only *sells* premium, which is correctly idle when IV Rank is low (3–36 on 2026-09-30/10-01); (2) delta-as-probability assumes zero edge, so EV rarely clears after bid/ask; (3) oversold + high-IV setups — historically the best time to sell puts — are skipped rather than traded the other way. Goal: a deterministic regime → strategy map that always has an appropriate, small, defined-risk trade, measured per playbook.

| Trend \ Volatility | High IV (Rank > 50) — sell premium | Low IV (Rank < 30) — buy premium / directional |
|---|---|---|
| Uptrend | ✅ bull put, ✅ CSP (Wheel) | ❌ call debit spread, ❌ PMCC |
| Downtrend | ✅ bear call | ❌ put debit spread |
| Sideways | ✅ iron condor / butterfly | ❌ calendar spread |
| Oversold / IV spike | ❌ bounce bull put (after stabilisation) | — |
| Capitulation | ✅ block entries, ❌ hedge holdings | — |

Order (paper only, one at a time, each with skill doc + tests + scoring in `decision_engine.py`):

- [x] **6.1 Fill-realistic pricing.** *(Done 2026-10-05: `PresetConfig.fill_model`, default natural; scorers, Wheel, iron butterfly, profit target.)* Compute EV, credit floors and screen yields at natural prices (sell at bid, buy at ask) instead of mid in all scorers — paper filled only at natural on 2026-09-30 (VZ entry) and 2026-10-01 (SPY close). Small change; makes every other number honest.
- [x] **6.2 Regime-state classifier + playbook table.** Per cycle: trend × IV Rank × RSI extreme → one playbook from the table above (extends §1's market risk-state overlay). Fully deterministic; journal the chosen playbook on every row. *(Done 2026-10-05, Phase 3 — `trading_agent/market_state.py`, skill 58: `playbook`/`playbook_implemented` on every signal row; market state gates entries; validation via `scripts/research/validate_market_state.py`.)*
- [x] **6.3 Debit spreads** for trend + low IV (call debit in uptrend, put debit in downtrend). Biggest trade-count gain in low-IV tapes. *(Done 2026-10-05, Phase 4 — skill 59; fallback after the credit plan in low volatility.)*
- [x] **6.4 Bounce bull put:** bearish + RSI < 30 + IV Rank elevated + stabilisation trigger (e.g. close reclaims the 5-day high) → bull put below the recent low. *(Done 2026-10-05, Phase 4 — trigger: price > N-day high close; short strike below the 2N-day low; replaces the bear call when RSI < 30.)*
- [x] **6.5 Calendar spreads** for sideways + low IV. *(Done 2026-10-05, Phase 4 — Black-Scholes model over the near-expiry distribution.)*
- [x] **6.6 Per-playbook scorecard:** tag every trade with regime + playbook; after 20–30 trades per playbook report win rate, avg win/loss, slippage vs mid, EV realised; use it to size (1–3 % risk by track record) or disable a playbook. *(Done 2026-10-05 — MCP `get_playbook_scorecard`; sizing advisory; skill 60.)*
- [x] **6.7 More shots at the same risk:** add liquid single names (via the Wheel fundamentals screen), ladder weekly expirations, smaller size per trade. *(Done 2026-10-05 — laddering: 2/ticker, later day, ≥7-day expiry gap (skill 37); per-trade risk 3 % → 2 %; META + AMZN added — the only large caps of 20 checked that pass both the quality screen and the per-leg quote gate (AAPL/MSFT/GOOGL/JPM/V/MA/XOM… had 7–41 % median option spreads mid-day; re-check later).)*
- [x] **6.8 Start now, in parallel:** the §3 realized-vol POP shadow log (no trading effect) so evidence accumulates while 6.1–6.5 are built. *(Done 2026-10-05 with the §3 shadow log.)*

**Expectation, stated plainly:** with the full playbook the agent might move from ~0 to roughly 5–15 smaller trades a week across conditions. Whether that is profitable is for the 6.6 scorecard to show after 1–2 months of paper data — no playbook makes money in every market; the aim is always having a fitting, small, measured trade.


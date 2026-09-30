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

- [ ] **Every morning:** `/portfolio` — open positions, fills, alerts, `cycle_minute_count_today` > 0 after 09:45 ET.
- [ ] **Every evening:** `/review` (skill 56) — reject-reason histogram, exits, errors.
- [ ] **Friday:** weekly review with the `journal-analyst` subagent. Record the answers below.
  - Trades opened / closed, win rate, realized P&L.
  - Top 5 reject reasons, and whether each is correct behaviour or a bug.
  - Slippage: fill price vs mid at entry and exit.
  - Any exit that fired on noise (closed and the position would have recovered).
  - Wheel: screen results at 09:45 ET each day; did any pick survive the earnings gate?
  - Open SPY iron condor (exp 2026-10-23): outcome and which rule acted.
- [ ] Decide the next build from what the week shows, not from this list alone.

---

## 1. Correction readiness — market risk-state overlay

Motivation: the agent judges each ticker alone. In a broad 10–20 % correction, single tickers can still read "sideways" and receive bull puts or condors, and the Wheel keeps selling puts into a falling market.

- [ ] **Risk-state classifier (new skill).** Each cycle, rate the market Normal / Caution / Defensive / Capitulation / Recovery from:
  - SPY vs 50- and 200-day averages
  - VIX level and z-score
  - Breadth: share of watchlist ETFs above their 50-day average
  - VIX / VIX3M term structure (needs the data source below)
- [ ] **Gates by state:** position-size multiplier (1.0 / 0.5 / 0.25 / 0), allowed strategies per state (Defensive = bear calls only), journal the state every cycle, show it in `/portfolio`.
- [ ] **Wheel pause:** `wheel_screen` refuses new cash-secured puts in Caution and Defensive; covered calls unaffected. Resume in Recovery at half size.
- [ ] **VIX3M data source** for term structure (Schwab `$VIX3M.X` or equivalent).
- [ ] **Hedge suggestion in `/portfolio`:** in Defensive, propose an SPY put spread sized to beta-weighted holdings; staged through `/propose`, never automatic.
- [ ] **Validate thresholds** on 2020, 2022 and 2025 SPY/VIX history. The backtester's synthetic option pricing cannot validate the option P&L itself.

## 2. Wheel improvements

- [x] Bid/ask width gate in `_score_cash_secured_put` / covered calls, reusing the spread scanner's `max_leg_spread_pct_mid` (5 %) and `max_leg_spread_cents`. Found 2026-09-30 pre-market: BMY $60P 0.40/1.11, VZ $44P 0.32/0.85 — mid-based yields of 20 %+ on untradeable quotes.
- [ ] `wheel_screen` treats a JSON-array string (`'["VZ"]'`, as some MCP clients send lists) as one ticker → `missing:fundamentals`. Parse JSON-looking strings before the comma split. Found 2026-09-30 during `/propose VZ`.
- [ ] 200-day trend filter: no cash-secured puts on a stock below its 200-day average.
- [ ] One pick per sector (banks and telecom cluster today).
- [ ] Wire `csp_*` / `cc_*` tunables into `PresetConfig` (skill 40 §3.3): `to_summary_line()`, Streamlit Strategy-Profile panel, `agent.py`.
- [ ] Fundamentals: map `short_int_to_float` to `None` when Schwab returns 0.0.

## 3. Strategy research

- [ ] Shadow mode: log a realized-volatility probability of profit next to the delta-based one on every candidate, no trading effect. Compare after 4+ weeks of paper data.
- [ ] Quality + momentum stock sleeve with ATR trailing stops (after the Wheel has a track record).

## 4. Hygiene

- [ ] `tests/test_after_hours_shutdown.py` SIGKILLs pytest when run after market close.
- [ ] Triage the 23 pre-existing test failures (mostly environment-dependent).
- [ ] Delete merged branches: `fix/iron-condor-wing-width`, `fix/mcp-dataserver-env-key`, `feat/wheel-lifecycle`.

## 5. Operator actions (not code)

- [ ] Add Telegram bot tokens to `.env` so outages and errors page you.
- [ ] Reconcile the July XLE iron condor P&L from the old paper account's order history.
- [ ] Refresh the June holdings paste in the dashboard.
- [ ] Confirm the rotated data-server token is the only valid one.

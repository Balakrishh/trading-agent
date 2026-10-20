# Agent backlog

Open work for the trading agent, in priority order within each group. Each code item follows the SDLC in `docs/skills/00_sdlc_and_conventions.md`: skill doc first, tests for every §4 edge case, `pytest tests/` and `scripts/checks/*` green.

*Created 2026-09-30.*

---

## Plan of record (2026-10-13) — read this first

All open items below, grouped by theme and ordered. Older sections keep the history and detail; IDs here point to them.

**Why no trades (2026-10-13 analysis):** the scorer's win probability is the option delta, i.e. the market's own odds, so a spread's expected value is ≈ 0 before costs and slightly negative after the bid/ask — near misses of −6¢ to −7.5¢/share every day. Edge has to come from an independent probability (realized volatility vs implied: XLF condor POP 46.5 % by delta vs 71.4 % by 20-day realized vol), from paying less to get in (C), and from gates that stop blocking trades the market supports (E). B1 + B2 produce the evidence before anything loosens.

### A. Keep it running
- [x] **A1 Schwab 7-day clock from the login, not the last refresh; Telegram reminder ≤ 2 days before** (skill 16 §4). *Done 2026-10-13.*
- [x] **A2 Telegram HTML escaping + plain-text resend** — the 10/13 EOD summary failed twice on a raw `<` (skill 32 §4). *Done 2026-10-13.*
- [x] **A3 `.mcp.json` prefers `myenv`** — after a reboot the bare `python` was the system Python and `/mcp` failed with `CONNECTION_CLOSED` (skill 48). *Done 2026-10-13.*
- [x] **A4 MCP `get_quote` for option symbols** via `/quotes` (skill 48). *Done 2026-10-13.*
- [x] **A5 Liveness watchdog** — Telegram page when no cycle for 10 min in session (from 09:40 ET) or the data server is down (skill 64). *Done 2026-10-13; enable `trading-agent-liveness.timer` on deploy.*
- [x] **A6 Daily review on Claude, not Ollama** (§9 "Reliable reviewer"). *Done 2026-10-14 — logged-in `claude -p`, one session across days, rotated after 20 reviews / 60k tokens (skill 56 §3.5.1).*
- [x] **A7 Telegram buttons** — *Dropped 2026-10-14: the operator runs the `claude` CLI from the phone (Termius) instead.*

### B. Find real edge
- [ ] **B1 Shadow book of near misses** — every cycle journal the best rejected candidate per ticker (also when capped: §7 "Evidence while capped") and track its P&L to expiry at natural prices. No orders.
- [x] **B2 Calibration study** — over 2019–2026 daily prices, does the realized-vol POP match realized win rates? *Done 2026-10-13 — `scripts/research/calibrate_pop_rv.py`, results in `research_pop_calibration.md`: `pop_rv` calibrated within ~3 pts (overconfident > 0.9); short puts survived 9–11 pts more than implied odds, short calls 2–6 pts less, in both trends.*
- [ ] **B3 Two-model expected value** — trade when EV under realized-vol POP clears a margin and EV under delta POP is not badly negative. Only if B1 + B2 support it. *B2: supported for the put side only (cap `pop_rv` at 0.90); calls showed no edge — see the research note.*
- [x] *(Done 2026-10-14 — `bear_call_enabled=False`, skill 59 §4.)* **B6 Bear call spreads vs put debit spreads in bearish regimes** — B2 found short calls survived 2–6 pts less than implied even in down-trends. Operator decision: prefer put debits, or demand a larger edge buffer on calls; skew condors put-heavy.
- [ ] **B5 Calibrated probability model** (skill 66) — conditional odds by horizon, asset, direction, vol regime, market regime, term structure; walk-forward. *Phase 1 (research report `research_prob_model.md`) and phase 2 (daily vol-surface timer) done 2026-10-16. Next: phase 3 — shadow `pop_cal` in the journal + weekly refit; phase 4 — feed B3 per strategy once live shadow agrees. Add per-ticker skew after 3–6 months of `vol_surface.jsonl`.*
- [ ] **B4 Rank candidates across the watchlist** — best 1–2 a day by EV per $ risk, instead of yes/no per ticker.
- [ ] **B5 Re-check large-cap option quotes** (§7).

### C. Pay less to get in and out
- [ ] **C1 Mid-first entry ladder** (10.3) — 10/13 debit near misses were 8–10¢ over the ceiling. *Shadow since 2026-10-16 — `debit_ladder_shadow` rows (skill 59 §3); decide on live after ~1 week of `saving_usd` / `missed_trade` / `would_open`.*
- [x] **C2 Slippage vs mid, not vs estimate** (§7). *Done 2026-10-13 — scorecard `entry_cost_vs_mid` (skill 60 §4).*

### D. Exits and open positions
- [x] **D1 No non-emergency exits before 09:45** (10.1). *Done 2026-10-13 — `no_exit_before_et` (skill 30 §4).*
- [x] **D2 Peak / trough P&L for every position from entry** (10.2). *Done 2026-10-13 — `position_excursion` rows (skills 30, 19).*
- [ ] **D3 Wheel puts: stop, roll or accept assignment** (10.4).
- [ ] **D4 Trailing profit live or not** (§7) — after 2–4 weeks of shadow data.
- [ ] **D5 Trade repair advisor** (§8).

### E. Gates that block good trades
- [ ] **E1 Breadth from the 11 sector SPDRs** (§7) — CAUTION 3 sessions on 4 of 8 tickers with VIX 15.
- [ ] **E2 Direction-aware sector cap** (10.6).
- [ ] **E3 Same-ticker add on strong two-model EV**, not only when the open position is profitable.
- [ ] **E4 Book direction balance** (§7, 11.1) — beta-weighted net-delta budget: warn, then prefer trades that pull the book back toward neutral. **Scheduled: build Fri 2026-10-23.**
- [ ] **E6 Correlated-risk groups** (11.2) — TLT + XLU (+ later rate-sensitive names) count as one "rates" group for position / entry limits; cap one-direction risk per group and for the whole book. **Scheduled: build Fri 2026-10-23.**
- [x] **E5 Market-state SPY input** (10.5). *Answered 2026-10-13 — SPY trend and breadth use the prior close by design, VIX is intraday (skill 58 §4).*

### F. Engineering hygiene
- [ ] **F6 Positions with contracts and total risk** (11.3) — `list_positions` and the daily reviewer get contract counts, total max loss, broker legs and mid P&L. **Scheduled: build Fri 2026-10-23.**
- [ ] **F1 `scripts/ci_local.sh`** (§7). **F2 Day-replay harness** (§9). **F3 Stage 2 as a gate pipeline** (§9). **F4 Promotion-by-evidence report** (§9). **F5 Delete merged branches** (§4, plus `feat/manual-close`, `feat/data-server-key-guard`, `feat/debit-target-on-cost`).

### G. Operator
- [ ] Schwab re-login by **Mon 2026-10-26 ~22:00 ET** (logged in 2026-10-19 22:01; the data server ran pre-A1 code until 2026-10-19, so the countdown is only trustworthy from that login). Refresh the June holdings paste. Confirm the rotated data-server token is the only valid one.

### H. Later
- [ ] Quality + momentum stock sleeve (§3).

**Order:** this week A + D1/D2/C2/E5 + B1/B2 (no trading change) → next week C1, E1 → from 10/26 B3/B4 if the evidence supports them, then E2/E3 → later D3/D4, A6/A7, D5, F.

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
- [x] *(Superseded by A6.)* The daily reviewer's LLM step needs Ollama running at `LLM_BASE_URL` (not running on 2026-09-30, so reviews contain stats only). Start Ollama, or accept stats-only reviews since the 16:22 Claude review covers the analysis.
- [x] *(Done — Telegram alerts deliver since 10/05.)* Add Telegram tokens, then drop `--no-telegram` from the installed plist to get the digest on your phone.

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
- [x] **Correct the cycle interval in docs.** Logs show a monitor cycle roughly every 75 s (10:21:48, 10:23:05, 10:24:21 …; supervisor restarts the agent ~60 s after each cycle), not every 5 minutes. The 3-cycle exit debounce is therefore ~4 min, not 15. Fix the Trading Day Flow artifact, skills 44 / 57 / PROJECT_MANIFEST, and decide whether the cadence is intended (API load, journal volume, debounce length). *(Trading Day Flow artifact updated 2026-10-05 to the ~75 s cadence and the phase 1–5 strategies; source now in `docs/presentations/trading_day_flow.html` — edit there and republish to https://claude.ai/artifact/3z9mQKunCPtepWpYzgqHFZ.)*
- [x] **Subagent `tools:` lists use bare MCP names** (`list_recent_trades` …), so `journal-analyst` (and likely `scanner-runner`, `risk-reviewer`) start with zero tools and Claude Code refuses to spawn them (found 2026-10-02 in the weekly review). Use `mcp__trading-agent__<name>`; check `scan_subagent_allowlists.py` accepts the prefixed form.
- [x] *(2026-10-05: no longer reproducible — 20/20 pass outside market hours in a clean worktree and with the local `.env`; watch CI on evening pushes.)* `tests/test_after_hours_shutdown.py` SIGKILLs pytest (exit 137) when run after market close — **CI risk for evening pushes**. `test_after_close_calls_graceful_exit_0` patches `_is_within_market_hours` and `graceful_exit`, so `run_cycle` continues past the mocked exit; suspects: `_maybe_send_eod_summary()` (agent.py:727, real clock) or the real `TradingAgent` picking up the local `.env`. Reproduce after 16:05 ET, in a clean worktree with `env -i`, to see whether CI is affected at all.
- [x] CI red since 2026-09-29 (fixed 2026-10-01): skill-freshness failures (footers stamped before commit / in local time vs CI's UTC), stale traceability matrix, missing fastapi/uvicorn/httpx2 in requirements, and four tests stale since the 29bb020 cache-off default. Before pushing: commit, then run `TZ=UTC python scripts/checks/scan_skill_freshness.py` and the CI steps in a clean worktree.
- [x] "(no reason recorded)" rejects were valid plans vetoed by RiskManager (reason only in `checks_failed`); the writer now records `risk: <first failed check>` and the reader falls back to `checks_failed` (2026-10-05).
- [x] The "23 pre-existing test failures" are local-environment only: a clean worktree with `env -i` (no `.env`, UTC) gives 0 failures (2026-10-01). Run local checks that way before trusting a failure count.
- [ ] Delete merged branches: `fix/iron-condor-wing-width`, `fix/mcp-dataserver-env-key`, `feat/wheel-lifecycle`.

## 5. Operator actions (not code)

- [x] *(Done.)* Add Telegram bot tokens to `.env` so outages and errors page you.
- [x] Reconcile the July XLE iron condor P&L from the old paper account's order history. *(2026-10-05: history unreachable (account replaced; strikes and size not in the journal, trade plan rotated) → `position_reconciled` row with P&L unknown via `trading_agent/journal_reconcile.py`; if you recover the P&L from the old account, append the real close with `--realized-pl`.)*
- [ ] Refresh the June holdings paste in the dashboard.
- [ ] Confirm the rotated data-server token is the only valid one.

## 7. Learnings from the first live playbook day (2026-10-05)

- [x] **Stagger entries.** Four positions opened in 8 minutes and used the whole 10 % budget by 13:37; the rest of the day was 933 `total_risk_cap` skips. Add `max_new_entries_per_hour` (e.g. 1) and/or `max_new_entries_per_day` (e.g. 3) to `PresetConfig`, so risk is spread across the session and across days. *(Done 2026-10-05 — `max_new_entries_per_hour` = 1, entries from 09:45 ET; skill 61.)*
- [ ] **Portfolio direction balance.** The book is net bearish (2 put debits + 1 call debit + calendar) while the market state is NORMAL with SPY above every average. Add a beta-weighted net-delta check (warn, then cap) so single-ticker regimes cannot stack one direction against the index.
- [ ] **Broader breadth input.** Breadth uses 8 watchlist ETFs (12.5 % steps) and sat exactly on the 50 % CAUTION line all day. Compute it from the 11 Select Sector SPDRs (data only, not traded) for a finer, steadier reading; re-run `validate_market_state.py`.
- [ ] **Evidence while capped.** The shadow POP log got zero rows after the cap engaged, because capped tickers are skipped before planning. Plan in shadow mode when capped (journal a `would_trade` row with `pop_delta` / `pop_rv`, no order) so the §3 / §6.6 data keeps accumulating.
- [ ] **Measure slippage vs mid, not vs estimate.** All 4 fills equalled the natural-price estimate, so `entry_slippage` reads 0 while the positions started ~$100 down (≈ 3.4 % of risk = the bid/ask paid on 9 legs). Store the mid at entry in the trade plan and report fill − mid in the scorecard (the §6.6 "slippage vs mid" metric).
- [x] **Entry regime persistence.** IWM entered as bearish and read sideways one cycle later (price between its 50- and 200-day). Require the regime to hold for N cycles (or a minimum distance from the SMA) before a directional entry; exits already debounce. *(Done 2026-10-05 — entry confirmation: same strategy + expiry on 3 consecutive cycles; skill 61.)*
- [ ] **Run CI's exact pytest command locally.** `pytest -q` hid an INTERNALERROR that `pytest -v --maxfail=10` (CI) hit. Add `scripts/ci_local.sh` mirroring every CI step in a clean `env -i` worktree, and use it before every push.
- [ ] **Trailing profit: decide live vs not.** `profit_trail_mode` is `shadow` since 2026-10-05; after 2–4 weeks compare `profit_trail_shadow` rows (`trail_minus_actual`) per structure and switch the winners to `live` (likely debit spreads first).
- [ ] **Re-check large-cap option quotes** (AAPL, MSFT, GOOGL, JPM, V, MA, XOM had 7–41 % median leg spreads mid-day on 10/05) at 10:00 and 15:30 ET; if they tighten, add the ones that pass the quality screen (§6.7).
- [x] *(Obsolete — the agent runs on the Pi since 10/07.)* *(Operator)* **Keep the Mac awake for the session:** `sudo pmset repeat wakeorpoweron MTWRF 09:15:00` and no sleep on power during 09:15–16:10 ET. The supervisor fix only helps once the machine is awake.

## 8. Trade repair advisor — fix losing trades with the same intelligence (added 2026-10-06)

**Why:** on 2026-10-06 the GLD 379/369 put debit (2 × $4.25, −$145 at mid) was analysed by hand: hold vs close vs butterfly vs bear-call add vs roll out / out-and-up, each priced at natural and scored with a realized-volatility model. Hold won (worth ≈ $3.92/sh vs $3.35 to exit — every repair paid the bid/ask again, costing ≈ $36–115 of expected value); the butterfly was the only repair that lowered risk ($850 → $508 for ≈ $100). That analysis should run automatically and consistently, not ad hoc.

**Plan (advisory first, shadow-measured, never automatic until proven):**

- [ ] **Trigger.** Each cycle, flag a position for repair review when any holds: loss ≥ 25 % of max loss; the underlying is past the long strike (debit) or within 1 % of a short strike (credit); the thesis weakened (regime no longer matches but has not reversed, or price crossed back over the 20-day average); or ≤ 10 DTE with the position out of the money. Journal `repair_review` once per position per day.
- [ ] **Repair menu per structure** (pure scorer in `decision_engine.py`, invariant 2):
  - all: **hold**, **close now**;
  - debit verticals: **convert to butterfly** (sell the next spread beyond the short strike), **roll out** (same strikes, later expiry), **roll out and re-centre** (at the money);
  - credit verticals: **roll out for a credit**, **convert to iron condor** (sell the untested side), the existing **defensive roll** (skill 31) as one candidate;
  - calendars: **roll the short leg**, **close**.
  Never offer a repair that raises max loss (e.g. the bear-call add, which took GLD's worst case from $850 to $1,550) unless explicitly allowed.
- [ ] **Score each repair the same way:** cost at natural price now; model value at expiry from a lognormal with 20-day realized volatility (`shadow_pop`), Black-Scholes for multi-expiry legs; Δ expected value vs hold; Δ max loss; P(profit); risk reduced per $ of expected value given up. Recommend hold unless a repair beats it on EV or buys risk reduction at ≤ a set price (e.g. ≤ $0.30 of EV per $1 of max-loss reduction).
- [ ] **Use the intelligence already built:** market state (no bullish repair in DEFENSIVE / CAPITULATION), regime and SMA position for the thesis check, the playbook scorecard (repairs on a losing playbook lean to close), entry confirmation (a repair must be recommended on 3 consecutive cycles) and entry timing (place the repair order at the best-priced cycle).
- [ ] **Surface it:** MCP `get_repair_options(ticker)` (read-only) and a `/triage` section; staged only through `/propose` → promote.
- [ ] **Prerequisite — position families.** A butterfly conversion adds legs that share a strike with the open spread (GLD: two more short 369P). `group_into_spreads` must group an original plan and its repair under one family id, with combined economics for exits, before any repair is staged.
- [ ] **Shadow measurement first:** journal `repair_shadow` with the recommended repair and what hold vs repair would have returned at expiry or exit; review after 4+ weeks per structure before any auto-repair is considered.

## 9. Autonomy with a human in the loop (added 2026-10-07)

**Why:** the agent now trades every market condition, but week 2 showed the operating model lagging: the host is a laptop (it slept through an open), nothing pages the operator (Telegram not configured), the daily AI review has been empty for days (Ollama down), most bugs were found by watching live paper trading, and four features sit in shadow with no path to promotion. Goal: the agent runs unattended, and every decision outside its rails reaches a human who can approve or stop it from a phone.

**Autonomy levels — every action class gets one; changing a level is a recorded human decision:**

| Level | Meaning | Action classes |
|---|---|---|
| L3 auto | acts within rails, reports after | entries, exits, stops, kill-switch-rule closes |
| L2 auto with veto | notifies, waits N minutes, acts unless vetoed | repairs (§8), scorecard-suggested sizing |
| L1 propose | stages; a human approves | Wheel orders, preset proposals, shadow → live promotions, new playbooks |
| L0 observe | logs only | anything new (shadow) |

- [x] *(Done — merged a2c9ee5.)* **Rename** away from "credit-spread agent" in everything people read (README, manifest / context, handoff prompt, preset descriptions, dashboard, Telegram, MCP descriptions); keep stored identifiers (journal actions, `net_credit`, class / file / skill names). *(Started 2026-10-07.)*
- [x] *(Done — merged a2c9ee5, skill 62.)* **Kill switch + drawdown governor.** One command pauses new entries (exits keep running); automatic pause at a daily / weekly loss limit; only a human resumes. *(Started 2026-10-07.)*
- [x] *(Done — Pi since 10/07; liveness = A5, skill 64.)* **Always-on host + liveness alerts.** Move the agent and data server off the laptop — chosen host: the operator's Raspberry Pi (runbook 08, `deploy/systemd/`, `scripts/migrate_to_pi.sh`, 2026-10-07); dead-man alert when no cycle by 09:50 ET or none for 10 min in session.
- [ ] **Phone approvals.** Telegram with approve / reject / pause buttons — prerequisite for L1 / L2 away from the terminal.
- [ ] **Promotion by evidence.** Weekly report from the playbook scorecard and the shadow logs (trailing profit, entry timing, realized-vol POP) with an explicit recommendation per feature, approved through L1.
- [ ] **Reliable reviewer.** Run the daily / weekly AI review on Claude instead of a local Ollama.
- [ ] **Day-replay harness.** Replay a recorded session (chains, quotes, journal) through the full agent cycle in tests — this week's sector-cap, plan-retention and regime-flicker bugs would have failed there first.
- [ ] **Stage 2 as a gate pipeline.** Split the entry path into small, separately tested gates (caps → filters → plan → risk → market state → ladder → confirmation → timing); `_run_cycle_impl` is at the 400-line cap.

## 10. Week-2 learnings — review week of 2026-10-12 (added 2026-10-10)

**Week 2 (10/05–10/09):** 5 entries Mon–Tue, none Wed–Fri; realized +$191 (IWM +$288 by hand, VZ CSP −$97); account $29,765 → $29,658. Shipped: manual close (skill 63), debit profit target on cost (`debit_profit_target_basis="debit"`, merged `267f727`). Suggested order: 10.1 + 10.2 first (small, safe, add data), then 10.3, then §7 broader breadth.

- [ ] **10.1 Opening-window guard for exits.** VZ $43 CSP gapped from Δ −0.12 (Thu close) through the strike; the delta stop closed it at 09:36 ET, six minutes into the session, at $1.18 vs a $1.045 mark (−$97 on a $21 credit). Entries wait until 09:45 (`no_entry_before_et`); exits have no equivalent. Add a preset field (e.g. `no_exit_before_et`) that defers non-emergency exits (delta stop, profit target, regime shift) — not hard stops at max loss — until the opening quotes settle. Skill 30 / Wheel skill.
- [ ] **10.2 Track peak P&L for every position from entry (shadow only).** `profit_trail.json` records `peak_pl` only after a position arms, so every open position shows a $0 peak and the IWM target question (+$444 peak at 11:07, closed at +$288) had to be answered from memory. Record peak and peak time (and trough) per position from entry; use it to tune debit targets / stops and the §7 trailing-profit decision. Skill 30 / profit trail.
- [ ] **10.3 Mid-first price ladder for debit entries.** The 5 debit entries paid $106 over mid (1.2–4.9 % of each debit), and ~1,000–1,300 debit candidates a day were refused as `debit_above_fair` (natural > mid × 1.05). `entry_timing_shadow` resolved only 1 entry all week, so it will not answer this. Try mid → halfway → natural with a short timeout per step (as the Wheel does), journal which step filled, and keep `debit_above_fair` measured against the price actually sent. Paper first; compare two weeks of fills. Skill 59.
- [ ] **10.4 CSP delta stop vs the Wheel.** A stop that turns a $21 credit into a $97 loss needs an ~82 % win rate to break even, and closing at Δ 0.45 means a Wheel put is never assigned on a name it was screened to own. Decide per name: roll down-and-out for a credit, accept assignment, or keep the stop (with 10.1). Design decision first, then the Wheel skill.
- [ ] **10.5 Market-state SPY input.** `market_state.json` showed SPY 773.93 all session on 2026-10-09 while SPY traded up to $778 — confirm whether the state is meant to use the prior daily close; if so, say so in skill 58. (Breadth on 8 tickers kept CAUTION all day at 3 of 8 — see §7 "Broader breadth input".)
- [ ] **10.6 Direction-aware sector cap.** IWM / DIA were `skipped_existing` 1,080 times a day because SPY (neutral calendar) + QQQ (bullish) fill the Broad Market cap of 2, while IWM (bearish) was the week's only big winner. Consider counting bullish and bearish positions per sector separately, or not counting calendars; weigh against concentration risk (ties to §7 "Portfolio direction balance").

## 11. Book-level risk — direction and correlation (added 2026-10-19)

**Evidence (2026-10-19).** Seven open positions, five of them put debit spreads: ≈ $1,935 of $2,974 total max loss bearish (6.6 % of equity), and TLT ×2 + XLU (≈ $839) are close to one bet on higher yields. `entry_rate_scope="sector"` treats Treasuries and Utilities as different sectors, so nothing limits it. Account daily P&L vs SPY daily return, 2026-09-29 → 10-19 (15 sessions, Alpaca portfolio history): correlation 0.15, and 0.0 without the noisy first two sessions; same sign on 7 days, opposite on 8. The book is not long the market — it swings with whichever direction the individual tickers' regimes pick, and realized losses came from directional bets the market went against (QQQ call debit on 10-14, TLT put debit on 10-15, VZ CSP on 10-09). No defined-risk strategy wins in every market (directional spreads need the direction, condors / calendars need a range, long volatility needs a big move); what the agent can control is which risks the book carries at once.

- [ ] **11.1 Net-delta budget (E4).** Beta-weighted net delta of the book (SPY-equivalent shares, from leg deltas × contracts × 100 × beta to SPY) computed each cycle and journaled. Preset `max_net_beta_delta_pct` (share of equity, SPY-equivalent notional). Above the budget: candidates that add to the tilt are rejected (`net_delta_budget`), candidates that reduce it are allowed, neutral structures (condor, calendar, butterfly) pass. Shadow first (would-block counts in the daily review), then live. Show the number in `/portfolio` and the dashboard.
- [ ] **11.2 Correlated-risk groups (E6).** A `risk_group` map beside `sector_map` (start: rates = TLT, XLU; gold = GLD, GDX; US large-cap = SPY, QQQ, DIA, XLK, SMH). Position and entry-rate limits count by group as well as sector; `max_group_direction_risk_pct` caps max loss in one direction per group (start 3 %), `max_book_direction_risk_pct` across the book (start 6 %). Ties to E2: counting bullish and bearish separately per group.
- [ ] **11.3 Positions with contracts and total risk (F6).** `list_positions` returns `contracts` and `total_max_loss` (the `max_loss` field is per contract — the 10-16 and 10-19 briefs understated risk as $1,769 vs $2,974). The reviewer context gets the same plus broker legs and mid P&L, so it stops asking about positions it cannot see (TLT count, the QQQ leg the broker no longer holds) and judges debit cost per dollar of width, not in dollars.

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


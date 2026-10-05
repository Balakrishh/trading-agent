# Skill Library

Atomic, reusable concepts extracted from the trading agent. Each file is self-contained: theory → formula → reference Python → edge cases. A new collaborator (LLM or human) should be able to read any one skill in 5 minutes and reproduce the math without reading the rest of the codebase.

**Philosophy.** This library is **derived** from `trading_agent/` — every skill cites a `file:line` source of truth. If the source moves, the skill moves with it. Skills don't introduce new logic, they document what already exists in a form that's easier to reason about.

> **Looking for operational runbooks instead?** See [`docs/runbooks/`](../runbooks/README.md). Skills are "why does the code do this"; runbooks are "what do I do when I see X."  The two are complementary — runbooks cross-link to skills for the explanatory background.

---

## Phase 1 (21 skills + meta + skill 28, dependency-ordered)

Read top-to-bottom on a first pass. Each row's "Depends on" column lists the prerequisite skills. **Skill 00 is mandatory first reading** — it's the meta-skill covering glossary, SDLC, and conventions that every other skill assumes you've internalised.

| # | Skill | Group | Source of truth |
|---|---|---|---|
| 00 | [SDLC, glossary, and code conventions](00_sdlc_and_conventions.md) | meta | this folder + `scripts/checks/scan_invariant_check.py` |
| 01 | [POP from short delta](01_pop_from_delta.md) | strategy | `chain_scanner.py:110-112` |
| 02 | [Strike snapping to grid](02_strike_snapping.md) | strategy | `strategy.py:684`, `chain_scanner.py:496` |
| 03 | [Credit-to-Width floor](03_credit_to_width_floor.md) | risk | `chain_scanner.py:160-162`, `risk_manager.py:108-114` |
| 04 | [Adaptive spread width](04_adaptive_spread_width.md) | strategy | `strategy.py:653-685` |
| 05 | [EV per $ risked scoring](05_ev_per_dollar_risked.md) | strategy | `chain_scanner.py:165-207` |
| 06 | [Stale-spread risk gate](06_stale_spread_risk_gate.md) | risk | `risk_manager.py:50, 79, 166-190` |
| 07 | [Anchor map for leadership](07_anchor_map_for_leadership.md) | bias | `regime.py:36-110` |
| 08 | [Leadership Z-score](08_leadership_zscore.md) | bias | `market_data.py:1075, 1138-1176` |
| 09 | [VIX Z-score inhibitor](09_vix_zscore_inhibitor.md) | bias | `regime.py:117, 293-299`; `strategy.py:255-265` |
| 10 | [ADX with Wilder smoothing](10_adx_wilder_smoothing.md) | regime | `multi_tf_regime.py:367-423` |
| 11 | [Six-regime classifier](11_six_regime_classifier.md) | regime | `regime.py:209, 412-455` |
| 12 | [Multi-timeframe regime resolution](12_multi_timeframe_resolution.md) | regime | `multi_tf_regime.py:225-361` |
| 13 | [Preset system & hot-reload](13_preset_system_hot_reload.md) | architecture | `strategy_presets.py:54-295` |
| 14 | [Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md) | architecture | `strategy.py:158-205, 302-334` |
| 15 | [Backtest↔live parity](15_backtest_live_parity.md) | architecture | `trading_agent/backtest/`, `streamlit/backtest_ui.py` |
| 16 | [Market-data provider routing](16_market_data_provider_routing.md) | architecture | `market_data_factory.py`, `market_data_schwab.py`, `market_data_yahoo.py`, `schwab_oauth.py` |
| 17 | [Close-failure action + cooldown + PDT](17_close_failure_and_cooldown.md) | risk | `agent.py:124-200, 867-1265`, `streamlit/live_monitor.py:1095-1316` |
| 18 | [Order-submission idempotency (`client_order_id` + retry)](18_order_submission_idempotency.md) | risk | `executor.py:60-76, 299-541` |
| 19 | [Signal-journal schema (action enum + dedup bypass)](19_journal_schema.md) | architecture | `journal_kb.py:95-101, 155-432`, `journal_reader.py` |
| 28 | [Position-monitor spread grouping (plan-match-then-infer)](28_position_monitor_spread_grouping.md) | architecture | `position_monitor.py:294-373` |
| 29 | [Per-leg liquidity gate](29_per_leg_liquidity_gate.md) | strategy | `chain_scanner.py:_leg_spread_too_wide`, `decision_engine.py` call-site |
| 30 | [Profit-target management](30_profit_target_management.md) | risk | `strategy_presets.py:PresetConfig`, `position_monitor.py:601-609`, `backtest/runner.py:258-271` |
| 31 | [Defensive roll](31_defensive_roll.md) | risk | `defensive_roll_evaluator.py`, `executor.py:roll_position_defensive`, `agent.py:_maybe_defensive_roll` |

## Phase 2 (operational hardening — added 2026-05-13 → 2026-05-22)

Skills 32–35 ship alongside the live-deployment hardening pass. Phase 1 covers "how the strategy decides"; Phase 2 covers "how the operator sees what's happening + how the code stays maintainable."

| # | Skill | Group | Source of truth |
|---|---|---|---|
| 32 | [Telegram operator alerts + stuck-position banner](32_telegram_operator_alerts.md) | ops | `telegram_notifier.py`, `agent.py:_send_telegram_alert`, `streamlit/live_monitor.py:_render_stuck_position_banner` |
| 33 | [PDT-aware DTE cap](33_pdt_dte_cap.md) | risk | `strategy_presets.py:PresetConfig`, `strategy.py:apply_pdt_dte_cap`, `agent.py` wiring |
| 34 | [Exception monitor — operator visibility for silenced failures](34_exception_monitor.md) | ops | `exception_monitor.py:ExceptionMonitor`, `agent.py` + `executor.py` + `strategy.py` + `market_data_schwab.py` call sites, `telegram_notifier.py:notify_silenced_exception`, `journal_reader.py:silenced_exceptions_today` |
| 35 | [Close-event collaborators (extracted from `_journal_close_event`)](35_close_event_collaborators.md) | architecture | `close_event_collaborators.py`, `agent.py` construction + delegation |
| 36 | [Ticker filters — early-return pipeline](36_ticker_filters.md) | architecture | `ticker_filters.py`, `agent.py` construction + delegation |
| 37 | [Position-cap dedup (per-ticker + per-sector)](37_position_caps.md) | risk | `position_caps.py`, `agent.py` Stage 2 call site |
| 38 | [Backtester slippage + commissions](38_backtest_slippage.md) | backtest | `backtest/runner.py`, `backtest/account.py`, `backtest/cycle.py` |
| 39 | [Backtester volatility skew](39_skew_model.md) | backtest | `backtest/skew_model.py`, `backtest/synthetic_chain.py` |

## Phase 3 (Wheel, MCP surface, ops, playbooks — 2026-06 → 2026-10)

Skills 40–60: the long-term / Wheel evaluator, the Claude Code MCP surface and slash-command playbooks, the supervisor, and the 2026-10 backlog phases — market risk state + playbook table (58), debit spreads / calendars / bounce bull put (59), trade measurement (60). Rows generated from each skill's header.

| # | Skill | Group | Source of truth |
|---|---|---|---|
| 40 | [Long-term options evaluator](40_long_term_options_evaluator.md) | strategy | `trading_agent/long_term_evaluator.py`, `trading_agent/decision_engine.py`, `trading_agent/streamlit/long_term_evaluator_ui.py` |
| 41 | [Positions provider — uniform holdings input](41_positions_provider.md) | data_quality | `trading_agent/positions_provider.py` |
| 42 | [Portfolio alert scheduler](42_portfolio_alert_scheduler.md) | ops | `trading_agent/portfolio_alert_scheduler.py`, `trading_agent/telegram_notifier.py:notify_portfolio_review`, `trading_agent/agent.py:_maybe_run_portfolio_review` |
| 44 | [Position-Monitor Scaling — Contract Count + Post-Fill Grace](44_position_monitor_scaling.md) | risk | `trading_agent/position_monitor.py:_check_exit`, `trading_agent/position_monitor.py:SpreadPosition` |
| 45 | [Iron Butterfly Scoring](45_iron_butterfly.md) | strategy | `trading_agent/chain_scanner.py:_score_iron_butterfly`, `trading_agent/chain_scanner.py:_pop_from_ib_structure`, `trading_agent/strategy_presets.py:PresetConfig` |
| 46 | [Broken-Wing Butterfly Scoring](46_broken_wing_butterfly.md) | strategy | `trading_agent/chain_scanner.py:_score_broken_wing_butterfly`, `trading_agent/chain_scanner.py:_pop_from_bwb_structure`, `trading_agent/strategy_presets.py:PresetConfig` |
| 47 | [Schwab Data API — Local HTTP Server](47_schwab_data_api.md) | ops | `trading_agent/data_server/app.py`, `trading_agent/data_server/auth.py`, `trading_agent/data_server/cache.py` |
| 48 | [Claude Code MCP Surface — Read-Only Tools](48_claude_code_mcp_surface.md) | ops | `trading_agent/mcp/__init__.py`, `trading_agent/mcp/server.py`, `trading_agent/mcp/tools/positions.py` |
| 49 | [Daily Portfolio Review — Playbook](49_daily_portfolio_review.md) | ops | `trading_agent/mcp/tools/positions.py`, `trading_agent/mcp/tools/strategy.py`, `trading_agent/mcp/tools/market.py` |
| 50 | [Position Triage — Playbook](50_position_triage.md) | ops | `trading_agent/mcp/tools/positions.py`, `trading_agent/mcp/tools/strategy.py`, `trading_agent/defensive_roll_evaluator.py` |
| 51 | [Pre-Trade Approval — Playbook](51_pre_trade_approval.md) | ops | `trading_agent/mcp/tools/strategy.py`, `trading_agent/pending_orders_writer.py`, `trading_agent/executor_promote.py` |
| 52 | [Watchlist Curation — Playbook](52_watchlist_curation.md) | ops | `trading_agent/mcp/tools/positions.py`, `trading_agent/watchlist_store.py`, `trading_agent/journal_reader.py` |
| 53 | [Incident Response — Playbook](53_incident_response.md) | ops | `trading_agent/mcp/tools/market.py`, `trading_agent/exception_monitor.py`, `trading_agent/journal_reader.py` |
| 54 | [Tax-Lot Review — Playbook](54_tax_lot_review.md) | ops | `trading_agent/mcp/tools/positions.py`, `trading_agent/journal_reader.py` |
| 55 | [Pending-Orders Promotion — Write Gate](55_pending_orders_promotion.md) | ops | `trading_agent/executor_promote.py`, `trading_agent/pending_orders_writer.py`, `trading_agent/strategy_presets.py` |
| 56 | [End-of-Day Trade Journal Reviewer](56_daily_journal_reviewer.md) | ops | `trading_agent/daily_reviewer.py`, `trading_agent/daily_reviewer_main.py`, `trading_agent/pending_preset_updates_writer.py` |
| 57 | [Agent Supervisor — Long-running Wrapper](57_agent_supervisor.md) | ops | `trading_agent/agent_supervisor.py`, `ops/launchd/com.trading-agent.headless.plist` |
| 58 | [Market Risk State & Playbook Table](58_market_state_playbook.md) | risk / regime | `trading_agent/market_state.py` |
| 59 | [Debit Spreads, Calendars & the Bounce Bull Put](59_debit_spreads_calendars.md) | strategy | `trading_agent/debit_policy.py`, `trading_agent/decision_engine.py` |
| 60 | [Trade Measurement — Entry Fills, Shadow POP, Playbook Scorecard](60_trade_measurement.md) | architecture / risk | `trading_agent/fill_reconciler.py`, `trading_agent/shadow_pop.py`, `trading_agent/playbook_scorecard.py` |

## Phase 2 (planned, not yet written)

Hygiene and diagnostics. Useful but not edge-defining.

- `20_open_bar_skip.md` — Why we drop the first N bars after the open auction.
- `21_stale_data_age_detection.md` — `last_bar_ts` + 30-min wall-clock badge.
- `22_signal_availability_sentinel.md` — The `*_signal_available: bool` design pattern.
- `23_trend_conflict_detector.md` — 200-SMA slope vs short-term — diagnostic only.
- `24_bollinger_bandwidth_regime.md` — The 4 % SIDEWAYS rule, in isolation.
- `25_account_risk_pct_sizing.md` — Conservative 1 % / Balanced 2 % / Aggressive 3 %.
- `26_regime_to_strategy_routing.md` — The dispatch table (now README "Strategy Selection"; routing is covered by skills 58 + 59).
- `27_width_aware_max_loss.md` — `(width − credit) × multiplier`.

---

## How to add a new skill

1. Copy `_template.md` to `NN_short_kebab_name.md` (next free number).
2. Fill in **all five sections**. Skipping §4 (edge cases) defeats the point.
3. Quote source verbatim — do not paraphrase. Future readers should be able to grep the code and find your snippet.
4. Add a row to the table above.
5. If your skill changes the inventory of phase 1 vs phase 2, update both lists.

## How to verify a skill is still accurate

For any skill `NN_*.md`:

1. Open the file linked in the **Source of truth** header.
2. Diff its contents against §3 of the skill file.
3. If they disagree, the source has drifted — update the skill or open a PR explaining why the divergence is intentional.

This is also a good first task for a new LLM joining the project: "pick one skill, verify the citation is still accurate."

## Reading order for a new contributor

If you only have 30 minutes:
- **Minute 0–5:** [`PROJECT_MANIFEST.md`](../../PROJECT_MANIFEST.md) at repo root.
- **Minute 5–10:** Skill 00 — glossary + SDLC + conventions. Sets vocabulary for everything else.
- **Minute 10–18:** Skills 01, 03, 04, 05 — the core spread-economics math.
- **Minute 18–25:** Skills 11, 12 — the regime classifier and how it composes across timeframes.
- **Minute 25–30:** Skill 13 — the preset system, which is the central control surface.

If you have a full afternoon, read 00 first, then all 21 in order. Skills 28, 29 are late-Phase-1 additions (position-monitor spread grouping; per-leg liquidity gate) — read them after 19. Skills 30–35 are Phase-2 operational hardening added 2026-05-13 → 2026-05-22:

- 30 (profit-target management) and 31 (defensive roll) extend the close-side risk surface.
- 32 (Telegram operator alerts), 33 (PDT-aware DTE cap), 34 (exception monitor) make the running system visible to the operator without log-scraping.
- 35 (close-event collaborators) decouples the ~300-line `_journal_close_event` into four constructor-injected classes — read after 17 + 19 + 32 since it integrates all three.

For **what the agent trades today**, read 58 → 59 → 37 → 60 (market state and playbook routing → debit structures → portfolio caps → measurement), then 40 for the Wheel.

---

*Last updated: 2026-10-05 against repo HEAD.*

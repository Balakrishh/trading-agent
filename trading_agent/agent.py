"""
The Trading Agent Orchestrator
================================
Runs a two-stage cycle:

  STAGE 1 — MONITOR existing positions
    → Fetch open positions from Alpaca
    → Classify current regime for each underlying
    → Evaluate exit signals (stop-loss, profit-target, regime-shift)
    → Close spreads that trigger an exit signal

  STAGE 2 — OPEN new positions (the original four-phase loop)
    I.   PERCEIVE   — fetch market data
    II.  CLASSIFY   — determine regime
    III. PLAN       — select strategy and strikes
    IV.  ACT        — validate risk, execute or log

5-minute cycle design notes
------------------------------
• run_cycle() is wrapped in a 270-second (4.5 min) hard-timeout guard.
  If the cycle has not completed by then, a TIMEOUT event is logged to
  JournalKB and the guard calls shutdown.hard_exit(1) so the cron
  scheduler can cleanly launch the next run without a zombie process.

• All historical price data is pre-fetched in parallel via
  prefetch_historical_parallel() before the per-ticker loop begins.

• All current prices are fetched in a single batch API call via
  fetch_batch_snapshots() before the per-ticker loop.

• JournalKB.log_signal() is called for EVERY ticker on EVERY cycle
  regardless of LLM enablement, execution mode, or failure type.

Week 3-4 modularization
-----------------------
Several concerns previously inlined in this file were extracted:

  • market_hours.py  — NYSE trading-hours guard
  • daily_state.py   — DailyStateStore + drawdown + debounce policy
  • thesis_builder.py — raw_signal thesis dict
  • shutdown.py      — graceful vs hard exit paths, signal handlers
  • file_locks.py    — locked appends + atomic JSON writes
  • logger_setup.py  — now uses RotatingFileHandler

The TradingAgent class is a thin orchestrator over those modules.
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from trading_agent.config import AppConfig, load_config
from trading_agent.journal_kb import JournalKB
from trading_agent.logger_setup import setup_logging
from trading_agent.market_data import MarketDataProvider, InsufficientDataError
from trading_agent.ports import (
    MarketDataPort,
    ExecutionPort,
    PositionsPort,
    OrdersPort,
)
from trading_agent.regime import Regime, RegimeClassifier, RegimeAnalysis
from trading_agent.rsi_gate import evaluate_rsi_gate
from trading_agent.strategy import StrategyPlanner, SpreadPlan
from trading_agent.strategy_presets import (
    PresetConfig,
    load_active_preset,
    regime_is_allowed,
)
from trading_agent.risk_manager import RiskManager, RiskVerdict
from trading_agent.defensive_roll_evaluator import (
    RollDecision,
    RollEvalInputs,
    evaluate_defensive_roll,
)
from trading_agent.executor import OrderExecutor
from trading_agent.telegram_notifier import TelegramNotifier
from trading_agent.position_monitor import (
    PositionMonitor, ExitSignal, SpreadPosition, IMMEDIATE_EXIT_SIGNALS,
)
from trading_agent.order_tracker import OrderTracker
from trading_agent.llm_client import LLMClient, LLMConfig
from trading_agent.trade_journal import TradeJournal
from trading_agent.knowledge_base import KnowledgeBase
from trading_agent.llm_analyst import LLMAnalyst, AnalystDecision
from trading_agent.fingpt_analyser import FinGPTAnalyser, SentimentReport
from trading_agent.news_aggregator import NewsAggregator
from trading_agent.sentiment_verifier import SentimentVerifier, VerifiedSentimentReport
from trading_agent.sentiment_pipeline import SentimentPipeline

# --- Week 3-4 extractions ---
from trading_agent.market_hours import (
    EASTERN,
    is_within_market_hours as _is_within_market_hours,
    market_window_str,
)
from trading_agent.daily_state import (
    DailyStateStore,
    check_daily_drawdown,
    tally_exit_vote,
)
from trading_agent.thesis_builder import build_thesis
from trading_agent import shutdown as _shutdown

logger = logging.getLogger(__name__)

# Kill the process if a cycle takes longer than this (seconds).
# The external scheduler (cron / APScheduler) will start the next run cleanly.
CYCLE_TIMEOUT_SECONDS = 270   # 4 min 30 sec

# Number of consecutive cycles an exit signal must repeat before acting.
EXIT_DEBOUNCE_REQUIRED = 3

# Stale-order policy.
#
# An open limit order whose age (since Alpaca's `created_at`) exceeds this
# threshold is cancelled at the start of every cycle — the next planning
# pass will re-price against fresh quotes.  Set high enough to give a
# midday combo a fair shot at filling, low enough to recover before
# theta has chewed through the original limit.
STALE_ORDER_MAX_AGE_MIN = 15

# ── Pattern Day Trading (PDT) threshold ─────────────────────────────────
# Alpaca enforces FINRA's PDT rule on accounts below this equity
# threshold: 4+ "day trades" (open + close same security same day) in
# a 5-day window triggers a 90-day flag that blocks further day trades.
# At $5K, a same-day open + REGIME_SHIFT close is exactly this pattern,
# producing the "trade denied due to pattern day trading protection"
# 403 we hit on 2026-05-06.  Below this threshold, suppress same-day
# REGIME_SHIFT exits and hold overnight; STRIKE_PROXIMITY / HARD_STOP
# / DTE_SAFETY still fire because those are real risk events that
# justify the PDT hit.
PDT_EQUITY_THRESHOLD = 25_000.0

# ── Partial-close cooldown ──────────────────────────────────────────────
# When ``executor.close_spread`` returns ``fill_status="partial"`` the
# position is in zombie state — some legs closed, some rejected by
# Alpaca.  Naive retry on every cycle (5 min) generates spam and never
# converges (the Alpaca-side block doesn't lift on its own).  After
# this many consecutive partial fills, suppress further auto-close
# attempts for ``CLOSE_COOLDOWN_MINUTES`` and require manual
# intervention.  See 2026-05-06 SPY zombie incident.
PARTIAL_CLOSE_COOLDOWN_THRESHOLD = 3
CLOSE_COOLDOWN_MINUTES = 60

# ── Hard cap on positions per underlying ────────────────────────────────────
# Belt-and-suspenders against the dedup gate failing — if anything lets the
# cycle slip a duplicate plan past ``_tickers_with_open_orders`` /
# ``fetch_open_positions``, this cap is the final stop.  Pre-2026-05-13 the
# dedup gate only filtered tickers whose positions reported ``signal=HOLD``,
# which silently allowed Stage 2 to fire for tickers whose positions had
# triggered exit signals (profit_target, regime_shift, etc.) but hadn't yet
# closed — see the 2026-05-12 GLD dedup-gate-bypass incident.
#
# Counting actual broker positions (any signal) and capping at 1 closes both
# failure modes structurally.  Set ``MAX_POSITIONS_PER_TICKER>1`` only if you
# explicitly want stacked spreads on a single underlying.
MAX_POSITIONS_PER_TICKER = 1

# Per-sector position cap. Imported alongside ``MAX_POSITIONS_PER_TICKER``
# from ``trading_agent.sector_map``. The cap is enforced in Stage 1.5
# (right after position fetch) — any ticker whose sector already has
# ``MAX_POSITIONS_PER_SECTOR`` filled spreads is added to
# ``tickers_with_positions`` so Stage 2 skips it. Prevents
# over-concentration when the trading universe has multiple tickers in
# the same sector (e.g., XLF + KRE both Financials).
from trading_agent.sector_map import (  # noqa: E402
    MAX_POSITIONS_PER_SECTOR,
    sector_for,
)

# OCC option symbol → underlying root.  Format ROOT(1-6) + YYMMDD(6) +
# C/P(1) + STRIKE*1000(8).  Used both for stale-order cancel scoping
# and open-order dedup, so identical to the dashboard helper in
# streamlit/live_monitor.py.
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def _root_from_occ(symbol: str) -> str:
    """Return the OCC root (e.g. ``GOOG260508P00337500`` → ``GOOG``)
    or the empty string if the symbol can't be parsed."""
    if not symbol:
        return ""
    m = _OCC_RE.match(symbol)
    return m.group(1) if m else ""


class TradingAgent:
    """
    Autonomous credit-spread trading agent.

    Lifecycle::
        agent = TradingAgent.from_env()
        results = agent.run_cycle()
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # ── Cycle singleton lock ────────────────────────────────────────
        # Only one cycle may run at a time per agent instance. The
        # Streamlit watchdog observer + auto-refresh + scheduled timer
        # can all fire simultaneously; on 2026-05-06 nine cycles
        # started within 38 seconds of each other after Schwab began
        # feeding chains successfully, and two of them raced to submit
        # SPY Bear Call spreads only 2 seconds apart (orders 462b61ba
        # and 9d271dc7 — same strikes, doubled risk).
        #
        # ``_cycle_lock.acquire(blocking=False)`` at the top of
        # ``run_cycle`` makes the second concurrent caller short-
        # circuit cleanly instead of running a full duplicate cycle.
        # Non-reentrant: if a cycle internally calls run_cycle (it
        # shouldn't), the second call returns the skip result.
        self._cycle_lock = threading.Lock()

        # ── Partial-close cooldown — journal-derived ───────────────────
        # Pre-2026-05-13 the cooldown state lived in two in-memory
        # dicts on this instance.  That broke in production because
        # the cycle subprocess exits after each cycle and the dicts
        # reset to empty — the streak never accumulated across cycles
        # and the cooldown never engaged.  See the 2026-05-13 XLF/GLD
        # post-mortem.
        #
        # State is now derived on every read from the journal's
        # ``close_failed`` and ``closed`` rows.  See
        # ``_close_failed_streak_within_window`` for the derivation
        # and ``_close_cooldown_minutes_remaining`` / ``_journal_close_event``
        # for the readers.  No instance dicts needed.

        # MarketDataProvider now satisfies both MarketDataPort and
        # AccountPort (its get_account_info/is_market_open methods no
        # longer accept base_url — the adapter owns its endpoint).
        #
        # The factory routes between Alpaca / Schwab / Yahoo based on
        # the surface tag and per-surface env vars. The agent's main
        # cycle is the LIVE surface, so it looks up
        # MARKET_DATA_PROVIDER_LIVE first, then MARKET_DATA_PROVIDER,
        # then defaults to Alpaca. Schwab gives retail brokerage
        # holders real-time options data without Alpaca's OPRA tier;
        # Alpaca remains the execution broker either way.
        from trading_agent.market_data_factory import build_market_data_provider
        self.data_provider: MarketDataPort = build_market_data_provider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=config.alpaca.data_url,
            alpaca_base_url=config.alpaca.base_url,
            surface="live",
        )

        # Load the active Strategy-Profile preset (Conservative / Balanced /
        # Aggressive / Custom) chosen via the Streamlit dashboard. The preset
        # bundles max_delta + per-strategy DTE + width policy + C/W floor +
        # max-risk %, plus the directional-bias filter. Falls back to BALANCED
        # if STRATEGY_PRESET.json is missing or malformed (logged at info-level).
        # Each subprocess re-reads the file on init, so dashboard changes apply
        # on the next 5-min cycle without restarting the loop.
        self.preset: PresetConfig = load_active_preset()
        logger.info("Strategy preset → %s", self.preset.to_summary_line())

        # Risk knobs come from the preset; everything else stays from the
        # AppConfig env-loaded baseline (liquidity floors, margin, etc).
        max_delta        = self.preset.max_delta
        min_credit_ratio = self.preset.min_credit_ratio
        max_risk_pct     = self.preset.max_risk_pct

        # ── .env-vs-preset mismatch warning ──────────────────────────────
        # Pre-2026-05-06 the executor read ``config.trading.max_risk_pct``
        # (.env) while the RiskManager read ``preset.max_risk_pct`` — when
        # the two disagreed, trades passed the risk gate at one budget
        # then got sized at another, breaking the C/W floor invariant.
        # The fix routes BOTH through the preset; this warning surfaces
        # the silent override that previously caused operator confusion
        # ("I changed MAX_RISK_PCT in .env, why isn't anything different?").
        env_risk = config.trading.max_risk_pct
        if abs(env_risk - max_risk_pct) > 1e-6:
            logger.warning(
                "MAX_RISK_PCT mismatch: .env=%.4f vs STRATEGY_PRESET.json=%.4f. "
                "The PRESET wins (it's the live control surface; .env is a "
                "fallback for tests predating the preset system). Edit the "
                "Strategy Profile panel in the dashboard, or set "
                "STRATEGY_PRESET.json:max_risk_pct directly to change risk "
                "sizing. Setting .env:MAX_RISK_PCT alone has no effect on "
                "live trading.",
                env_risk, max_risk_pct,
            )

        self.regime_classifier = RegimeClassifier(self.data_provider)
        self.strategy_planner = StrategyPlanner(
            data_provider=self.data_provider,
            max_delta=max_delta,
            min_credit_ratio=min_credit_ratio,
            dte_vertical=self.preset.dte_vertical,
            dte_iron_condor=self.preset.dte_iron_condor,
            dte_mean_reversion=self.preset.dte_mean_reversion,
            dte_window_days=self.preset.dte_window_days,
            width_mode=self.preset.width_mode,
            width_value=self.preset.width_value,
            preset=self.preset,
        )
        self.risk_manager = RiskManager(
            max_risk_pct=max_risk_pct,
            min_credit_ratio=min_credit_ratio,
            max_delta=max_delta,
            liquidity_max_spread=config.trading.liquidity_max_spread,
            liquidity_bps_of_mid=config.trading.liquidity_bps_of_mid,
            stale_spread_pct=config.trading.stale_spread_pct,
            max_buying_power_pct=config.trading.max_buying_power_pct,
            margin_multiplier=config.trading.margin_multiplier,
            # Adaptive mode: replace the static C/W floor with a Δ-aware one —
            # same formula the scanner uses: |Δshort| × (1 + edge_buffer).
            delta_aware_floor=(self.preset.scan_mode == "adaptive"),
            edge_buffer=self.preset.edge_buffer,
        )
        # The three broker-facing adapters are typed as ports so the
        # agent core never reaches into vendor-specific internals.  The
        # concrete classes still satisfy these Protocols via structural
        # typing — no inheritance required.
        self.executor: ExecutionPort = OrderExecutor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
            trade_plan_dir=config.logging.trade_plan_dir,
            dry_run=config.trading.dry_run,
            data_provider=self.data_provider,   # for live quote refresh on execution
            # CRITICAL: must match the values passed to RiskManager above
            # — the executor's position sizer uses these to compute qty,
            # and a mismatch with the validator means trades pass the
            # risk gate at one budget and get sized at another (breaking
            # the C/W floor invariant).  Pre-2026-05-06 these read from
            # config.trading (.env), but the RiskManager read from the
            # preset; the two diverged whenever the operator changed
            # one without the other.  Now both route through the preset.
            max_risk_pct=max_risk_pct,                           # shared w/ RiskManager #4
            min_credit_ratio=min_credit_ratio,                   # shared w/ RiskManager #2
            # Adaptive mode: live-credit recheck + 1-tick haircut both use the
            # same Δ-aware floor RiskManager is enforcing, so a scanner-picked
            # plan can never be vetoed at execution time by a stale static
            # floor.  Mirrors the kwargs passed to RiskManager above.
            delta_aware_floor=(self.preset.scan_mode == "adaptive"),
            edge_buffer=self.preset.edge_buffer,
        )
        self.position_monitor: PositionsPort = PositionMonitor(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
            # Profit-take threshold comes from the preset (skill 30). 50% is
            # the Balanced default; Conservative rides to 60%, Aggressive
            # banks at 40%. A mismatch between agent.py and the
            # PositionMonitor default would silently apply the wrong rule.
            profit_target_pct=self.preset.profit_target_pct,
        )
        self.order_tracker: OrdersPort = OrderTracker(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            base_url=config.alpaca.base_url,
        )

        # JournalKB writes signals.jsonl + signals.md into the same
        # trade_journal/ directory — no extra folder needed.
        journal_dir = (
            config.intelligence.journal_dir
            if config.intelligence and config.intelligence.journal_dir
            else "trade_journal"
        )
        # Dry-run uses a SEPARATE journal file (skill 19 §1 decoupling).
        # Pre-2026-05-21 dry-run wrote into ``signals_live.jsonl`` with
        # a ``mode="DRY-RUN"`` tag; three readers (EOD recap, realized
        # P&L tile, _render_closed_today) forgot to filter and produced
        # the -$2,860 phantom-loss family of bugs. Now:
        #   live cycles → signals_live.jsonl
        #   dry-run     → signals_dryrun.jsonl
        # Production consumers (Telegram EOD, dashboard P&L tile) read
        # only signals_live.jsonl → dry-run pollution is structurally
        # impossible, not "carefully defended against".
        is_dry = bool(config.trading.dry_run)
        journal_run_mode = "dryrun" if is_dry else "live"
        self.journal_kb = JournalKB(
            journal_dir, run_mode=journal_run_mode, dry_run=is_dry,
        )
        logger.info(
            "Journal stream: %s (file: %s)",
            journal_run_mode, self.journal_kb.jsonl_path,
        )

        # ── Telegram alerter (skill 32, opt-in via env) ──────────────
        # No-op when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are unset,
        # so existing installs without an env tweak see no behavioral
        # change. Only fires for operator-actionable events (PDT block,
        # close-cooldown engagement, post-close open failure).
        self.telegram = TelegramNotifier()
        if self.telegram.is_active:
            logger.info("Telegram alerter active (operator notifications enabled)")
        else:
            logger.debug(
                "Telegram alerter inactive — set TELEGRAM_BOT_TOKEN + "
                "TELEGRAM_CHAT_ID to enable operator notifications"
            )

        # Daily state store (drawdown + exit debounce)
        self.daily_state = DailyStateStore(config.logging.trade_plan_dir)

        # Register graceful-shutdown handlers so SIGTERM (docker stop,
        # systemctl stop, cron cancel) flushes the journal + logs
        # instead of losing the in-flight write buffer.
        _shutdown.install_signal_handlers(journal=self.journal_kb)

        # Intelligence layer (LLM + RAG + Journal) — optional
        self.llm_analyst = self._init_intelligence(config)

        # Tiered sentiment pipeline (news → FinGPT → verifier) behind a
        # single facade.  The facade owns a cycle-scoped ThreadPoolExecutor
        # so the worker is always drained on cycle exit — no more
        # instance-lifetime pool surviving a SIGTERM mid-call.  Short
        # circuits (earnings calendar, content-hash cache) live inside
        # the facade, not the agent, so the orchestration call site
        # stays trivial.
        # Short-circuit when the intelligence layer is fully disabled so we
        # don't pay the SentimentPipeline factory cost (transitively imports
        # NewsAggregator + FinGPTAnalyser + EarningsCalendar) for tests and
        # rule-only deployments. Was a measurable per-test hit because every
        # `TradingAgent(...)` instantiation triggered the factory.
        intel_cfg = config.intelligence
        intel_disabled = (
            intel_cfg is None
            or not getattr(intel_cfg, "enabled", False)
        )
        self.sentiment_pipeline: Optional[SentimentPipeline] = (
            None if intel_disabled
            else SentimentPipeline.from_config(intel_cfg)
        )
        # Back-compat handles — a few tests and journal helpers still
        # reach for these instance attributes by name.  Expose the
        # underlying components without reintroducing ownership.
        self.fingpt_analyser: Optional[FinGPTAnalyser] = (
            self.sentiment_pipeline.fingpt if self.sentiment_pipeline else None
        )
        self.news_aggregator: Optional[NewsAggregator] = (
            self.sentiment_pipeline.news_aggregator if self.sentiment_pipeline else None
        )
        self.sentiment_verifier: Optional[SentimentVerifier] = (
            self.sentiment_pipeline.verifier if self.sentiment_pipeline else None
        )

    def _init_intelligence(self, config: AppConfig):
        """Initialize the LLM intelligence layer if enabled."""
        intel_cfg = config.intelligence
        if not intel_cfg or not intel_cfg.enabled:
            logger.info("Intelligence layer DISABLED — rule-based mode only")
            return None

        try:
            llm_config = LLMConfig(
                provider=intel_cfg.llm_provider,
                base_url=intel_cfg.llm_base_url,
                model=intel_cfg.llm_model,
                embedding_model=intel_cfg.llm_embedding_model,
                api_key=intel_cfg.llm_api_key,
                temperature=intel_cfg.llm_temperature,
            )
            llm_client = LLMClient(llm_config)
            journal = TradeJournal(journal_dir=intel_cfg.journal_dir)
            kb = KnowledgeBase(
                kb_dir=intel_cfg.knowledge_base_dir,
                embed_fn=llm_client.embed,
            )
            analyst = LLMAnalyst(
                llm_client=llm_client,
                journal=journal,
                knowledge_base=kb,
                enabled=True,
            )
            logger.info(
                "Intelligence layer ENABLED — model=%s, provider=%s",
                intel_cfg.llm_model, intel_cfg.llm_provider,
            )
            return analyst

        except Exception as exc:
            logger.warning(
                "Failed to initialize intelligence layer: %s — "
                "continuing in rule-based mode",
                exc,
            )
            return None

    # Sentiment pipeline construction lives in
    # ``SentimentPipeline.from_config`` (wired once in __init__).  The
    # per-cycle call site is `_with_sentiment_pipeline()` below, which
    # context-manages the facade's background pool.

    @classmethod
    def from_env(cls, env_path: str = None) -> "TradingAgent":
        """Factory: create agent from environment / .env file."""
        config = load_config(env_path)
        setup_logging(config.logging.log_level, config.logging.log_dir)
        return cls(config)

    # ==================================================================
    # Main cycle — public entry point
    # ==================================================================

    def run_cycle(self) -> Dict:
        """
        Execute one full cycle with a hard timeout guard.

        If the cycle exceeds CYCLE_TIMEOUT_SECONDS the guard logs a
        TIMEOUT event to JournalKB and terminates the process so the
        scheduler can launch the next run cleanly.

        Concurrency
        -----------
        Singleton via ``self._cycle_lock``.  Only one cycle runs at a
        time per agent instance.  Concurrent callers (Streamlit's
        watchdog observer + auto-refresh + scheduled timer all firing
        on the same tick) get a fast-path "skip" result without
        starting a second cycle, so the dedup gate never sees a
        race-window where Alpaca hasn't yet listed an in-flight order.
        """
        # ── Singleton acquire (non-blocking) ─────────────────────────────
        if not self._cycle_lock.acquire(blocking=False):
            logger.info(
                "run_cycle skipped — another cycle is already in progress; "
                "this trigger is dropped to prevent duplicate submissions.")
            return {
                "status": "skipped_concurrent",
                "reason": "another cycle already in progress",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        try:
            return self._run_cycle_with_timeout_guard()
        finally:
            self._cycle_lock.release()

    def _run_cycle_with_timeout_guard(self) -> Dict:
        """Inner: timeout guard + exception handling, original body of
        ``run_cycle`` before the singleton wrapper was added."""
        # --- Timeout guard -----------------------------------------------
        def _on_timeout():
            reason = (
                f"Cycle TIMEOUT after {CYCLE_TIMEOUT_SECONDS}s "
                "— killing process to unblock scheduler"
            )
            logger.error(reason)
            self.journal_kb.log_cycle_error(
                "cycle_timeout",
                {"timeout_seconds": CYCLE_TIMEOUT_SECONDS},
            )
            # Use hard_exit — process may be hung on a syscall or
            # deadlocked on a mutex; clean teardown is unsafe.
            _shutdown.hard_exit(1, reason=reason)

        timer = threading.Timer(CYCLE_TIMEOUT_SECONDS, _on_timeout)
        timer.daemon = True
        timer.start()
        # -----------------------------------------------------------------

        cycle_start = time.monotonic()
        # Cycle-scope the sentiment pipeline's worker pool so its thread
        # is drained cleanly on every exit path — including SIGTERM and
        # uncaught exceptions.  The nullcontext fallback covers the
        # case where intelligence is disabled entirely.
        pipeline_ctx: contextlib.AbstractContextManager = (
            self.sentiment_pipeline
            if self.sentiment_pipeline is not None
            else contextlib.nullcontext()
        )
        try:
            with pipeline_ctx:
                result = self._run_cycle_impl()
        except Exception as exc:
            logger.exception("CYCLE FAILED with unhandled exception: %s", exc)
            self.journal_kb.log_cycle_error(
                str(exc), {"tickers": self.config.trading.tickers},
            )
            result = {
                "status": "error",
                "reason": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            timer.cancel()

        elapsed = time.monotonic() - cycle_start
        logger.info(
            "Cycle completed in %.1fs / %ds budget",
            elapsed, CYCLE_TIMEOUT_SECONDS,
        )
        if elapsed > CYCLE_TIMEOUT_SECONDS * 0.8:
            logger.warning(
                "Cycle used %.0f%% of time budget — consider "
                "reducing ticker count or increasing interval",
                100 * elapsed / CYCLE_TIMEOUT_SECONDS,
            )
        return result

    # ==================================================================
    # Cycle implementation
    # ==================================================================

    def _run_cycle_impl(self) -> Dict:
        """Core cycle logic — called inside run_cycle()'s timeout guard."""
        # --- After-hours shutdown guard -----------------------------------
        # Exit cleanly (code 0) when invoked outside NYSE market hours so
        # that a cron scheduler does not waste cycles on a closed market.
        # Set FORCE_MARKET_OPEN=true (or force_market_open=True in config)
        # to bypass this check in tests or paper-trading outside hours.
        if not self.config.trading.force_market_open and not _is_within_market_hours():
            now_et = datetime.now(EASTERN)
            reason = (
                f"Outside NYSE market hours "
                f"({now_et.strftime('%A %H:%M ET')}) — shutting down cleanly"
            )
            logger.info(reason)
            self.journal_kb.log_cycle_error(
                "after_hours_shutdown",
                {
                    "local_time_et": now_et.isoformat(),
                    "market_window": market_window_str(),
                },
            )
            # ── EOD Telegram recap (skill 32 §3.8) ─────────────────────
            # Best-effort, deduped to once per trading day, only fires
            # AFTER today's market close. A failure inside this helper
            # never blocks graceful_exit — the alert is paging, not
            # business-critical.
            try:
                self._maybe_send_eod_summary()
            except Exception as exc:                            # noqa: BLE001
                logger.warning("EOD summary attempt failed: %s", exc)
            # graceful_exit: we decided to stop; logs + journal are healthy.
            _shutdown.graceful_exit(0, reason="after_hours_shutdown", context={
                "local_time_et": now_et.isoformat(),
            })
        # ------------------------------------------------------------------

        logger.info("=" * 70)
        logger.info(
            "TRADING CYCLE START — %s",
            datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "Tickers: %s | Mode: %s | Dry-run: %s",
            self.config.trading.tickers,
            self.config.trading.mode,
            self.config.trading.dry_run,
        )
        logger.info("=" * 70)

        # Pre-flight: fetch account info
        account = self.data_provider.get_account_info()
        if not account:
            msg = "Cannot fetch account info — aborting cycle."
            logger.error(msg)
            self.journal_kb.log_cycle_error(msg)
            return {
                "status": "error",
                "reason": "Account info unavailable",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        account_balance = float(account.get("equity", 0))
        account_buying_power = float(account.get("buying_power", account_balance))
        account_type = "paper" if "paper" in self.config.alpaca.base_url else "live"
        market_open = self.data_provider.is_market_open()

        logger.info(
            "Account: balance=$%s, buying_power=$%s, type=%s, market_open=%s",
            f"{account_balance:,.2f}",
            f"{account_buying_power:,.2f}",
            account_type, market_open,
        )
        logger.info("Schedule interval: %s", self.config.trading.schedule_interval)

        # --- Daily Drawdown Circuit Breaker ---
        if check_daily_drawdown(
            self.daily_state,
            current_equity=account_balance,
            drawdown_limit=self.config.trading.daily_drawdown_limit,
            journal_kb=self.journal_kb,
        ):
            reason = (
                f"Daily drawdown limit "
                f"({self.config.trading.daily_drawdown_limit * 100:.0f}%) "
                f"exceeded — stopping all trading"
            )
            logger.critical(reason)
            # graceful_exit: drawdown is a decided policy stop, not a hang.
            _shutdown.graceful_exit(1, reason="daily_drawdown_breaker", context={
                "equity": account_balance,
                "limit_pct": self.config.trading.daily_drawdown_limit * 100,
            })

        # --- Liquidation Mode Check ---
        liquidation_mode = self._check_liquidation_mode(
            account_balance, account_buying_power)
        if liquidation_mode:
            logger.warning(
                "LIQUIDATION MODE: buying power >%.0f%% used — "
                "closing positions only, no new trades",
                self.config.trading.max_buying_power_pct * 100,
            )

        # ------------------------------------------------------------------
        # Pre-fetch data for all tickers in parallel (5-min optimisation)
        # ------------------------------------------------------------------
        tickers = self.config.trading.tickers
        logger.info("Pre-fetching market data for %d ticker(s)…", len(tickers))
        self.data_provider.prefetch_historical_parallel(tickers)
        self.data_provider.fetch_batch_snapshots(tickers)

        # ------------------------------------------------------------------
        # Stage 1: MONITOR existing positions
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("STAGE 1 — MONITOR EXISTING POSITIONS")
        logger.info("=" * 70)

        monitor_results = self._stage_monitor(account_balance)

        # ── Per-ticker position count + dedup set ──────────────────────
        # Pre-2026-05-13 this only added tickers whose positions reported
        # ``signal=HOLD``.  That silently bypassed the dedup for any
        # ticker whose positions had triggered an exit signal but
        # hadn't yet closed (in particular: GLD on 2026-05-12 was
        # ``regime_shift`` exit-pending but not yet in HOLD, so Stage 2
        # tried to open a new GLD spread).  Now we count every reported
        # position regardless of signal and cap at MAX_POSITIONS_PER_TICKER.
        positions_per_ticker: Dict[str, int] = {}
        positions_per_sector: Dict[str, int] = {}
        for sr in monitor_results.get("positions", []):
            underlying = sr.get("underlying", "")
            if underlying:
                positions_per_ticker[underlying] = (
                    positions_per_ticker.get(underlying, 0) + 1
                )
                sec = sector_for(underlying)
                positions_per_sector[sec] = (
                    positions_per_sector.get(sec, 0) + 1
                )

        tickers_with_positions: Set[str] = {
            t for t, n in positions_per_ticker.items()
            if n >= MAX_POSITIONS_PER_TICKER
        }
        # Per-sector cap: any ticker in the universe whose sector is
        # already at MAX_POSITIONS_PER_SECTOR gets blocked. Stage 2's
        # existing dedup uses ``tickers_with_positions`` so we just
        # union the sector-blocked tickers into that set.
        sectors_at_cap: Set[str] = {
            s for s, n in positions_per_sector.items()
            if n >= MAX_POSITIONS_PER_SECTOR
        }
        if sectors_at_cap:
            tickers_with_positions |= {
                t for t in tickers
                if sector_for(t) in sectors_at_cap
                # Don't fire the sector block for tickers ALREADY caught
                # by the per-ticker cap — keeps the log noise focused on
                # the sector-as-additional-gate signal.
                and t not in tickers_with_positions
            }
        if positions_per_ticker:
            logger.info(
                "Open positions snapshot — %s (cap: %d/ticker, "
                "%d/sector); sectors at cap: %s",
                {t: n for t, n in sorted(positions_per_ticker.items())},
                MAX_POSITIONS_PER_TICKER,
                MAX_POSITIONS_PER_SECTOR,
                sorted(sectors_at_cap) or "[]",
            )

        # ------------------------------------------------------------------
        # Stage 1.5: Stale-order maintenance
        # ------------------------------------------------------------------
        # Cancel stuck limits that have been on the book longer than
        # STALE_ORDER_MAX_AGE_MIN so the next planning pass can re-price
        # against a fresh mid.  Then collect the tickers of every
        # remaining open order and union them into tickers_with_positions
        # — the original dedup only blocked tickers with FILLED spreads,
        # which let the cycle stack identical limit orders on top of an
        # unfilled one (root cause of the duplicate-GOOG issue).
        try:
            self._cancel_stale_orders(tickers)
        except Exception as exc:
            logger.warning("Stale-order maintenance failed: %s", exc)

        try:
            tickers_with_open_orders = self._tickers_with_open_orders()
            if tickers_with_open_orders:
                logger.info(
                    "Open orders pending fill on: %s — these tickers will "
                    "be skipped in Stage 2 to avoid duplicate submissions",
                    sorted(tickers_with_open_orders),
                )
                tickers_with_positions |= tickers_with_open_orders
        except Exception as exc:
            logger.warning("Open-order dedup failed: %s", exc)

        # ------------------------------------------------------------------
        # Stage 2: OPEN new positions
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("STAGE 2 — OPEN NEW POSITIONS")
        logger.info("=" * 70)

        # ── Fail-closed dedup gate ──────────────────────────────────────
        # If Stage 1 couldn't read open positions from the broker (RPC
        # failure — Connection reset, timeout, 5xx), we don't actually
        # know what's already on the book.  In that state, opening new
        # positions is a duplicate-submission risk: on 2026-05-05 a
        # 100ms TCP reset led to a duplicate DIA Iron Condor.  Skip
        # the new-trade loop entirely and try again next cycle when
        # the broker call recovers.  Stage 1's exit/close path already
        # short-circuited (no spreads to evaluate), so this is purely
        # about not adding new exposure under uncertainty.
        if monitor_results.get("fetch_failed"):
            logger.warning(
                "STAGE 2 SKIPPED — broker position fetch failed in "
                "Stage 1.  Refusing to open new positions until the "
                "next cycle confirms current account state.  This is "
                "the fail-closed dedup gate; transient broker outages "
                "should not cause duplicate orders.")
            self.journal_kb.log_cycle_error(
                "stage2_skipped_position_fetch_failed",
                {"reason": "broker_position_fetch_returned_none"},
            )
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "account_balance": account_balance,
                "monitor": monitor_results,
                "new_trades": [],
                "order_summary": {
                    "open_orders": {"total": 0},
                    "recent_fills": {"total": 0},
                    "skipped_reason": "stage2_skipped_position_fetch_failed",
                },
            }

        new_trade_results = []
        for ticker in tickers:
            # Check for shutdown between tickers so a SIGTERM mid-cycle
            # stops the loop cleanly at the next safe point.
            if _shutdown.shutdown_requested():
                logger.warning(
                    "Shutdown requested — aborting ticker loop at %s", ticker)
                break

            if ticker in tickers_with_positions:
                # tickers_with_positions includes BOTH filled spreads
                # (Stage 1) and pending limit orders (Stage 1.5 dedup),
                # so this branch covers both.  The journal reason is
                # kept generic to avoid log churn for a single string.
                logger.info(
                    "[%s] Already has an open spread or pending order — skipping",
                    ticker,
                )
                self.journal_kb.log_signal(
                    ticker=ticker,
                    action="skipped_existing",
                    price=self._cached_price(ticker),
                    raw_signal={"reason": "Existing open position or pending order"},
                )
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": "Existing open position or pending order",
                })
                continue

            if liquidation_mode:
                self.journal_kb.log_signal(
                    ticker=ticker,
                    action="skipped_liquidation_mode",
                    price=self._cached_price(ticker),
                    raw_signal={"reason": "Liquidation Mode — buying power exhausted"},
                )
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": "Liquidation Mode",
                })
                continue

            try:
                result = self._process_ticker(
                    ticker, account_balance, account_buying_power,
                    account_type, market_open,
                )
                new_trade_results.append(result)
            except InsufficientDataError as exc:
                # Expected condition — ticker has too little history for a
                # reliable SMA-200 classification. Log as a warning and
                # skip cleanly; this is not an error worth paging on.
                logger.warning("[%s] Skipped — %s", ticker, exc)
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": f"insufficient_data: {exc}",
                })
            except Exception as exc:
                logger.exception("[%s] Unhandled error: %s", ticker, exc)
                self.journal_kb.log_error(
                    ticker=ticker,
                    error=str(exc),
                    price=self._cached_price(ticker),
                )
                new_trade_results.append({
                    "ticker": ticker,
                    "status": "error",
                    "reason": str(exc),
                })

        # ------------------------------------------------------------------
        # Order status summary
        # ------------------------------------------------------------------
        order_summary = self._check_order_statuses()

        logger.info("=" * 70)
        logger.info(
            "TRADING CYCLE COMPLETE — %d tickers processed",
            len(new_trade_results),
        )
        self._print_summary(new_trade_results)
        logger.info("=" * 70)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account_balance": account_balance,
            "monitor": monitor_results,
            "new_trades": new_trade_results,
            "order_summary": order_summary,
        }

    # ==================================================================
    # Stage 1: Position monitoring
    # ==================================================================

    def _stage_monitor(self, account_balance: float) -> Dict:
        """
        Fetch positions, classify regimes, evaluate exit signals,
        and close spreads that need closing.

        Return-shape contract
        ---------------------
        Always returns a dict with keys ``total_spreads`` (int),
        ``positions`` (list), ``closed`` (list).  When the broker
        position fetch fails, the dict additionally carries
        ``fetch_failed=True`` so the calling cycle can fail closed
        on Stage 2 (don't open new positions when we don't know what
        already exists).  An empty broker response (``[]``) is
        distinct from a failed RPC (``None``) — the former returns
        ``fetch_failed=False`` (clean slate), the latter
        ``fetch_failed=True`` (unknown state).
        """
        positions = self.position_monitor.fetch_open_positions()

        if positions is None:
            # RPC failed — propagate the unknown state up so Stage 2
            # can short-circuit. Don't conflate with "no positions."
            logger.warning(
                "Position fetch failed — Stage 2 will skip new entries "
                "this cycle to avoid duplicate-submission bugs.")
            # Surface to operator via journal so the dashboard's
            # Recent Journal Entries panel shows the connectivity
            # blip without requiring a log scrape.  Bypasses dedup
            # so successive cycles all surface the issue (warnings
            # are material — see _DEDUP_BYPASS_ACTIONS).
            try:
                self.journal_kb.log_warning(
                    source="position_monitor",
                    message=(
                        "Broker position fetch failed (retries "
                        "exhausted) — Stage 2 entries skipped this "
                        "cycle to prevent duplicate submission. The "
                        "dedup gate is failing closed by design."
                    ),
                )
            except Exception as exc:                              # noqa: BLE001
                logger.warning("Failed to journal position-fetch warning: %s", exc)
            return {"total_spreads": 0, "positions": [], "closed": [],
                    "fetch_failed": True}

        if not positions:
            logger.info("No open option positions found.")
            return {"total_spreads": 0, "positions": [], "closed": [],
                    "fetch_failed": False}

        trade_plans = self._load_trade_plans()
        spreads = self.position_monitor.group_into_spreads(positions, trade_plans)
        if not spreads:
            logger.info("Could not match positions to any trade plans.")
            return {"total_spreads": 0, "positions": [], "closed": []}

        underlyings = {s.underlying for s in spreads}
        current_regimes: Dict = {}
        underlying_prices: Dict[str, float] = {}
        for ticker in underlyings:
            try:
                analysis = self.regime_classifier.classify(ticker)
                current_regimes[ticker] = analysis.regime
                logger.info(
                    "[%s] Current regime: %s",
                    ticker, analysis.regime.value,
                )
            except Exception as exc:
                logger.warning("[%s] Could not classify regime: %s", ticker, exc)
            price = self._cached_price(ticker)
            if price > 0:
                underlying_prices[ticker] = price

        spreads = self.position_monitor.evaluate(
            spreads, current_regimes, underlying_prices)

        # ── PDT same-day-open detection ─────────────────────────────────
        # Build the set of tickers with action="submitted" today (UTC).
        # Used below to suppress REGIME_SHIFT exits on small accounts so
        # we don't trip pattern-day-trading protection.  See the SPY
        # zombie-close incident on 2026-05-06.
        same_day_tickers = self._tickers_opened_today()
        pdt_restricted = account_balance < PDT_EQUITY_THRESHOLD
        if pdt_restricted and same_day_tickers:
            logger.info(
                "PDT-restricted account ($%.2f < $25K). Same-day-open "
                "tickers: %s. REGIME_SHIFT exits will be suppressed for "
                "these (real-risk exits like STRIKE_PROXIMITY still fire).",
                account_balance, sorted(same_day_tickers),
            )

        # ── PDT-aware DTE cap (skill 33) ─────────────────────────────
        # When sub-$25K, cap each strategy's DTE at preset.pdt_dte_cap
        # so the planner picks shorter expirations. Reduces overnight
        # drift risk for positions that can't be closed same-day. When
        # the account is at-or-above the PDT threshold, restore the
        # preset's original DTE (no cap). Called every cycle so an
        # intraday balance transition adjusts the planner reactively.
        cap = self.preset.pdt_dte_cap if pdt_restricted else None
        try:
            self.strategy_planner.apply_pdt_dte_cap(cap)
            if pdt_restricted:
                logger.info(
                    "PDT DTE cap engaged — vertical/IC/mean-reversion "
                    "capped at %d days (was preset value).",
                    cap,
                )
        except Exception as exc:                                # noqa: BLE001
            logger.warning(
                "apply_pdt_dte_cap raised — preset DTE unchanged: %s", exc,
            )

        # Reactive PDT-block set (skill 17 §4). Read journal once per
        # cycle for tickers whose close attempt earlier today was rejected
        # by Alpaca with code 40310100. Subsequent same-day attempts on
        # these tickers will short-circuit in the close loop below — no
        # speculative suppression, only after a real broker response.
        pdt_blocked_today = self._pdt_blocked_today_tickers()
        if pdt_blocked_today:
            logger.info(
                "PDT-blocked-today tickers (Alpaca confirmed code 40310100): "
                "%s. Auto-close suppressed for these until UTC midnight.",
                sorted(pdt_blocked_today),
            )

        closed = []
        for spread in spreads:
            if spread.exit_signal != ExitSignal.HOLD and self._should_exit_spread(spread):
                # ── Cooldown guard ────────────────────────────────────────
                # If this ticker has hit PARTIAL_CLOSE_COOLDOWN_THRESHOLD
                # consecutive partial fills, park further auto-closes for
                # CLOSE_COOLDOWN_MINUTES.  Naive retry is hopeless on
                # zombie partial-fill states (Alpaca's reason — PDT,
                # uncovered, buying-power — doesn't lift on its own; the
                # operator has to manually clean up via the broker UI).
                cooldown_left = self._close_cooldown_minutes_remaining(
                    spread.underlying)
                if cooldown_left > 0:
                    streak, _ = self._close_failed_streak_within_window(
                        spread.underlying)
                    logger.warning(
                        "[%s] Close cooldown active — %d min remaining "
                        "(%d consecutive partial fills). Skipping retry. "
                        "Manually close the position on Alpaca's UI to "
                        "clear the zombie state.",
                        spread.underlying, cooldown_left,
                        streak,
                    )
                    continue

                # ── PDT REGIME_SHIFT suppression ────────────────────────
                # On a sub-$25K account, opening AND closing the same
                # ticker on the same day counts as a "day trade" under
                # FINRA's PDT rule — Alpaca rejects with code 40310100
                # (`pattern day trading protection`).  Hold overnight
                # and let tomorrow's cycle close it without the same-day
                # day-trade flag.  Real-risk exits (STRIKE_PROXIMITY,
                # HARD_STOP, DTE_SAFETY) still fire because those are
                # worth the PDT hit — UNLESS Alpaca has actually
                # confirmed the PDT block via API response earlier
                # today (see next gate).
                if (pdt_restricted
                        and spread.underlying in same_day_tickers
                        and spread.exit_signal == ExitSignal.REGIME_SHIFT):
                    logger.warning(
                        "[%s] PDT-suppressed REGIME_SHIFT exit — position "
                        "opened today, account equity $%.2f < $25K PDT "
                        "threshold. Holding overnight to avoid day-trade "
                        "flag. Will reconsider on next session's cycle.",
                        spread.underlying, account_balance,
                    )
                    continue

                # ── Reactive PDT-block suppression (skill 17 §4) ─────────
                # Skip the close attempt ONLY when Alpaca has already
                # responded with code 40310100 (pattern day trading
                # protection) on this ticker earlier today. The flag
                # was written by ``_journal_close_event`` after the
                # first failed attempt; the helper reads the journal
                # and looks for a same-UTC-day pdt_blocked_today marker.
                # This is REACTIVE — we don't speculatively gate based
                # on PDT-restriction + same-day-open. The first close
                # attempt always proceeds; only AFTER seeing the broker
                # respond with PDT do we stop retrying. Reduces API
                # spam from 18+ failed close attempts per day per
                # stuck position (observed in pi-diagnostics 2026-05-20)
                # down to a single attempt + clean journal record.
                if spread.underlying in pdt_blocked_today:
                    self.journal_kb.log_signal(
                        ticker=spread.underlying,
                        action="skipped_pdt_blocked",
                        price=self._cached_price(spread.underlying),
                        raw_signal={
                            "reason": (
                                "Alpaca returned code 40310100 (pattern "
                                "day trading protection) earlier today. "
                                "Subsequent close attempts suppressed "
                                "until UTC midnight."
                            ),
                            "exit_signal": spread.exit_signal.value,
                            "exit_reason": spread.exit_reason,
                            "account_balance": account_balance,
                            "spread_strategy": spread.strategy_name,
                        },
                        notes=(
                            f"skipped_pdt_blocked: {spread.strategy_name}, "
                            f"{spread.exit_signal.value} — broker confirmed "
                            f"PDT block earlier today"
                        ),
                    )
                    logger.warning(
                        "[%s] Auto-close suppressed — Alpaca confirmed PDT "
                        "block earlier today (code 40310100). Position is "
                        "at risk (%s: %s) but retrying is futile until "
                        "UTC midnight. Manual close via Alpaca UI accepts "
                        "the day-trade flag if intervention is needed.",
                        spread.underlying,
                        spread.exit_signal.value,
                        spread.exit_reason,
                    )
                    continue

                # ── Defensive-roll dispatcher (skill 31) ─────────────────
                # When STRIKE_PROXIMITY would otherwise just close at a loss,
                # try evaluating a defensive roll (close + reopen further-OTM).
                # _maybe_defensive_roll returns a result dict if the roll
                # ran (success or failure — already journalled), else None
                # if the roll was declined and we should fall through to
                # the normal close path. The six-predicate evaluator (skill 31)
                # ensures the roll only fires when it's economically positive.
                if (spread.exit_signal == ExitSignal.STRIKE_PROXIMITY
                        and self.preset.defensive_roll_enabled):
                    roll_outcome = self._maybe_defensive_roll(
                        spread, account_balance,
                    )
                    if roll_outcome is not None:
                        closed.append(roll_outcome)
                        continue

                # Capture pre-close state for the audit-trail journal entry —
                # spread.legs may be partially populated after close_spread()
                # mutates leg statuses, so snapshot what the user is paying
                # for now.
                close_context = {
                    "strategy":       spread.strategy_name,
                    "exit_signal":    spread.exit_signal.value,
                    "exit_reason":    spread.exit_reason,
                    "exit_immediate": spread.exit_signal in IMMEDIATE_EXIT_SIGNALS,
                    "net_unrealized_pl": float(getattr(spread, "net_unrealized_pl", 0) or 0),
                    "original_credit":   float(getattr(spread, "original_credit", 0) or 0),
                    "max_loss":          float(getattr(spread, "max_loss", 0) or 0),
                    "spread_width":      float(getattr(spread, "spread_width", 0) or 0),
                    "expiration":        getattr(spread, "expiration", "") or "",
                    "short_strikes":     list(getattr(spread, "short_strikes", []) or []),
                    "regime_at_close":   (current_regimes.get(spread.underlying).value
                                          if current_regimes.get(spread.underlying) is not None
                                          else "unknown"),
                    "origin":            getattr(spread, "origin", "trade_plan"),
                }
                if self.config.trading.dry_run:
                    logger.info(
                        "[%s] DRY RUN — would close %s (%s: %s)",
                        spread.underlying, spread.strategy_name,
                        spread.exit_signal.value, spread.exit_reason,
                    )
                    closed.append({
                        "underlying": spread.underlying,
                        "signal": spread.exit_signal.value,
                        "reason": spread.exit_reason,
                        "action": "dry_run_close",
                    })
                    self._journal_close_event(
                        spread, close_context, leg_results=[],
                        fill_status="dry_run", dry_run=True,
                    )
                else:
                    result = self.executor.close_spread(spread)
                    closed.append(result)
                    leg_results = result.get("leg_results", []) if isinstance(result, dict) else []
                    fill_status = (
                        "complete" if (isinstance(result, dict) and result.get("all_closed"))
                        else "partial"
                    )
                    # ── Cooldown bookkeeping ──────────────────────────────
                    # IMPORTANT: do this BEFORE journaling so the row can
                    # capture the resulting cooldown state.  On a complete
                    # close, clear any prior partial-fill state so the
                    # next opening of this ticker starts fresh.  On a
                    # partial fill, increment the per-ticker counter; if
                    # it reaches PARTIAL_CLOSE_COOLDOWN_THRESHOLD, park
                    # the ticker in a cooldown window so the cycle stops
                    # hammering the broker with hopeless retries.  The
                    # cooldown timestamp is then surfaced in the journal
                    # row so the dashboard can render a manual-
                    # intervention warning banner without re-deriving the
                    # state from log scraping.
                    if fill_status == "complete":
                        self._clear_close_cooldown(spread.underlying)
                    else:
                        self._record_partial_close(spread.underlying)
                    self._journal_close_event(
                        spread, close_context, leg_results=leg_results,
                        fill_status=fill_status, dry_run=False,
                    )

                if self.llm_analyst:
                    self._learn_from_close(spread)

        summary = self.position_monitor.summary(spreads)
        summary["closed"] = closed
        return summary

    # ==================================================================
    # Defensive roll (skill 31)
    # ==================================================================

    def _maybe_defensive_roll(self, spread, account_balance: float
                              ) -> Optional[Dict]:
        """Run the six-predicate evaluator and execute the roll if all pass.

        Skill 31 §3 — orchestrates the close-then-open atomic sequence.

        Returns:
          * ``Dict`` if the roll ran (whether it succeeded, partially
            failed, or got declined post-predicate) — caller adds to
            ``closed`` list and skips the normal close path
          * ``None`` if the evaluator declined OR an upstream step
            (regime classify / plan / risk) failed — caller falls
            through to the normal close path, which closes the
            threatened spread at the STRIKE_PROXIMITY signal

        All outcomes are journalled inside this method — caller doesn't
        need to add a separate journal call.
        """
        ticker = spread.underlying
        cur_price = self._cached_price(ticker)
        if cur_price <= 0:
            logger.warning(
                "[%s] Defensive-roll declined — no cached price; "
                "falling through to normal close.", ticker)
            return None

        # ── Build candidate ─────────────────────────────────────────
        # Re-run the planner so the candidate uses the current chain
        # and current regime. If the regime has flipped (e.g. the
        # underlying broke through what was sideways into bearish),
        # the planner returns a different strategy and the evaluator's
        # SAME-DIRECTION predicate will reject — that's the correct
        # behaviour (we don't want to roll a bull put into a bear
        # call mid-defense).
        try:
            analysis = self.regime_classifier.classify(ticker)
            new_plan = self.strategy_planner.plan(ticker, analysis)
        except Exception as exc:
            logger.warning(
                "[%s] Defensive-roll: plan attempt crashed (%s) — "
                "falling through to normal close.", ticker, exc)
            return None

        if not getattr(new_plan, "valid", False):
            logger.info(
                "[%s] Defensive-roll: planner returned no valid candidate "
                "— falling through to normal close.", ticker)
            self.journal_kb.log_signal(
                ticker=ticker, action="defensive_roll_evaluated",
                price=cur_price,
                raw_signal={
                    "decision": "skip_no_candidate",
                    "reason": "planner returned no valid candidate",
                    "exit_signal": spread.exit_signal.value,
                },
            )
            return None

        # ── Compute evaluator inputs ────────────────────────────────
        # debit_to_close estimated from unrealized P&L. For a credit
        # spread opened at $X credit currently showing $Y loss, the
        # debit to close ≈ X + Y per spread. Crude but conservative —
        # the predicate compares against new credit which is also
        # a planner estimate.
        original_credit_per_spread = float(
            getattr(spread, "original_credit", 0.0) or 0.0
        )
        unrealized_per_spread = float(
            getattr(spread, "net_unrealized_pl_per_spread",
                    getattr(spread, "net_unrealized_pl", 0.0) or 0.0)
        )
        debit_to_close = max(
            0.0, original_credit_per_spread - unrealized_per_spread
        )

        # DTE remaining
        from datetime import date as _date
        try:
            exp = spread.expiration  # "YYYY-MM-DD"
            exp_date = _date.fromisoformat(exp)
            dte = max(0, (exp_date - _date.today()).days)
        except (ValueError, TypeError):
            dte = 0

        # The threatened-spread's short strike is the one closest to spot.
        short_strikes = list(getattr(spread, "short_strikes", []) or [])
        if not short_strikes:
            logger.warning(
                "[%s] Defensive-roll: no short_strikes recorded on "
                "spread; falling through.", ticker)
            return None
        closest_short = min(
            short_strikes, key=lambda k: abs(cur_price - k)
        )

        # New short leg delta — from the new plan's "sell" legs.
        new_sell_legs = [l for l in new_plan.legs if l.action == "sell"]
        if not new_sell_legs:
            logger.info(
                "[%s] Defensive-roll: new plan has no sell legs — "
                "falling through.", ticker)
            return None
        new_short_delta = max(abs(l.delta) for l in new_sell_legs)

        roll_inputs = RollEvalInputs(
            spot=cur_price,
            short_strike=closest_short,
            dte=dte,
            strategy_name=spread.strategy_name,
            debit_to_close=debit_to_close,
            current_roll_count=0,    # MVP: budget enforced per-position-id
                                     # via journal lookup is a follow-up.
                                     # max_defensive_rolls_per_position=1
                                     # still bounds total rolls in steady
                                     # state because a successful roll
                                     # opens a NEW position with no roll
                                     # history.
            new_short_delta=new_short_delta,
            new_strategy_name=new_plan.strategy_name,
            new_projected_credit=float(new_plan.net_credit),
            new_spread_width=float(new_plan.spread_width),
            new_cw_ratio=float(new_plan.credit_to_width_ratio),
            roll_trigger_min_pct=self.preset.roll_trigger_min_pct,
            roll_trigger_max_pct=self.preset.roll_trigger_max_pct,
            min_dte_for_roll=self.preset.min_dte_for_roll,
            max_defensive_rolls_per_position=(
                self.preset.max_defensive_rolls_per_position
            ),
            edge_buffer=self.preset.edge_buffer,
        )

        decision = evaluate_defensive_roll(roll_inputs)
        eval_payload = {
            "decision":                decision.value,
            "spot":                    cur_price,
            "short_strike":            closest_short,
            "proximity_pct":           roll_inputs.proximity_pct(),
            "dte":                     dte,
            "debit_to_close":          debit_to_close,
            "new_strategy":            new_plan.strategy_name,
            "new_credit":              float(new_plan.net_credit),
            "new_cw_ratio":            float(new_plan.credit_to_width_ratio),
            "new_short_delta":         new_short_delta,
            "preset_trigger_band":     [
                self.preset.roll_trigger_min_pct,
                self.preset.roll_trigger_max_pct,
            ],
            "preset_min_dte":          self.preset.min_dte_for_roll,
        }

        if decision != RollDecision.ROLL:
            # Predicate failed — journal the decision and fall through
            # to the normal close.
            self.journal_kb.log_signal(
                ticker=ticker, action="defensive_roll_evaluated",
                price=cur_price, raw_signal=eval_payload,
            )
            logger.info(
                "[%s] Defensive-roll declined (%s) — falling through "
                "to normal close at STRIKE_PROXIMITY.",
                ticker, decision.value,
            )
            return None

        # ── All six predicates passed — run the roll ────────────────
        # Build a RiskVerdict for the new plan so the executor's
        # standard live-or-dry-run submission path handles broker calls.
        try:
            verdict: RiskVerdict = self.risk_manager.evaluate(
                new_plan, account_balance, "margin", True,
                self.config.trading.force_market_open,
            )
        except Exception as exc:
            logger.warning(
                "[%s] Defensive-roll: risk check crashed (%s) — "
                "falling through to normal close.", ticker, exc)
            self.journal_kb.log_signal(
                ticker=ticker, action="defensive_roll_evaluated",
                price=cur_price,
                raw_signal={**eval_payload,
                            "post_eval_error": f"risk_check: {exc}"},
            )
            return None

        if not verdict.approved:
            logger.info(
                "[%s] Defensive-roll: new plan REJECTED at risk gate (%s) "
                "— falling through to normal close.",
                ticker, verdict.summary,
            )
            self.journal_kb.log_signal(
                ticker=ticker, action="defensive_roll_evaluated",
                price=cur_price,
                raw_signal={**eval_payload,
                            "post_eval_risk_rejection": verdict.summary},
            )
            return None

        logger.info(
            "[%s] Defensive-roll ROLL — closing threatened spread + "
            "opening %s @ short Δ-%.2f for $%.2f credit (was $%.2f, "
            "C/W %.3f)",
            ticker, new_plan.strategy_name, new_short_delta,
            new_plan.net_credit, original_credit_per_spread,
            new_plan.credit_to_width_ratio,
        )

        roll_result = self.executor.roll_position_defensive(spread, verdict)

        # Journal the outcome under one canonical action so the
        # dashboard / traceability can attribute it.
        action_map = {
            "roll_completed":     "defensive_roll_completed",
            "roll_dry_run":       "defensive_roll_dry_run",
            "roll_close_failed":  "defensive_roll_close_failed",
            "roll_open_failed":   "defensive_roll_open_failed",
        }
        outcome_status = roll_result.get("status", "roll_open_failed")
        journal_action = action_map.get(outcome_status, "defensive_roll_open_failed")

        self.journal_kb.log_signal(
            ticker=ticker, action=journal_action,
            price=cur_price,
            raw_signal={
                **eval_payload,
                "roll_status":     outcome_status,
                "close_all_ok":    bool(
                    (roll_result.get("close_result") or {}).get("all_closed")
                ),
                "open_status":     (
                    (roll_result.get("open_result") or {}).get("status")
                    if roll_result.get("open_result") else None
                ),
                "from_short_strikes": short_strikes,
                "from_expiration":    spread.expiration,
                "to_short_delta":     new_short_delta,
                "to_expiration":      getattr(new_plan, "expiration", ""),
            },
            exec_status=outcome_status,
            notes=(
                f"Defensive roll {outcome_status}; "
                f"close_ok={(roll_result.get('close_result') or {}).get('all_closed')}"
            ),
        )

        # Operator alert (skill 32) — CRITICAL case: close filled but
        # open failed, leaving us flat. Always-on (no dedup): this is
        # a once-per-position emergency that the operator MUST see.
        if outcome_status == "roll_open_failed":
            open_reason = (
                ((roll_result.get("open_result") or {}).get("reason"))
                or "unspecified — see executor log"
            )
            self._send_telegram_alert(
                ticker=ticker,
                alert_type="roll_open_failed",
                send_fn=self.telegram.notify_open_failed_after_close,
                strategy=spread.strategy_name,
                reason=str(open_reason),
            )

        return {
            "underlying":  ticker,
            "signal":      spread.exit_signal.value,
            "action":      journal_action,
            "roll_status": outcome_status,
            "result":      roll_result,
        }

    def _journal_close_event(self, spread, ctx: Dict,
                             leg_results: List[Dict], fill_status: str,
                             dry_run: bool) -> None:
        """
        Emit a structured close-attempt row to ``signals_live.jsonl``.

        Action mapping (added 2026-05-06)
        ---------------------------------
        Pre-fix every close attempt — complete OR partial-fill zombie —
        was tagged ``action="closed"``.  That made the dashboard's
        "Closed Today" tile lie: a SPY position that hit the partial-
        fill loop 11 times produced 11 ``closed`` rows even though the
        position was still open on the broker side.  Operators saw
        "11 closed today" and panicked.

        Now the action depends on ``fill_status``:

          * ``complete`` / ``dry_run``  → ``action="closed"``
            (the position is gone from the broker; exit attribution is
            valid; "Closed Today" can count it.)
          * ``partial`` / anything else → ``action="close_failed"``
            (the position is still open; one or more legs were rejected
            by Alpaca — typically PDT, uncovered, or insufficient
            buying power; "Close Failures" tile counts these
            separately so the operator can intervene.)

        Why this exists
        ---------------
        Pre-2026-05-06 the agent logged exit signals only to
        ``logs/trading_agent.log`` (rolling Python log).  The
        ``signals_live.jsonl`` journal — which feeds the dashboard's
        Risk Guardrail grid, the Latest Trades panel, and any future
        replay tooling — only had ``submitted`` / ``rejected`` /
        ``skipped_*`` rows.  Closed positions effectively "disappeared"
        from the dashboard's view: they vanished from Open Positions
        when the broker reported them gone, but no journal row
        explained why.  This helper closes the gap so every close has
        an audit-trail entry with the exit signal as justification.

        Schema
        ------
        ::

          action: "closed" | "close_failed"
          notes:
              "closed: <Strategy>, P&L=<±$X>, <exit_reason>"
                                         (fill_status=complete/dry_run)
              "close_failed: <Strategy>, <exit_reason>, partial fill"
                                         (fill_status=partial)
          raw_signal:
              {strategy, exit_signal, exit_reason, exit_immediate,
               net_unrealized_pl, original_credit, max_loss,
               spread_width, expiration, short_strikes,
               regime_at_close, origin,
               leg_close_results: [{symbol, status}, ...],
               fill_status: "complete" | "partial" | "failed" | "dry_run",
               mode: "live" | "dry_run"}
        """
        try:
            pl = ctx["net_unrealized_pl"]
            sign = "+" if pl >= 0 else ""
            # Action + notes branch on fill_status — see method docstring.
            # SKILL 19 §2 — three distinct close actions:
            #   * "closed"         → real broker close, all legs filled.
            #                        P&L counts toward realized total.
            #   * "dry_run_close"  → synthetic close in dry-run mode.
            #                        Position is NOT actually closed at the
            #                        broker; the row is informational only.
            #                        Must NOT count toward realized P&L.
            #                        Pre-2026-05-21 this used action="closed"
            #                        which on a stuck-in-dry-run position
            #                        accumulated 22 phantom -$130 closes
            #                        on a single day → -$2,860 false loss.
            #   * "close_failed"   → broker rejected one or more legs
            #                        (partial fill). Position still open.
            if fill_status == "complete":
                action = "closed"
                note = (
                    f"closed: {ctx['strategy']}, P&L={sign}${pl:.2f}, "
                    f"{ctx['exit_signal']}"
                )
                exec_status = f"closed_{ctx['exit_signal']}"
            elif fill_status == "dry_run":
                action = "dry_run_close"
                note = (
                    f"dry_run_close: {ctx['strategy']}, "
                    f"would-be P&L={sign}${pl:.2f}, "
                    f"{ctx['exit_signal']}"
                )
                exec_status = f"dry_run_close_{ctx['exit_signal']}"
            else:
                action = "close_failed"
                # Surface the leg-level failure summary so an operator
                # can immediately tell which leg(s) Alpaca rejected.
                failed_legs = [
                    leg.get("symbol", "?")
                    for leg in (leg_results or [])
                    if isinstance(leg, dict)
                       and leg.get("status") != "closed"
                ]
                failed_str = (
                    f" failed_legs={','.join(failed_legs)}"
                    if failed_legs else ""
                )
                note = (
                    f"close_failed: {ctx['strategy']}, "
                    f"{ctx['exit_signal']}, fill_status={fill_status}"
                    f"{failed_str}"
                )
                exec_status = f"close_failed_{ctx['exit_signal']}"

            payload: Dict[str, Any] = dict(ctx)
            payload["leg_close_results"] = [
                {"symbol": leg.get("symbol", ""),
                 "status": leg.get("status", "unknown")}
                for leg in (leg_results or [])
                if isinstance(leg, dict)
            ]
            payload["fill_status"] = fill_status
            # Mode tag MUST match the value written by _log_signal so the
            # dashboard's current_mode filter (case-sensitive equality) keeps
            # close rows in the same view as their corresponding submitted
            # rows.  Pre-2026-05-07 this wrote lowercase ("live") while
            # _log_signal wrote uppercase ("LIVE"), which dropped close rows
            # whenever the dashboard filtered by mode — and that hid them
            # from the supersede-PENDING-on-close logic in the grid (the
            # SPY 2026-05-07 PENDING bug).
            payload["mode"] = "DRY-RUN" if dry_run else "LIVE"

            # ── Cooldown surface ──────────────────────────────────────
            # When a close_failed row is written AND the ticker has hit
            # the partial-fill threshold, embed the cooldown deadline +
            # a human-readable reason so the dashboard can render a
            # "manual intervention required" banner without re-deriving
            # the state.  Pre-cooldown rows still get the running
            # streak count so the operator can see "we're at 2/3,
            # one more partial fills the cooldown."
            if action == "close_failed":
                # Streak counts EXISTING close_failed rows in the
                # cooldown window (since the most recent ``closed``).
                # +1 represents the row we're about to write — so a
                # fresh first failure produces "1/3", three in a row
                # produces "3/3" and engages the cooldown.  This is
                # the journal-derived replacement for the pre-2026-05-13
                # in-memory counter that didn't survive process restarts.
                existing_streak, _ = self._close_failed_streak_within_window(
                    spread.underlying)
                streak = existing_streak + 1
                payload["partial_close_streak"] = streak
                payload["partial_close_threshold"] = (
                    PARTIAL_CLOSE_COOLDOWN_THRESHOLD
                )
                if streak >= PARTIAL_CLOSE_COOLDOWN_THRESHOLD:
                    deadline = (
                        datetime.now(timezone.utc)
                        + timedelta(minutes=CLOSE_COOLDOWN_MINUTES)
                    )
                    payload["close_cooldown_until"] = deadline.isoformat()
                    payload["close_cooldown_reason"] = (
                        f"{streak} consecutive partial fills "
                        f"≥ threshold {PARTIAL_CLOSE_COOLDOWN_THRESHOLD}; "
                        f"auto-close suppressed for "
                        f"{CLOSE_COOLDOWN_MINUTES} min — manual broker "
                        f"intervention required to clear zombie state."
                    )
                    # Operator alert (skill 32). Fires once when the
                    # cooldown FIRST engages today — dedup helper
                    # ensures subsequent cooldown re-engagements on
                    # the same ticker the same day don't re-spam.
                    failed_legs = (
                        ",".join(
                            (leg.get("symbol") or "?")
                            for leg in (leg_results or [])
                            if isinstance(leg, dict)
                               and leg.get("status") != "closed"
                        ) or "—"
                    )
                    self._send_telegram_alert(
                        ticker=spread.underlying,
                        alert_type="close_cooldown",
                        send_fn=self.telegram.notify_close_cooldown,
                        strategy=ctx.get("strategy", spread.strategy_name),
                        streak=streak,
                        threshold=PARTIAL_CLOSE_COOLDOWN_THRESHOLD,
                        cooldown_until_iso=deadline.isoformat(),
                        failed_legs=failed_legs,
                    )

                # ── Reactive PDT-block detection (skill 17 §4) ──────────
                # If ANY leg's error string contains Alpaca's PDT code
                # (40310100) OR the literal "pattern day trading" phrase,
                # mark this row so the next cycle's close loop can read
                # the journal and short-circuit retries that are doomed
                # for the rest of UTC-today. The flag is date-keyed so
                # the marker self-expires at midnight UTC — when the
                # next trading day begins, the position is no longer
                # "same-day-open" from FINRA's perspective and the close
                # will succeed normally. Reactive design: we only mark
                # AFTER the broker has actually responded with PDT, so
                # legitimate closes are never speculatively suppressed.
                pdt_signals = ("40310100", "pattern day trading")
                pdt_blocked = any(
                    isinstance(leg, dict)
                    and any(
                        s in str(leg.get("error", "")).lower()
                        for s in pdt_signals
                    )
                    for leg in (leg_results or [])
                )
                if pdt_blocked:
                    today_utc = (
                        datetime.now(timezone.utc).date().isoformat()
                    )
                    payload["pdt_blocked_today"] = True
                    payload["pdt_blocked_date"] = today_utc
                    payload["pdt_blocked_reason"] = (
                        "Alpaca returned code 40310100 (pattern day "
                        "trading protection) on one or more legs. "
                        "Subsequent close attempts on this ticker will "
                        "be suppressed until UTC midnight, when the "
                        "position is no longer same-day-open."
                    )
                    logger.critical(
                        "[%s] PDT block detected from Alpaca response — "
                        "marking ticker pdt_blocked_today=%s. Further "
                        "auto-closes will be suppressed for the rest of "
                        "the trading day. Manual close via Alpaca UI is "
                        "still possible if you accept the day-trade flag.",
                        spread.underlying, today_utc,
                    )
                    # Operator alert (skill 32). Dedup helper short-circuits
                    # re-sends so DIA's 18 daily detections collapse to 1
                    # Telegram message.
                    self._send_telegram_alert(
                        ticker=spread.underlying,
                        alert_type="pdt_block",
                        send_fn=self.telegram.notify_pdt_block,
                        strategy=ctx.get("strategy", spread.strategy_name),
                        exit_signal=ctx.get("exit_signal",
                                            spread.exit_signal.value),
                        exit_reason=ctx.get("exit_reason",
                                            spread.exit_reason),
                        account_balance=float(
                            getattr(spread, "account_balance", 0.0) or 0.0
                        ),
                    )

            self.journal_kb.log_signal(
                ticker=spread.underlying,
                action=action,
                price=self._cached_price(spread.underlying),
                exec_status=exec_status,
                notes=note,
                raw_signal=payload,
            )

            # ── Position-closed alert (skill 32 §3.6) ────────────────
            # Fire only when the close fully filled (action="closed",
            # not "close_failed" — partial fills are surfaced via the
            # stuck-position banner + cooldown alert instead).
            #
            # Dedup gate: same close event must not re-fire. Pi-deploy
            # 2026-05-20 hotfix — operators in DRY-RUN mode received
            # 3 identical "DIA closed" alerts every 5 minutes because
            # the synthetic dry-run close path journals action="closed"
            # every cycle without actually closing the position at the
            # broker; STRIKE_PROXIMITY then re-fires next cycle and the
            # alert repeats. Dedup key combines ticker + expiration +
            # exit_signal + UTC date so a legitimate same-day re-trade
            # (different expiration) still alerts, but the same stuck
            # position can't spam.
            if action == "closed" and self.telegram.is_active:
                exit_sig_val = ctx.get("exit_signal",
                                       spread.exit_signal.value)
                exp_val = ctx.get("expiration", spread.expiration or "")
                dedup_alert_type = (
                    f"position_closed:{exp_val}:{exit_sig_val}"
                )
                self._send_telegram_alert(
                    ticker=spread.underlying,
                    alert_type=dedup_alert_type,
                    send_fn=self.telegram.notify_position_closed,
                    strategy=ctx.get("strategy",
                                     spread.strategy_name),
                    exit_signal=exit_sig_val,
                    exit_reason=ctx.get("exit_reason",
                                        spread.exit_reason or ""),
                    realized_pl=float(
                        ctx.get("net_unrealized_pl", 0.0) or 0.0
                    ),
                    original_credit=float(
                        ctx.get("original_credit", 0.0) or 0.0
                    ),
                    max_loss=float(ctx.get("max_loss", 0.0) or 0.0),
                )
        except Exception as exc:                                # noqa: BLE001
            # Never let a journaling failure break the cycle — the close
            # itself already happened.  Log the error and move on.
            logger.warning(
                "[%s] Failed to journal close event: %s",
                getattr(spread, "underlying", "?"), exc,
            )

    # ==================================================================
    # PDT + partial-close cooldown helpers
    # ==================================================================

    def _tickers_opened_today(self) -> Set[str]:
        """
        Return the set of underlyings that submitted a new spread today
        (UTC).  Used to suppress same-day REGIME_SHIFT exits on PDT-
        restricted accounts (< $25K equity).

        Reads the journal directly so the answer survives a Streamlit
        restart or agent-loop restart — the in-memory state is
        deliberately *not* the source of truth here.  We tolerate ANY
        parse error by returning an empty set, because the worst-case
        consequence of a false-empty is "we attempted a close that
        Alpaca might reject for PDT" — which is exactly the failure
        mode that already had to be handled before this helper existed.
        """
        try:
            jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
            if not jsonl_path or not os.path.isfile(jsonl_path):
                return set()
            today_utc = datetime.now(timezone.utc).date()
            tickers: Set[str] = set()
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("action") != "submitted":
                        continue
                    ts_str = rec.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        continue
                    # Normalise to UTC for the date comparison.
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts.astimezone(timezone.utc).date() != today_utc:
                        continue
                    tk = rec.get("ticker")
                    if tk:
                        tickers.add(tk)
            return tickers
        except Exception as exc:                                # noqa: BLE001
            logger.warning(
                "Failed to read same-day-open tickers from journal: %s",
                exc,
            )
            return set()

    def _telegram_alert_already_sent_today(self, ticker: str,
                                            alert_type: str) -> bool:
        """True if a successful Telegram alert for this (ticker, alert_type)
        was already journalled earlier the same UTC day.

        Skill 32 §3.4 — dedup gate. The first time the agent detects a
        PDT block on DIA, ``notify_pdt_block`` fires and a
        ``telegram_alert_sent`` row gets written. Across the next ~78
        cycles of the same trading day, the same DIA detection would
        otherwise re-fire the alert — this helper short-circuits the
        send so the operator sees ONE alert per ticker per day per type.

        Date-keyed → self-clears at UTC midnight, matching the
        pdt_blocked_today marker's lifetime.
        """
        try:
            jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
            if not jsonl_path or not os.path.isfile(jsonl_path):
                return False
            today_iso = datetime.now(timezone.utc).date().isoformat()
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("action") != "telegram_alert_sent":
                        continue
                    if rec.get("ticker") != ticker:
                        continue
                    rs = rec.get("raw_signal") or {}
                    if rs.get("alert_type") != alert_type:
                        continue
                    if rs.get("alert_date") != today_iso:
                        continue
                    return True
            return False
        except Exception as exc:                                # noqa: BLE001
            logger.warning(
                "Failed to read telegram-alert dedup state: %s", exc,
            )
            # Fail-open: if we can't read dedup state, prefer one extra
            # alert over no alert at all — duplicates are cheap, missing
            # a stuck-position alert is not.
            return False

    def _send_telegram_alert(self, ticker: str, alert_type: str,
                              send_fn, **payload) -> None:
        """Helper that combines dedup + send + journal write.

        ``send_fn`` is a bound method on ``self.telegram``. Returns
        nothing — failures are logged but never propagate. On
        success, writes a journal row so the same
        ``(ticker, alert_type, UTC date)`` won't re-fire today.

        Ticker propagation (2026-05-21 hotfix). Five of the six
        ``notify_*`` methods take ``ticker`` as a required first
        positional/named arg; the sixth (``notify_eod_summary``)
        uses ``ticker="__eod__"`` as a sentinel and doesn't accept
        it in its signature. We introspect ``send_fn``'s parameters
        and pass ``ticker=ticker`` only when the signature accepts
        it. Pre-hotfix the helper called ``send_fn(**payload)``
        unconditionally — ``notify_position_closed("DIA", ...)``
        failed with ``TypeError: missing 1 required positional
        argument: 'ticker'`` on every real close. Caught by the
        exception handler, silently lost the alert. See pi
        2026-05-21 09:34:19 trace.
        """
        if not self.telegram.is_active:
            return
        if self._telegram_alert_already_sent_today(ticker, alert_type):
            logger.debug(
                "[%s] Telegram %s alert already sent today — skipping",
                ticker, alert_type,
            )
            return
        try:
            import inspect
            try:
                sig_params = inspect.signature(send_fn).parameters
            except (TypeError, ValueError):
                sig_params = {}
            call_kwargs = dict(payload)
            if "ticker" in sig_params:
                call_kwargs["ticker"] = ticker
            ok = send_fn(**call_kwargs)
        except Exception as exc:                                # noqa: BLE001
            logger.warning(
                "[%s] Telegram %s alert raised: %s",
                ticker, alert_type, exc,
            )
            return
        if not ok:
            # Notifier already logged the failure detail — don't journal
            # a fake-success row that would block legitimate retries.
            return
        try:
            today_iso = datetime.now(timezone.utc).date().isoformat()
            self.journal_kb.log_signal(
                ticker=ticker,
                action="telegram_alert_sent",
                price=self._cached_price(ticker),
                raw_signal={
                    "alert_type": alert_type,
                    "alert_date": today_iso,
                    **{k: v for k, v in payload.items()
                       if isinstance(v, (str, int, float, bool))},
                },
                notes=f"telegram_alert_sent: {alert_type}",
            )
        except Exception as exc:                                # noqa: BLE001
            # Journal write failure here means dedup will be lossy for
            # this ticker today — but the alert DID go out, so the
            # operator isn't blind. Log + continue.
            logger.warning(
                "[%s] Telegram alert sent but journal-dedup write failed: %s",
                ticker, exc,
            )

    def _build_eod_summary(self) -> Dict:
        """Aggregate today's journal rows into the EOD recap payload.

        Skill 32 §3.8 — pure read of signals_live.jsonl. Returns a dict
        the notifier consumes directly. Never raises — partial data is
        better than no alert.

        Aggregations:
          * opens_today / closes_today: per-event mini-summaries
          * realized_pl_today: sum of net_unrealized_pl on every closed row
          * cycles_today: count of distinct cycle_start journal events
          * errors_today: count of cycle_error / warning rows
          * last_balance: most-recent account_balance value seen today
          * starting_balance: earliest account_balance value seen today
          * stuck_tickers: tickers with pdt_blocked_today=today OR
                           future close_cooldown_until
        """
        result: Dict = {
            "opens_today": [],
            "closes_today": [],
            "realized_pl_today": 0.0,
            "cycles_today": 0,
            "errors_today": 0,
            "last_balance": 0.0,
            "starting_balance": None,
            "stuck_tickers": [],
        }
        try:
            jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
            if not jsonl_path or not os.path.isfile(jsonl_path):
                return result
            # Trading-session date filter (2026-05-21 hotfix). Earlier
            # version used datetime.utcnow().date() for ``today_iso``,
            # which pulled in Wed-evening journal writes (20:00 ET Wed
            # = 00:00 UTC Thu) into Thursday's EOD recap — that's how
            # 22 phantom -$130 dry-run rows showed up in the Telegram
            # summary. The right boundary is the ET calendar date that
            # matches the trading session this recap is summarising.
            today_et_iso = datetime.now(EASTERN).date().isoformat()
            # UTC today retained for fields whose markers are
            # UTC-keyed (pdt_blocked_date specifically — written by
            # _journal_close_event with UTC today).
            today_utc_iso = datetime.now(timezone.utc).date().isoformat()
            now_utc = datetime.now(timezone.utc)
            seen_balances: List[float] = []
            seen_stuck: Dict[str, str] = {}   # ticker → reason
            # The journal has no explicit "cycle_start" row — count
            # distinct ISO minutes instead. Each cycle takes ~30s, so
            # one cycle ≈ one minute-bucket of journal writes. This is
            # a proxy that works without changing every writer to emit
            # a cycle marker.
            seen_minutes: Set[str] = set()
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts_str = rec.get("timestamp", "")
                    if not ts_str:
                        continue
                    # Convert UTC timestamp to ET date for the filter.
                    try:
                        ts_dt = datetime.fromisoformat(ts_str)
                        if ts_dt.tzinfo is None:
                            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                        row_et_date = ts_dt.astimezone(EASTERN).date().isoformat()
                    except ValueError:
                        continue
                    if row_et_date != today_et_iso:
                        continue
                    action = rec.get("action", "") or ""
                    rs = rec.get("raw_signal") or {}
                    if not isinstance(rs, dict):
                        rs = {}
                    ticker = rec.get("ticker", "") or ""

                    # Cycle-bucket — distinct minutes of any journal write
                    if len(ts_str) >= 16:
                        seen_minutes.add(ts_str[:16])

                    # Account-balance tracking
                    bal = rs.get("account_balance")
                    if isinstance(bal, (int, float)) and bal > 0:
                        seen_balances.append(float(bal))

                    # Opens
                    if action == "submitted" and ticker:
                        result["opens_today"].append({
                            "ticker": ticker,
                            "strategy": rs.get("strategy", "?"),
                            "credit": rs.get("net_credit") or 0.0,
                        })

                    # Closes — defense against pre-2026-05-21 mislabeled
                    # rows where action="closed" + fill_status="dry_run"
                    # (synthetic dry-run closes that pre-date the
                    # action-label split). Skip them so realised P&L
                    # stays clean. New code writes action="dry_run_close"
                    # so this filter is a no-op going forward.
                    if (action == "closed" and ticker
                            and rs.get("fill_status") != "dry_run"):
                        pl = float(rs.get("net_unrealized_pl") or 0.0)
                        result["closes_today"].append({
                            "ticker": ticker,
                            "strategy": rs.get("strategy", "?"),
                            "exit_signal": rs.get("exit_signal", "?"),
                            "realized_pl": pl,
                        })
                        result["realized_pl_today"] += pl

                    # Error counts
                    if action in ("error", "cycle_error", "warning"):
                        result["errors_today"] += 1

                    # Stuck positions — PDT-blocked. ``pdt_blocked_date``
                    # is UTC-keyed (written by _journal_close_event using
                    # UTC today); compare to UTC today for backward-
                    # compat with that writer. The outer row filter is
                    # ET-keyed but this internal comparison stays UTC.
                    if (action == "close_failed"
                            and rs.get("pdt_blocked_today")
                            and rs.get("pdt_blocked_date") == today_utc_iso):
                        seen_stuck[ticker] = (
                            "PDT block (Alpaca 40310100) — close in "
                            "Alpaca UI or wait for next session"
                        )
                    # Stuck positions — close cooldown still active
                    elif action == "close_failed":
                        cd = rs.get("close_cooldown_until")
                        if cd:
                            try:
                                cd_dt = datetime.fromisoformat(cd)
                                if cd_dt.tzinfo is None:
                                    cd_dt = cd_dt.replace(tzinfo=timezone.utc)
                                if cd_dt > now_utc and ticker not in seen_stuck:
                                    seen_stuck[ticker] = (
                                        "Close cooldown active — manual "
                                        "close required to clear zombie state"
                                    )
                            except ValueError:
                                pass

            if seen_balances:
                result["last_balance"] = seen_balances[-1]
                result["starting_balance"] = seen_balances[0]
            result["cycles_today"] = len(seen_minutes)
            result["stuck_tickers"] = [
                {"ticker": t, "reason": r} for t, r in sorted(seen_stuck.items())
            ]
        except Exception as exc:                                # noqa: BLE001
            logger.warning("Failed to build EOD summary: %s", exc)
        return result

    def _maybe_send_eod_summary(self) -> None:
        """Fire the end-of-day Telegram recap once per trading day.

        Skill 32 §3.8 — called from the after-hours shutdown path. Only
        sends:
          * when the Telegram alerter is configured (env-gated, same as
            every other notify_*)
          * AFTER today's market close in ET (post-16:00 weekday, or
            any time on weekend so Friday's recap goes out)
          * when no eod_summary alert has been journalled for THIS
            ET trading date (dedup helper compares UTC dates, but
            we want ET trading-session uniqueness — see hotfix note)
          * when today's journal contains at least one open or close
            (no trades → no summary worth sending)

        EOD-specific dedup (2026-05-21 hotfix). Wednesday's recap
        fires at 23:41 ET Wed = 03:41 UTC Thu — the journal row
        records ``alert_date=2026-05-21`` (UTC). Without further
        handling, Thursday's intended recap at 16:43 ET Thu =
        20:43 UTC also computes ``today_iso=2026-05-21`` (UTC)
        → false dedup match → Thursday's recap silently
        suppressed. Fix: embed the ET trading session date into
        the ``alert_type`` string so Wed and Thu use distinct
        dedup keys regardless of UTC alignment.
        """
        if not self.telegram.is_active:
            return
        try:
            now_et = datetime.now(EASTERN)
        except Exception:                                       # noqa: BLE001
            return
        is_weekday = now_et.weekday() < 5
        # Pre-market on weekdays: skip (the day hasn't happened yet).
        if is_weekday and now_et.hour < 16:
            return
        # ET trading session date embedded in the alert_type so the
        # dedup helper (which already matches by alert_type) treats
        # each trading day as a unique key. Wed's row is
        # alert_type="eod_summary:2026-05-20"; Thu's intended row
        # is "eod_summary:2026-05-21" — no collision possible.
        trading_session_date = now_et.date().isoformat()
        eod_alert_type = f"eod_summary:{trading_session_date}"
        if self._telegram_alert_already_sent_today(
            ticker="__eod__", alert_type=eod_alert_type,
        ):
            return

        summary = self._build_eod_summary()
        if not summary["opens_today"] and not summary["closes_today"]:
            # No trading activity → nothing to recap. Don't burn a
            # message on an empty day.
            return

        self._send_telegram_alert(
            ticker="__eod__",
            alert_type=eod_alert_type,
            send_fn=self.telegram.notify_eod_summary,
            date_label=now_et.strftime("%A %Y-%m-%d"),
            account_balance=float(summary["last_balance"] or 0.0),
            starting_balance=summary["starting_balance"],
            opens_today=summary["opens_today"],
            closes_today=summary["closes_today"],
            realized_pl_today=summary["realized_pl_today"],
            unrealized_pl_today=0.0,
            cycles_today=summary["cycles_today"],
            errors_today=summary["errors_today"],
            stuck_tickers=summary["stuck_tickers"],
        )

    def _pdt_blocked_today_tickers(self) -> Set[str]:
        """Return underlyings whose journal carries an unexpired
        ``pdt_blocked_today=True`` marker from earlier this UTC day.

        Skill 17 §4 — reactive PDT suppression. The marker is written
        only after Alpaca *actually* responds with code 40310100
        (pattern day trading protection) on a leg DELETE — see
        ``_journal_close_event`` PDT detection block. So a ticker
        only ends up in this set after a real broker response, never
        speculatively.

        State self-clears at UTC midnight because the marker is
        date-keyed (``pdt_blocked_date`` field). When today's date
        moves forward, yesterday's markers stop matching and the
        ticker is once again eligible for close attempts.
        """
        try:
            jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
            if not jsonl_path or not os.path.isfile(jsonl_path):
                return set()
            today_iso = datetime.now(timezone.utc).date().isoformat()
            blocked: Set[str] = set()
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("action") != "close_failed":
                        continue
                    rs = rec.get("raw_signal") or {}
                    if not rs.get("pdt_blocked_today"):
                        continue
                    if rs.get("pdt_blocked_date") != today_iso:
                        continue
                    tk = rec.get("ticker")
                    if tk:
                        blocked.add(tk)
            return blocked
        except Exception as exc:                                # noqa: BLE001
            logger.warning(
                "Failed to derive PDT-blocked tickers from journal: %s",
                exc,
            )
            return set()

    def _close_failed_streak_within_window(
        self, ticker: str, window_min: int = CLOSE_COOLDOWN_MINUTES,
    ) -> Tuple[int, Optional[datetime]]:
        """
        Count consecutive ``action="close_failed"`` rows for ``ticker``
        within the last ``window_min`` minutes, since the most recent
        ``action="closed"`` row (which resets the streak).

        Journal-derived rather than in-memory because the trading-agent
        process exits at the end of each cycle in subprocess-per-cycle
        deployments.  Pre-2026-05-13 this state lived in an in-memory
        dict on ``TradingAgent``, and every cycle restart blew it away
        — the cooldown protection never engaged in production (see the
        2026-05-13 XLF/GLD post-mortem).  Now we count rows in the
        signals journal, which persists across cycles by construction.

        Returns
        -------
        (streak, last_failure_timestamp)
            ``streak`` — number of unsuperseded close_failed rows in
                         the window.
            ``last_failure_timestamp`` — UTC datetime of the most
                         recent counted row, or None if streak == 0.
        """
        jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
        if not jsonl_path or not os.path.isfile(jsonl_path):
            return 0, None

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=window_min)
        # (timestamp, action) tuples for this ticker, in window
        events: List[Tuple[datetime, str]] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("ticker") != ticker:
                        continue
                    if rec.get("action") not in ("close_failed", "closed"):
                        continue
                    ts_str = rec.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        continue
                    events.append((ts, rec.get("action", "")))
        except Exception as exc:                                # noqa: BLE001
            logger.warning(
                "Could not read journal for cooldown derivation: %s", exc,
            )
            return 0, None

        events.sort()  # chronological

        # Find the most recent 'closed' row in window — anything before
        # it doesn't count toward the current streak (the position was
        # closed and the streak resets).
        last_closed_ts: Optional[datetime] = None
        for ts, action in events:
            if action == "closed":
                last_closed_ts = ts

        streak = 0
        last_fail_ts: Optional[datetime] = None
        for ts, action in events:
            if action != "close_failed":
                continue
            if last_closed_ts is not None and ts <= last_closed_ts:
                continue
            streak += 1
            if last_fail_ts is None or ts > last_fail_ts:
                last_fail_ts = ts
        return streak, last_fail_ts

    def _close_cooldown_minutes_remaining(self, ticker: str) -> int:
        """
        Minutes remaining in the cooldown window for ``ticker``, or 0
        if no cooldown is active.

        Derived from the journal: cooldown is engaged once the streak
        reaches ``PARTIAL_CLOSE_COOLDOWN_THRESHOLD``, and lasts
        ``CLOSE_COOLDOWN_MINUTES`` from the most recent partial-close
        timestamp.  Rounds remaining time up to the nearest minute so
        a 30-second remainder still surfaces as 1 (operator should
        know the cooldown hasn't fully elapsed).
        """
        streak, last_fail_ts = self._close_failed_streak_within_window(ticker)
        if streak < PARTIAL_CLOSE_COOLDOWN_THRESHOLD or last_fail_ts is None:
            return 0
        deadline = last_fail_ts + timedelta(minutes=CLOSE_COOLDOWN_MINUTES)
        now = datetime.now(timezone.utc)
        if now >= deadline:
            return 0
        delta = deadline - now
        return max(
            1,
            int(delta.total_seconds() // 60)
            + (1 if delta.total_seconds() % 60 else 0),
        )

    def _record_partial_close(self, ticker: str) -> None:
        """
        No-op since 2026-05-13 — kept for caller compatibility.

        Pre-2026-05-13 this incremented an in-memory counter.  That
        counter was reset every cycle in production (each cycle =
        new process = empty dict), so the cooldown never engaged.
        State is now derived from journal rows on every read; there
        is nothing to "record" in-memory.  The actual partial-fill
        information is journaled by ``_journal_close_event`` via
        ``action="close_failed"``.

        Method retained so existing call sites in ``_stage_monitor``
        don't have to change shape.  Emits an info-level streak log
        line for operator situational awareness.
        """
        streak, _ = self._close_failed_streak_within_window(ticker)
        # +1 for the row about to be journaled by the caller's path.
        upcoming = streak + 1
        if upcoming >= PARTIAL_CLOSE_COOLDOWN_THRESHOLD:
            logger.warning(
                "[%s] %d consecutive partial closes — entering "
                "%d-min cooldown.  Manually clean up the position on "
                "Alpaca's UI to clear the zombie state; the cooldown "
                "will expire automatically thereafter.",
                ticker, upcoming, CLOSE_COOLDOWN_MINUTES,
            )
        else:
            logger.info(
                "[%s] Partial-close streak %d/%d.  Will park in "
                "cooldown after %d.",
                ticker, upcoming, PARTIAL_CLOSE_COOLDOWN_THRESHOLD,
                PARTIAL_CLOSE_COOLDOWN_THRESHOLD,
            )

    def _clear_close_cooldown(self, ticker: str) -> None:
        """
        No-op since 2026-05-13 — kept for caller compatibility.

        Pre-2026-05-13 this purged in-memory cooldown state on a
        successful close.  Now the state is derived from journal
        rows, and the ``action="closed"`` row written by the
        caller's path is itself the supersede signal — the next call
        to ``_close_failed_streak_within_window`` will see it and
        reset the streak to 0.  Nothing to purge here.

        Method retained so existing call sites don't have to change.
        Emits an info log for traceability.
        """
        # Read current state to decide whether to log.  If there's no
        # active streak, the close is just a normal close (no cooldown
        # to clear); stay quiet.
        streak, _ = self._close_failed_streak_within_window(ticker)
        if streak > 0:
            logger.info(
                "[%s] Successful close clears journal-derived "
                "cooldown streak (was %d/%d).",
                ticker, streak, PARTIAL_CLOSE_COOLDOWN_THRESHOLD,
            )

    def _load_trade_plans(self) -> List[Dict]:
        """
        Load trade plans from the plan directory.

        Handles two formats:
          • New  — trade_plan_{TICKER}.json  (state_history array)
          • Old  — trade_plan_{TICKER}_{TS}.json  (flat dict, legacy)
        """
        plan_dir = self.config.logging.trade_plan_dir
        if not os.path.isdir(plan_dir):
            return []

        plans = []
        for path in sorted(glob.glob(
                os.path.join(plan_dir, "trade_plan_*.json"))):
            try:
                with open(path) as fh:
                    data = json.load(fh)

                if "state_history" in data:
                    # New format: flatten all approved history entries
                    for entry in data["state_history"]:
                        plans.append(entry)
                else:
                    # Old timestamped format
                    plans.append(data)

            except Exception as exc:
                logger.warning("Could not load plan %s: %s", path, exc)

        logger.info("Loaded %d trade plan(s) from %s", len(plans), plan_dir)
        return plans

    # ==================================================================
    # Stage 2: New trade entry
    # ==================================================================

    def _process_ticker(self, ticker: str, balance: float,
                        buying_power: float,
                        acct_type: str, market_open: bool) -> Dict:
        """Full four-phase pipeline for a single ticker (+ LLM Phase V)."""
        logger.info("-" * 50)
        logger.info("[%s] Phase I  — PERCEIVE", ticker)

        # Liquidity check on underlying
        underlying_bid_ask = self.data_provider.get_underlying_bid_ask(ticker)

        logger.info("[%s] Phase II — CLASSIFY", ticker)
        analysis: RegimeAnalysis = self.regime_classifier.classify(ticker)

        # --- Directional-bias filter (Strategy Profile) ------------------
        # The active preset can restrict which regimes are tradeable.  This
        # check runs immediately after classify so we short-circuit before
        # spinning up the sentiment pipeline or option-chain fetch — both
        # of which are expensive and pointless when the regime would be
        # filtered out anyway.  Mean-reversion is always allowed (the 3-σ
        # touch override is a fear-spike signal, not a directional view).
        if not regime_is_allowed(
            analysis.regime.value, self.preset.directional_bias
        ):
            reason = (
                f"DirectionalBias={self.preset.directional_bias} blocks "
                f"regime={analysis.regime.value}"
            )
            logger.info("[%s] %s — skipping ticker", ticker, reason)
            self.journal_kb.log_signal(
                ticker=ticker,
                action="skipped_bias",
                price=analysis.current_price,
                raw_signal={
                    "regime": analysis.regime.value,
                    "directional_bias": self.preset.directional_bias,
                    "preset": self.preset.name,
                    "reason": reason,
                },
            )
            return {
                "ticker": ticker,
                "regime": analysis.regime.value,
                "strategy": "skipped_bias",
                "plan_valid": False,
                "risk_approved": False,
                "status": "skipped",
                "reason": reason,
            }

        # --- RSI gate (opt-in, off by default) ---------------------------
        # Refines the strategy choice using RSI alongside the regime. See
        # ``trading_agent/rsi_gate.py`` for the full decision matrix and
        # rationale. Three possible outcomes:
        #   * skip cycle      — RSI says momentum is too active for the
        #                       regime's default strategy
        #   * proceed as-is   — gate has no opinion, planner uses
        #                       ``analysis.regime`` unchanged
        #   * regime override — gate downgrades a sideways/IC plan to a
        #                       single-side vertical (Bull Put or Bear Call);
        #                       we clone ``analysis`` with the new regime
        #                       so the planner picks the right strategy
        #                       AND the journal records the actual choice
        #
        # Toggle via the RSI_GATE_ENABLED env var (default off) so the
        # change can be A/B tested in the backtester without code rolls.
        # When the gate is disabled OR the env var is unset, this block
        # is a no-op and the original regime → strategy mapping applies.
        rsi_gate_enabled = os.environ.get(
            "RSI_GATE_ENABLED", "false"
        ).strip().lower() in ("true", "1", "yes", "on")
        if rsi_gate_enabled:
            decision = evaluate_rsi_gate(analysis.regime, analysis.rsi_14)
            if not decision.allow:
                logger.info(
                    "[%s] RSI gate skipped cycle — %s", ticker, decision.reason
                )
                self.journal_kb.log_signal(
                    ticker=ticker,
                    action="skipped_rsi_gate",
                    price=analysis.current_price,
                    raw_signal={
                        "regime":   analysis.regime.value,
                        "rsi_14":   analysis.rsi_14,
                        "reason":   decision.reason,
                        "preset":   self.preset.name,
                    },
                )
                return {
                    "ticker": ticker,
                    "regime": analysis.regime.value,
                    "strategy": "skipped_rsi_gate",
                    "plan_valid": False,
                    "risk_approved": False,
                    "status": "skipped",
                    "reason": decision.reason,
                }
            if decision.override_regime is not None:
                logger.info(
                    "[%s] RSI gate override — %s", ticker, decision.reason
                )
                # Substitute the analysis with the new regime so the
                # planner picks Bull Put / Bear Call instead of the
                # original Iron Condor. ``RegimeAnalysis`` is a regular
                # dataclass; ``replace`` clones it field-for-field with
                # only the regime swapped. Every downstream consumer
                # (planner, risk manager, journal) sees a consistent
                # view of the cycle's intent.
                import dataclasses
                analysis = dataclasses.replace(
                    analysis, regime=decision.override_regime
                )

        # --- High-IV block: IV rank > 95th pct blocks all new entries ---
        if getattr(analysis, "high_iv_warning", False):
            reason = (
                f"HighIV: IV rank {getattr(analysis, 'iv_rank', 0):.1f} > 95th pct "
                f"— extreme volatility, blocking all new entries"
            )
            logger.warning("[%s] %s | strategy_mode=defense_first", ticker, reason)
            self.journal_kb.log_defense_first(
                ticker, reason, analysis.current_price,
                {
                    "regime": analysis.regime.value,
                    "iv_rank": getattr(analysis, "iv_rank", 0.0),
                    "high_iv_warning": True,
                },
            )
            return {
                "ticker": ticker,
                "regime": analysis.regime.value,
                "strategy": "skipped",
                "plan_valid": False,
                "risk_approved": False,
                "status": "skipped",
                "reason": reason,
                "strategy_mode": "defense_first",
            }

        # Launch tiered sentiment pipeline (Tier-0 earnings → Tier-1 cache →
        # Tier-2 FinGPT + verifier) in the background immediately after
        # Phase II so it runs concurrently with Phase III + IV, adding
        # near-zero wall-clock latency when the Tier-0/1 short-circuit
        # applies (the common case once the cache is warm).
        fingpt_future: Optional[Future] = None
        if self.sentiment_pipeline is not None:
            fingpt_future = self.sentiment_pipeline.submit(
                ticker,
                analysis.regime.value,
                analysis.current_price,
                analysis.rsi_14,
                getattr(analysis, "iv_rank", 0.0),
                self._regime_to_strategy(analysis.regime),
            )

        logger.info(
            "[%s] Phase III — PLAN (%s → %s)",
            ticker, analysis.regime.value,
            self._regime_to_strategy(analysis.regime),
        )
        plan: SpreadPlan = self.strategy_planner.plan(ticker, analysis)

        # Snapshot adaptive-scan results immediately so the next ticker's
        # plan() call doesn't overwrite ``last_scan_candidates`` before
        # we get a chance to journal them.  Returns None in static mode.
        scan_results = self._snapshot_scan_results()

        logger.info("[%s] Phase IV — RISK CHECK", ticker)
        verdict: RiskVerdict = self.risk_manager.evaluate(
            plan, balance, acct_type, market_open,
            self.config.trading.force_market_open,
            underlying_bid_ask=underlying_bid_ask,
            account_buying_power=buying_power,
        )

        # Resolve tiered sentiment pipeline result (earnings → cache →
        # FinGPT + verifier).  Timeout is 60s: news fetching adds
        # ~5-15s on top of inference; the Tier-0/1 short-circuits
        # return in <100 ms so the typical case is effectively free.
        #
        # Efficiency gate: if Phase III/IV produced no tradeable
        # candidate (invalid plan or risk rejection), the sentiment
        # readout is not consumed downstream — cancel the future to
        # skip the LLM calls entirely.
        sentiment: Optional[VerifiedSentimentReport] = None
        if fingpt_future is not None:
            if not (plan.valid and verdict.approved):
                # Best-effort cancellation — if the worker has already
                # begun the LLM call, it'll finish; we just drop the
                # result.  Either way we don't block the cycle.
                fingpt_future.cancel()
                logger.debug(
                    "[%s] No tradeable candidate — sentiment future dropped",
                    ticker,
                )
            else:
                try:
                    sentiment = fingpt_future.result(timeout=60)
                except Exception as exc:
                    logger.warning(
                        "[%s] Sentiment pipeline future failed: %s",
                        ticker, exc,
                    )

        # Phase V: LLM Analysis (if enabled)
        llm_decision = None
        if self.llm_analyst and plan.valid and verdict.approved:
            logger.info("[%s] Phase V  — LLM ANALYSIS", ticker)
            llm_decision = self.llm_analyst.analyze_trade(
                ticker, analysis, plan, verdict, sentiment=sentiment)

            if llm_decision.action == "skip":
                logger.warning(
                    "[%s] LLM SKIPPED trade (confidence=%.2f): %s",
                    ticker, llm_decision.confidence,
                    llm_decision.reasoning[:150],
                )

                self._log_signal(
                    ticker, "skipped_by_llm", analysis, plan, verdict,
                    llm_decision, exec_result=None,
                    scan_results=scan_results,
                )

                return {
                    "ticker": ticker,
                    "regime": analysis.regime.value,
                    "strategy": plan.strategy_name,
                    "plan_valid": plan.valid,
                    "risk_approved": verdict.approved,
                    "llm_decision": "skip",
                    "llm_reasoning": llm_decision.reasoning,
                    "llm_confidence": llm_decision.confidence,
                    "execution": {"status": "skipped_by_llm"},
                    "analysis": self._analysis_dict(analysis),
                }

        # Execute trade
        logger.info("[%s] Phase VI — EXECUTE", ticker)
        exec_result = self.executor.execute(verdict)

        # ── Surface executor errors as journal warnings ─────────────────
        # status="error" means retries were exhausted on a transport-level
        # failure (timeout, connection reset).  The order MAY have landed
        # at Alpaca despite the timeout — the client_order_id makes
        # duplicate submission impossible, but the operator still needs
        # to confirm whether ANY attempt succeeded.  Surface this
        # explicitly in the journal so the dashboard's Recent Journal
        # Entries panel shows the warning + the client_order_id to
        # search for in the broker UI.
        if isinstance(exec_result, dict) and exec_result.get("status") == "error":
            err_msg = exec_result.get("error", "unknown")
            client_order_id = exec_result.get("client_order_id", "n/a")
            attempts = exec_result.get("retry_attempts", 1)
            try:
                self.journal_kb.log_warning(
                    source="executor",
                    ticker=ticker,
                    message=(
                        f"Order submission failed after {attempts} "
                        f"attempt(s): {err_msg}. Search Alpaca for "
                        f"client_order_id={client_order_id} to confirm "
                        f"whether the order ever landed (the broker "
                        f"dedupes duplicates by this key)."
                    ),
                    context={
                        "client_order_id": client_order_id,
                        "retry_attempts": attempts,
                        "strategy": plan.strategy_name,
                    },
                )
            except Exception as exc:                              # noqa: BLE001
                logger.warning(
                    "[%s] Failed to journal executor warning: %s",
                    ticker, exc,
                )

        # Journal the trade (if LLM enabled)
        if (self.llm_analyst and llm_decision
                and exec_result.get("status") in ("submitted", "dry_run")):
            try:
                entry = self.llm_analyst.create_journal_entry(
                    ticker, analysis, plan, verdict, llm_decision)
                entry.order_status = exec_result.get("status", "")
                entry.order_id = exec_result.get("order_id", "")
                trade_id = self.llm_analyst.journal.open_trade(entry)
                exec_result["trade_journal_id"] = trade_id
            except Exception as exc:
                logger.warning("[%s] Failed to journal trade: %s", ticker, exc)

        # Always log to JournalKB
        self._log_signal(
            ticker, exec_result.get("status", "unknown"),
            analysis, plan, verdict, llm_decision, exec_result,
            scan_results=scan_results,
        )

        result = {
            "ticker": ticker,
            "regime": analysis.regime.value,
            "strategy": plan.strategy_name,
            "plan_valid": plan.valid,
            "risk_approved": verdict.approved,
            "execution": exec_result,
            "analysis": self._analysis_dict(analysis),
        }

        if llm_decision:
            result["llm_decision"] = llm_decision.action
            result["llm_confidence"] = llm_decision.confidence
            result["llm_reasoning"] = llm_decision.reasoning
            result["llm_warnings"] = llm_decision.warnings

        if sentiment:
            # Use the SentimentReadout surface (verified_* fields exposed
            # as plain attribute aliases) so the journal emits
            # identical keys regardless of which pipeline tier produced
            # the result (earnings short-circuit, cache hit, or full
            # FinGPT + verifier chain).
            result["fingpt_sentiment"] = sentiment.sentiment_score
            result["fingpt_event_risk"] = sentiment.event_risk
            result["fingpt_recommendation"] = sentiment.recommendation
            result["fingpt_themes"] = sentiment.key_themes
            result["fingpt_agreement"] = sentiment.agreement_score
            result["fingpt_hallucination_flags"] = sentiment.hallucination_flags
            result["fingpt_verified_by"] = sentiment.verifier_model

        return result

    # ==================================================================
    # JournalKB signal helper
    # ==================================================================

    # ------------------------------------------------------------------
    # Adaptive-scan journal helper
    # ------------------------------------------------------------------

    # Top-K candidates persisted per cycle.  10 is enough to reconstruct
    # the scanner's decision (best + a few near-misses) without bloating
    # signals.jsonl on a 12-ticker, 4-grid sweep.
    _SCAN_JOURNAL_TOPK = 10

    def _snapshot_scan_results(self) -> Optional[Dict]:
        """
        Capture the planner's most recent scanner output as a journal-safe
        dict, or return ``None`` when the planner is in static mode.

        MUST be called immediately after ``strategy_planner.plan(...)`` —
        the next plan() invocation resets ``last_scan_candidates`` and
        the snapshot would otherwise reflect a different ticker.

        The returned shape is::

            {
              "side":           "bull_put" | "bear_call",
              "scan_mode":      "adaptive",
              "edge_buffer":    0.10,
              "min_pop":        0.55,
              "candidates_total": 8,
              "selected_index":  0,             # index into the K below
              "top_k": [ <SpreadCandidate.to_journal_dict()>, ... ],
              "diagnostics": {
                  "grid_points_total":    16,
                  "grid_points_priced":   12,
                  "expirations_resolved": 4,
                  "rejects_by_reason":    {"cw_below_floor": 11, ...},
                  "best_near_miss":       {<SpreadCandidate-like dict>}
              }
            }

        ``selected_index`` is 0 when the scanner picked a candidate
        (top-of-list) and -1 when no candidate cleared the floor.

        The ``diagnostics`` block is the actionable answer to *"why didn't
        the scanner pass?"*. ``best_near_miss`` is the single highest-EV
        candidate that failed only the C/W floor — quoting it lets the
        user see "the closest we came was C/W=0.18, needed 0.22" without
        digging through trading_agent.log.
        """
        planner = self.strategy_planner
        if not getattr(planner, "is_adaptive", False):
            return None
        candidates = list(getattr(planner, "last_scan_candidates", []) or [])
        side = getattr(planner, "last_scan_side", None)
        diagnostics = getattr(planner, "last_scan_diagnostics", None)
        # No scan ran this ticker (e.g. iron condor or mean-reversion path
        # in adaptive preset — those still use the static builders today).
        if not candidates and side is None and diagnostics is None:
            return None
        top_k = candidates[: self._SCAN_JOURNAL_TOPK]
        block: Dict = {
            "scan_mode":        "adaptive",
            "side":             side,
            "edge_buffer":      float(getattr(self.preset, "edge_buffer", 0.10)),
            "min_pop":          float(getattr(self.preset, "min_pop", 0.55)),
            "candidates_total": len(candidates),
            "selected_index":   0 if candidates else -1,
            "top_k":            [c.to_journal_dict() for c in top_k],
        }
        if diagnostics is not None:
            block["diagnostics"] = diagnostics
        return block

    def _log_signal(
        self,
        ticker: str,
        action: str,
        analysis: "RegimeAnalysis",
        plan: "SpreadPlan",
        verdict: "RiskVerdict",
        llm_decision: Optional["AnalystDecision"],
        exec_result: Optional[Dict],
        *,
        scan_results: Optional[Dict] = None,
    ) -> None:
        """Build raw_signal dict and write to JournalKB."""
        thesis = build_thesis(analysis, plan, verdict)

        raw: Dict = {
            # ``mode`` distinguishes dry-run cycles from live cycles within
            # the same signals_live.jsonl stream.  The dashboard uses this
            # to scope its "latest verdict" guardrail panel to the active
            # mode so a stale LIVE row doesn't display while DRY-RUN is
            # selected (or vice-versa).  Legacy rows without this field
            # are treated as LIVE for filtering — see
            # streamlit/live_monitor.py:_guardrail_status_from_journal.
            "mode": "DRY-RUN" if self.config.trading.dry_run else "LIVE",
            "regime": analysis.regime.value,
            "strategy": plan.strategy_name,
            "plan_valid": plan.valid,
            "rejection_reason": plan.rejection_reason if not plan.valid else None,
            "risk_approved": verdict.approved,
            "net_credit": plan.net_credit if plan.valid else None,
            "max_loss": plan.max_loss if plan.valid else None,
            "credit_to_width_ratio": (
                plan.credit_to_width_ratio if plan.valid else None
            ),
            "spread_width": plan.spread_width if plan.valid else None,
            "expiration": plan.expiration if plan.valid else None,
            "sma_50": analysis.sma_50,
            "sma_200": analysis.sma_200,
            "rsi_14": analysis.rsi_14,
            "mean_reversion_signal": getattr(analysis, "mean_reversion_signal", False),
            "mean_reversion_direction": getattr(analysis, "mean_reversion_direction", ""),
            "leadership_anchor": getattr(analysis, "leadership_anchor", ""),
            "leadership_zscore": getattr(analysis, "leadership_zscore", 0.0),
            "leadership_raw_diff": getattr(analysis, "leadership_raw_diff", 0.0),
            "vix_zscore": getattr(analysis, "vix_zscore", 0.0),
            "inter_market_inhibit_bullish": getattr(
                analysis, "inter_market_inhibit_bullish", False),
            "account_balance": verdict.account_balance,
            "checks_passed": verdict.checks_passed,
            "checks_failed": verdict.checks_failed,
            "llm_decision": llm_decision.action if llm_decision else None,
            "llm_confidence": llm_decision.confidence if llm_decision else None,
            "order_id": (
                exec_result.get("order_id") if exec_result else None
            ),
            "run_id": exec_result.get("run_id") if exec_result else None,
            "thesis": thesis,
        }

        # Adaptive-scan diagnostics: top-K candidates + selected pick. Only
        # set when the planner ran the scanner this cycle; static mode emits
        # nothing so the journal stays compact.
        if scan_results is not None:
            raw["scan_results"] = scan_results

        exec_status = exec_result.get("status") if exec_result else action

        try:
            self.journal_kb.log_signal(
                ticker=ticker,
                action=action,
                price=analysis.current_price,
                raw_signal=raw,
                exec_status=exec_status,
            )
        except Exception as exc:
            logger.warning("[%s] JournalKB log failed: %s", ticker, exc)

        # ── Position-opened alert (skill 32 §3.6) ────────────────────
        # Fire once when a submission to Alpaca succeeds. action="submitted"
        # is the canonical "we have a new live position" journal row;
        # action="dry_run" fires the same alert so paper-flow operators
        # still see the lifecycle. action="rejected" / "skip" do NOT
        # alert (the position never existed).
        #
        # Dedup gate: pi-deploy 2026-05-20 hotfix. In DRY-RUN mode the
        # planner re-emits the same trade plan every cycle as long as
        # the regime + chain math agrees, journaling action="dry_run"
        # repeatedly for the same notional position. Dedup key uses
        # ticker + expiration + UTC date so a real same-day re-open
        # (different expiration) still alerts but the same dry-run
        # plan being re-emitted cycle after cycle does not.
        if action in ("submitted", "dry_run") and plan.valid:
            short_strikes_str = ", ".join(
                f"${l.strike:g}" for l in plan.legs
                if l.action == "sell"
            ) or "—"
            exp_str = str(plan.expiration or "")
            dedup_alert_type = f"position_opened:{exp_str}"
            self._send_telegram_alert(
                ticker=ticker,
                alert_type=dedup_alert_type,
                send_fn=self.telegram.notify_position_opened,
                strategy=plan.strategy_name,
                regime=analysis.regime.value,
                net_credit=float(plan.net_credit or 0.0),
                max_loss=float(plan.max_loss or 0.0),
                spread_width=float(plan.spread_width or 0.0),
                expiration=exp_str,
                short_strikes=short_strikes_str,
                thesis=str(thesis or ""),
            )

    # ==================================================================
    # Risk guardrail helpers (delegated to daily_state module)
    # ==================================================================

    def _should_exit_spread(self, spread: SpreadPosition) -> bool:
        """
        3-cycle debounce guard for non-immediate exit signals.

        Immediate signals (HARD_STOP, STRIKE_PROXIMITY, DTE_SAFETY) bypass
        debounce and return True immediately.  All other signals require
        the SAME signal on 3 consecutive cycles before this returns True.
        """
        if spread.exit_signal == ExitSignal.HOLD:
            return False

        if spread.exit_signal in IMMEDIATE_EXIT_SIGNALS:
            logger.warning(
                "[%s] IMMEDIATE exit signal %s — bypassing debounce",
                spread.underlying, spread.exit_signal.value,
            )
            return True

        count = tally_exit_vote(
            self.daily_state,
            ticker=spread.underlying,
            signal_val=spread.exit_signal.value,
            required=EXIT_DEBOUNCE_REQUIRED,
        )

        if count >= EXIT_DEBOUNCE_REQUIRED:
            logger.warning(
                "[%s] Exit signal %s confirmed after %d cycles — acting",
                spread.underlying, spread.exit_signal.value, count,
            )
            return True

        logger.info(
            "[%s] Exit signal %s vote %d/%d — debouncing (next check in ~5 min)",
            spread.underlying, spread.exit_signal.value,
            count, EXIT_DEBOUNCE_REQUIRED,
        )
        return False

    def _check_liquidation_mode(self, equity: float,
                                buying_power: float) -> bool:
        """
        Returns True if buying power usage exceeds the configured threshold,
        signalling the agent should close positions rather than open new ones.
        """
        if equity <= 0:
            logger.warning("Equity <= 0 (%.2f) — Emergency Liquidation Mode", equity)
            return True
        initial_bp = equity * self.config.trading.margin_multiplier
        pct_used = 1.0 - (buying_power / initial_bp)
        limit = self.config.trading.max_buying_power_pct
        if pct_used > limit:
            logger.warning(
                "Buying power %.1f%% used (limit=%.0f%%) — Liquidation Mode",
                pct_used * 100, limit * 100,
            )
            self.journal_kb.log_cycle_error(
                "liquidation_mode_activated",
                {
                    "buying_power_used_pct": round(pct_used * 100, 1),
                    "limit_pct": limit * 100,
                    "equity": equity,
                    "buying_power": buying_power,
                },
            )
            return True
        return False

    # ==================================================================
    # Open-order dedup + stale-order maintenance
    # ==================================================================

    def _tickers_with_open_orders(self) -> Set[str]:
        """
        Return the set of underlying tickers that currently have at least
        one open (pending-fill) order on Alpaca.

        For multi-leg orders the broker's top-level ``symbol`` is empty,
        so we recover the underlying by parsing the OCC root from each
        leg.  Equity / single-leg orders carry the underlying directly.
        """
        out: Set[str] = set()
        try:
            open_orders = self.order_tracker.fetch_open_orders()
        except Exception as exc:
            logger.warning("Could not fetch open orders for dedup: %s", exc)
            return out

        for o in open_orders:
            top = (o.symbol or "").upper().strip()
            if top:
                # Equity / single-leg: top-level symbol is the ticker
                # (e.g. "SPY") — keep it as-is unless it's an OCC string.
                root = _root_from_occ(top) or top
                if root:
                    out.add(root)
            for leg in (o.legs or []):
                root = _root_from_occ((leg.get("symbol") or "").upper())
                if root:
                    out.add(root)
        return out

    def _cancel_stale_orders(self, agent_tickers: List[str]) -> None:
        """
        Cancel limit orders that have been on the broker's book longer
        than ``STALE_ORDER_MAX_AGE_MIN`` minutes.

        Scoping
        -------
        Only orders whose underlying matches one of ``agent_tickers``
        are cancelled — this keeps any manual trade you placed on a
        ticker the agent doesn't manage off the chopping block.  Orders
        younger than the threshold are also untouched, so a manual
        order placed in the last 15 minutes is safe.

        Why this matters
        ----------------
        The executor submits at the planning-time mid with
        ``time_in_force="day"`` and never re-prices.  On a 7-DTE put
        spread the achievable credit erodes minute by minute as theta
        drains, so an unfilled limit becomes structurally un-fillable.
        Cancelling and re-planning is the only way to keep the limit
        anywhere near the live mid.
        """
        try:
            open_orders = self.order_tracker.fetch_open_orders()
        except Exception as exc:
            logger.warning("Could not fetch open orders for stale check: %s", exc)
            return

        if not open_orders:
            return

        ticker_set = {t.upper() for t in (agent_tickers or [])}
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_ORDER_MAX_AGE_MIN)
        cancelled = 0

        for o in open_orders:
            # Recover the underlying.  Multi-leg orders have empty
            # top-level symbol; fall back to the first leg's OCC root.
            roots = set()
            top = (o.symbol or "").upper().strip()
            if top:
                roots.add(_root_from_occ(top) or top)
            for leg in (o.legs or []):
                r = _root_from_occ((leg.get("symbol") or "").upper())
                if r:
                    roots.add(r)
            if not roots & ticker_set:
                continue   # not on a ticker the agent manages

            # Parse Alpaca's RFC3339 created_at.  It's already UTC.
            created_raw = o.created_at or ""
            if not created_raw:
                continue
            try:
                created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            except ValueError:
                logger.debug("Order %s: unparseable created_at %r — skipping",
                             o.order_id, created_raw)
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created > cutoff:
                continue   # younger than the threshold

            age_min = (datetime.now(timezone.utc) - created).total_seconds() / 60.0
            logger.warning(
                "[%s] Cancelling stale order %s (age %.1f min > %d min) — "
                "next cycle will re-price",
                next(iter(roots & ticker_set)), o.order_id, age_min,
                STALE_ORDER_MAX_AGE_MIN,
            )
            if self.order_tracker.cancel_order(o.order_id):
                cancelled += 1

        if cancelled:
            self.journal_kb.log_cycle_error(
                "stale_orders_cancelled",
                {
                    "count": cancelled,
                    "max_age_minutes": STALE_ORDER_MAX_AGE_MIN,
                },
            )

    # ==================================================================
    # Order status check
    # ==================================================================

    def _check_order_statuses(self) -> Dict:
        """Fetch recent orders and log a summary."""
        try:
            open_orders = self.order_tracker.fetch_open_orders()
            recent_fills = self.order_tracker.fetch_recent_fills(limit=10)

            open_summary = self.order_tracker.summarize_orders(open_orders)
            fill_summary = self.order_tracker.summarize_orders(recent_fills)

            logger.info(
                "Open orders: %d | Recent fills: %d",
                open_summary["total"], fill_summary["total"],
            )

            return {
                "open_orders": open_summary,
                "recent_fills": fill_summary,
            }
        except Exception as exc:
            logger.warning("Could not check order statuses: %s", exc)
            return {"error": str(exc)}

    # ==================================================================
    # Helpers
    # ==================================================================

    def _cached_price(self, ticker: str) -> float:
        """Return cached price for *ticker* or 0.0 if unavailable.

        Delegates to the MarketDataPort.get_cached_price method so the
        agent never reaches into adapter-private caches.  Pre-week-5-6
        this poked ``data_provider._snapshot_cache`` and
        ``_price_cache`` directly, which was the classic "leaky
        abstraction" symptom the port refactor eliminates.
        """
        price = self.data_provider.get_cached_price(ticker)
        return float(price) if price is not None else 0.0

    def _check_daily_drawdown(self, current_equity: float) -> bool:
        """Thin wrapper for tests / legacy callers.

        The canonical implementation is
        :func:`trading_agent.daily_state.check_daily_drawdown`, which
        was extracted during the week 3-4 refactor.  This method keeps
        the pre-refactor instance-method shape so existing integration
        tests (and anyone who held on to the previous surface) don't
        have to rewire.  All real policy lives in daily_state.
        """
        return check_daily_drawdown(
            self.daily_state,
            current_equity=current_equity,
            drawdown_limit=self.config.trading.daily_drawdown_limit,
            journal_kb=self.journal_kb,
        )

    def _learn_from_close(self, spread: SpreadPosition):
        """Run post-trade LLM analysis when a spread is closed."""
        try:
            recent = self.llm_analyst.journal.get_trades_by_ticker(
                spread.underlying, limit=5)
            for trade in recent:
                if (trade.strategy_name == spread.strategy_name
                        and not trade.timestamp_closed):
                    self.llm_analyst.journal.close_trade(
                        trade_id=trade.trade_id,
                        exit_signal=spread.exit_signal.value,
                        exit_reason=spread.exit_reason,
                        realized_pl=spread.net_unrealized_pl,
                    )
                    trade = self.llm_analyst.journal.get_trade(trade.trade_id)
                    if trade:
                        self.llm_analyst.analyze_outcome(trade)
                        # Back-fill the outcome into the KB document so
                        # future RAG searches return outcome-labelled results
                        self.llm_analyst.knowledge_base.update_trade_outcome(
                            trade_id=trade.trade_id,
                            outcome_label=trade.outcome_label,
                            realized_pl=trade.realized_pl,
                            exit_signal=trade.exit_signal,
                            exit_reason=trade.exit_reason,
                            updated_text=trade.to_embedding_text(),
                        )
                    break
        except Exception as exc:
            logger.warning(
                "[%s] Post-trade learning failed: %s",
                spread.underlying, exc,
            )

    @staticmethod
    def _regime_to_strategy(regime) -> str:
        return {
            Regime.BULLISH: "Bull Put Spread",
            Regime.BEARISH: "Bear Call Spread",
            Regime.SIDEWAYS: "Iron Condor",
            Regime.MEAN_REVERSION: "Mean Reversion Spread",
        }.get(regime, "Unknown")

    @staticmethod
    def _analysis_dict(analysis: "RegimeAnalysis") -> Dict:
        return {
            "price": analysis.current_price,
            "sma_50": analysis.sma_50,
            "sma_200": analysis.sma_200,
            "rsi": analysis.rsi_14,
            "reasoning": analysis.reasoning,
        }

    def _print_summary(self, results: List[Dict]):
        """Log a human-readable summary table."""
        logger.info(
            "\n%-6s | %-10s | %-18s | %-8s | %-10s | %s",
            "Ticker", "Regime", "Strategy", "Valid", "Risk OK", "Status",
        )
        logger.info("-" * 80)
        for r in results:
            logger.info(
                "%-6s | %-10s | %-18s | %-8s | %-10s | %s",
                r.get("ticker", "?"),
                r.get("regime", "?"),
                r.get("strategy", "?"),
                r.get("plan_valid", "?"),
                r.get("risk_approved", "?"),
                r.get("execution", {}).get("status", r.get("status", "?")),
            )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    """Run a single trading cycle from the command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Autonomous Options Trading Agent")
    parser.add_argument("--env", type=str, default=None,
                        help="Path to .env file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Override: force dry-run mode")
    args = parser.parse_args()

    agent = TradingAgent.from_env(args.env)
    if args.dry_run:
        agent.executor.dry_run = True

    results = agent.run_cycle()
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

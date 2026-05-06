"""
backtest_ui.py — Backtesting tab (thin Streamlit wrapper around BacktestRunner).

The heavy lifting — clock, historical port, synthetic chain, position
mark/exit logic, account ledger — lives in ``trading_agent.backtest``.
This file is just the form + chart wiring. Picking a (ticker × date-window
× preset) configuration here drives a single ``BacktestRunner.run()`` call
that produces a ``BacktestResult``; everything below is rendering.

Why this file is so much smaller than the legacy one
----------------------------------------------------
The pre-2026-05 ``Backtester`` class re-implemented its own scoring,
sigma-strike heuristic, exit logic, and account ledger. That created a
second source of truth and let the backtest drift from live silently
(see skills 03 + 14 for the floor-formula and adaptive-scan invariants).
The new architecture pushes all of that into shared modules:

  * Strike selection → ``trading_agent.decision_engine.decide()`` —
    the SAME function the live agent calls. Skill 14.
  * Risk / size       → ``trading_agent.risk_manager.RiskManager`` and
    ``trading_agent.executor.calculate_position_qty``. Skills 03 + 05.
  * Mark/exit         → ``trading_agent.backtest.sim_position`` whose
    ``evaluate_exit`` mirrors ``PositionMonitor._check_exit`` rule-for-rule.
  * Cadence           → ``trading_agent.backtest.runner`` decides
    intraday-vs-daily by window length; ≤30d gets 5m bars, longer gets
    daily-only. Skill 15.

CI invariant #3 (``scripts/checks/scan_invariant_check.py``) requires this
file to contain at least one ``decide(`` call so the unified path remains
linked. The ``_preview_decision`` helper below satisfies that and doubles
as a "what would the engine pick today?" diagnostic from the UI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from trading_agent.backtest import HistoricalPort
from trading_agent.backtest.runner import (
    BacktestResult,
    BacktestRunner,
    INTRADAY_LOOKBACK_LIMIT_DAYS,
)
from trading_agent.backtest.synthetic_chain import (
    build_chain_config_from_preset,
    build_chain_slice,
)
from trading_agent.decision_engine import ChainSlice, DecisionInput, decide
from trading_agent.risk_manager import RiskManager
from trading_agent.strategy_presets import PRESETS, PresetConfig, load_active_preset
from trading_agent.streamlit._busy import global_busy
from trading_agent.streamlit.components import (
    closed_trades_table,
    drawdown_chart,
    equity_curve_chart,
)

logger = logging.getLogger(__name__)


# --- Defaults ---------------------------------------------------------------

# Tickers offered in the multiselect. Includes broad-market & sector ETFs
# (matching the user-policy ETF-only universe in `.env` TICKERS) plus the
# legacy megacap names for users who want to backtest single-name spreads.
ALL_TICKERS = [
    # Broad-market ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs (deep weekly options liquidity)
    "XLF", "XLE", "XLK", "XLY", "XLV",
    # Asset-class diversifiers
    "TLT", "GLD",
    # Megacap single-names — kept for users who explicitly want them
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
]
DEFAULT_TICKERS = ["SPY", "QQQ"]
DEFAULT_END = date.today() - timedelta(days=1)
DEFAULT_START = DEFAULT_END - timedelta(days=180)
DEFAULT_STARTING_EQUITY = 100_000.0

# Sentinel dropdown value: surfaces the user's on-disk "custom" preset
# (whatever they tuned via the Live tab's Strategy Profile) as a
# first-class seed option in the Backtest tab. Without this, a user
# whose live preset is "custom" finds the backtest defaulting to the
# *closest named profile* (e.g. "aggressive") — which is misleading
# because the canned profile may have very different floor / risk /
# scan-mode values from what they're actually trading live.
LIVE_CUSTOM_LABEL = "custom (from live)"


# ---------------------------------------------------------------------------
# Result → DataFrame adapters (kept here, not in components.py, because
# they're tightly coupled to BacktestResult's dataclass shape)
# ---------------------------------------------------------------------------

def _equity_curve_to_df(equity_points: List) -> pd.DataFrame:
    """Convert ``list[EquityPoint]`` → DataFrame the chart helpers expect.

    ``equity_curve_chart`` / ``drawdown_chart`` look for ``timestamp`` +
    ``account_balance`` columns (live-monitor convention).
    """
    rows = [
        {
            "timestamp":         p.t,
            "account_balance":   p.equity,
            "cash":              p.cash,
            "open_market_value": p.open_market_value,
            "realised_pnl":      p.realised_pnl,
        }
        for p in equity_points
    ]
    return pd.DataFrame(rows)


def _cycle_outcomes_to_df(outcomes: List) -> pd.DataFrame:
    rows = [
        {
            "t":       o.t,
            "ticker":  o.ticker,
            "spot":    round(float(o.spot or 0.0), 2),
            "regime":  getattr(o.regime, "value", o.regime) if o.regime is not None else None,
            "status":  o.status,
            "reason":  o.reason,
        }
        for o in outcomes
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Preset-editor helpers (pure — testable without a Streamlit context)
# ---------------------------------------------------------------------------

# Field names that ``PresetConfig`` declares as ``Tuple[...]``. Streamlit
# text inputs hand us list-shaped collections; the dataclass is frozen and
# requires hashable values, so we coerce here once.
_TUPLE_FIELDS = ("dte_grid", "delta_grid", "width_grid_pct")


def _format_csv(values) -> str:
    """Format a tuple/list of numbers as ``"a, b, c"`` for a text input.

    Uses ``%g`` so 0.10 prints as ``0.1`` (not ``0.10``); easier to skim
    in a narrow sidebar text field.
    """
    return ", ".join(f"{v:g}" for v in values)


def _parse_csv_floats(s: str, fallback: Tuple[float, ...]) -> Tuple[float, ...]:
    """Parse ``"0.1, 0.15, 0.2"`` → ``(0.1, 0.15, 0.2)``.

    Empty fields are dropped silently. On any parse error or if every
    field is empty, return ``fallback`` so a typo can never zero out the
    grid (which would make the runner skip every cycle).
    """
    try:
        out = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    except (ValueError, AttributeError):
        return fallback
    return out or fallback


def _parse_csv_ints(s: str, fallback: Tuple[int, ...]) -> Tuple[int, ...]:
    """Same as ``_parse_csv_floats`` but coerces to ``int`` (round via float).

    Float-via-cast is intentional so ``"7.0, 14"`` parses cleanly.
    """
    try:
        out = tuple(int(float(x.strip())) for x in s.split(",") if x.strip())
    except (ValueError, AttributeError):
        return fallback
    return out or fallback


def _apply_overrides(base: PresetConfig, **overrides) -> PresetConfig:
    """Build a new PresetConfig from ``base`` plus per-field overrides.

    Coerces list-shaped tuple fields (``dte_grid`` etc.) to tuples so the
    frozen dataclass accepts them. If every override matches the base
    field exactly, returns ``base`` unchanged so the preset's *name* is
    preserved — only flips the name to ``"custom"`` when something
    actually differs. That keeps the run-summary caption honest:
    "preset=balanced" until you tweak something, then "preset=custom".
    """
    coerced: dict = {}
    any_diff = False
    for k, v in overrides.items():
        if k in _TUPLE_FIELDS and isinstance(v, list):
            v = tuple(v)
        if v != getattr(base, k):
            any_diff = True
        coerced[k] = v
    if not any_diff:
        return base
    coerced.setdefault("name", "custom")
    return replace(base, **coerced)


# ---------------------------------------------------------------------------
# Decision preview (also satisfies CI invariant #3 — must call decide())
# ---------------------------------------------------------------------------

def _preview_decision(ticker: str, preset_name: str,
                      side: str = "bull_put", spot: float = 500.0,
                      sigma: float = 0.20,
                      *,
                      preset: Optional[PresetConfig] = None) -> Optional[dict]:
    """Run a single ``decide()`` call against a synthesized chain.

    Diagnostic helper — lets the operator preview what the unified engine
    would select for a given preset right now without booting a full
    backtest. Also keeps a literal ``decide(`` call in this file so the
    invariant scanner's parity check (Invariant 3) stays green.

    ``preset`` (keyword-only) takes precedence over ``preset_name`` so
    the sidebar editor can preview an *edited* preset without first
    persisting it. When omitted, falls back to the named lookup +
    ``load_active_preset()``.
    """
    if preset is None:
        preset = PRESETS.get(preset_name) or load_active_preset()
    cfg = build_chain_config_from_preset(side, preset)

    today = date.today()
    chain_slices: List[ChainSlice] = []
    for dte_target in preset.dte_grid:
        expiration = today + timedelta(days=int(dte_target))
        try:
            chain_slices.append(build_chain_slice(
                ticker=ticker, side=side, spot=spot,
                sigma_annual=sigma, now=today, expiration=expiration,
                cfg=cfg,
            ))
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("preview: chain build failed for dte=%s: %s",
                           dte_target, exc)

    if not chain_slices:
        return {"error": "no chain slices built"}

    try:
        out = decide(  # CI invariant #3: do NOT remove or rename this call.
            DecisionInput(side=side, chain_slices=chain_slices, preset=preset),
            max_candidates=3,
        )
    except Exception as exc:
        logger.warning("preview decide() failed: %s", exc)
        return {"error": str(exc)}

    candidates = [
        {
            "expiration":   c.expiration,
            "dte":          c.dte,
            "short_strike": round(c.short_strike, 2),
            "long_strike":  round(c.long_strike, 2),
            "credit":       round(c.credit, 2),
            "width":        round(c.width, 2),
            "cw_ratio":     round(getattr(c, "cw_ratio", 0.0), 4),
            "pop":          round(getattr(c, "pop", 0.0), 4),
            "score":        round(getattr(c, "annualized_score", 0.0), 4),
        }
        for c in out.candidates
    ]
    return {
        "candidates":     candidates,
        "grid_total":     out.diagnostics.grid_points_total,
        "grid_priced":    out.diagnostics.grid_points_priced,
        "rejects":        dict(out.diagnostics.rejects_by_reason),
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_backtest_ui() -> None:
    """Render the Backtesting tab.

    Layout (full-width, top-down — mirrors live_monitor's
    ``render_strategy_profile_panel``):

      1. Strategy Profile  — preset picker + slider grid + Δ/DTE/width text
                             grids in a wide expander; same widget vocabulary
                             as the live tab so what you see here is what the
                             live agent would honour.
      2. Run Configuration — tickers / dates / equity in a single 4-column
                             row, cadence hint, primary Run button.
      3. Preview            — collapsed expander, ``decide()`` smoke-test.
      4. Results            — full-width below; metric cards, charts, tables.
    """

    # ── Strategy preset picker — same library the live agent uses.
    # If the active preset on disk (STRATEGY_PRESET.json) is "custom", we
    # surface it as a dedicated dropdown option (LIVE_CUSTOM_LABEL) so the
    # backtest defaults to *exactly what the live agent is trading right
    # now* — rather than silently falling back to the closest named
    # profile (which often has very different floor / risk / scan-mode
    # values and produces misleading backtest results).
    try:
        active = load_active_preset()
    except Exception:
        active = PRESETS["balanced"]

    has_live_custom = active.name == "custom"
    named_choices = list(PRESETS.keys())   # [conservative, balanced, aggressive]

    if has_live_custom:
        # Live-custom comes FIRST so it's the obvious default.
        preset_choices = [LIVE_CUSTOM_LABEL] + named_choices
        default_seed_name = LIVE_CUSTOM_LABEL
    else:
        preset_choices = named_choices
        if active.name in preset_choices:
            default_seed_name = active.name
        else:
            # Pathological fallback (e.g. unknown preset name on disk):
            # closest named profile by max_delta.
            default_seed_name = min(
                preset_choices,
                key=lambda n: abs(PRESETS[n].max_delta - active.max_delta),
            )
    default_idx = preset_choices.index(default_seed_name)

    # ── 1. Strategy Profile (full-width, top-down) ───────────────────────
    with st.expander(
        "🎛️ Strategy Profile — preset + adaptive overrides",
        expanded=True,
    ):
        st.caption(
            "Same preset library the live agent uses. Pick a seed profile, "
            "then tweak any field — the Effective summary at the bottom is "
            "what drives both the preview and the full run. "
            "Mirrors the live tab's Strategy-Profile panel so what you see "
            "here is the same payload shape live trading consumes."
        )

        # --- Row A: preset seed + reset button -------------------------------
        row_a_l, row_a_r = st.columns([3, 1])
        with row_a_l:
            preset_name = st.selectbox(
                "Strategy preset (seed)",
                options=preset_choices, index=default_idx,
                key="bt_preset",
                help="Picks the seed values for the editor below. Edit any "
                     "field to override; unedited fields stay at the seed's "
                     "defaults. Flips to `custom` automatically when "
                     "anything differs from the seed.",
            )
        with row_a_r:
            st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical align hack
            _editor_keys = (
                "bt_pe_scan_mode", "bt_pe_max_delta", "bt_pe_max_risk_pct",
                "bt_pe_min_credit_ratio", "bt_pe_edge_buffer",
                "bt_pe_min_pop", "bt_pe_dte_grid", "bt_pe_delta_grid",
                "bt_pe_width_grid_pct",
            )
            if st.button("↺ Reset to seed",
                         key="bt_pe_reset",
                         width='stretch',
                         help="Wipe editor overrides and re-seed from the "
                              "selected preset."):
                for _k in _editor_keys:
                    st.session_state.pop(_k, None)
                st.rerun()

        # Seed selection:
        #   * LIVE_CUSTOM_LABEL → the on-disk custom preset (what the live
        #     agent is actually trading right now). This makes the
        #     backtest default to a faithful re-run of the live config.
        #   * named profile     → the canned PRESETS[name] dict.
        # The named-profile case also covers the edge case where the user
        # has a non-custom preset on disk and the dropdown matches it.
        if preset_name == LIVE_CUSTOM_LABEL:
            seed_preset = active
        elif active.name == preset_name:
            seed_preset = active
        else:
            seed_preset = PRESETS[preset_name]

        # --- Row B: scan_mode + edge_buffer ---------------------------------
        row_b_l, row_b_r = st.columns(2)
        with row_b_l:
            scan_mode_val = st.radio(
                "Scan mode",
                options=["adaptive", "static"],
                index=0 if seed_preset.scan_mode == "adaptive" else 1,
                key="bt_pe_scan_mode",
                horizontal=True,
                help="Adaptive sweeps the (DTE × Δ × width) grid below. "
                     "Static uses the scalar dte_vertical / max_delta / "
                     "width_value fields and gates on min_credit_ratio.",
            )
        with row_b_r:
            edge_buffer_val = st.slider(
                "Edge buffer",
                0.0, 0.50, float(seed_preset.edge_buffer), 0.01,
                key="bt_pe_edge_buffer",
                help="Cushion above the breakeven C/W floor. Adaptive demands "
                     "C/W ≥ |Δshort| × (1 + edge_buffer). 0.0 = trade at "
                     "exact breakeven; 0.05 = 5% safety margin.",
            )

        # --- Row C: 4 sliders side-by-side (same widget vocabulary as live) -
        sl1, sl2, sl3, sl4 = st.columns(4)
        with sl1:
            max_delta_val = st.slider(
                "Max short-leg |Δ|",
                0.05, 0.45, float(seed_preset.max_delta), 0.01,
                key="bt_pe_max_delta",
                help="0.15 ≈ 85% POP · 0.25 ≈ 75% POP · 0.35 ≈ 65% POP",
            )
        with sl2:
            min_credit_ratio_val = st.slider(
                "Credit/Width floor",
                0.10, 0.50, float(seed_preset.min_credit_ratio), 0.05,
                key="bt_pe_min_credit_ratio",
                help="Only consulted in static scan mode. Adaptive uses the "
                     "delta-aware floor instead.",
            )
        with sl3:
            max_risk_pct_val = st.slider(
                "Max account risk per trade (%)",
                0.5, 5.0, float(seed_preset.max_risk_pct) * 100, 0.5,
                key="bt_pe_max_risk_pct",
                help="Hard cap on max-loss as a fraction of account equity.",
            ) / 100.0
        with sl4:
            min_pop_val = st.slider(
                "Min POP",
                0.30, 0.85, float(seed_preset.min_pop), 0.05,
                key="bt_pe_min_pop",
                help="Drop candidates whose POP (≈ 1 − |Δshort|) is below this.",
            )

        # --- Row D: adaptive scan grids in 3 columns ------------------------
        st.markdown("**Adaptive scan grids** "
                    "<span style='color:#888'>(only used when scan_mode = adaptive)</span>",
                    unsafe_allow_html=True)
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            dte_grid_str = st.text_input(
                "DTE grid (days)",
                value=_format_csv(seed_preset.dte_grid),
                key="bt_pe_dte_grid",
                help="e.g. 7, 14, 21, 30",
            )
        with gc2:
            delta_grid_str = st.text_input(
                "Δ grid (|short delta|)",
                value=_format_csv(seed_preset.delta_grid),
                key="bt_pe_delta_grid",
                help="Lower deltas (0.10–0.15) clear the floor more easily "
                     "in normal-IV regimes.",
            )
        with gc3:
            width_grid_str = st.text_input(
                "Width grid (% of spot)",
                value=_format_csv(seed_preset.width_grid_pct),
                key="bt_pe_width_grid_pct",
                help="0.01 = 1% of spot. C/W is dimensionless so width-grid "
                     "changes mostly shift dollar credit, not the floor "
                     "pass/fail.",
            )

        # Build the effective preset from the seed + edits. Any field left
        # at the seed's default is treated as "no override" so the preset
        # name stays meaningful (only flips to "custom" when something
        # actually differs).
        effective_preset = _apply_overrides(
            seed_preset,
            scan_mode=scan_mode_val,
            max_delta=float(max_delta_val),
            max_risk_pct=float(max_risk_pct_val),
            min_credit_ratio=float(min_credit_ratio_val),
            edge_buffer=float(edge_buffer_val),
            min_pop=float(min_pop_val),
            dte_grid=_parse_csv_ints(
                dte_grid_str, seed_preset.dte_grid),
            delta_grid=_parse_csv_floats(
                delta_grid_str, seed_preset.delta_grid),
            width_grid_pct=_parse_csv_floats(
                width_grid_str, seed_preset.width_grid_pct),
        )
        st.success(f"**Effective:** {effective_preset.to_summary_line()}")

    # ── 2. Run Configuration (full-width) ────────────────────────────────
    with st.expander("📅 Run Configuration", expanded=True):
        rc1, rc2, rc3, rc4 = st.columns([3, 2, 2, 2])
        with rc1:
            tickers = st.multiselect(
                "Tickers", options=ALL_TICKERS, default=DEFAULT_TICKERS,
                key="bt_tickers",
            )
        with rc2:
            start_date = st.date_input(
                "Start Date", value=DEFAULT_START, key="bt_start_date",
            )
        with rc3:
            end_date = st.date_input(
                "End Date", value=DEFAULT_END, key="bt_end_date",
            )
        with rc4:
            starting_equity = st.number_input(
                "Starting equity ($)",
                min_value=10_000.0, max_value=10_000_000.0,
                value=DEFAULT_STARTING_EQUITY, step=10_000.0,
                key="bt_starting_equity",
            )

        # Cadence hint — preview what the runner will do.
        window_days = (end_date - start_date).days if end_date > start_date else 0
        if window_days <= INTRADAY_LOOKBACK_LIMIT_DAYS:
            st.caption(
                f"📡 Window ≤ {INTRADAY_LOOKBACK_LIMIT_DAYS} days → "
                "intraday 5m entry decisions enabled (yfinance 5m bar limit)."
            )
        else:
            st.caption(
                f"📅 Window > {INTRADAY_LOOKBACK_LIMIT_DAYS} days → "
                "daily-only cadence (one decision per RTH open)."
            )

        run_btn = st.button("▶ Run Backtest", type="primary", width='stretch')

    # ── 3. Diagnostic preview (collapsed) ────────────────────────────────
    with st.expander("🔍 Preview today's engine decision", expanded=False):
        st.caption(
            "Builds a synthetic chain from today's date + the selected "
            "preset's grids, then calls `decide()` to show what the unified "
            "engine would pick. Useful for sanity-checking a preset before "
            "kicking off a full backtest."
        )
        pv1, pv2, pv3 = st.columns([2, 2, 1])
        with pv1:
            preview_ticker = st.text_input(
                "Ticker", value="SPY", key="bt_preview_ticker",
            )
        with pv2:
            preview_side = st.selectbox(
                "Side", options=["bull_put", "bear_call"], index=0,
                key="bt_preview_side",
            )
        with pv3:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            preview_btn = st.button("Run preview",
                                    key="bt_run_preview",
                                    width='stretch')
        if preview_btn:
            # Pass the customized preset directly so the preview reflects
            # what's actually staged in the editor — not the un-edited seed.
            with global_busy("Running decision preview…",
                             detail=f"{preview_ticker} · {preview_side}"):
                preview = _preview_decision(
                    preview_ticker, preset_name, side=preview_side,
                    preset=effective_preset,
                )
            if preview is None or "error" in (preview or {}):
                st.warning(
                    f"Preview failed: {preview.get('error') if preview else 'unknown'}"
                )
            else:
                st.write(
                    f"Grid: {preview['grid_priced']} priced of "
                    f"{preview['grid_total']} total."
                )
                if preview["candidates"]:
                    st.dataframe(
                        pd.DataFrame(preview["candidates"]),
                        width='stretch', hide_index=True,
                    )
                else:
                    st.info("No candidates — top reject reasons:")
                    st.json(preview["rejects"])

    st.divider()

    # ── 4. Results (full-width below) ────────────────────────────────────
    st.subheader("Results")

    if not run_btn and "backtest_result" not in st.session_state:
        st.info(
            "Configure parameters above and click **Run Backtest**.\n\n"
            f"• Window ≤ {INTRADAY_LOOKBACK_LIMIT_DAYS} days → "
            "intraday-decision + daily-mark cadence.\n"
            "• Otherwise → daily-only cadence.\n\n"
            "Strikes are synthesized from Black-Scholes; mid-life marks "
            "use VIX-proxy IV scaling so vol regime shifts feed back "
            "into mark-to-market."
        )
        return

    if run_btn:
        if not tickers:
            st.error("Select at least one ticker.")
            return
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return

        # Use the editor's effective preset (seed + overrides) — not the
        # dropdown name alone, because that would silently drop any tweaks
        # the user staged in the Strategy Profile expander.
        preset = effective_preset
        risk_manager = RiskManager(
            max_risk_pct=preset.max_risk_pct,
            min_credit_ratio=preset.min_credit_ratio,
            max_delta=preset.max_delta,
            # Adaptive presets need the delta-aware floor so RM and the
            # scanner agree on what's a passing C/W ratio. Skill 03.
            delta_aware_floor=(preset.scan_mode == "adaptive"),
            edge_buffer=preset.edge_buffer,
        )
        port = HistoricalPort()

        run_caption = (
            f"Backtest · {len(tickers)} ticker(s) · "
            f"{start_date} → {end_date} · preset={preset.name} · "
            f"equity=${starting_equity:,.0f}"
        )
        run_detail = (
            f"{(end_date - start_date).days} day window · "
            f"{len(tickers)} ticker(s) · preset={preset.name}"
        )
        run_status = st.empty()
        run_status.info(f"⏳ {run_caption}")
        t0 = time.monotonic()
        try:
            # global_busy paints a fullscreen overlay so the user can't
            # accidentally fire stale clicks at other tabs while the
            # runner is mid-flight. Streamlit already serialises script
            # reruns, but the overlay makes the busy state visually
            # explicit (the small top-right "Running" badge is easy to
            # miss on long runs).
            with global_busy("Running backtest…", detail=run_detail):
                runner = BacktestRunner(
                    tickers=tuple(sorted(tickers)),
                    start=start_date, end=end_date,
                    preset=preset,
                    starting_equity=float(starting_equity),
                    risk_manager=risk_manager,
                    port=port,
                )
                result = runner.run()
        except Exception as exc:
            logger.exception("backtest run failed")
            run_status.error(f"❌ Backtest failed: {exc}")
            return

        elapsed = time.monotonic() - t0
        run_status.success(
            f"✅ Backtest complete in {elapsed:.1f}s — "
            f"{result.trade_count} trade(s) · cadence={result.cadence}."
        )
        st.session_state["backtest_result"] = result
        st.session_state["backtest_preset_name"] = preset.name

    result: Optional[BacktestResult] = st.session_state.get("backtest_result")
    if result is None:
        return

    # ── Run summary caption ─────────────────────────────────────────────
    st.caption(
        f"Preset: **{st.session_state.get('backtest_preset_name', '?')}** · "
        f"Cadence: **{result.cadence}** · "
        f"{result.run_t0.date()} → {result.run_t1.date()} · "
        f"{len(result.tickers)} ticker(s)"
    )

    # ── Summary metric cards ────────────────────────────────────────────
    cols = st.columns(5)
    cols[0].metric("Trades", result.trade_count)
    cols[1].metric("Win Rate", f"{result.win_rate_pct:.1f}%")
    cols[2].metric("Realised P&L", f"${result.realised_pnl:,.2f}")
    cols[3].metric("Total Return", f"{result.total_return_pct:+.2f}%")
    cols[4].metric("Ending Equity", f"${result.ending_equity:,.2f}")

    st.divider()

    # ── Equity + drawdown charts ────────────────────────────────────────
    eq_df = _equity_curve_to_df(result.equity_curve)
    if not eq_df.empty:
        st.plotly_chart(equity_curve_chart(eq_df), width='stretch')
        st.plotly_chart(drawdown_chart(eq_df), width='stretch')
        st.divider()

    # ── Closed trades table ─────────────────────────────────────────────
    st.subheader("Closed Trades")
    if result.closed_trades:
        closed_trades_table(result.closed_trades)

        # CSV export of the closed trades list.
        export_rows = [
            {
                "ticker":         t.ticker,
                "side":           t.side,
                "entry_t":        t.entry_t,
                "exit_t":         t.exit_t,
                "expiration":     t.expiration,
                "short_strike":   t.short_strike,
                "long_strike":    t.long_strike,
                "spread_width":   t.spread_width,
                "qty":            t.qty,
                "credit_open":    t.credit_open,
                "debit_close":    t.debit_close,
                "realised_pnl":   t.realised_pnl,
                "exit_signal":    t.exit_signal,
                "exit_reason":    t.exit_reason,
                "sigma_entry":    t.sigma_entry,
                "sigma_exit":     t.sigma_exit,
                "days_held":      t.days_held,
            }
            for t in result.closed_trades
        ]
        export_df = pd.DataFrame(export_rows)
        st.download_button(
            "Export trades CSV",
            data=export_df.to_csv(index=False).encode(),
            file_name=(
                f"backtest_{result.run_t0.date()}_to_{result.run_t1.date()}_"
                f"{st.session_state.get('backtest_preset_name','preset')}.csv"
            ),
            mime="text/csv",
        )
    else:
        st.info(
            "No trades were opened. Common causes:\n\n"
            "• The window is too short for the preset's DTE grid to "
            "place a position.\n"
            "• The regime classifier returned UNKNOWN for every cycle "
            "(expand 'Cycle outcomes' below to see why).\n"
            "• The preset's `min_pop`/`edge_buffer` floors rejected "
            "every candidate — try the 'Preview today's decision' "
            "panel to see live reject reasons."
        )

    # ── Cycle outcomes diagnostic (PERCEIVE→PLAN→RISK→EXECUTE) ──────────
    if result.cycle_outcomes:
        with st.expander(
            f"Cycle outcomes ({len(result.cycle_outcomes)} entry decisions evaluated)",
            expanded=False,
        ):
            st.caption(
                "One row per (ticker × intraday-decision-event). The "
                "`status` column tells you which phase the cycle "
                "exited on: `opened` (executed), `no_candidate` (engine "
                "found nothing), `risk_rejected` (RM blocked it), "
                "`no_data` (port returned empty)."
            )
            outcomes_df = _cycle_outcomes_to_df(result.cycle_outcomes)
            # Filter chips
            status_filter = st.multiselect(
                "Filter status",
                options=sorted(outcomes_df["status"].unique().tolist()),
                default=sorted(outcomes_df["status"].unique().tolist()),
                key="bt_status_filter",
            )
            if status_filter:
                outcomes_df = outcomes_df[outcomes_df["status"].isin(status_filter)]
            st.dataframe(outcomes_df, width='stretch', hide_index=True)

    # ── Open positions diagnostic (anything left at end of window) ──────
    if result.open_positions:
        with st.expander(
            f"Open positions at run end ({len(result.open_positions)} — "
            "force-closed by runner)",
            expanded=False,
        ):
            st.caption(
                "These positions were still open when the backtest "
                "window expired and were force-closed at the final "
                "synthetic mark."
            )
            rows = [
                {
                    "ticker":       p.ticker,
                    "side":         p.side,
                    "short_strike": p.short_strike,
                    "long_strike":  p.long_strike,
                    "qty":          p.qty,
                    "entry_t":      p.entry_t,
                    "current_mark": p.current_mark,
                }
                for p in result.open_positions
            ]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

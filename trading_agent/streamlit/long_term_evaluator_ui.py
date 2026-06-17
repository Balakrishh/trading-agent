"""
long_term_evaluator_ui.py — Streamlit tab for the long-term options evaluator.

Skill: ``docs/skills/40_long_term_options_evaluator.md``.

Walking-skeleton scope (this session)
-------------------------------------
- Sidebar: positions source selector (Manual paste only this session;
  Alpaca + Schwab pickers stubbed for next session).
- Holdings textarea (JSON paste) with inline validation.
- Section 1 — Portfolio snapshot: equity, sector pie, position counts.
- Section 2 — Manage existing options: lists open option positions and
  their current state vs the skill-40 §2.6 TP/SL anchors (read-only).
- Section 3 — Income overlay: covered-call recommendations for tickers
  held with ≥ 100 shares that are also on the watchlist.
- Sections 4 (entry vehicles) and 5 (portfolio gaps) are reserved
  placeholders this session.

Chain-fetch
-----------
The evaluator needs an option chain to score against. The walking
skeleton wires this through a thin ``_make_chain_fetcher`` that uses
the same ``MarketDataProvider`` the credit-spread agent uses. Failures
degrade gracefully — the panel shows a warning rather than crashing.

Architectural safety
--------------------
Mirrors ``watchlist_ui.py``: this module imports the long-term
evaluator, the positions provider, and the watchlist store — it does
NOT import any executor / order-placement module. Order placement
lands in Phase 5 (dedicated session).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from trading_agent.holdings_store import (
    clear_holdings,
    load_holdings,
    update_paste,
)
from trading_agent.long_term_evaluator import (
    EvaluatorConfig,
    LongTermEvaluator,
    Recommendation,
)
from trading_agent.positions_provider import (
    ManualPositionsProvider,
    Position,
    PositionsProvider,
    aggregate_snapshot,
)
from trading_agent.sector_map import sector_for
from trading_agent.watchlist_store import load_watchlist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_long_term_evaluator() -> None:
    """Top-level Streamlit renderer for the Long-Term Evaluator tab.

    Hooked into ``trading_agent/streamlit/app.py`` alongside the existing
    Live / Backtest / LLM / Watchlist tabs.
    """
    st.subheader("📈 Long-Term Options Evaluator")
    st.caption(
        "Portfolio-aware recommendations across covered calls "
        "(this session) and CSP / LEAPS / PMCC / debit spreads "
        "(next session). See `docs/skills/40_long_term_options_evaluator.md`."
    )

    # Diagnostic dict the section renderers will populate as they run.
    # Surfaced via a "🔍 Diagnostics" expander at the bottom of the tab
    # so the operator can see exactly where the pipeline dropped data
    # without having to scroll the streamlit terminal log.
    diag: Dict[str, Any] = {
        "stage": "init",
        "activation": st.session_state.get("lt_evaluator_activated", False),
        "holdings_file_path": "knowledge_base/holdings.json",
        "holdings_file_exists": False,
        "holdings_textarea_chars": 0,
        "parsed_positions": 0,
        "watchlist_size": 0,
        "intersection_held_and_watched": [],
        "cc_eligible_qty_100_plus": [],
        "chain_fetches": {},
        "recommendation_count": 0,
        "errors": [],
    }
    try:
        from pathlib import Path as _Path
        diag["holdings_file_exists"] = _Path(
            "knowledge_base/holdings.json"
        ).exists()
    except Exception:
        pass

    # Gate the heavy work behind an explicit activation — same pattern as
    # the Watchlist tab. Streamlit reruns every tab body on every event;
    # without the gate this tab would chain-fetch on every keystroke.
    if "lt_evaluator_activated" not in st.session_state:
        st.session_state.lt_evaluator_activated = False

    if not st.session_state.lt_evaluator_activated:
        if st.button("▶ Activate evaluator", key="lt_activate_btn"):
            st.session_state.lt_evaluator_activated = True
            st.rerun()
        st.info(
            "Activation gates the chain-fetch + scoring work so this tab "
            "doesn't run on every script rerun. Click to start."
        )
        return

    # ── Holdings input ────────────────────────────────────────────────
    diag["stage"] = "holdings_input"
    positions = _render_holdings_input()
    diag["holdings_textarea_chars"] = len(
        st.session_state.get("lt_holdings_textarea", "") or ""
    )
    diag["parsed_positions"] = len(positions) if positions else 0
    if positions is None:
        # Operator hasn't pasted holdings yet — render a stub snapshot
        # and bail before chain-fetching.
        st.warning("Paste your current holdings above to see recommendations.")
        _render_diagnostics_expander(diag)
        return

    provider = ManualPositionsProvider(positions=positions)
    diag["stage"] = "portfolio_snapshot"

    # ── Portfolio snapshot ────────────────────────────────────────────
    _render_portfolio_snapshot(provider)

    # ── Recommendations ───────────────────────────────────────────────
    watchlist_symbols = load_watchlist().symbols()
    diag["watchlist_size"] = len(watchlist_symbols)
    held_qty = {p.ticker: p.qty for p in positions if p.kind == "stock"}
    diag["intersection_held_and_watched"] = sorted(
        set(watchlist_symbols) & set(held_qty.keys())
    )
    diag["cc_eligible_qty_100_plus"] = sorted(
        t for t in diag["intersection_held_and_watched"]
        if held_qty.get(t, 0) >= 100
    )

    if not watchlist_symbols:
        st.info(
            "Your watchlist is empty. Add tickers in the **Watchlist** "
            "tab to see recommendations here."
        )
        _render_diagnostics_expander(diag)
        return

    # Wrap the chain fetcher so we can capture per-ticker outcomes for
    # the diagnostics panel without changing the fetcher's interface.
    raw_fetcher = _make_chain_fetcher()

    def _instrumented_fetcher(ticker: str):
        try:
            result = raw_fetcher(ticker)
        except Exception as exc:  # noqa: BLE001
            diag["chain_fetches"][ticker] = f"ERROR: {exc!s}"
            diag["errors"].append(f"chain {ticker}: {exc!s}")
            return []
        diag["chain_fetches"][ticker] = f"{len(result)} contracts"
        return result

    diag["stage"] = "scoring"
    evaluator = LongTermEvaluator(
        positions_provider=provider,
        call_chain_fetcher=_instrumented_fetcher,
        preset=_active_preset_or_none(),
        config=EvaluatorConfig(),
    )

    with st.spinner("Scoring covered-call candidates…"):
        recommendations = evaluator.recommend(watchlist_symbols)
    diag["recommendation_count"] = len(recommendations)
    diag["stage"] = "render"

    _render_manage_existing(provider, recommendations)
    _render_income_overlay(recommendations, diag=diag)
    _render_reserved_sections()
    _render_diagnostics_expander(diag)


# ---------------------------------------------------------------------------
# Holdings input
# ---------------------------------------------------------------------------

_HOLDINGS_PLACEHOLDER = """[
  {"ticker": "AAPL", "qty": 100, "avg_cost": 215.40, "kind": "stock"},
  {"ticker": "NVDA", "qty": 200, "avg_cost": 132.10, "kind": "stock"},
  {"ticker": "MSFT", "qty": 1,   "avg_cost": 18.50,  "kind": "option",
   "occ_symbol": "MSFT  270115C00400000", "side": "long"}
]

— OR paste your Schwab portfolio export directly. The parser
auto-detects the {Symbol, Qty (Quantity), Cost Basis, Asset Type}
shape, strips $/commas from Cost Basis, computes per-share avg_cost,
and skips the Cash & Positions Total summary rows."""


def _render_holdings_input() -> Optional[List[Position]]:
    """Holdings textarea + parse. Returns None until a valid paste lands.

    Persistence (skill 41 §3.4):
      * On first render after activation, load any saved paste from
        ``knowledge_base/holdings.json`` and pre-populate the textarea +
        st.session_state[<widget-key>] so the operator's last book is
        restored across Streamlit restarts.
      * On a successful parse, atomically save the raw paste + parsed
        count back to the file. Next restart loads the same book.
      * The "Reset saved holdings" button clears both the on-disk file
        and the session-state cache, leaving a blank textarea.

    Streamlit state pattern note: when ``st.text_area`` is given both a
    ``value=`` and a ``key=``, the WIDGET KEY wins on subsequent reruns
    (the ``value`` is only the initial fallback). To pre-populate, we
    set ``st.session_state[<widget-key>]`` BEFORE the widget renders and
    omit ``value=`` so the widget reads its own key. This is the only
    way the Schwab paste survives a Streamlit hot-reload.
    """
    st.markdown("### Your holdings")

    _TEXTAREA_KEY = "lt_holdings_textarea"
    _HYDRATED_FLAG = "lt_holdings_hydrated"

    # ── First-render hydration from persistent store ─────────────────
    # Hydration runs once per session — guard with a flag so re-runs
    # don't clobber the operator's in-flight edits with the on-disk blob.
    if not st.session_state.get(_HYDRATED_FLAG):
        saved = load_holdings()
        if not saved.is_empty:
            st.session_state[_TEXTAREA_KEY] = saved.raw_paste
            st.session_state.lt_holdings_saved_at = saved.saved_at
            st.session_state.lt_holdings_saved_count = saved.parsed_count
        st.session_state[_HYDRATED_FLAG] = True

    saved_at = st.session_state.get("lt_holdings_saved_at", "")
    saved_count = st.session_state.get("lt_holdings_saved_count", 0)
    if saved_at:
        st.caption(
            f"📁 Last saved {saved_count} positions at **{saved_at}**. "
            "Edit + click _Parse & save_ to update; _Reset_ clears the file."
        )
    else:
        st.caption(
            "Paste your current positions as a JSON array (canonical or "
            "Schwab portfolio-export format). Parsed pastes are saved to "
            "`knowledge_base/holdings.json` so they survive Streamlit "
            "restarts. The evaluator never writes to your brokerage — "
            "this is read-only input."
        )

    raw = st.text_area(
        "Holdings (JSON)",
        height=240,
        placeholder=_HOLDINGS_PLACEHOLDER,
        key=_TEXTAREA_KEY,
    )

    col_parse, col_reset, col_status = st.columns([1, 1, 3])
    with col_parse:
        parse_clicked = st.button(
            "💾 Parse & save", key="lt_parse_btn",
            help="Validate the JSON and persist it to disk.",
        )
    with col_reset:
        reset_clicked = st.button(
            "🗑 Reset saved", key="lt_reset_btn",
            help="Delete the saved holdings file and clear the textarea.",
        )

    # ── Reset path — wipes disk + session state, then reruns clean ──
    if reset_clicked:
        clear_holdings()
        for k in (
            _TEXTAREA_KEY,
            _HYDRATED_FLAG,
            "lt_holdings_saved_at",
            "lt_holdings_saved_count",
        ):
            st.session_state.pop(k, None)
        st.rerun()

    if not (raw or "").strip():
        return None

    try:
        provider = ManualPositionsProvider.from_json_text(raw)
    except ValueError as exc:
        st.error(f"Holdings JSON could not be parsed: {exc}")
        return None

    snapshot = provider.snapshot()
    if not snapshot:
        st.warning("Holdings JSON parsed, but no positions were found.")
        return None

    # ── Persist on every successful parse so the file reflects the
    # last operator-confirmed-working blob, never an in-flight edit. ─
    if parse_clicked:
        saved = update_paste(raw_paste=raw, parsed_count=len(snapshot))
        st.session_state.lt_holdings_saved_at = saved.saved_at
        st.session_state.lt_holdings_saved_count = saved.parsed_count
        with col_status:
            st.success(
                f"Saved {len(snapshot)} positions to "
                "`knowledge_base/holdings.json`."
            )
    else:
        with col_status:
            st.info(f"Parsed {len(snapshot)} positions (not saved yet).")
    return snapshot


# ---------------------------------------------------------------------------
# § Portfolio snapshot
# ---------------------------------------------------------------------------

def _render_portfolio_snapshot(provider: PositionsProvider) -> None:
    st.markdown("### Portfolio snapshot")
    snap = aggregate_snapshot(provider.snapshot(), sector_for=sector_for)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock value (cost basis)", f"${snap.total_stock_market_value:,.0f}")
    c2.metric("Option premium paid", f"${snap.total_option_premium_paid:,.0f}")
    c3.metric("Stock positions", snap.position_count_by_kind.get("stock", 0))
    c4.metric("Option positions", snap.position_count_by_kind.get("option", 0))

    if snap.by_sector:
        st.markdown("**Sector exposure (by cost basis)**")
        st.bar_chart(snap.by_sector)
    else:
        st.caption("No stock positions to break down by sector.")


# ---------------------------------------------------------------------------
# § Manage existing options
# ---------------------------------------------------------------------------

def _render_manage_existing(
    provider: PositionsProvider,
    recommendations: List[Recommendation],
) -> None:
    st.markdown("### Manage existing options")
    opt_positions = [p for p in provider.snapshot() if p.kind == "option"]
    if not opt_positions:
        st.caption("No open option positions. Nothing to surveil.")
        return

    rows = []
    for p in opt_positions:
        rows.append({
            "Ticker": p.ticker,
            "OCC symbol": p.occ_symbol,
            "Side": p.side,
            "Qty": p.qty,
            "Avg cost / contract": f"${p.avg_cost:,.2f}",
            "Status": "Surveillance lands next session",
        })
    st.dataframe(rows, use_container_width=True)


# ---------------------------------------------------------------------------
# § Income overlay (covered calls)
# ---------------------------------------------------------------------------

def _render_income_overlay(
    recommendations: List[Recommendation],
    *,
    diag: Optional[Dict[str, Any]] = None,
) -> None:
    st.markdown("### Income overlay — covered calls")
    cc = [r for r in recommendations if r.strategy == "covered_call"]
    if not cc:
        # Explain *why* this section is empty using the diagnostic dict
        # the orchestrator passes in. Skill 40 §4 gate ordering reproduced
        # here in operator-readable form.
        d = diag or {}
        intersect = d.get("intersection_held_and_watched", []) or []
        eligible = d.get("cc_eligible_qty_100_plus", []) or []
        chain_attempts = d.get("chain_fetches", {}) or {}
        with st.expander(
            "🛈 Why no covered-call suggestions?", expanded=True,
        ):
            st.markdown(
                f"- **Held ∩ Watchlist:** {len(intersect)} tickers "
                f"({', '.join(intersect) if intersect else '_none_'})"
            )
            st.markdown(
                f"- **Of those, ≥ 100 shares:** {len(eligible)} tickers "
                f"({', '.join(eligible) if eligible else '_none_'})"
            )
            if not eligible:
                st.warning(
                    "Covered calls require **100 shares per contract**. "
                    "None of your watchlist-held positions clear that "
                    "floor. Options: (a) add a ticker you already own "
                    "100+ shares of to the watchlist, (b) wait for the "
                    "LEAPS-overlay and debit-spread evaluators landing "
                    "next session — they work on smaller lots."
                )
            elif chain_attempts:
                st.markdown("- **Chain fetch outcomes per eligible ticker:**")
                for tk in eligible:
                    outcome = chain_attempts.get(
                        tk, "_chain not fetched_",
                    )
                    st.markdown(f"  - `{tk}`: {outcome}")
                st.info(
                    "If chains returned 0 contracts, your Alpaca data feed "
                    "(`ALPACA_OPTIONS_FEED` env var) may not include "
                    "options for those tickers. The default `indicative` "
                    "feed has gaps; the `opra` feed (paid) covers more. "
                    "If chains returned N>0 but no recommendations, the "
                    "preset's `cc_*` gates filtered everything — see "
                    "skill 40 §2.1 for the gate values."
                )
            else:
                st.info(
                    "Eligible tickers exist but no chains were fetched. "
                    "Check the Streamlit terminal log for "
                    "`Chain fetch for ... failed:` lines."
                )
        return

    for rec in cc:
        with st.expander(
            f"**{rec.ticker}** · ann. yield "
            f"{rec.metrics['annualised_return']*100:.1f}% · "
            f"POP {rec.metrics['pop']*100:.0f}%",
            expanded=False,
        ):
            st.markdown(f"_{rec.rationale}_")
            entry_total = rec.legs[0].limit_price * rec.legs[0].qty * 100.0
            ca_a, ca_b, ca_c, ca_d = st.columns(4)
            ca_a.metric("Entry limit (per share)", f"${rec.legs[0].limit_price:.2f}")
            ca_b.metric("Take-profit limit", f"${rec.take_profit_limit:.2f}")
            ca_c.metric("Stop trigger (underlying)", f"${rec.stop_trigger:.2f}")
            ca_d.metric("Score", f"{rec.score:.3f}")
            st.caption(
                f"**Bracket sketch:** STO {rec.legs[0].qty}× "
                f"`{rec.legs[0].occ_symbol}` @ ${rec.legs[0].limit_price:.2f} "
                f"limit → on fill, OCO with "
                f"(BTC @ ${rec.take_profit_limit:.2f}) and "
                f"(stop if underlying < ${rec.stop_trigger:.2f}). "
                f"Total credit if filled: **${entry_total:,.2f}**."
            )
            st.json({
                "ticker": rec.ticker,
                "strategy": rec.strategy,
                "metrics": {k: round(float(v), 4) for k, v in rec.metrics.items()},
                "legs": [
                    {
                        "action": leg.action, "occ_symbol": leg.occ_symbol,
                        "qty": leg.qty, "side": leg.side,
                        "limit_price": leg.limit_price,
                    }
                    for leg in rec.legs
                ],
            })


# ---------------------------------------------------------------------------
# § Diagnostics expander — operator self-debug
# ---------------------------------------------------------------------------

def _render_diagnostics_expander(diag: Dict[str, Any]) -> None:
    """Always-on bottom-of-tab expander revealing pipeline state.

    Reveals: activation flag, holdings file existence + saved-at, parse
    state, watchlist size, held∩watched intersection, ≥100-share
    eligibility, per-ticker chain fetch outcomes, total recommendation
    count, error trail.

    The expander defaults to *collapsed* so it doesn't clutter the
    healthy path, but stays visible at every render so the operator
    can self-debug an empty panel without scrolling the streamlit log.
    """
    expanded_default = (
        diag.get("recommendation_count", 0) == 0
        and bool(diag.get("parsed_positions"))
    )
    with st.expander(
        "🔍 Diagnostics (pipeline state)", expanded=expanded_default,
    ):
        st.markdown(f"**Stage reached:** `{diag.get('stage', 'unknown')}`")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Holdings file?", "yes" if diag.get(
            "holdings_file_exists") else "no")
        col_a.metric("Activated", "yes" if diag.get("activation") else "no")
        col_b.metric("Textarea chars", diag.get("holdings_textarea_chars", 0))
        col_b.metric("Parsed positions", diag.get("parsed_positions", 0))
        col_c.metric("Watchlist size", diag.get("watchlist_size", 0))
        col_c.metric(
            "Recommendations", diag.get("recommendation_count", 0),
        )

        intersect = diag.get("intersection_held_and_watched", []) or []
        eligible = diag.get("cc_eligible_qty_100_plus", []) or []
        st.markdown(
            f"- **Held ∩ Watchlist** ({len(intersect)}): "
            f"`{', '.join(intersect) if intersect else 'none'}`"
        )
        st.markdown(
            f"- **CC-eligible (≥100 shares)** ({len(eligible)}): "
            f"`{', '.join(eligible) if eligible else 'none'}`"
        )

        chain_fetches = diag.get("chain_fetches", {}) or {}
        if chain_fetches:
            st.markdown("- **Per-ticker chain fetch outcomes:**")
            for tk, outcome in sorted(chain_fetches.items()):
                st.markdown(f"  - `{tk}` → {outcome}")
        else:
            st.markdown(
                "- **Chain fetches:** _none attempted_ "
                "(usually means no CC-eligible tickers to fetch for)."
            )

        errors = diag.get("errors", []) or []
        if errors:
            st.markdown("- **Errors observed:**")
            for err in errors:
                st.markdown(f"  - {err}")

        st.caption(
            "If this expander is open by default, the pipeline ran but "
            "produced no recommendations — the breakdown above shows "
            "where the data dropped. Send me a screenshot of this "
            "expander and I'll know what's wrong."
        )


# ---------------------------------------------------------------------------
# § Reserved sections (next session)
# ---------------------------------------------------------------------------

def _render_reserved_sections() -> None:
    st.markdown("### Entry vehicles · _next session_")
    st.caption(
        "Cash-secured puts and LEAPS calls for watchlist tickers you "
        "don't currently own. Lands in the next session — see skill 40 §2.2 / §2.3."
    )

    st.markdown("### Portfolio gaps · _next session_")
    st.caption(
        "Sector / delta exposure analysis with diversification "
        "suggestions. Lands in the next session — see skill 40 §3.1."
    )


# ---------------------------------------------------------------------------
# Chain fetcher + preset wiring
# ---------------------------------------------------------------------------

def _make_chain_fetcher():
    """Return a ``ChainFetcher`` callable for the active market data provider.

    Walking-skeleton behavior: if the live ``MarketDataProvider`` is
    importable AND a recent-DTE call chain is fetchable, use it.
    Otherwise return an empty-chain stub so the panel renders without
    crashing and the operator sees an informative empty state.

    Uses ``market_data_factory.build_market_data_provider`` (the
    canonical factory) and threads the operator's Alpaca creds in from
    the active ``AppConfig`` — same plumbing the credit-spread agent
    and the existing Watchlist tab use.
    """
    try:
        from datetime import date

        from trading_agent.calendar_utils import next_weekly_expiration
        from trading_agent.config import load_config
        from trading_agent.market_data_factory import (
            build_market_data_provider,
        )

        config = load_config()
        # Surface = "long_term" — picks up MARKET_DATA_PROVIDER_LONG_TERM.
        # Default = "schwab": Schwab's options-chain coverage (including
        # ADRs like NOK + small-caps) is materially better than Alpaca's
        # `indicative` feed, which has gaps the covered-call scorer
        # silently absorbs as "0 candidates today". The operator's
        # SchwabMarketDataProvider OAuth is already wired for the agent's
        # credit-spread chain fetches; we just reuse it here.
        provider = build_market_data_provider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=getattr(
                config.alpaca, "data_url", "https://data.alpaca.markets/v2",
            ),
            alpaca_base_url=getattr(
                config.alpaca, "base_url",
                "https://paper-api.alpaca.markets/v2",
            ),
            surface="long_term",
            default_provider="schwab",
        )
    except Exception as exc:  # noqa: BLE001 — fail-open per skill 40 §4
        logger.warning("Could not initialise market-data provider: %s", exc)
        st.warning(
            f"Could not connect to a market-data provider: `{exc!s}`. "
            "Showing snapshot only — chain-fetch lives in the next pass."
        )
        return lambda _ticker: []

    def fetch(ticker: str) -> List[Dict[str, Any]]:
        # Walking-skeleton: pick the nearest weekly expiration in the
        # CC DTE band (30-60d). Sweep one candidate expiration this
        # session; next session sweeps the full band.
        today = date.today()
        try:
            exp = next_weekly_expiration(
                today, target_dte=45, dte_min=30, dte_max=60,
            )
            raw = provider.fetch_option_chain(
                underlying=ticker,
                expiration_date=(
                    exp.isoformat() if hasattr(exp, "isoformat") else str(exp)
                ),
                option_type="call",
            ) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chain fetch for %s failed: %s", ticker, exc)
            return []

        # Normalise to the dict shape the scorer expects.
        normalised = []
        for c in raw:
            try:
                dte = int(c.get("dte") or (exp - today).days)
            except Exception:
                dte = 45
            normalised.append({
                "strike": float(c.get("strike", 0.0)),
                "delta": float(c.get("delta", 0.0)),
                "bid": float(c.get("bid", 0.0)),
                "ask": float(c.get("ask", 0.0)),
                "dte": dte,
                "symbol": str(c.get("symbol", "")),
                # iv_rank is optional; skill 40 §4 fail-open if absent.
                **({"iv_rank": float(c["iv_rank"])} if "iv_rank" in c else {}),
            })
        return normalised

    return fetch


def _active_preset_or_none():
    """Return the active PresetConfig, or None to fall through to defaults."""
    try:
        from trading_agent.strategy_presets import get_active_preset
        return get_active_preset()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load active preset: %s", exc)
        return None

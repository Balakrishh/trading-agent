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
    positions = _render_holdings_input()
    if positions is None:
        # Operator hasn't pasted holdings yet — render a stub snapshot
        # and bail before chain-fetching.
        st.warning("Paste your current holdings above to see recommendations.")
        return

    provider = ManualPositionsProvider(positions=positions)

    # ── Portfolio snapshot ────────────────────────────────────────────
    _render_portfolio_snapshot(provider)

    # ── Recommendations ───────────────────────────────────────────────
    watchlist_symbols = load_watchlist().symbols()
    if not watchlist_symbols:
        st.info(
            "Your watchlist is empty. Add tickers in the **Watchlist** "
            "tab to see recommendations here."
        )
        return

    chain_fetcher = _make_chain_fetcher()
    evaluator = LongTermEvaluator(
        positions_provider=provider,
        call_chain_fetcher=chain_fetcher,
        preset=_active_preset_or_none(),
        config=EvaluatorConfig(),
    )

    with st.spinner("Scoring covered-call candidates…"):
        recommendations = evaluator.recommend(watchlist_symbols)

    _render_manage_existing(provider, recommendations)
    _render_income_overlay(recommendations)
    _render_reserved_sections()


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
        st.session_state so the operator's last book is restored across
        Streamlit restarts.
      * On a successful parse, atomically save the raw paste + parsed
        count back to the file. Next restart loads the same book.
      * The "Reset saved holdings" button clears both the on-disk file
        and the session-state cache, leaving a blank textarea.
    """
    st.markdown("### Your holdings")

    # ── First-render hydration from persistent store ─────────────────
    if "lt_holdings_blob" not in st.session_state:
        saved = load_holdings()
        if not saved.is_empty:
            st.session_state.lt_holdings_blob = saved.raw_paste
            st.session_state.lt_holdings_saved_at = saved.saved_at
            st.session_state.lt_holdings_saved_count = saved.parsed_count
        else:
            st.session_state.lt_holdings_blob = ""

    saved_at = st.session_state.get("lt_holdings_saved_at", "")
    saved_count = st.session_state.get("lt_holdings_saved_count", 0)
    if saved_at:
        st.caption(
            f"📁 Last saved {saved_count} positions at **{saved_at}**. "
            "Edit + click _Parse_ to update; _Reset_ clears the saved file."
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
        value=st.session_state.get("lt_holdings_blob", ""),
        height=240,
        placeholder=_HOLDINGS_PLACEHOLDER,
        key="lt_holdings_textarea",
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
            "lt_holdings_blob",
            "lt_holdings_saved_at",
            "lt_holdings_saved_count",
        ):
            st.session_state.pop(k, None)
        st.rerun()

    if not raw.strip():
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
        st.session_state.lt_holdings_blob = raw
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

def _render_income_overlay(recommendations: List[Recommendation]) -> None:
    st.markdown("### Income overlay — covered calls")
    cc = [r for r in recommendations if r.strategy == "covered_call"]
    if not cc:
        st.info(
            "No covered-call candidates today. Reasons might include: "
            "no holdings ≥ 100 shares on the watchlist; or the active "
            "preset's `cc_*` gates filtered every chain candidate. "
            "See **skill 40 §4** for the gate ordering."
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
            surface="watchlist",  # reuse the watchlist surface routing
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

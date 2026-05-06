"""
watchlist_ui.py — Streamlit Watchlist tab.

PR #3 scope (this file)
-----------------------
- Add / remove tickers (persisted to knowledge_base/watchlist.json).
- Per-ticker multi-timeframe regime table (1d / 4h / 1h / 15m / 5m).
- Macro context strip (VIX z-score + agreement summary).

PR #4 added the candlestick chart panel below the table — see
``watchlist_chart.py``. The chart panel is imported lazily so an
unactivated tab pays no import cost (Plotly is heavy).

Caching strategy
----------------
``@st.cache_data`` keyed on ``(ticker, intervals_tuple, refresh_token)``
so a Streamlit rerun within the same refresh window hits the cache. The
``refresh_token`` is bumped manually by the "Refresh" button — there is
no auto-refresh, because Streamlit re-evaluates *all* tab bodies on every
rerun and an auto-bump would re-fetch the world even when the user is
sitting on the Live tab.

Lazy activation
---------------
Streamlit's ``st.tabs`` does NOT lazy-render tabs — every tab body runs
on every script rerun. With 13+ tickers each costing 5–7 yfinance/Alpaca
RPCs, that's 60+ blocking network calls *per click anywhere in the app*.
To prevent that, the tab body is gated behind
``st.session_state.watchlist_activated``: until the user explicitly
clicks "Load watchlist data" once per session, the tab renders only a
lightweight skeleton + activation button. After activation, normal
behaviour resumes.

Parallel classification
-----------------------
``_classify_all`` fans the per-ticker work out across a thread pool
(``WATCHLIST_PARALLEL_WORKERS``, default 8). Each worker thread inherits
the current ScriptRunContext so ``@st.cache_data`` doesn't emit
"missing ScriptRunContext" warnings. Returns rows in original ticker
order regardless of completion order — the table layout depends on it.

Architectural safety
--------------------
This module ONLY imports the data layer, the multi-tf wrapper, the
watchlist store, and the regime enum. It deliberately does NOT import
``decision_engine``, ``chain_scanner``, ``executor``, or ``risk_manager``
— so the watchlist cannot affect trade decisions, only display.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ScriptRunContext propagation — without this, worker threads spawned from
# the main Streamlit thread get "missing ScriptRunContext" warnings on
# every @st.cache_data call. The pattern is documented in Streamlit's own
# `concurrent.futures` cookbook entry.
try:
    from streamlit.runtime.scriptrunner import (
        add_script_run_ctx,
        get_script_run_ctx,
    )
except ImportError:  # pragma: no cover — older Streamlit fallback
    def add_script_run_ctx(thread, ctx=None):  # type: ignore[no-redef]
        return thread

    def get_script_run_ctx():  # type: ignore[no-redef]
        return None

from trading_agent.config import load_config
from trading_agent.market_data import MarketDataProvider
from trading_agent.market_hours import is_within_market_hours
from trading_agent.multi_tf_regime import (
    DEFAULT_TIMEFRAMES,
    MultiTFRegime,
    adx_strength,
    adx_strength_label,
    classify_multi_tf,
)
from trading_agent.regime import Regime, RegimeClassifier
from trading_agent.streamlit._busy import global_busy
from trading_agent.streamlit.components import REGIME_COLORS
from trading_agent.watchlist_store import (
    DEFAULT_WATCHLIST_PATH,
    add_ticker,
    load_watchlist,
    remove_ticker,
)

logger = logging.getLogger(__name__)


# Refresh cadence — short enough for an analyst tab, long enough that
# yfinance doesn't rate-limit us. Same env-var convention as
# LIVE_MONITOR_REFRESH_SECS used by the live tab.
WATCHLIST_REFRESH_SECS = int(os.environ.get("WATCHLIST_REFRESH_SECS", "60"))

# Thread-pool size for parallel ticker classification. yfinance/Alpaca
# rate limits comfortably accommodate 8 concurrent connections per process
# at our request volume; bump higher only if you've validated the upstream
# limit. 1 = effectively serial (debug mode).
WATCHLIST_PARALLEL_WORKERS = max(
    1, int(os.environ.get("WATCHLIST_PARALLEL_WORKERS", "8"))
)


# ----------------------------------------------------------------------
# Per-process singletons — built once via @st.cache_resource so the
# data provider's snapshot / intraday caches are shared across reruns.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_data_provider() -> MarketDataProvider:
    """
    Build the market-data provider for the Watchlist surface.

    Routes via :func:`build_market_data_provider` so an operator can
    set ``MARKET_DATA_PROVIDER_WATCHLIST=yahoo`` to read indicators
    without spending Alpaca quota, or ``=schwab`` for parity with the
    live agent.  Defaults to Alpaca when no env var is set.
    """
    from trading_agent.market_data_factory import build_market_data_provider
    cfg = load_config()
    return build_market_data_provider(
        alpaca_api_key=cfg.alpaca.api_key,
        alpaca_secret_key=cfg.alpaca.secret_key,
        alpaca_data_url=cfg.alpaca.data_url,
        alpaca_base_url=cfg.alpaca.base_url,
        surface="watchlist",
    )


@st.cache_resource(show_spinner=False)
def _get_daily_classifier() -> RegimeClassifier:
    return RegimeClassifier(_get_data_provider())


# ----------------------------------------------------------------------
# Cached classification — wraps classify_multi_tf with a Streamlit cache.
# ``refresh_token`` is the manual cache-bust knob (bumped on user click /
# auto-refresh tick).
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=WATCHLIST_REFRESH_SECS)
def _classify_cached(
    ticker: str,
    intervals: Tuple[str, ...],
    refresh_token: int,
) -> Dict:
    """
    Returns a serialisable dict so Streamlit's cache hash is stable.
    The UI rebuilds Regime enum values on read.
    """
    provider = _get_data_provider()
    daily = _get_daily_classifier()
    out = classify_multi_tf(ticker, provider,
                            intervals=intervals,
                            daily_classifier=daily)

    by_interval = {}
    adx_by_interval = {}
    for tf, analysis in out.by_interval.items():
        # ``last_bar_ts`` is serialised as an ISO-8601 string so the
        # @st.cache_data hash stays stable (datetime objects from
        # different tz instances would otherwise miss the cache).
        last_bar_iso = (
            analysis.last_bar_ts.isoformat()
            if analysis.last_bar_ts is not None else None
        )
        by_interval[tf] = {
            "regime":           analysis.regime.value,
            "current_price":    analysis.current_price,
            "rsi_14":           analysis.rsi_14,
            "iv_rank":          analysis.iv_rank,
            "leadership_z":     analysis.leadership_zscore,
            "leadership_anchor": analysis.leadership_anchor,
            "leadership_signal_available": analysis.leadership_signal_available,
            "vix_z":            analysis.vix_zscore,
            "reasoning":        analysis.reasoning,
            "trend_conflict":   analysis.trend_conflict,
            "last_bar_ts":      last_bar_iso,
        }
        # ADX strength — only meaningful on intraday/daily bars; skip if
        # we can't fetch them cheaply (e.g. for the daily delegate).
        try:
            if tf == "1d":
                bars = provider.fetch_historical_prices(ticker)
            else:
                bars = provider.fetch_intraday_bars(ticker, tf)
            adx_by_interval[tf] = adx_strength(bars)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] ADX skipped at %s: %s", ticker, tf, exc)
            adx_by_interval[tf] = None

    return {
        "ticker":     ticker,
        "by_interval": by_interval,
        "adx":         adx_by_interval,
        "errors":      dict(out.errors),
        "agreement":   out.agreement_score,
    }


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_watchlist() -> None:
    """Top-level entry point — registered as the 4th tab in app.py.

    Lazy-load gate
    --------------
    Streamlit re-renders every tab body on every rerun. To avoid running
    the heavy per-ticker classification pipeline (5-7 RPCs × N tickers)
    on every click anywhere in the app, the body is gated behind
    ``st.session_state.watchlist_activated``. Until the user clicks the
    activation button once per session, the tab shows only the controls
    + a stub. After activation, behaviour matches the original.

    This also defers the lazy import of ``watchlist_chart`` (which pulls
    in plotly) until the user genuinely wants the tab — keeping app
    cold-start as light as possible for users who only use Live + Backtest.
    """
    st.subheader("📊 Watchlist")
    st.caption(
        "Multi-timeframe regime view. **Read-only** — does not influence "
        "the live agent's trade decisions. Same regime rule "
        "(`_determine_regime`) as the daily classifier, fed intraday bars."
    )

    if "watchlist_refresh_token" not in st.session_state:
        st.session_state.watchlist_refresh_token = int(time.time())
    if "watchlist_activated" not in st.session_state:
        st.session_state.watchlist_activated = False

    # Controls (add/remove/refresh) are always rendered — they're cheap
    # and let the user manage the ticker list before paying for a load.
    _render_controls()

    wl = load_watchlist(DEFAULT_WATCHLIST_PATH)
    if not wl.tickers:
        st.info(
            "No tickers yet — add some above. The macro VIX context strip "
            "and per-ticker regime table will appear once you do."
        )
        return

    # ── Lazy-load gate ──────────────────────────────────────────────────
    # Until the user clicks "Load watchlist data" once per session, render
    # a skeleton showing what's available + how many tickers will load.
    # This skips the 13×5-RPC sequential storm when the user is on a
    # different tab and Streamlit reruns the whole script.
    #
    # IMPORTANT — no st.rerun(): when the activation button is clicked,
    # we set the session-state flag and *fall through* to render the heavy
    # section in the SAME script run. Calling ``st.rerun()`` here would
    # reset ``st.tabs`` to the first tab (Live Monitoring) because the
    # tabs widget has no ``selected_index`` API — every rerun starts the
    # tab widget at index 0 unless the user has interacted with a widget
    # *inside* that tab during the same client connection. Falling
    # through preserves the active tab AND lets the overlay render on the
    # tab the user is actually looking at.
    if not st.session_state.watchlist_activated:
        n = len(wl.tickers)
        st.info(
            f"📋 Watchlist has **{n} ticker(s)**: "
            f"`{', '.join(wl.symbols())}`.\n\n"
            "Click below to fetch live indicators. "
            "(Skipping this on cold-start prevents the watchlist from "
            "running on every click in other tabs — Streamlit re-renders "
            "all tabs on every rerun.)"
        )
        activate_clicked = st.button(
            "▶ Load watchlist data",
            type="primary",
            use_container_width=True,
            key="wl_activate_btn",
        )
        if activate_clicked:
            # Flip the flag and fall through — DO NOT rerun (see above).
            st.session_state.watchlist_activated = True
        else:
            return  # Stub state — show the skeleton, exit the function.

    # ── Activated path: classify + render ──────────────────────────────
    # The overlay only fires when there's *new* work to do. A `st.rerun()`
    # from the Live tab (e.g. its "Refresh Now" button) re-executes this
    # block but ``_classify_cached`` returns instantly from the
    # @st.cache_data layer — flashing the overlay in that case made the
    # Live-tab Refresh button look like it was refreshing the watchlist.
    # We track the refresh-token last consumed and only paint the overlay
    # when it advances (first activation or explicit Refresh click).
    n = len(wl.tickers)
    detail = (
        f"{n} ticker(s) · {len(DEFAULT_TIMEFRAMES)} timeframes · "
        f"≤{WATCHLIST_PARALLEL_WORKERS} parallel"
    )
    current_token = st.session_state.watchlist_refresh_token
    last_loaded_token = st.session_state.get("watchlist_last_loaded_token")
    fetching_fresh = last_loaded_token != current_token

    if fetching_fresh:
        with global_busy("Refreshing watchlist…", detail=detail):
            rows = _classify_all(wl.symbols())
        st.session_state["watchlist_last_loaded_token"] = current_token
    else:
        # Cache-hit fast path — no overlay, no busy state. The
        # _classify_cached call still runs but returns from the @st.cache_data
        # layer in microseconds; we keep the call (rather than skipping it)
        # so any in-flight cache eviction triggered by TTL still rebuilds
        # cleanly without a state mismatch.
        rows = _classify_all(wl.symbols())

    _render_macro_strip(rows)
    _render_table(rows)

    # PR #4 chart panel — full Plotly stack with indicator toggles.
    # Imported here (not at module level) so an unactivated tab pays
    # neither the plotly import nor the chart-state cost.
    from trading_agent.streamlit.watchlist_chart import render_chart_panel
    render_chart_panel(_get_data_provider(), wl.symbols())


def _label_spacer() -> None:
    """
    Reserve vertical space equal to one Streamlit input label.

    Use inside a column whose sibling column has a labelled widget
    (text_input, selectbox, …) so a button or unlabelled control in this
    column lands on the same horizontal baseline as the input. Without
    this, Streamlit aligns each column's content to the top of the
    column and a button appears ~1 label-row higher than the input it
    operates on.
    """
    st.markdown(
        "<div style='height: 1.9rem; visibility: hidden;'>label</div>",
        unsafe_allow_html=True,
    )


def _render_controls() -> None:
    """Add / remove ticker inputs + manual refresh button.

    Layout: [text_input | Add | Remove select | Remove btn | Refresh]
    Column ratios are tuned so the Remove dropdown has room for ticker
    symbols (5-char tickers like "GOOGL" overflow at the old 1.0 ratio).
    Every non-input column begins with ``_label_spacer()`` so its
    button lands on the same baseline as the labelled text input.

    Why no ``st.rerun()`` calls
    ---------------------------
    Each button click already triggers a Streamlit rerun on its own —
    calling ``st.rerun()`` *additionally* fires a second one, and that
    second rerun resets ``st.tabs`` to index 0 (Live Monitoring) because
    the tabs widget has no ``selected_index`` API. Instead, we mutate
    state in place (``add_ticker``/``remove_ticker`` write to disk,
    ``_bump_refresh`` clears the cache) and let the rest of the script
    re-read fresh state in the SAME run. The reload below picks up the
    new ticker list, and ``_classify_all`` further down picks up the
    cleared cache. Everything stays on whichever tab the user is on.
    """
    cols = st.columns([3, 1, 1.4, 1, 1])
    with cols[0]:
        new_ticker = st.text_input(
            "Add ticker (uppercased automatically)",
            key="wl_add_input",
            placeholder="e.g. SPY, QQQ, AAPL",
        ).strip().upper()
    with cols[1]:
        _label_spacer()
        if st.button("Add", use_container_width=True, key="wl_add_btn"):
            if new_ticker:
                add_ticker(new_ticker, path=DEFAULT_WATCHLIST_PATH)
                _bump_refresh()
                # No st.rerun() — disk write is visible to the
                # load_watchlist() call below within this same script run.
            else:
                st.warning("Enter a ticker first")

    wl = load_watchlist(DEFAULT_WATCHLIST_PATH)
    with cols[2]:
        _label_spacer()
        to_remove = st.selectbox(
            "Remove",
            options=["—"] + (wl.symbols() if wl.tickers else []),
            key="wl_remove_select",
            label_visibility="collapsed",
            disabled=not wl.tickers,
        )
    with cols[3]:
        _label_spacer()
        remove_clicked = st.button(
            "✕ Remove",
            use_container_width=True,
            key="wl_remove_btn",
            disabled=(not wl.tickers or to_remove == "—"),
        )
        if remove_clicked and to_remove != "—":
            remove_ticker(to_remove, path=DEFAULT_WATCHLIST_PATH)
            _bump_refresh()
            # No st.rerun() — same rationale as Add.
    with cols[4]:
        _label_spacer()
        if st.button("⟳ Refresh", use_container_width=True,
                     key="wl_refresh_btn"):
            _bump_refresh()
            # No st.rerun() — _classify_cached.clear() inside _bump_refresh
            # already empties the cache, so the next _classify_all call in
            # this same script run re-fetches everything fresh.


def _bump_refresh() -> None:
    st.session_state.watchlist_refresh_token = int(time.time())
    # Streamlit's cache_data TTL also self-expires every
    # WATCHLIST_REFRESH_SECS, but bumping the token forces immediate
    # invalidation when the user explicitly clicks Refresh.
    _classify_cached.clear()


def _classify_one(ticker: str, intervals: Tuple[str, ...],
                  token: int, ctx=None) -> Dict:
    """Worker wrapper run inside the thread pool.

    Attaches the main thread's ScriptRunContext so ``@st.cache_data``
    inside ``_classify_cached`` can resolve the active session without
    emitting "missing ScriptRunContext" warnings on every call. Catches
    any per-ticker exception and returns a stub row so a single bad
    ticker can't poison the whole table.
    """
    if ctx is not None:
        try:
            add_script_run_ctx(threading.current_thread(), ctx)
        except Exception:  # pragma: no cover — defensive, shouldn't fire
            pass
    try:
        return _classify_cached(ticker, intervals, token)
    except Exception as exc:  # noqa: BLE001 — keep UI alive
        logger.warning("[%s] watchlist classification failed: %s",
                       ticker, exc)
        return {
            "ticker": ticker,
            "by_interval": {},
            "adx": {},
            "errors": {"_top": str(exc)},
            "agreement": 0.0,
        }


def _classify_all(tickers: List[str]) -> List[Dict]:
    """Classify every ticker in parallel.

    Each ticker's pipeline is I/O-bound (yfinance / Alpaca RPCs +
    indicator math), so a thread pool gives a near-linear speedup up to
    the rate-limit ceiling. With 13 tickers and 8 workers, total wall
    time drops from ~13×(1-3s)=13-39s sequential to ~3-6s.

    Result order matches input order (the dataframe builder in
    ``_render_table`` assumes this so user-facing row order is stable).
    """
    if not tickers:
        return []

    token = st.session_state.watchlist_refresh_token
    ctx = get_script_run_ctx()
    workers = min(WATCHLIST_PARALLEL_WORKERS, len(tickers))

    # Map future → original index so we can re-sort to input order.
    rows: List[Optional[Dict]] = [None] * len(tickers)
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="watchlist") as ex:
        future_to_idx = {
            ex.submit(_classify_one, t, DEFAULT_TIMEFRAMES, token, ctx): i
            for i, t in enumerate(tickers)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                rows[idx] = future.result()
            except Exception as exc:  # noqa: BLE001 — final defensive net
                # _classify_one already catches per-ticker errors, so this
                # only fires on programming bugs (invalid future, etc).
                logger.exception("watchlist worker crashed for %s",
                                 tickers[idx])
                rows[idx] = {
                    "ticker": tickers[idx],
                    "by_interval": {},
                    "adx": {},
                    "errors": {"_top": str(exc)},
                    "agreement": 0.0,
                }

    elapsed = time.monotonic() - started
    logger.info(
        "Watchlist refreshed %d tickers in %.2fs (workers=%d)",
        len(tickers), elapsed, workers,
    )
    # Strip Optional[Dict] for the type checker — every slot is filled
    # by the time we get here.
    return [r for r in rows if r is not None]


def _render_macro_strip(rows: List[Dict]) -> None:
    """Top-of-page summary: VIX z-score, # tickers, avg agreement."""
    # Pull the VIX z-score from the first row that has a 1d cell — it's
    # a market-wide signal so any successful row exposes the same value.
    vix_z = next(
        (r["by_interval"]["1d"]["vix_z"]
         for r in rows
         if "1d" in r["by_interval"]),
        0.0,
    )
    avg_agreement = (
        sum(r["agreement"] for r in rows) / len(rows) if rows else 0.0
    )

    cols = st.columns(3)
    # NOTE: every metric carries a ``delta`` so the three cards reserve
    # identical vertical space.  For Tickers and Avg TF agreement we use a
    # neutral spacer (``" "`` with ``delta_color="off"``) — Streamlit lays
    # out the delta row even when it's blank, which keeps the macro strip
    # baseline-aligned.  Removing the spacer here will make VIX taller than
    # its siblings on every render.
    cols[0].metric("Tickers", len(rows), delta=" ", delta_color="off")
    cols[1].metric(
        "VIX z-score (5min Δ)",
        f"{vix_z:+.2f}",
        delta="Bullish-inhibit" if vix_z > 2.0 else "OK",
        delta_color="inverse" if vix_z > 2.0 else "normal",
    )
    cols[2].metric(
        "Avg TF agreement",
        f"{avg_agreement:.0%}",
        delta=" ",
        delta_color="off",
        help="Share of timeframes whose trend matches each ticker's "
             "longest interval (typically 1d). 100% = all aligned.",
    )


# Anchors with reliably-good Alpaca IEX coverage during RTH.  When
# rows anchored to these still come back signal-unavailable, the cause
# is almost certainly market-closed / off-hours / bad credentials —
# NOT IEX feed thinness.  Used by ``_lead_z_health`` to classify the
# failure mode the user is seeing.
_BROAD_MARKET_ANCHORS = frozenset({"SPY", "QQQ", "IWM", "DIA"})


def _lead_z_health(rows: List[Dict]) -> Dict[str, int]:
    """Bucket Lead-z signal availability across the table.

    Returns counts so the caller can pick the right banner copy:
      * ``broad_ok / broad_total``   — rows anchored to SPY/QQQ/IWM/DIA
      * ``sector_ok / sector_total`` — rows anchored to a sector ETF
                                        (XLF, XLK, …)
      * ``no_anchor``                — rows with leadership_anchor == ""
                                        (special tickers, unknown to the map)
      * ``total``                    — rows with a 1d cell (rows with no
                                        1d cell are excluded; we can't
                                        say anything about them)

    The classification is intentionally based on the **anchor**, not the
    ticker itself, because the failure mode is per-anchor: if XLF's 5-min
    series is empty, every JPM/BAC/WFC row anchored to XLF fails together.
    """
    counts = {
        "broad_ok": 0, "broad_total": 0,
        "sector_ok": 0, "sector_total": 0,
        "no_anchor": 0, "total": 0,
    }
    for r in rows:
        d = r.get("by_interval", {}).get("1d")
        if not d:
            continue
        counts["total"] += 1
        anchor = d.get("leadership_anchor", "")
        ok = bool(d.get("leadership_signal_available", False))
        if not anchor:
            counts["no_anchor"] += 1
            continue
        if anchor in _BROAD_MARKET_ANCHORS:
            counts["broad_total"] += 1
            if ok:
                counts["broad_ok"] += 1
        else:
            # Sector ETFs and any other non-broad anchor.
            counts["sector_total"] += 1
            if ok:
                counts["sector_ok"] += 1
    return counts


def _render_lead_z_health_banner(counts: Dict[str, int]) -> None:
    """Render a single banner explaining the current Lead-z source state.

    Picks one of four messages based on the per-anchor success counts so
    the user can self-diagnose whether the failure is *systemic*
    (markets closed / creds bad → all rows fail) vs *per-anchor*
    (only sector-ETF anchors fail → IEX feed thinness).
    """
    broad_ok = counts["broad_ok"]
    broad_total = counts["broad_total"]
    sector_ok = counts["sector_ok"]
    sector_total = counts["sector_total"]
    total = counts["total"]
    if total == 0:
        return  # Empty watchlist — nothing to report.

    anchored_total = broad_total + sector_total
    anchored_ok = broad_ok + sector_ok

    # Happy path: every anchored row has a real reading.  Stay quiet.
    if anchored_total > 0 and anchored_ok == anchored_total:
        st.success(
            f"✅ Lead-z source healthy — {anchored_ok}/{anchored_total} "
            "tickers have a real reading."
        )
        return

    # Total blackout: zero anchored rows succeeded.
    if anchored_total > 0 and anchored_ok == 0:
        st.error(
            f"❌ Lead-z unavailable for all {anchored_total} anchored "
            "tickers.  Likely cause: markets closed (weekend / off-hours), "
            "Alpaca credentials missing or rejected, or the Alpaca data "
            "URL is unreachable.  Check `APCA_API_KEY_ID` / "
            "`APCA_API_SECRET_KEY` and confirm the time matches a US "
            "equities session."
        )
        return

    # Sector-only failure: broad-market anchors work but every sector
    # anchor fails.  Classic IEX feed thinness on XL* ETFs.
    if (broad_total > 0 and broad_ok == broad_total
            and sector_total > 0 and sector_ok == 0):
        st.warning(
            f"⚠️ Lead-z works for SPY/QQQ-anchored rows ({broad_ok}/"
            f"{broad_total}) but fails for all {sector_total} "
            "sector-ETF-anchored rows.  Cause: Alpaca's free IEX feed "
            "has thin volume on sector ETFs (XLF, XLK, XLY, …) so "
            "after `OPEN_BAR_SKIP` is dropped there are <2 5-min bars "
            "to compute a rolling stdev.  Fix: set "
            "`ALPACA_STOCKS_FEED=sip` for full-market coverage, or "
            "remap the affected tickers' anchors to SPY in "
            "`LEADERSHIP_ANCHORS`."
        )
        return

    # Mixed / partial failure.  Surface the breakdown so the user
    # knows which anchors to investigate.
    st.warning(
        f"⚠️ Lead-z partially available: "
        f"{anchored_ok}/{anchored_total} anchored tickers have a real "
        f"reading "
        f"(broad anchors {broad_ok}/{broad_total}, "
        f"sector anchors {sector_ok}/{sector_total}).  "
        "See the legend below for what '— (no data vs X)' means."
    )


def _render_table(rows: List[Dict]) -> None:
    """One row per ticker; one column per timeframe + ADX + IV rank."""
    intervals = list(DEFAULT_TIMEFRAMES)

    # Detect stale data once per render. ``_stale_minutes`` picks the
    # *freshest* bar across all (row × timeframe) cells — during RTH the
    # 5m bar will be < 5 min old, so this naturally suppresses the
    # warning while markets are open. When no row has any usable
    # timestamp, default to "fresh" — better to omit the chip than
    # display a misleading one.
    stale_age_minutes = _stale_minutes(rows)

    # Lead-z source-health banner — rendered above the table so the user
    # can interpret "no data" rows immediately.  Computed once per render
    # from the 1d cells (cheap; just dict access).
    _render_lead_z_health_banner(_lead_z_health(rows))

    # Build the dataframe in display order.
    display_rows = []
    for r in rows:
        record = {"Ticker": r["ticker"]}
        for tf in intervals:
            cell = r["by_interval"].get(tf)
            adx = r["adx"].get(tf)
            label = adx_strength_label(adx)
            if cell:
                regime = cell["regime"]
                # ⚠ when long & medium SMAs disagree on direction —
                # informational only, doesn't change the regime label.
                conflict_mark = " ⚠" if cell.get("trend_conflict") else ""
                record[tf] = (f"{_emoji(regime)} {regime}{conflict_mark}"
                              f" · ADX {label}")
            elif tf in r["errors"]:
                record[tf] = "—"
            else:
                record[tf] = ""
        # IV rank + leadership are daily-context fields — pull from 1d.
        d = r["by_interval"].get("1d", {})
        record["IV Rank"] = (f"{d['iv_rank']:.0f}"
                             if d.get("iv_rank") is not None else "—")
        # Lead-z formatting:
        #   * No anchor configured (leadership_anchor == "")  → "—"
        #     (silent zeros pre-fix were misleading: the system literally
        #      had nothing to compare against, but printed +0.00)
        #   * RPC-failed / degenerate stdev / IEX feed empty
        #     (leadership_signal_available == False) → "—" + "(no data)"
        #     so the user can tell "computed and got 0" from "couldn't
        #     compute".  Common cause: sector ETF anchors (XLF, XLK, …)
        #     trade primarily on NYSE Arca and Alpaca's free IEX feed
        #     returns <2 5-min bars after open-skip, which makes
        #     get_5min_return_series return None.
        #   * Anchor configured AND signal available → 3 decimals so
        #     genuinely-tiny but non-zero deltas (SPY vs QQQ on a quiet
        #     day) become visible instead of rounding to +0.00.
        anchor = d.get("leadership_anchor", "")
        signal_ok = d.get("leadership_signal_available", False)
        if not anchor:
            record["Lead z"] = "—"
        elif d.get("leadership_z") is None:
            record["Lead z"] = "—"
        elif not signal_ok:
            record["Lead z"] = f"— (no data vs {anchor})"
        else:
            record["Lead z"] = f"{d['leadership_z']:+.3f} (vs {anchor})"
        record["TF agree"] = f"{r['agreement']:.0%}"
        display_rows.append(record)

    df = pd.DataFrame(display_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Freshness banner — three-state output (silent / feed-health hint /
    # stale-data warning) gated by NYSE market hours via
    # ``is_within_market_hours()``. See ``_render_freshness_banner`` for
    # the decision matrix. The previous version fired the "stale"
    # warning unconditionally whenever the 1d-bar timestamp was old,
    # which is essentially always (1d closes only update once per day,
    # so during RTH on Monday the 1d bar is Friday's close — naturally
    # 50-80h stale by design).
    _render_freshness_banner(rows, stale_age_minutes)

    # Color-key footnote so the user can decode emoji/label combos.
    legend_parts = [
        f"<span style='color:{REGIME_COLORS[k]}'>{_emoji(k)} {k}</span>"
        for k in ("bullish", "bearish", "sideways", "mean_reversion")
    ]
    st.caption(
        "Regime legend: " + " · ".join(legend_parts)
        + " · ADX label: weak (<20) · developing (20-40) · strong (40+)"
        + " · ⚠ = SMA50/SMA200 slope conflict"
        + " · Lead-z \"— (no data vs X)\" = anchor X exists but the "
        "intraday return series was unavailable (Alpaca IEX feed limit, "
        "weekend / pre-market, or degenerate stdev)",
        unsafe_allow_html=True,
    )


def _stale_minutes(rows: List[Dict]) -> Optional[float]:
    """Return age (minutes) of the *freshest* bar across all rows + timeframes.

    Why we scan ALL intervals (was 1d-only)
    ---------------------------------------
    The 1d bar's ``last_bar_ts`` is the close timestamp of the most
    recently *completed* daily session — which during an active RTH
    session is yesterday's close (~16-24h old) and after a weekend is
    Friday's close (~50-80h old). Using only the 1d timestamp made the
    "Data stale" warning fire constantly during normal Monday-morning
    use even though the 5m / 15m / 1h feeds were updating live.

    The fix is to compute the *freshest* timestamp across every
    (row × timeframe) cell. During RTH that's the 5m or 15m bar (< 5-15
    min old). After hours, intraday bars also stop advancing, so the
    minimum age grows naturally — and the warning fires when it
    *should* (markets actually closed / weekend / off-hours).

    Tz-handling
    -----------
    yfinance daily bars come back tz-naive ("2026-05-01T00:00:00"),
    Alpaca intraday bars come back tz-aware ("2026-05-04T13:30:00+00:00").
    Mixing them in a ``ts > youngest`` comparison would raise
    ``TypeError: can't compare offset-naive and offset-aware datetimes``,
    so we normalise every parsed timestamp to aware-UTC immediately
    after parsing — before any cross-timeframe comparison.

    Returns None when no row has *any* usable timestamp — caller treats
    that as "unknown freshness", which suppresses the stale chip.
    """
    youngest: Optional[datetime] = None
    for r in rows:
        for tf_cell in r.get("by_interval", {}).values():
            ts_iso = tf_cell.get("last_bar_ts") if isinstance(tf_cell, dict) else None
            if not ts_iso:
                continue
            try:
                ts = datetime.fromisoformat(ts_iso)
            except (TypeError, ValueError):
                continue
            # Normalise BEFORE comparing — daily bars are tz-naive,
            # intraday bars are tz-aware, and Python refuses to compare
            # the two. Treating naive timestamps as UTC matches what the
            # data provider does internally when materialising bars.
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if youngest is None or ts > youngest:
                youngest = ts
    if youngest is None:
        return None
    now = datetime.now(timezone.utc)
    return (now - youngest).total_seconds() / 60


def _per_tf_freshness(rows: List[Dict]) -> Dict[str, Optional[float]]:
    """Return ``{tf: freshest_age_minutes_or_None}`` per timeframe.

    The age is the *freshest* (smallest) age across all rows for that
    timeframe, in minutes. ``None`` means no row had a usable
    ``last_bar_ts`` for that tf — typically because the underlying RPC
    failed for every ticker (Alpaca IEX feed empty, yfinance rate limit,
    sector-ETF anchor with no 5m bars, etc).

    Used by the table's freshness banner to distinguish:
      * 5m/15m/1h fresh → markets open, feed healthy → no warning
      * Only 1d populated → intraday RPCs failing → feed-health hint
      * Everything stale  → market actually closed → "data stale" warning
    """
    now = datetime.now(timezone.utc)
    out: Dict[str, Optional[float]] = {}
    for tf in DEFAULT_TIMEFRAMES:
        youngest: Optional[datetime] = None
        for r in rows:
            cell = r.get("by_interval", {}).get(tf)
            if not isinstance(cell, dict):
                continue
            ts_iso = cell.get("last_bar_ts")
            if not ts_iso:
                continue
            try:
                ts = datetime.fromisoformat(ts_iso)
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if youngest is None or ts > youngest:
                youngest = ts
        out[tf] = (now - youngest).total_seconds() / 60 if youngest else None
    return out


def _format_age(minutes: Optional[float]) -> str:
    """Compact age string for the freshness banner: '3m', '4h', '2.1d'."""
    if minutes is None:
        return "—"
    if minutes < 60:
        return f"{minutes:.0f}m"
    if minutes < 60 * 24:
        return f"{minutes / 60:.1f}h"
    return f"{minutes / (60 * 24):.1f}d"


def _render_freshness_banner(rows: List[Dict],
                             stale_age_minutes: Optional[float]) -> None:
    """Render the data-freshness banner.

    Decision matrix
    ---------------
    | Market | Intraday fresh | Banner                               |
    |--------|----------------|--------------------------------------|
    | open   | yes (<30m)     | hidden (success state)               |
    | open   | no             | feed-health hint + per-tf ages       |
    | closed | n/a            | "Data stale, markets closed" + ages  |

    "Markets open" is determined by ``is_within_market_hours()`` (NYSE
    calendar, weekends + holidays excluded, with the configured 5-min
    open/close buffers). It's the same primitive the live agent uses,
    so the watchlist tab and the live cycle agree on what "open" means.
    """
    market_open = False
    try:
        market_open = is_within_market_hours()
    except Exception as exc:  # pragma: no cover — defensive
        # If the calendar lookup fails (e.g. pandas_market_calendars import
        # error in some sandboxed runtime), don't block the table render —
        # fall back to the time-based-only check.
        logger.debug("market-hours lookup failed, falling back: %s", exc)

    per_tf = _per_tf_freshness(rows)
    intraday_tfs = [tf for tf in DEFAULT_TIMEFRAMES if tf != "1d"]
    intraday_ages = [per_tf[tf] for tf in intraday_tfs if per_tf.get(tf) is not None]
    has_fresh_intraday = any(a is not None and a <= 30 for a in intraday_ages)

    # Build the per-tf age summary once — reused in two banner variants.
    age_summary = " · ".join(
        f"{tf}={_format_age(per_tf.get(tf))}" for tf in DEFAULT_TIMEFRAMES
    )

    if market_open and has_fresh_intraday:
        # Happy path during RTH — feed is alive. Don't render anything;
        # the table itself signals success.
        return

    if market_open and not has_fresh_intraday:
        # Markets ARE open but no intraday data made it through. This is
        # an intraday-feed health problem, NOT a "stale" condition. Most
        # common causes: missing/wrong Alpaca creds (falls back to
        # yfinance which is rate-limited), Alpaca IEX feed empty for
        # sector-ETF anchors, or yfinance returning only daily bars.
        st.warning(
            "⚠️ Markets are open but no intraday bars are reaching the "
            "watchlist. The Lead-z / VIX-z values you see reflect the "
            "last completed session, not live conditions.\n\n"
            f"**Per-timeframe freshness:** {age_summary}\n\n"
            "Common causes: missing/invalid Alpaca credentials "
            "(`APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`), Alpaca IEX "
            "feed empty for sector ETFs (try `ALPACA_STOCKS_FEED=sip`), "
            "or yfinance rate-limit. The Live Monitoring tab uses the "
            "same data path — check there for the underlying RPC error."
        )
        return

    # Markets closed — the original "stale" warning, but only fires when
    # we genuinely have no recent data.
    if stale_age_minutes is not None and stale_age_minutes > 30:
        hours = stale_age_minutes / 60
        st.warning(
            f"⏰ Data stale by ~{hours:.1f}h — markets likely closed. "
            "Lead-z / VIX-z reflect the last completed session, not "
            "live conditions.\n\n"
            f"**Per-timeframe freshness:** {age_summary}"
        )


def _emoji(regime_value: str) -> str:
    return {
        Regime.BULLISH.value:        "🟢",
        Regime.BEARISH.value:        "🔴",
        Regime.SIDEWAYS.value:       "🟡",
        Regime.MEAN_REVERSION.value: "🟣",
    }.get(regime_value, "⚪")


# Chart panel lives in trading_agent/streamlit/watchlist_chart.py and is
# imported lazily inside render_watchlist() — keeps cold-start light when
# the user never opens this tab.

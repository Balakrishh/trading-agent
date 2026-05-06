"""
app.py — Streamlit dashboard entry point.

Run with:
    streamlit run trading_agent/streamlit/app.py

Logging
-------
We initialise logging at import time so every ``logger.info(...)`` call
inside the agent / backtester reaches the terminal Streamlit was launched
from.  Without this, Python's root logger defaults to WARNING level and
INFO-level diagnostics (rate-limiter sleeps, per-ticker progress, strike
selection, etc.) are silently dropped — which made long backtests
indistinguishable from genuine hangs.

The level can be overridden via the ``LOG_LEVEL`` env var:

    LOG_LEVEL=DEBUG streamlit run trading_agent/streamlit/app.py
"""

import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `trading_agent` is importable
# regardless of how Streamlit sets the working directory.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# IMPORTANT: setup_logging() must run BEFORE any other trading_agent module
# is imported.  Sub-modules grab their loggers at import time via
# ``logging.getLogger(__name__)``; if the root logger isn't configured yet
# those loggers inherit Python's default WARNING level and INFO calls
# from the backtester are silently dropped.
from trading_agent.logger_setup import setup_logging  # noqa: E402

setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Trading Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Page header with Market-Status badge in the top-right ────────────────
# Layout: title + caption on the left, OPEN/CLOSED badge floated to the
# right. Lives at the page root (above ``st.tabs``) so the badge is
# visible from every tab — the operator never has to scroll into a
# specific tab to know whether the market is currently in session.
#
# The badge consults ``is_within_market_hours()`` which uses the NYSE
# pandas_market_calendar (correctly handles weekends + holidays + the
# 5-minute open/close buffers). Same primitive the live agent loop uses,
# so the badge and the agent's "should-I-trade" decision agree by
# construction.
from trading_agent.market_hours import is_within_market_hours  # noqa: E402

_hdr_left, _hdr_right = st.columns([5, 1])
with _hdr_left:
    st.title("📈 Trading Agent Dashboard")
    st.caption("Credit-spread options agent · Paper trading · Alpaca Markets")
with _hdr_right:
    # Inline-styled badge — full background fill so it reads at a glance
    # even on a dense dashboard. Right-aligned via a flex wrapper so the
    # badge hugs the column's right edge regardless of text length.
    try:
        _market_open = is_within_market_hours()
    except Exception:
        _market_open = None  # calendar lookup failed — show neutral state

    if _market_open is True:
        _bg, _label = "#1b8a3a", "🟢 MARKET OPEN"
    elif _market_open is False:
        _bg, _label = "#a62a2a", "🔴 MARKET CLOSED"
    else:
        _bg, _label = "#5a5a5a", "⚪ MARKET STATUS UNKNOWN"

    st.markdown(
        f"""
        <div style="display:flex;justify-content:flex-end;
                    align-items:center;height:100%;padding-top:1rem">
          <span style="background:{_bg};color:#fff;padding:0.4rem 0.9rem;
                       border-radius:6px;font-weight:600;
                       font-size:0.95rem;letter-spacing:0.02em;
                       white-space:nowrap;
                       box-shadow:0 2px 6px rgba(0,0,0,0.18)">
            {_label}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Inject the global busy-overlay stylesheet AND reserve a top-level
# placeholder for the overlay BEFORE st.tabs(). The placeholder MUST live
# above the tabs, because Streamlit hides inactive tab panels with
# `display: none` — and CSS `position: fixed` cannot escape a
# `display: none` ancestor. An overlay rendered inside the (currently
# hidden) watchlist tab while the user is on Live tab would be invisible.
# See trading_agent/streamlit/_busy.py for the full rationale.
from trading_agent.streamlit._busy import (  # noqa: E402
    inject_overlay_css,
    register_top_level_slot,
)

inject_overlay_css()
register_top_level_slot()

tab_live, tab_backtest, tab_llm, tab_watchlist = st.tabs(
    ["📡 Live Monitoring", "📊 Backtesting", "🤖 LLM Extension", "📊 Watchlist"]
)

with tab_live:
    from trading_agent.streamlit.live_monitor import render_live_monitor
    render_live_monitor()

with tab_backtest:
    from trading_agent.streamlit.backtest_ui import render_backtest_ui
    render_backtest_ui()

with tab_llm:
    from trading_agent.streamlit.llm_extension import render_llm_extension
    render_llm_extension()

with tab_watchlist:
    # Read-only analyst tab — does not import decision_engine,
    # chain_scanner, executor, or risk_manager. See watchlist_ui.py
    # docstring for the architectural-safety rationale.
    from trading_agent.streamlit.watchlist_ui import render_watchlist
    render_watchlist()

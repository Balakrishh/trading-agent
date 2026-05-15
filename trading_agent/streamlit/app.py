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
    # "auto" collapses the sidebar on narrow screens (<768 px) so the
    # dashboard isn't covered by an open sidebar on phones; on desktop it
    # behaves as if "expanded" was set. Pre-2026-05-15 this was hardcoded
    # to "expanded" which made the dashboard unusable on mobile.
    initial_sidebar_state="auto",
)

# ── Mobile responsive overlay (quick-win shim) ───────────────────────────
# Streamlit's ``layout="wide"`` plus ``st.columns([...])`` produces fixed
# horizontal splits that don't collapse on narrow viewports — a 4-column
# metric row stays 4 columns at 375 px wide and becomes unreadable. This
# stylesheet targets Streamlit's stable ``data-testid`` selectors to:
#   1. Stack ``st.columns`` containers vertically below 768 px
#   2. Tighten main-content padding (mobile browsers eat margin already)
#   3. Reduce heading + metric font sizes for thumb-sized screens
#   4. Compact the tab strip so all tabs are reachable without scrolling
#
# This is a shim — not a real mobile UI. Data-heavy widgets (st.dataframe,
# the guardrail grid, the journal table) still scroll horizontally on
# phones because Streamlit doesn't expose a way to stack table columns.
# Tracked in task #104: build a dedicated mobile-essentials page that
# shows only the on-the-go monitoring fields (Start/Stop, day P&L, open
# positions count, last cycle status). Until then, this CSS makes the
# existing dashboard at-least-glanceable on a phone.
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        /* Stack st.columns rows vertically instead of side-by-side */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.5rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"],
        [data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        /* Reclaim the wide-layout side margins on a small screen */
        .block-container,
        [data-testid="stAppViewBlockContainer"] {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-top: 1rem !important;
        }
        /* Shrink headings — desktop sizes overflow on a 375 px viewport */
        h1 { font-size: 1.4rem !important; line-height: 1.25 !important; }
        h2 { font-size: 1.2rem !important; line-height: 1.25 !important; }
        h3 { font-size: 1.05rem !important; line-height: 1.25 !important; }
        /* Compact metric tiles so 3+ stacked tiles still fit a phone screen */
        [data-testid="stMetric"] {
            padding: 0.25rem 0.5rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
        /* Tighter tab strip — Streamlit's default tabs are >40 px wide each */
        [data-baseweb="tab"] {
            padding: 0.5rem 0.75rem !important;
            font-size: 0.85rem !important;
        }
        [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
        }
        /* When the sidebar IS opened on mobile, let it take full width
           so its content is actually readable (default is ~244 px which
           clips longer preset names). */
        [data-testid="stSidebar"] {
            min-width: 85vw !important;
            max-width: 85vw !important;
        }
        /* Market-status badge in the header — when the header columns
           stack on mobile the badge wrapper still right-aligns; pull it
           back to flex-start so it sits flush with the title above it. */
        [data-testid="stMarkdownContainer"] div[style*="justify-content:flex-end"] {
            justify-content: flex-start !important;
            padding-top: 0.25rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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

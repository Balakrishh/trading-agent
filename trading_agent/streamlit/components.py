"""
components.py — Reusable Plotly charts and Streamlit UI primitives.

All chart-building logic lives here so live_monitor, backtest_ui, and
llm_extension never import plotly directly.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REGIME_COLORS: Dict[str, str] = {
    "bullish": "#00c853",
    "bearish": "#d50000",
    "sideways": "#ff6d00",
    "mean_reversion": "#6200ea",
    "unknown": "#9e9e9e",
}

GUARDRAIL_NAMES: List[str] = [
    "Plan Validity",
    "Credit/Width Ratio",
    "Delta ≤ Max Delta",
    "Max Loss ≤ 2% Equity",
    "Paper Account",
    "Market Open",
    "Bid/Ask Spread",
    "Buying Power ≤ 80%",
]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of portfolio equity over time.

    Expected columns: timestamp (datetime-like), account_balance (float).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["account_balance"],
            mode="lines",
            name="Equity",
            line=dict(color="#1976d2", width=2),
            fill="tozeroy",
            fillcolor="rgba(25,118,210,0.08)",
            hovertemplate="$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Account Balance ($)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        plot_bgcolor="white",
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """
    Area chart showing rolling drawdown as a percentage.

    Expected columns: timestamp, account_balance.
    """
    equity = df["account_balance"]
    running_max = equity.cummax()
    # Use np.nan (not pd.NA) to keep dtype float64 — pd.NA coerces to
    # object, which breaks downstream numeric ops like .ewm().mean().
    drawdown_pct = (equity - running_max) / running_max.replace(0, np.nan) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=drawdown_pct,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#d32f2f", width=2),
            fill="tozeroy",
            fillcolor="rgba(211,47,47,0.12)",
            hovertemplate="%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Time",
        yaxis_title="Drawdown %",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        plot_bgcolor="white",
    )
    return fig


def regime_bar_chart(regime_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of total P&L grouped by regime.

    Expected columns: regime (str), pnl (float), trade_count (int).
    """
    colors = [REGIME_COLORS.get(str(r).lower(), "#9e9e9e") for r in regime_df["regime"]]
    fig = go.Figure(
        go.Bar(
            x=regime_df["pnl"],
            y=regime_df["regime"],
            orientation="h",
            marker_color=colors,
            text=[f"{n} trades" for n in regime_df["trade_count"]],
            textposition="outside",
            hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="P&L by Regime",
        xaxis_title="Total P&L ($)",
        yaxis_title="",
        height=280,
        margin=dict(l=0, r=60, t=40, b=0),
        plot_bgcolor="white",
        xaxis=dict(tickformat="$,.0f"),
    )
    return fig


# ---------------------------------------------------------------------------
# UI Primitives
# ---------------------------------------------------------------------------

def metric_row(
    equity: float,
    pnl: float,
    regime: str,
    cycle_secs: Optional[int],
) -> None:
    """Four-column metrics bar: equity · P&L · regime badge · cycle countdown.

    ``cycle_secs`` is ``None`` when the agent loop is stopped — we render an
    em-dash and a ``Stopped`` caption rather than a wall-clock countdown,
    because there is no real cycle to count down to. Passing the wall-clock
    value when the loop isn't running misleads the operator into believing
    the agent is about to act.
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Equity", f"${equity:,.2f}")

    delta_color = "normal" if pnl >= 0 else "inverse"
    c2.metric("Unrealized P&L", f"${pnl:+,.2f}", delta_color=delta_color)

    with c3:
        color = REGIME_COLORS.get(regime.lower(), "#9e9e9e")
        st.markdown(
            f"""<div style="text-align:center;padding-top:6px">
            <p style="margin:0;font-size:0.85em;color:#888">Dominant Regime</p>
            <span style="background:{color};color:#fff;padding:3px 14px;
            border-radius:12px;font-weight:600;font-size:1em">{regime.upper()}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    if cycle_secs is None:
        c4.metric("Next Cycle In", "—", delta="Stopped", delta_color="off")
    else:
        c4.metric("Next Cycle In", f"{cycle_secs}s", delta=" ", delta_color="off")


def guardrail_cards(guardrail_status: List[Dict]) -> None:
    """
    Two rows of 4 guardrail status cards.

    Each dict must have keys: name (str), passed (bool), detail (str).

    Optional key ``state`` (str): one of ``"ok"``, ``"warn"``, ``"fail"``.
    When present, overrides the green/red colour scheme derived from
    ``passed``.  ``"warn"`` renders an amber card with a ⚠️ icon — used
    for dry-run forced-open and other "passed but synthetic" states so
    the operator can see *why* the check passed.  When absent we fall
    back to the legacy two-state ``passed → ok|fail`` mapping.
    """
    state_styles = {
        "ok":   ("✅", "#e8f5e9", "#4caf50"),
        "warn": ("⚠️", "#fff8e1", "#ffb300"),
        "fail": ("❌", "#ffebee", "#ef5350"),
    }
    for row_start in (0, 4):
        cols = st.columns(4)
        for offset, col in enumerate(cols):
            idx = row_start + offset
            if idx >= len(guardrail_status):
                break
            g = guardrail_status[idx]
            state = g.get("state") or ("ok" if g["passed"] else "fail")
            icon, bg, border = state_styles.get(state, state_styles["ok"])
            col.markdown(
                f"""<div style="background:{bg};border-left:4px solid {border};
                padding:8px 10px;border-radius:4px;margin-bottom:6px;min-height:60px">
                <b>{icon} {g['name']}</b><br>
                <small style="color:#555">{g.get('detail', '')[:70]}</small>
                </div>""",
                unsafe_allow_html=True,
            )


def guardrail_grid(grid_rows: List[Dict]) -> None:
    """Per-ticker × per-guardrail status grid for the latest cycle.

    ``grid_rows`` comes from
    :func:`live_monitor._guardrail_grid_from_journal` — see its
    docstring for the row-shape contract.

    We render a hand-rolled HTML table because Streamlit's native
    ``st.dataframe`` cannot colour cells AND show hover tooltips for
    the per-check detail string.  The 8 guardrails become the columns
    so an operator can scan a single row to see which check failed for
    a given ticker, or scan a single column to see which tickers tripped
    the same gate this cycle.

    Hover any cell to reveal the verbatim check string from the
    journal.  Cells are colour-coded:

      ✅ green   — guardrail passed
      ⚠️ amber   — guardrail passed under a synthetic override
                   (currently only ``FORCED`` market-open in dry-run)
      ❌ red     — guardrail failed
    """
    if not grid_rows:
        st.info(
            "No guardrail data for the current mode yet — start a "
            "cycle to populate."
        )
        return

    # ``skipped`` is the fallback state for tickers with an open
    # position whose entry-trade history has been rotated out of the
    # journal — em-dash cells on grey, signalling "we hold something
    # but don't have its entry context any more". HOLDING rows (the
    # common case — entry trade is still on record) substitute the
    # entry trade's actual checks_passed/failed values, so they render
    # with the same green ``ok`` cells as a fresh APPROVED row.
    state_emoji   = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skipped": "—"}
    state_bg      = {"ok": "#e8f5e9", "warn": "#fff8e1",
                     "fail": "#ffebee", "skipped": "#f5f5f5"}
    state_text_fg = {"ok": "#1b5e20", "warn": "#8d6e00",
                     "fail": "#b71c1c", "skipped": "#888888"}

    cols = GUARDRAIL_NAMES
    th_style = (
        "text-align:center;padding:6px 6px;font-size:0.78em;"
        "color:#555;font-weight:600;background:#fafafa;"
        "border-bottom:1px solid #e0e0e0;white-space:nowrap"
    )
    th_first = (
        "text-align:left;padding:6px 8px;font-size:0.85em;"
        "color:#555;font-weight:600;background:#fafafa;"
        "border-bottom:1px solid #e0e0e0"
    )
    header = ["Ticker", "Cycle", "Approved"] + cols
    header_html = "".join(
        f"<th style='{th_first if i == 0 else th_style}' title='{name}'>{name}</th>"
        for i, name in enumerate(header)
    )

    def _esc(s: str) -> str:
        # Escape for use inside a title="..." attribute.
        return (
            s.replace("&", "&amp;")
             .replace("'", "&#39;")
             .replace('"', "&quot;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
        )

    body_parts: List[str] = []
    for row in grid_rows:
        ts = pd.Timestamp(row["timestamp"]).strftime("%H:%M:%S")
        # Five-state rendering for the verdict cell:
        #   approved → ✅ APPROVED   on green (risk manager ran, all checks passed)
        #   rejected → ❌ REJECTED   on red   (risk manager ran, ≥1 check failed)
        #   holding  → 🔒 HOLDING    on green (substituted entry-trade
        #              checks for a FILLED position — the broker is
        #              currently holding this spread)
        #   pending  → ⏳ PENDING    on amber (substituted entry-trade
        #              checks for a SUBMITTED but UNFILLED limit
        #              order — order is on the book waiting for a
        #              counterparty; not yet a real position)
        #   skipped  → ⏸ SKIPPED     on grey  (no entry trade in journal,
        #              cells render as em-dashes with the journal's
        #              skip reason as a hover tooltip)
        # The grid's ``status`` field is the source of truth; we keep
        # the legacy ``approved`` boolean for back-compat callers.
        status = row.get("status") or (
            "approved" if row.get("approved") else "rejected"
        )
        if status == "skipped":
            approved_emoji = "⏸"
            approved_bg    = state_bg["skipped"]
            approved_label = "SKIPPED"
            approved_fg    = state_text_fg["skipped"]
        elif status == "holding":
            # Same green palette as APPROVED — visually communicates
            # "this position cleared the risk manager when it was
            # opened AND is currently held at the broker". The 🔒
            # emoji + HOLDING label disambiguates from a freshly-
            # approved row.
            approved_emoji = "🔒"
            approved_bg    = state_bg["ok"]
            approved_label = "HOLDING"
            approved_fg    = state_text_fg["ok"]
        elif status == "pending":
            # Amber/warn palette — visually distinct from HOLDING's
            # green so the operator can scan the grid and tell at a
            # glance which positions are filled vs which are still
            # waiting. The cells themselves carry the same checks-
            # passed values as HOLDING (substituted from the entry
            # trade), so the per-guardrail context is identical.
            approved_emoji = "⏳"
            approved_bg    = state_bg["warn"]
            approved_label = "PENDING"
            approved_fg    = state_text_fg["warn"]
        elif status == "closed":
            # Position-monitor exit fired (added 2026-05-06).  Greyish-
            # purple distinguishes a closed-this-cycle row from
            # SKIPPED's "we did nothing" grey and APPROVED's "live
            # holding" green.  The verdict label says EXITED so the
            # operator immediately knows the spread is out — and the
            # journal row's ``raw_signal.exit_signal`` /
            # ``exit_reason`` carry the justification (rendered into
            # the row tooltip via ``skip_detail`` upstream).
            approved_emoji = "🚪"
            approved_bg    = "#ede7f6"     # very light purple
            approved_label = "EXITED"
            approved_fg    = "#4527a0"     # deep purple text
        elif status == "approved":
            approved_emoji = state_emoji["ok"]
            approved_bg    = state_bg["ok"]
            approved_label = "APPROVED"
            approved_fg    = state_text_fg["ok"]
        else:  # "rejected"
            approved_emoji = state_emoji["fail"]
            approved_bg    = state_bg["fail"]
            approved_label = "REJECTED"
            approved_fg    = state_text_fg["fail"]

        cells_html = [
            f"<td style='padding:6px 8px;font-weight:600;white-space:nowrap'>{row['ticker']}</td>",
            f"<td style='padding:6px 8px;color:#777;font-size:0.85em;white-space:nowrap'>{ts}</td>",
            (
                f"<td style='padding:6px 8px;text-align:center;background:{approved_bg};"
                f"color:{approved_fg};font-weight:600;font-size:0.78em;white-space:nowrap'>"
                f"{approved_emoji} {approved_label}</td>"
            ),
        ]
        for cell in row["cells"]:
            state    = cell["state"]
            emoji    = state_emoji.get(state, "·")
            bg       = state_bg.get(state, "#ffffff")
            fg       = state_text_fg.get(state, "#444")
            summary  = cell.get("summary", "") or ""
            chip     = f"{emoji} {summary}" if summary else emoji
            cells_html.append(
                f"<td style='padding:6px 8px;text-align:center;background:{bg};"
                f"color:{fg};font-size:0.78em;white-space:nowrap;cursor:help' "
                f"title='{_esc(cell['detail'])}'>{chip}</td>"
            )
        body_parts.append(f"<tr>{''.join(cells_html)}</tr>")

    table_html = (
        "<div style='overflow-x:auto;border:1px solid #e0e0e0;border-radius:4px'>"
        "<table style='border-collapse:collapse;width:100%;"
        "font-family:-apple-system,BlinkMacSystemFont,sans-serif'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_parts)}</tbody>"
        "</table>"
        "</div>"
        "<p style='font-size:0.75em;color:#888;margin-top:6px'>"
        "Numbers in each cell are the values that drove the verdict; "
        "hover for the verbatim check string from the risk manager. "
        "⚠️ amber = passed under a synthetic override (e.g. dry-run "
        "FORCED market-open)."
        "</p>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def positions_table(spreads: List[Dict]) -> None:
    """Styled positions table. Each dict is a serialised SpreadPosition.

    The ``Source`` column distinguishes plan-matched rows ("📋 plan") from
    inferred rows ("🔮 inferred") — the latter are spreads reconstructed
    from broker leg structure when no ``trade_plan_*.json`` matched.
    Both kinds carry live P&L and exit-signal info; only the recorded
    entry credit differs in fidelity.
    """
    if not spreads:
        st.info("No open positions.")
        return
    rows = []
    for s in spreads:
        credit = round(s.get("original_credit", 0) * 100, 2)
        pnl = round(s.get("net_unrealized_pl", 0), 2)
        pct = f"{pnl / credit * 100:.1f}%" if credit else "—"
        origin = s.get("origin", "trade_plan")
        source_label = "📋 plan" if origin == "trade_plan" else "🔮 inferred"
        rows.append(
            {
                "Symbol": s.get("underlying", ""),
                "Strategy": s.get("strategy_name", ""),
                "Credit ($)": credit,
                "Unreal. P&L ($)": pnl,
                "% Profit": pct,
                "Expiry": s.get("expiration", ""),
                "Exit Signal": s.get("exit_signal", "hold"),
                "Source": source_label,
            }
        )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def ungrouped_legs_table(legs: List[Dict]) -> None:
    """
    Render Alpaca option legs that are NOT matched to any local
    trade_plan_*.json file.

    These legs are typically positions opened outside the agent
    (manual entry via the Alpaca web UI, a different machine, or
    runs whose plan files were rotated/deleted). Showing them here
    keeps the dashboard a faithful mirror of the broker's view.
    """
    if not legs:
        return  # silent — nothing to show
    rows = []
    for L in legs:
        rows.append(
            {
                "Symbol":          L.get("symbol", ""),
                "Underlying":      L.get("underlying", ""),
                "Type":            L.get("type", ""),
                "Strike":          L.get("strike", ""),
                "Expiry":          L.get("expiration", ""),
                "Side":            L.get("side", ""),
                "Qty":             L.get("qty", 0),
                "Avg Entry ($)":   round(float(L.get("avg_entry_price", 0)), 2),
                "Current ($)":     round(float(L.get("current_price", 0)), 2),
                "Unreal. P&L ($)": round(float(L.get("unrealized_pl", 0)), 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def alert_box(message: str, level: str = "warning") -> None:
    """Render an alert at the given severity level (info/warning/error/success)."""
    getattr(st, level, st.warning)(message)


# ---------------------------------------------------------------------------
# Backtest closed-trades table
# ---------------------------------------------------------------------------

def closed_trades_table(closed_trades: List) -> None:
    """Render a sortable table of ``ClosedTrade`` records from a backtest run.

    Mirrors the live dashboard's ``positions_table`` styling so the operator
    can scan a backtest the same way they scan the live portfolio. Each
    ``ClosedTrade`` is the immutable record produced by
    ``trading_agent.backtest.sim_position.SimPosition.close()``.
    """
    if not closed_trades:
        st.info("No closed trades.")
        return

    rows = []
    for t in closed_trades:
        credit = round(float(t.credit_open) * 100.0 * int(t.qty), 2)
        pnl = round(float(t.realised_pnl or 0.0), 2)
        pct = f"{pnl / credit * 100:.1f}%" if credit else "—"
        rows.append(
            {
                "Ticker":         t.ticker,
                "Side":           t.side,
                "Entry":          getattr(t, "entry_t", None),
                "Exit":           getattr(t, "exit_t", None),
                "Days Held":      getattr(t, "days_held", None),
                "Short K":        round(float(t.short_strike), 2),
                "Long K":         round(float(t.long_strike), 2),
                "Width":          round(float(t.spread_width), 2),
                "Qty":             int(t.qty),
                "Credit ($)":     round(float(t.credit_open), 2),
                "Debit Close ($)": round(float(t.debit_close or 0.0), 2),
                "Realised P&L ($)": pnl,
                "% Profit":       pct,
                "σ Entry":        round(float(t.sigma_entry), 4),
                "σ Exit":         round(float(t.sigma_exit or 0.0), 4),
                "Exit Signal":    t.exit_signal,
                "Reason":         t.exit_reason,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

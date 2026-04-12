"""
backtest_ui.py — Backtesting tab.

Provides a Backtester class that simulates credit-spread P&L on
historical prices, then renders the results with metrics, charts,
a sortable trade log, and CSV/JSON/Journal export.
"""

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from trading_agent.streamlit.components import (
    drawdown_chart,
    equity_curve_chart,
    regime_bar_chart,
)

STARTING_EQUITY = 100_000.0
SPREAD_WIDTH = 5.0
COMMISSION_ROUND_TRIP = 2.60  # 4 legs × $0.65

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM"]
DEFAULT_START = date.today() - timedelta(days=365)   # needs 200+ bars for SMA warmup
DEFAULT_END = date.today() - timedelta(days=1)

# Yahoo Finance hard limits for intraday data
INTRADAY_MAX_DAYS = 29          # 5m data only available for last ~30 days
INTRADAY_WARMUP_BARS = 20       # 20 × 5-min bars ≈ 1.5 hours warmup for intraday SMA
INTRADAY_HOLD_BARS = 12         # 12 × 5-min bars = 1 hour hold per intraday trade

# OTM % for short strike placement
# Daily: 3% is realistic over a 45-day hold (SPY moves ~1% per day)
# Intraday: 3% in 60 minutes is nearly impossible → use 0.5% so losses actually occur
DAILY_OTM_PCT   = 0.03
INTRADAY_OTM_PCT = 0.005

ALL_TICKERS = [
    "SPY", "QQQ", "IWM", "GOOG", "AAPL",
    "MSFT", "AMZN", "META", "SOFI", "TSLA", "JPM",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    ticker: str
    strategy: str
    regime: str
    entry_date: date
    expiry_date: date
    credit: float
    max_loss: float
    outcome: str = ""   # "win" | "loss"
    pnl: float = 0.0
    hold_days: int = 0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict
    regime_stats: pd.DataFrame
    skipped: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Simulate credit-spread strategy on historical OHLCV data.

    Regime classification mirrors the real RegimeClassifier rules:
      price > SMA-200 AND slope > 0  → bullish  → Bull Put Spread
      price < SMA-200 AND slope < 0  → bearish  → Bear Call Spread
      otherwise                      → sideways → Iron Condor

    P&L simulation:
      - Credit = spread_width × credit_pct  (default 30 % of width)
      - Win if short strike is never breached within hold period
      - Short strike placed 3 % OTM for puts/calls, ±3 % band for condors
      - Win: capture profit_target_pct of credit minus commission
      - Loss: pay max_loss (width − credit) minus commission
    """

    def __init__(
        self,
        starting_equity: float = STARTING_EQUITY,
        spread_width: float = SPREAD_WIDTH,
        credit_pct: float = 0.30,
        target_dte: int = 45,
        profit_target_pct: float = 0.50,
        commission: float = COMMISSION_ROUND_TRIP,
    ) -> None:
        self.starting_equity = starting_equity
        self.spread_width = spread_width
        self.credit_pct = credit_pct
        self.target_dte = target_dte
        self.profit_target_pct = profit_target_pct
        self.commission = commission

    # ── Regime helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _classify(prices: pd.Series, idx: int) -> str:
        """Legacy daily-bar classifier (200-bar window). Kept for test compatibility."""
        return Backtester._classify_bars(prices, idx, warmup=200)

    @staticmethod
    def _classify_bars(prices: pd.Series, idx: int, warmup: int = 200) -> str:
        """
        Generalized regime classifier.

        Uses `warmup` bars as the long SMA window and warmup//4 as the short
        SMA window. Works for both daily bars (warmup=200) and intraday bars
        (warmup=20, short window=5).
        """
        if idx < warmup:
            return "sideways"
        short_window = max(5, warmup // 4)
        window = prices.iloc[max(0, idx - warmup): idx + 1]
        sma_short = window.iloc[-min(short_window, len(window)):].mean()
        sma_long = window.mean()
        lookback = min(10, len(window))
        slope = (
            window.iloc[-lookback // 2:].mean() - window.iloc[-lookback: -lookback // 2].mean()
            if lookback >= 4
            else 0.0
        )
        price = window.iloc[-1]
        if price > sma_long and slope > 0:
            return "bullish"
        if price < sma_long and slope < 0:
            return "bearish"
        return "sideways"

    @staticmethod
    def _strategy(regime: str) -> str:
        return {
            "bullish": "Bull Put Spread",
            "bearish": "Bear Call Spread",
            "sideways": "Iron Condor",
        }.get(regime, "Iron Condor")

    # ── Outcome simulation ──────────────────────────────────────────────────

    def _simulate(
        self, prices: pd.Series, entry_idx: int, regime: str, credit: float,
        hold_bars: int = None, otm_pct: float = DAILY_OTM_PCT,
    ) -> tuple:
        if hold_bars is None:
            hold_bars = self.target_dte
        end_idx = min(entry_idx + hold_bars, len(prices) - 1)
        fwd = prices.iloc[entry_idx: end_idx + 1]
        entry_p = prices.iloc[entry_idx]

        lower = entry_p * (1 - otm_pct)
        upper = entry_p * (1 + otm_pct)

        if regime == "bullish":
            breached = (fwd < lower).any()
        elif regime == "bearish":
            breached = (fwd > upper).any()
        else:  # sideways / iron condor
            breached = ((fwd < lower) | (fwd > upper)).any()

        hold_count = end_idx - entry_idx
        if breached:
            pnl = -(self.spread_width * 100 - credit * 100) - self.commission
            return "loss", round(pnl, 2), hold_count
        pnl = credit * 100 * self.profit_target_pct - self.commission
        return "win", round(pnl, 2), hold_count

    # ── Main run ────────────────────────────────────────────────────────────

    def run(
        self,
        tickers: List[str],
        start: date,
        end: date,
        timeframe: str = "1Day",
        use_alpaca: bool = False,
    ) -> BacktestResult:
        """
        Execute the backtest and return a BacktestResult.

        Parameters
        ----------
        tickers     : list of ticker symbols
        start / end : date range
        timeframe   : "1Day" or "5Min"
        use_alpaca  : reserved for future Alpaca data source (currently no-op)

        Notes on 5Min timeframe
        -----------------------
        Yahoo Finance only serves 5-minute data for the last ~30 days.
        Any start date older than that returns an empty DataFrame.
        When timeframe="5Min", start is automatically clamped to today-29 days.
        The warmup period is reduced from 200 daily bars to 20 intraday bars
        (≈ 1.5 hours) and the hold period from 45 bars to 12 bars (≈ 1 hour).
        """
        is_intraday = timeframe == "5Min"
        yf_interval = "1d" if not is_intraday else "5m"

        # ── Yahoo Finance 5m hard limit: clamp start to last 29 days ──────
        warnings: List[str] = []
        if is_intraday:
            earliest_allowed = date.today() - timedelta(days=INTRADAY_MAX_DAYS)
            if start < earliest_allowed:
                warnings.append(
                    f"5-minute data is only available for the last {INTRADAY_MAX_DAYS} days "
                    f"(Yahoo Finance limitation). Start date clamped from {start} "
                    f"to {earliest_allowed}."
                )
                start = earliest_allowed

        warmup_bars = INTRADAY_WARMUP_BARS if is_intraday else 200
        hold_bars = INTRADAY_HOLD_BARS if is_intraday else self.target_dte

        all_trades: List[SimTrade] = []
        equity = self.starting_equity
        equity_curve: List[Dict] = [{"timestamp": pd.Timestamp(start), "account_balance": equity}]

        skipped: List[str] = warnings.copy()

        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=True,
                )
                if raw.empty:
                    skipped.append(
                        f"{ticker} (no data returned — "
                        + ("Yahoo only serves 5m data for the last 30 days; "
                           "try a more recent date range" if is_intraday
                           else "check ticker symbol or try a wider date range")
                        + ")"
                    )
                    continue
                # Handle multi-level columns returned by recent yfinance versions
                if isinstance(raw.columns, pd.MultiIndex):
                    raw = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) else raw
                    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                prices: pd.Series = raw["Close"].dropna()
            except Exception as exc:
                skipped.append(f"{ticker} (download error: {exc})")
                continue

            min_bars = warmup_bars + hold_bars + 1
            if len(prices) < min_bars:
                unit = "intraday bars" if is_intraday else "daily bars"
                skipped.append(
                    f"{ticker} (only {len(prices)} {unit} downloaded — "
                    f"need {min_bars}+ for {warmup_bars}-bar warmup; "
                    + ("Yahoo Finance only provides ~1560 bars of 5m data over 30 days"
                       if is_intraday
                       else "extend your date range to at least 1 year")
                    + ")"
                )
                continue

            last_entry_idx = -hold_bars
            for i in range(warmup_bars, len(prices)):
                if i - last_entry_idx < hold_bars:
                    continue
                if equity <= 0:
                    break

                regime = self._classify_bars(prices, i, warmup_bars)
                strategy = self._strategy(regime)
                credit = self.spread_width * self.credit_pct
                max_loss = self.spread_width - credit

                otm_pct = INTRADAY_OTM_PCT if is_intraday else DAILY_OTM_PCT
                outcome, pnl, hold_count = self._simulate(prices, i, regime, credit, hold_bars, otm_pct)
                equity = round(equity + pnl, 2)
                last_entry_idx = i

                raw_date = prices.index[i]
                entry_date = raw_date.date() if hasattr(raw_date, "date") else start

                # For intraday, hold_days represents bars (each = 5 min), not calendar days
                all_trades.append(
                    SimTrade(
                        ticker=ticker,
                        strategy=strategy,
                        regime=regime,
                        entry_date=entry_date,
                        expiry_date=entry_date + timedelta(days=1 if is_intraday else hold_count),
                        credit=credit,
                        max_loss=max_loss,
                        outcome=outcome,
                        pnl=pnl,
                        hold_days=hold_count,
                    )
                )
                equity_curve.append(
                    {"timestamp": pd.Timestamp(raw_date), "account_balance": equity}
                )

        if not all_trades:
            empty_trades = pd.DataFrame(
                columns=["ticker", "strategy", "regime", "entry_date",
                         "expiry_date", "credit", "max_loss", "outcome", "pnl", "hold_days"]
            )
            empty_eq = pd.DataFrame(
                [{"timestamp": pd.Timestamp(start), "account_balance": self.starting_equity}]
            )
            return BacktestResult(
                trades=empty_trades,
                equity_curve=empty_eq,
                metrics=self._metrics([], self.starting_equity),
                regime_stats=pd.DataFrame(columns=["regime", "pnl", "trade_count"]),
                skipped=skipped,
            )

        trades_df = pd.DataFrame([vars(t) for t in all_trades])
        equity_df = pd.DataFrame(equity_curve)

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_df,
            metrics=self._metrics(all_trades, self.starting_equity),
            regime_stats=self._regime_stats(trades_df),
            skipped=skipped,
        )

    # ── Stats helpers ───────────────────────────────────────────────────────

    def _metrics(self, trades: List[SimTrade], starting: float) -> Dict:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
                "avg_hold_days": 0.0,
            }
        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        pnl_arr = np.array([t.pnl for t in trades])
        sharpe = (
            float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252))
            if pnl_arr.std() > 0
            else 0.0
        )

        # Max drawdown from trade-by-trade equity
        eq = starting
        peak = starting
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x.entry_date):
            eq += t.pnl
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return {
            "total_trades": len(trades),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "avg_hold_days": round(sum(t.hold_days for t in trades) / len(trades), 1),
        }

    @staticmethod
    def _regime_stats(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["regime", "pnl", "trade_count"])
        return (
            df.groupby("regime")
            .agg(pnl=("pnl", "sum"), trade_count=("pnl", "count"))
            .reset_index()
        )


# ---------------------------------------------------------------------------
# Cached runner
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running backtest…")
def _run_cached(
    tickers: tuple,
    start: date,
    end: date,
    timeframe: str,
    use_alpaca: bool,
) -> BacktestResult:
    return Backtester().run(list(tickers), start, end, timeframe, use_alpaca)


# ---------------------------------------------------------------------------
# Journal export helper
# ---------------------------------------------------------------------------

def _export_to_journal(result: BacktestResult) -> None:
    try:
        from trading_agent.journal_kb import JournalKB
        journal = JournalKB(journal_dir="trade_journal")
        m = result.metrics
        journal.log_signal(
            ticker="BACKTEST",
            action="dry_run",
            price=0.0,
            raw_signal={
                "regime": "backtest",
                "strategy": "Backtest Export",
                "plan_valid": True,
                "risk_approved": False,
                "account_balance": STARTING_EQUITY,
                "checks_passed": [],
                "checks_failed": [],
                "backtest_metrics": m,
                "backtest_trade_count": len(result.trades),
            },
            notes=(
                f"Backtest: {m['total_trades']} trades, "
                f"WR={m['win_rate']}%, Sharpe={m['sharpe']:.2f}"
            ),
        )
        st.success("Backtest summary exported to trade journal.")
    except Exception as exc:
        st.error(f"Journal export failed: {exc}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_backtest_ui() -> None:
    """Render the Backtesting tab."""

    # ── Two-column layout: settings panel (left) + results (right) ────────
    # All controls are in the left column, never in st.sidebar, so they only
    # appear when the Backtesting tab is active.
    settings_col, results_col = st.columns([1, 3], gap="large")

    with settings_col:
        st.subheader("Settings")

        timeframe = st.selectbox(
            "Timeframe", options=["1Day", "5Min"], index=0, key="bt_timeframe"
        )
        is_intraday = timeframe == "5Min"

        if is_intraday:
            st.info(
                f"Yahoo Finance only serves 5m data for the last "
                f"{INTRADAY_MAX_DAYS} days. Start date is auto-clamped.",
                icon="⚠️",
            )

        # Reset defaults when timeframe changes
        prev_tf = st.session_state.get("_bt_prev_timeframe")
        if prev_tf != timeframe:
            st.session_state["_bt_prev_timeframe"] = timeframe
            st.session_state.pop("backtest_result", None)
            st.session_state.pop("backtest_timeframe", None)
            st.session_state["bt_start_date"] = (
                date.today() - timedelta(days=INTRADAY_MAX_DAYS)
                if is_intraday else DEFAULT_START
            )

        default_start = (
            date.today() - timedelta(days=INTRADAY_MAX_DAYS)
            if is_intraday else DEFAULT_START
        )
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.get("bt_start_date", default_start),
            key="bt_start_date",
        )
        end_date = st.date_input("End Date", value=DEFAULT_END, key="bt_end_date")
        tickers = st.multiselect(
            "Tickers", options=ALL_TICKERS, default=DEFAULT_TICKERS, key="bt_tickers"
        )
        use_alpaca = st.toggle(
            "Use Alpaca Data",
            value=False,
            key="bt_use_alpaca",
            help="Prefer Alpaca historical bars over yfinance (requires live API key).",
        )
        st.divider()
        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

    # ── Results rendered in the right column ───────────────────────────────
    with results_col:
        st.subheader("Results")

        if not run_btn and "backtest_result" not in st.session_state:
            if is_intraday:
                st.info(
                    "**5-Min timeframe selected.**\n\n"
                    f"Yahoo Finance only provides 5-minute bars for the last {INTRADAY_MAX_DAYS} days. "
                    "Start date has been set automatically. Click **Run Backtest** to continue.\n\n"
                    "Use **1Day** timeframe for multi-year historical analysis."
                )
            else:
                st.info("Configure parameters on the left and click **Run Backtest**.")
            return

        if run_btn:
            if not tickers:
                st.error("Select at least one ticker.")
                return
            if start_date >= end_date:
                st.error("Start date must be before end date.")
                return
            result = _run_cached(
                tuple(sorted(tickers)), start_date, end_date, timeframe, use_alpaca
            )
            st.session_state["backtest_result"] = result
            st.session_state["backtest_timeframe"] = timeframe

        result: Optional[BacktestResult] = st.session_state.get("backtest_result")
        if result is None:
            return

        result_timeframe = st.session_state.get("backtest_timeframe", timeframe)
        is_result_intraday = result_timeframe == "5Min"

        m = result.metrics

        # ── Skipped / clamped warnings ─────────────────────────────────────
        if result.skipped:
            for msg in result.skipped:
                st.warning(msg)

        if result.trades.empty:
            if is_result_intraday:
                st.error(
                    "No trades were simulated for 5-Min timeframe. Possible causes:\n\n"
                    "- **Yahoo Finance 30-day limit** — 5-minute data is only available "
                    f"for the last {INTRADAY_MAX_DAYS} days. Dates older than that return empty data.\n"
                    "- **Not enough bars** — the intraday backtester needs at least "
                    f"{INTRADAY_WARMUP_BARS + INTRADAY_HOLD_BARS + 1} bars "
                    f"({INTRADAY_WARMUP_BARS}-bar warmup + {INTRADAY_HOLD_BARS}-bar hold).\n"
                    "- **Try 1Day timeframe** for multi-year historical analysis."
                )
            else:
                st.error(
                    "No trades were simulated. The most common cause is a date range that is "
                    "too short — the backtester needs at least **201 daily bars** (≈ 1 year) "
                    "to compute the SMA-200 warmup before it can place the first trade. "
                    "Try setting Start Date to at least 1 year ago."
                )
            return

        # ── Summary metric cards ───────────────────────────────────────────
        hold_label = "Avg Hold (bars)" if is_result_intraday else "Avg Hold (days)"
        pf = m["profit_factor"]
        pf_display = "∞" if pf == float("inf") else f"{pf:.2f}"
        metric_cols = st.columns(6)
        for col, (label, value) in zip(
            metric_cols,
            [
                ("Trades", m["total_trades"]),
                ("Win Rate", f"{m['win_rate']:.1f}%"),
                ("Profit Factor", pf_display),
                ("Max DD", f"{m['max_drawdown_pct']:.1f}%"),
                ("Sharpe", f"{m['sharpe']:.2f}"),
                (hold_label, m["avg_hold_days"]),
            ],
        ):
            col.metric(label, value)

        st.divider()

        # ── Per-regime table + bar chart ───────────────────────────────────
        if not result.regime_stats.empty:
            left, right = st.columns([2, 3])
            with left:
                st.subheader("Results by Regime")
                st.dataframe(
                    result.regime_stats.assign(
                        pnl=result.regime_stats["pnl"].map("${:,.2f}".format)
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            with right:
                st.plotly_chart(regime_bar_chart(result.regime_stats), use_container_width=True)
            st.divider()

        # ── Equity + drawdown charts ───────────────────────────────────────
        if not result.equity_curve.empty:
            st.plotly_chart(equity_curve_chart(result.equity_curve), use_container_width=True)
            st.plotly_chart(drawdown_chart(result.equity_curve), use_container_width=True)
            st.divider()

        # ── Sortable trade log ─────────────────────────────────────────────
        st.subheader("Trade Log")
        if not result.trades.empty:
            sort_col = st.selectbox(
                "Sort by", options=["entry_date", "pnl", "hold_days", "ticker"], index=0,
                key="bt_sort_col",
            )
            ascending = st.checkbox("Ascending", value=False, key="bt_ascending")
            st.dataframe(
                result.trades.sort_values(sort_col, ascending=ascending),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ── Export row ─────────────────────────────────────────────────────
        col_csv, col_json, col_journal = st.columns(3)

        with col_csv:
            st.download_button(
                "Export CSV",
                data=result.trades.to_csv(index=False).encode(),
                file_name="backtest_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_json:
            st.download_button(
                "Export JSON",
                data=result.trades.to_json(orient="records", date_format="iso", indent=2).encode(),
                file_name="backtest_trades.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_journal:
            if st.button("Export to Journal", use_container_width=True):
                _export_to_journal(result)

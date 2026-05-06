"""
runner.py — drive the clock, dispatch cycles, return a BacktestResult.

The runner is intentionally the *only* module in this package that
mutates state across cycles. Every other module is either pure
(``black_scholes``, ``synthetic_chain``) or owns a single bounded
slice of state (``HistoricalPort`` owns its cursor, ``SimAccount``
owns its ledger, ``SimPosition`` owns its mark). The runner stitches
them together via the clock event stream.

Cadence selection — the Option E choice
---------------------------------------
yfinance's 5-minute interval covers ≤ 30 calendar days back from now
(see ``trading_agent/market_data.py:62-72``). For backtest windows
that fit, we run *intraday-decision* cadence — fire the live
PERCEIVE → CLASSIFY → PLAN → RISK → EXECUTE pipeline every 5 minutes
inside RTH, exactly like the live agent. For older windows we
collapse to *daily-decision* cadence — run the pipeline once per day
at the close. Marks are *always* daily (close-of-day) because that's
the granularity ^VIX is published at, and finer marks would just
interpolate the same number.

This keeps the *most* parity (per-bar entry decisions) for windows
the user is most likely to look at (last 30 days), while still
allowing useful long-horizon studies (last 6 months, last year) at
daily granularity.

Output shape
------------
``BacktestResult`` carries everything the UI needs to render the new
backtest tab:

  * ``equity_curve``  — list[EquityPoint] from SimAccount
  * ``closed_trades`` — list[ClosedTrade] from SimPosition.close
  * ``cycle_outcomes`` — list[CycleOutcome] for every PLAN-stage event
                        (the journal exporter writes one JSONL row per)
  * ``starting_equity`` / ``ending_equity`` — for the headline metrics
  * ``run_t0`` / ``run_t1`` — wall-clock bracket of the run itself
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Dict, List, Optional, Sequence

from trading_agent.backtest.account import EquityPoint, SimAccount
from trading_agent.backtest.clock import (
    INTRADAY_BAR_MINUTES,
    iter_events,
    trading_days_in_range,
)
from trading_agent.backtest.cycle import CycleOutcome, run_one_cycle
from trading_agent.backtest.historical_port import HistoricalPort
from trading_agent.backtest.sim_position import ClosedTrade, SimPosition
from trading_agent.position_monitor import ExitSignal
from trading_agent.regime import Regime, RegimeClassifier
from trading_agent.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Yahoo's 5m interval is documented to cover ≤30 calendar days. We keep
# 1-day headroom so a backtest started near the boundary doesn't
# silently drop the first hour of bars. Above this threshold the
# runner switches to daily cadence.
INTRADAY_LOOKBACK_LIMIT_DAYS = 29


@dataclass
class BacktestResult:
    """
    Complete output of one ``BacktestRunner.run()`` call. The Streamlit
    UI consumes this directly — see ``backtest_ui.py`` (rewritten in
    PR 3) for the renderer.
    """
    starting_equity:  float
    ending_equity:    float
    equity_curve:     List[EquityPoint] = field(default_factory=list)
    closed_trades:    List[ClosedTrade] = field(default_factory=list)
    cycle_outcomes:   List[CycleOutcome] = field(default_factory=list)
    open_positions:   List[SimPosition] = field(default_factory=list)
    run_t0:           Optional[datetime] = None
    run_t1:           Optional[datetime] = None
    cadence:          str = "intraday"  # "intraday" | "daily"
    tickers:          Sequence[str] = ()

    @property
    def total_return_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.ending_equity - self.starting_equity) / self.starting_equity * 100.0

    @property
    def trade_count(self) -> int:
        return len(self.closed_trades)

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.closed_trades if t.realised_pnl > 0)

    @property
    def win_rate_pct(self) -> float:
        if not self.closed_trades:
            return 0.0
        return self.win_count / len(self.closed_trades) * 100.0

    @property
    def realised_pnl(self) -> float:
        return round(sum(t.realised_pnl for t in self.closed_trades), 2)


class BacktestRunner:
    """
    Glue together the historical port, account, classifier, risk
    manager, and clock.

    Parameters
    ----------
    tickers
        Universe to walk. Each ticker is processed independently per
        cycle (matching the live agent's per-ticker block).
    start, end
        Inclusive backtest window in calendar dates.
    preset
        Live ``PresetConfig`` — drives both the synthetic chain shape
        (DTE/Δ/width grids) and the planner's edge_buffer / min_pop.
        **Pin ``scan_mode="adaptive"``** so the backtester always runs
        through ``decide()`` — required by CI invariant #3.
    starting_equity
        SimAccount seed cash. Default 100k matches the legacy
        backtester so equity-curve magnitudes are comparable.
    risk_manager
        Optional override. If None, constructs a default
        ``RiskManager`` whose tunables mirror ``preset.max_risk_pct``
        and ``preset.edge_buffer`` so the planning-time and
        validation-time floors match.
    port
        Optional override for the historical port — useful in tests
        that want to inject a stubbed yfinance loader.
    """

    def __init__(self, *, tickers: Sequence[str], start: date, end: date,
                 preset, starting_equity: float = 100_000.0,
                 risk_manager: Optional[RiskManager] = None,
                 port: Optional[HistoricalPort] = None,
                 progress_callback: Optional[Callable[[float, str], None]] = None):
        self.tickers = tuple(tickers)
        self.start = start
        self.end = end
        self.preset = preset
        self.starting_equity = float(starting_equity)
        self.port = port or HistoricalPort()
        # Default RM mirrors the preset so adaptive-floor parity holds.
        self.risk_manager = risk_manager or RiskManager(
            max_risk_pct=getattr(preset, "max_risk_pct", 0.02),
            min_credit_ratio=getattr(preset, "min_credit_ratio", 0.33),
            max_delta=getattr(preset, "max_delta", 0.20),
            delta_aware_floor=(getattr(preset, "scan_mode", "static") == "adaptive"),
            edge_buffer=getattr(preset, "edge_buffer", 0.10),
        )
        self.progress_callback = progress_callback
        self._open_positions: Dict[str, List[SimPosition]] = {
            t: [] for t in self.tickers
        }
        # The classifier needs *some* MarketDataProvider to satisfy its
        # __init__ signature, but our ``cycle._classify_from_history``
        # only invokes static methods on it — never the instance methods
        # that touch Alpaca. We pass None to fail loudly if anyone tries.
        self._classifier = RegimeClassifier(data_provider=None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Cadence selection
    # ------------------------------------------------------------------

    def _pick_cadence(self, today: date) -> str:
        """
        intraday if the *entire* requested range lies within
        ``INTRADAY_LOOKBACK_LIMIT_DAYS`` of today; daily otherwise.

        We use the *start* of the range as the test, not the end —
        even a 5-day backtest starting 60 days back has no 5m data.
        """
        lookback = (today - self.start).days
        return "intraday" if lookback <= INTRADAY_LOOKBACK_LIMIT_DAYS else "daily"

    # ------------------------------------------------------------------
    # Per-event handlers
    # ------------------------------------------------------------------

    def _handle_intraday_decision(self, t: datetime,
                                  account: SimAccount,
                                  outcomes: List[CycleOutcome]) -> None:
        """One PLAN/RISK/EXECUTE pass per ticker."""
        self.port.set_cursor(t)
        for ticker in self.tickers:
            try:
                outcome = run_one_cycle(
                    ticker=ticker, t=t, port=self.port,
                    preset=self.preset, account=account,
                    classifier=self._classifier,
                    risk_manager=self.risk_manager,
                )
            except Exception as exc:
                logger.exception("[%s] cycle crashed at %s: %s", ticker, t, exc)
                outcome = CycleOutcome(ticker=ticker, t=t, spot=0.0,
                                       regime=Regime.SIDEWAYS,
                                       status="error", reason=str(exc)[:200])
            outcomes.append(outcome)
            if outcome.status == "opened" and outcome.new_position is not None:
                self._open_positions[ticker].append(outcome.new_position)

    def _spot_at(self, ticker: str, t: datetime) -> float:
        """
        Cursor-bounded spot lookup for re-marks. Daily close from the
        already-cached daily frame; falls back to 0 (caller skips mark).
        """
        df = self.port.fetch_underlying_daily(ticker)
        if df is None or df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])

    def _handle_daily_mark(self, t: datetime, account: SimAccount,
                           closed_trades: List[ClosedTrade]) -> None:
        """
        Re-mark every open position, then evaluate exit signals and
        close out anything that fires (HOLD signals are no-ops).
        """
        self.port.set_cursor(t)
        vix_t = self.port.vix_at(t)
        # We need a current Regime for the regime-shift exit check.
        # Reuse the per-ticker classification we just ran — but on a
        # daily-only run the regime has been computed only at the
        # daily_mark cadence, so re-classify here.
        regimes_now: Dict[str, Regime] = {}
        for ticker in self.tickers:
            df = self.port.fetch_underlying_daily(ticker)
            if df is None or df.empty or len(df) < 200:
                continue
            try:
                regime, _, _ = _classify_passthrough(ticker, df, self._classifier)
                regimes_now[ticker] = regime
            except Exception:
                continue

        total_open_mv = 0.0
        for ticker in self.tickers:
            spot = self._spot_at(ticker, t)
            if spot <= 0:
                continue
            current_regime = regimes_now.get(ticker)
            still_open: List[SimPosition] = []
            for pos in self._open_positions[ticker]:
                if pos.closed:
                    continue
                pos.remark(t=t, spot=spot, vix_t=vix_t)
                signal, reason = pos.evaluate_exit(
                    t=t, spot=spot, current_regime=current_regime,
                    profit_target_pct=0.50,
                    hard_stop_multiplier=3.0,
                    strike_proximity_pct=0.01,
                )
                # Force-close at expiration even on HOLD — the spread
                # has settled and continuing to mark it makes no sense.
                if signal == ExitSignal.HOLD and t.date() >= pos.expiration:
                    signal, reason = (ExitSignal.EXPIRED,
                                      f"Expiration {pos.expiration} reached")
                if signal == ExitSignal.HOLD:
                    still_open.append(pos)
                    total_open_mv += pos.current_mark_dollars
                else:
                    closed = pos.close(t=t, exit_signal=signal, reason=reason)
                    account.apply_close(
                        credit_per_share=pos.credit_open,
                        qty=pos.qty,
                        closing_debit_per_share=pos.current_mark,
                    )
                    closed_trades.append(closed)
            self._open_positions[ticker] = still_open
        account.apply_mark(total_open_market_value=total_open_mv)
        account.snapshot(t)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        Walk the clock from ``start`` to ``end`` and return a
        ``BacktestResult``. Idempotent — calling twice rebuilds from
        scratch (no state retained on the runner across runs).
        """
        run_t0 = datetime.utcnow()
        cadence = self._pick_cadence(date.today())
        intraday = (cadence == "intraday")

        account = SimAccount.fresh(self.starting_equity)
        outcomes: List[CycleOutcome] = []
        closed_trades: List[ClosedTrade] = []

        days = trading_days_in_range(self.start, self.end)
        if not days:
            return BacktestResult(
                starting_equity=self.starting_equity,
                ending_equity=self.starting_equity,
                run_t0=run_t0, run_t1=datetime.utcnow(),
                cadence=cadence, tickers=self.tickers,
            )

        for ev in iter_events(self.start, self.end, intraday=intraday):
            if ev.kind == "day_open":
                if self.progress_callback:
                    pct = (days.index(ev.timestamp.date()) + 1) / len(days)
                    self.progress_callback(pct, f"Replaying {ev.timestamp.date()}")
            elif ev.kind == "intraday_decision":
                self._handle_intraday_decision(ev.timestamp, account, outcomes)
            elif ev.kind == "daily_mark":
                # On daily cadence (no intraday decisions), run the
                # decision cycle once per day at the close mark before
                # we re-mark / exit.
                if not intraday:
                    self._handle_intraday_decision(ev.timestamp, account, outcomes)
                self._handle_daily_mark(ev.timestamp, account, closed_trades)
            # day_close currently a no-op marker — kept on the event
            # stream so future per-day journal flushes have a hook.

        # Force-close any positions still open at end of run so the
        # equity curve reflects realised cash, not phantom unrealised.
        # We don't synthesize a fake exit signal — just mark them as
        # EXPIRED (the most honest label for "ran out of backtest").
        end_t = datetime.combine(self.end, datetime.min.time())
        end_t = end_t.replace(hour=16)
        leftover_open: List[SimPosition] = []
        for ticker in self.tickers:
            for pos in self._open_positions[ticker]:
                if pos.closed:
                    continue
                spot = self._spot_at(ticker, end_t)
                if spot <= 0:
                    leftover_open.append(pos)
                    continue
                pos.remark(t=end_t, spot=spot, vix_t=self.port.vix_at(end_t))
                closed = pos.close(t=end_t, exit_signal=ExitSignal.EXPIRED,
                                   reason="Backtest window ended")
                account.apply_close(
                    credit_per_share=pos.credit_open, qty=pos.qty,
                    closing_debit_per_share=pos.current_mark,
                )
                closed_trades.append(closed)
        account.apply_mark(total_open_market_value=0.0)
        account.snapshot(end_t)

        return BacktestResult(
            starting_equity=self.starting_equity,
            ending_equity=account.equity,
            equity_curve=list(account.equity_curve),
            closed_trades=closed_trades,
            cycle_outcomes=outcomes,
            open_positions=leftover_open,
            run_t0=run_t0,
            run_t1=datetime.utcnow(),
            cadence=cadence,
            tickers=self.tickers,
        )


# Imported lazily inside _handle_daily_mark to avoid a top-level cycle.
def _classify_passthrough(ticker, df, classifier):
    from trading_agent.backtest.cycle import _classify_from_history
    return _classify_from_history(ticker, df, classifier)


__all__ = [
    "BacktestRunner",
    "BacktestResult",
    "INTRADAY_LOOKBACK_LIMIT_DAYS",
]

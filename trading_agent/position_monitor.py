"""
Position Monitor
=================
Fetches open positions from Alpaca, computes unrealized P&L against
the original trade plan, and generates exit signals based on:

  1. HARD_STOP:        Spread value ≥ 3× initial credit (immediate, no debounce)
  2. PROFIT_TARGET:    50% of max credit captured
  3. STRIKE_PROXIMITY: Underlying within 1% of any short strike (immediate)
  4. DTE_SAFETY:       Last trading day before expiry after 15:30 ET (immediate)
  5. REGIME_SHIFT:     Current regime contradicts the position's strategy

Signals marked "immediate" bypass the 3-cycle debounce in agent.py and
trigger a market-order close without waiting for confirmation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import requests

from trading_agent.calendar_utils import is_last_trading_day_before
from trading_agent.market_data import ALPACA_TIMEOUT
from trading_agent.regime import Regime

logger = logging.getLogger(__name__)


class ExitSignal(Enum):
    HOLD = "hold"
    STOP_LOSS = "stop_loss"          # legacy alias kept for compatibility
    HARD_STOP = "hard_stop"          # spread value ≥ 3× credit (immediate)
    PROFIT_TARGET = "profit_target"  # 50% of credit captured
    REGIME_SHIFT = "regime_shift"    # regime no longer matches strategy
    STRIKE_PROXIMITY = "strike_proximity"  # underlying within 1% of short strike
    DTE_SAFETY = "dte_safety"        # Thursday before expiry ≥ 15:30 ET
    EXPIRED = "expired"


# Signals that bypass the 3-cycle debounce — close immediately
IMMEDIATE_EXIT_SIGNALS = {
    ExitSignal.HARD_STOP,
    ExitSignal.STRIKE_PROXIMITY,
    ExitSignal.DTE_SAFETY,
}

# Which regime each strategy is compatible with
STRATEGY_REGIME_MAP = {
    "Bull Put Spread":      Regime.BULLISH,
    "Bear Call Spread":     Regime.BEARISH,
    "Iron Condor":          Regime.SIDEWAYS,
    "Mean Reversion Spread": None,   # direction-neutral; never regime-shift closed
}


@dataclass
class PositionSnapshot:
    """A single option leg position from Alpaca."""
    symbol: str
    qty: int
    side: str               # "long" or "short"
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    asset_class: str         # "us_option" for options


@dataclass
class SpreadPosition:
    """
    Aggregated view of a credit spread (2 or 4 legs) linked back to
    the original trade plan.

    Origin
    ------
    ``origin == "trade_plan"`` (default) — every field was sourced from
    the matching ``trade_plan_*.json``. ``original_credit``, ``max_loss``,
    ``spread_width`` are the recorded entry economics.

    ``origin == "inferred"`` — the broker has these legs but no
    ``trade_plan_*.json`` entry matches their symbols (history was
    rotated out, or the position was opened manually outside the agent).
    The ``strategy_name`` was reconstructed from the leg structure;
    ``original_credit`` falls back to the ``cost_basis`` derived from
    each leg's ``avg_entry_price`` (still useful for monitoring),
    ``max_loss`` and ``spread_width`` are computed from the strikes.
    The exit-signal logic still works because it consumes
    ``current_price`` and ``unrealized_pl`` directly off the legs.
    """
    underlying: str
    strategy_name: str
    legs: List[PositionSnapshot]
    original_credit: float        # net credit received at entry (per share)
    max_loss: float               # defined max loss from the trade plan ($)
    spread_width: float
    net_unrealized_pl: float      # sum of all legs' unrealized P&L
    expiration: str = ""          # option expiration date YYYY-MM-DD
    short_strikes: List[float] = field(default_factory=list)  # short-leg strikes
    exit_signal: ExitSignal = ExitSignal.HOLD
    exit_reason: str = ""
    origin: str = "trade_plan"    # "trade_plan" | "inferred"


class PositionMonitor:
    """
    Monitors open option positions and generates exit signals.

    Parameters
    ----------
    profit_target_pct : float
        Close when unrealized profit ≥ this fraction of the initial credit
        collected.  Default 0.50 (50% profit target — capital retainment).
    hard_stop_multiplier : float
        Close immediately when the spread has lost this multiple of the
        original credit.  Default 3.0 (hard stop at 3× credit).
    strike_proximity_pct : float
        Close immediately when underlying price is within this fraction of
        any short strike.  Default 0.01 (1%).
    """

    def __init__(self, api_key: str, secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets/v2",
                 stop_loss_pct: float = 0.50,    # kept for legacy compat
                 profit_target_pct: float = 0.50,  # 50% profit taker
                 hard_stop_multiplier: float = 3.0,
                 strike_proximity_pct: float = 0.01):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.hard_stop_multiplier = hard_stop_multiplier
        self.strike_proximity_pct = strike_proximity_pct

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Fetch positions from Alpaca
    # ------------------------------------------------------------------

    def fetch_open_positions(self) -> Optional[List[PositionSnapshot]]:
        """GET /v2/positions — filters to us_option only.

        Returns
        -------
        list[PositionSnapshot]
            One entry per open option leg.  Empty list (``[]``) when
            the broker genuinely reports zero positions — a clean slate.
        None
            The HTTP call failed (connection reset, DNS, 5xx, timeout).
            **The caller MUST treat this as "I don't know what's open"
            and fail closed** — do NOT confuse it with the empty-list
            success case, or you'll re-open positions that already
            exist on the broker.

        Why distinguish None from []
        ----------------------------
        Pre-2026-05-05 this method returned ``[]`` on RequestException,
        which made transient broker outages indistinguishable from a
        truly empty account.  On 2026-05-05 a single 100 ms TCP reset
        during Stage 1 of a cycle caused the dedup gate to fail open
        and submit a duplicate DIA Iron Condor on top of an existing
        one.  See ``docs/skills/12_multi_timeframe_resolution.md`` §4
        for the same `*_signal_available: bool` pattern applied
        elsewhere; this method uses the simpler ``Optional[List]``
        shape because the caller already had to handle ``not positions``
        either way.
        """
        url = f"{self.base_url}/positions"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=ALPACA_TIMEOUT)
            resp.raise_for_status()
            positions_data = resp.json()

            positions = []
            for p in positions_data:
                snap = PositionSnapshot(
                    symbol=p.get("symbol", ""),
                    qty=int(p.get("qty", 0)),
                    side=p.get("side", ""),
                    avg_entry_price=float(p.get("avg_entry_price", 0)),
                    current_price=float(p.get("current_price", 0)),
                    market_value=float(p.get("market_value", 0)),
                    cost_basis=float(p.get("cost_basis", 0)),
                    unrealized_pl=float(p.get("unrealized_pl", 0)),
                    unrealized_plpc=float(p.get("unrealized_plpc", 0)),
                    asset_class=p.get("asset_class", ""),
                )
                positions.append(snap)

            option_positions = [p for p in positions if p.asset_class == "us_option"]
            # Hot-path: fires every Stage-1 tick AND every Streamlit
            # broker-state refresh (BROKER_STATE_TTL_SECS=30s default).
            # DEBUG so the default INFO log focuses on actionable events.
            logger.debug("Fetched %d total positions, %d are options",
                         len(positions), len(option_positions))
            return option_positions

        except requests.RequestException as exc:
            logger.error(
                "Failed to fetch positions: %s — returning None so the "
                "cycle's dedup gate can fail closed (see method docstring).",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Group legs into spread positions
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_inner_plan(plan: Dict) -> Dict:
        """Return the inner trade_plan dict regardless of caller shape.

        Two callers feed this method, and they historically built
        ``trade_plans`` differently:

          * ``agent.py:_load_trade_plans`` appends each ``state_history``
            entry verbatim — i.e. the *envelope*
            ``{run_id, timestamp, trade_plan: {...}, risk_verdict, ...}``.
          * ``streamlit/live_monitor.py:_load_positions_with_plans``
            pre-unwraps in the loop and appends ``entry["trade_plan"]``
            — i.e. the *inner* plan ``{ticker, strategy, legs, ...}``
            directly.

        Before this helper existed, ``group_into_spreads`` blindly did
        ``plan.get("trade_plan", {})``. With the agent's envelope this
        unwrapped correctly; with the UI's pre-unwrapped shape it
        returned ``{}`` and silently produced an empty spread list — so
        the broker had filled positions but the dashboard's
        Open-Positions panel rendered nothing. The agent's Stage 1
        monitor still saw them because it passes the envelope shape.

        The helper now supports both shapes: if the dict carries a
        ``"trade_plan"`` sub-key, use that; otherwise treat the whole
        dict as the inner plan. A plan is *only* the envelope if it has
        BOTH the ``"trade_plan"`` key AND that value is itself a dict —
        guarding against an inner plan that happens to have a key
        called ``trade_plan`` (it doesn't, but defensive belt+braces).
        """
        inner = plan.get("trade_plan")
        if isinstance(inner, dict):
            return inner
        return plan

    def group_into_spreads(self, positions: List[PositionSnapshot],
                           trade_plans: List[Dict]) -> List[SpreadPosition]:
        """
        Match open option positions to their original trade plans using
        the option symbols in each plan's legs. Any leg that doesn't
        match a recorded plan is then INFERRED into a spread by leg
        structure (see ``_infer_spreads_from_legs``) so the dashboard
        always shows a meaningful aggregated view rather than dumping
        legs into a separate "ungrouped" section.

        Accepts both envelope-shaped (``{trade_plan: {...}}``) and
        inner-shaped (``{ticker, legs, ...}``) plan dicts — see
        ``_extract_inner_plan`` for the rationale.
        """
        spreads = []
        matched_symbols: set = set()

        for plan in trade_plans:
            tp = self._extract_inner_plan(plan)
            plan_legs = tp.get("legs", [])
            plan_symbols = {leg["symbol"] for leg in plan_legs}
            if not plan_symbols:
                continue

            # Skip plans whose every leg has already been claimed by an
            # earlier matching plan. ``state_history`` typically retains
            # several plan entries with the same leg-symbol set (re-runs,
            # duplicate fills, planner re-emissions); without this guard
            # the same broker spread shows up as N duplicate rows in the
            # dashboard.
            if plan_symbols.issubset(matched_symbols):
                continue

            matched_legs = [
                p for p in positions
                if p.symbol in plan_symbols and p.symbol not in matched_symbols
            ]
            if not matched_legs:
                continue

            # Extract short-strike prices from the plan for proximity checks
            short_strikes = [
                leg["strike"] for leg in plan_legs
                if leg.get("action") == "sell" and "strike" in leg
            ]

            net_pl = sum(leg.unrealized_pl for leg in matched_legs)

            spread = SpreadPosition(
                underlying=tp.get("ticker", ""),
                strategy_name=tp.get("strategy", ""),
                legs=matched_legs,
                original_credit=tp.get("net_credit", 0),
                max_loss=tp.get("max_loss", 0),
                spread_width=tp.get("spread_width", 0),
                net_unrealized_pl=net_pl,
                expiration=tp.get("expiration", ""),
                short_strikes=short_strikes,
                origin="trade_plan",
            )
            spreads.append(spread)
            matched_symbols.update(p.symbol for p in matched_legs)

        # ── Inference fallback ──────────────────────────────────────────
        # Anything still in `positions` but not in matched_symbols belongs
        # to a spread whose trade_plan was rotated out of state_history,
        # or was opened manually outside the agent. Reconstruct those
        # spreads from leg structure so the user sees them in the same
        # table — strategy name, breakeven, P&L all derived from what we
        # know about the legs themselves.
        unmatched = [p for p in positions if p.symbol not in matched_symbols]
        inferred = self._infer_spreads_from_legs(unmatched)
        spreads.extend(inferred)

        # Same hot-path: per-tick + per-Streamlit-refresh. DEBUG.
        logger.debug(
            "Grouped positions into %d spread(s) (%d matched, %d inferred)",
            len(spreads), len(spreads) - len(inferred), len(inferred),
        )
        return spreads

    @staticmethod
    def _parse_occ(symbol: str) -> Optional[Dict]:
        """Decode an OCC option symbol → {underlying, expiration, type, strike}.

        OCC format: ROOT(1-6) + YYMMDD(6) + C/P(1) + STRIKE(8, x1000).
        Example: ``DIA260529P00483000`` → underlying=DIA, expiration
        2026-05-29, type=put, strike=483.0. Returns None on a malformed
        symbol so the caller can skip it without raising.
        """
        if len(symbol) < 16:
            return None
        try:
            # Find the date prefix — ROOT is 1-6 chars, then 6 digits.
            for root_len in range(1, 7):
                if (len(symbol) >= root_len + 15
                        and symbol[root_len:root_len + 6].isdigit()
                        and symbol[root_len + 6] in ("C", "P")
                        and symbol[root_len + 7:root_len + 15].isdigit()):
                    underlying = symbol[:root_len]
                    yymmdd = symbol[root_len:root_len + 6]
                    cp = symbol[root_len + 6]
                    strike = int(symbol[root_len + 7:root_len + 15]) / 1000.0
                    expiration = (
                        f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
                    )
                    return {
                        "underlying": underlying,
                        "expiration": expiration,
                        "type":       "call" if cp == "C" else "put",
                        "strike":     strike,
                    }
        except (ValueError, IndexError):
            return None
        return None

    @classmethod
    def _infer_spreads_from_legs(
            cls, legs: List[PositionSnapshot]) -> List[SpreadPosition]:
        """Infer spread structure for legs with no matching trade plan.

        Groups by (underlying, expiration), then classifies the strategy
        from the count + sign of put-vs-call legs:

          * 2 puts (one short, one long) only       → "Bull Put Spread"
          * 2 calls (one short, one long) only      → "Bear Call Spread"
          * 4 legs spanning both calls + puts       → "Iron Condor"
          * 1 short leg only                        → "Naked Short"
          * everything else                         → "Multi-leg Position"

        The credit/max-loss math uses each leg's ``avg_entry_price`` to
        recover the entry economics. The reconstruction is best-effort —
        if the leg-mix doesn't fit a clean credit-spread shape, the
        strategy is labelled "Multi-leg Position" but the table row
        still aggregates so the user can see what's open and the live
        P&L without each leg being its own row.
        """
        # Bucket legs by (underlying, expiration) — that's the natural
        # spread grouping. Two strategies on the same underlying with
        # different expirations are independent positions.
        buckets: Dict[tuple, List[PositionSnapshot]] = {}
        for leg in legs:
            occ = cls._parse_occ(leg.symbol)
            if occ is None:
                continue
            key = (occ["underlying"], occ["expiration"])
            buckets.setdefault(key, []).append(leg)

        out: List[SpreadPosition] = []
        for (underlying, expiration), grp in buckets.items():
            # Decode each leg's strike + type for strategy inference.
            decoded = []
            for leg in grp:
                occ = cls._parse_occ(leg.symbol)
                if occ is None:
                    continue
                decoded.append({
                    "leg":    leg,
                    "type":   occ["type"],
                    "strike": occ["strike"],
                    "side":   leg.side,         # "short" or "long"
                })

            shorts = [d for d in decoded if d["side"] == "short"]
            longs  = [d for d in decoded if d["side"] == "long"]
            puts   = [d for d in decoded if d["type"] == "put"]
            calls  = [d for d in decoded if d["type"] == "call"]

            # Classify
            if len(decoded) == 4 and puts and calls and shorts and longs:
                strategy = "Iron Condor"
            elif (len(decoded) == 2 and len(puts) == 2 and len(shorts) == 1):
                strategy = "Bull Put Spread"
            elif (len(decoded) == 2 and len(calls) == 2 and len(shorts) == 1):
                strategy = "Bear Call Spread"
            elif len(decoded) == 1 and shorts:
                strategy = "Naked Short"
            else:
                strategy = "Multi-leg Position"

            # Compute economics from leg snapshots.
            #   credit (per share) = Σ(short avg_entry) − Σ(long avg_entry)
            credit = (
                sum(d["leg"].avg_entry_price for d in shorts)
                - sum(d["leg"].avg_entry_price for d in longs)
            )
            # Max loss for credit spreads = max single-side spread width − credit.
            # We compute width per side (call wing + put wing for ICs) and
            # take the wider as the max single-side loss bound.
            def _wing_width(side_legs):
                if len(side_legs) != 2:
                    return 0.0
                strikes = sorted(d["strike"] for d in side_legs)
                return strikes[1] - strikes[0]

            put_width  = _wing_width(puts)
            call_width = _wing_width(calls)
            spread_width = max(put_width, call_width, 0.0)
            max_loss = max(0.0, (spread_width - credit) * 100)

            short_strikes = [d["strike"] for d in shorts]
            net_pl = sum(d["leg"].unrealized_pl for d in decoded)

            out.append(SpreadPosition(
                underlying=underlying,
                strategy_name=strategy,
                legs=[d["leg"] for d in decoded],
                original_credit=round(credit, 2),
                max_loss=round(max_loss, 2),
                spread_width=spread_width,
                net_unrealized_pl=net_pl,
                expiration=expiration,
                short_strikes=short_strikes,
                origin="inferred",
            ))

        return out

    # ------------------------------------------------------------------
    # Evaluate exit signals
    # ------------------------------------------------------------------

    def evaluate(self, spreads: List[SpreadPosition],
                 current_regimes: Dict[str, Regime],
                 underlying_prices: Optional[Dict[str, float]] = None
                 ) -> List[SpreadPosition]:
        """
        Check each spread against all exit rules and assign exit_signal.

        Parameters
        ----------
        underlying_prices : dict mapping ticker → current price, used for
            the strike-proximity guard.
        """
        prices = underlying_prices or {}

        for spread in spreads:
            signal, reason = self._check_exit(
                spread, current_regimes, prices.get(spread.underlying, 0.0))
            spread.exit_signal = signal
            spread.exit_reason = reason

            if signal != ExitSignal.HOLD:
                immediate = signal in IMMEDIATE_EXIT_SIGNALS
                logger.warning(
                    "[%s] EXIT SIGNAL: %s%s — %s | P&L=$%.2f",
                    spread.underlying, signal.value,
                    " (IMMEDIATE)" if immediate else " (debounce)",
                    reason, spread.net_unrealized_pl)
            else:
                logger.info(
                    "[%s] HOLD — P&L=$%.2f (credit=$%.2f, max_loss=$%.2f)",
                    spread.underlying, spread.net_unrealized_pl,
                    spread.original_credit, spread.max_loss)

        return spreads

    def _check_exit(self, spread: SpreadPosition,
                    current_regimes: Dict[str, Regime],
                    underlying_price: float = 0.0):
        """Return (ExitSignal, reason) for a single spread."""

        credit_value = spread.original_credit * 100   # per-contract dollar value

        # --- 1. Hard stop: spread has lost 3× the initial credit (IMMEDIATE) ---
        hard_stop_threshold = credit_value * self.hard_stop_multiplier
        loss = -spread.net_unrealized_pl   # positive when losing
        if loss >= hard_stop_threshold > 0:
            return (
                ExitSignal.HARD_STOP,
                f"Loss ${loss:.2f} ≥ {self.hard_stop_multiplier:.0f}× credit "
                f"${credit_value:.2f} (threshold=${hard_stop_threshold:.2f})"
            )

        # --- 2. Legacy stop-loss: loss ≥ 50% of defined max-loss ---
        loss_threshold = spread.max_loss * self.stop_loss_pct
        if loss >= loss_threshold > 0:
            return (
                ExitSignal.STOP_LOSS,
                f"Loss ${loss:.2f} ≥ {self.stop_loss_pct*100:.0f}% of "
                f"max loss ${spread.max_loss:.2f}"
            )

        # --- 3. Profit target: 50% of credit captured ---
        profit_threshold = credit_value * self.profit_target_pct
        if spread.net_unrealized_pl >= profit_threshold > 0:
            return (
                ExitSignal.PROFIT_TARGET,
                f"Profit ${spread.net_unrealized_pl:.2f} ≥ "
                f"{self.profit_target_pct*100:.0f}% of credit "
                f"${credit_value:.2f}"
            )

        # --- 4. Strike proximity guard (IMMEDIATE) ---
        if underlying_price > 0 and spread.short_strikes:
            for strike in spread.short_strikes:
                proximity = abs(underlying_price - strike) / strike
                if proximity <= self.strike_proximity_pct:
                    return (
                        ExitSignal.STRIKE_PROXIMITY,
                        f"Underlying ${underlying_price:.2f} is within "
                        f"{proximity*100:.2f}% of short strike ${strike:.0f} "
                        f"— closing to prevent ITM assignment"
                    )

        # --- 5. DTE safety: liquidate by 15:30 ET on Thursday before expiry ---
        dte_signal = self._check_dte_safety(spread.expiration)
        if dte_signal:
            return (ExitSignal.DTE_SAFETY, dte_signal)

        # --- 6. Regime shift ---
        ticker = spread.underlying
        if ticker in current_regimes:
            current_regime = current_regimes[ticker]
            expected_regime = STRATEGY_REGIME_MAP.get(spread.strategy_name)
            if expected_regime and current_regime != expected_regime:
                return (
                    ExitSignal.REGIME_SHIFT,
                    f"Regime shifted to {current_regime.value} but holding "
                    f"{spread.strategy_name} (expects {expected_regime.value})"
                )

        return (ExitSignal.HOLD, "")

    @staticmethod
    def _check_dte_safety(expiration: str) -> str:
        """
        Return a non-empty reason string if the DTE safety rule triggers.

        Rule: if today is the **last NYSE trading day strictly before
        expiration** AND current time is ≥ 15:30 ET, return a warning.

        We avoid carrying an option into its final day of life to prevent
        last-day gamma explosion and assignment risk. Using the NYSE
        calendar (pandas_market_calendars) correctly handles holiday
        weeks — e.g. when Good Friday closes the market, the last trading
        day before a Friday expiration is Thursday; when a Wednesday
        expiration week lands (unusual), the rule fires on Tuesday.
        """
        if not expiration:
            return ""
        try:
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            # Convert to ET for the time check
            now_utc = datetime.now(timezone.utc)
            # ET = UTC-4 (EDT) or UTC-5 (EST); use UTC-4 (market hours)
            now_et_hour = (now_utc.hour - 4) % 24
            now_et_minute = now_utc.minute
            today = now_utc.date()

            after_cutoff = (now_et_hour > 15 or
                            (now_et_hour == 15 and now_et_minute >= 30))
            last_day = is_last_trading_day_before(today, exp_date)

            if last_day and after_cutoff:
                return (
                    f"DTE safety: expiration {expiration} is the next "
                    f"trading day. Liquidating by 15:30 ET to avoid "
                    f"last-day gamma risk."
                )
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, spreads: List[SpreadPosition]) -> Dict:
        total_pl = sum(s.net_unrealized_pl for s in spreads)
        signals: Dict[str, int] = {}
        for s in spreads:
            sig = s.exit_signal.value
            signals[sig] = signals.get(sig, 0) + 1

        return {
            "total_spreads": len(spreads),
            "total_unrealized_pl": round(total_pl, 2),
            "signals": signals,
            "positions": [
                {
                    "underlying": s.underlying,
                    "strategy": s.strategy_name,
                    "pl": round(s.net_unrealized_pl, 2),
                    "signal": s.exit_signal.value,
                    "reason": s.exit_reason,
                    "expiration": s.expiration,
                    "short_strikes": s.short_strikes,
                }
                for s in spreads
            ],
        }

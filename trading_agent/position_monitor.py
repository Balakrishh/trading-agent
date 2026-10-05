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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional

import requests

from trading_agent.calendar_utils import is_last_trading_day_before
from trading_agent.market_data import ALPACA_TIMEOUT
from trading_agent.regime import Regime
from trading_agent.wheel_policy import (
    CSP_STOP_ABS_DELTA, CSP_STRATEGY, TAKE_PROFIT_PCT_OF_CREDIT, WHEEL_STRATEGIES,
)
from trading_agent.debit_policy import (
    BOUNCE_BULL_PUT_STRATEGY, CALENDAR_STRATEGY, CALL_DEBIT_STRATEGY,
    DEBIT_STRATEGIES, DEBIT_VERTICALS, PUT_DEBIT_STRATEGY,
)


# ── Position-fetch retry policy ─────────────────────────────────────────────
# Two attempts on transient transport errors.  The dedup gate fails closed
# on a final None, so a one-off TCP blip used to skip an entire cycle of
# exit-signal evaluation; with retry we recover from the typical 100ms
# reset that triggered the 2026-05-05 duplicate-submission incident.
POSITION_FETCH_RETRY_ATTEMPTS = 2
# Brief sleep between attempts.  Total worst-case latency stays well
# inside the 270s cycle hard guard.
POSITION_FETCH_RETRY_BACKOFF_S = 0.5

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
    DELTA_STOP = "delta_stop"        # Wheel CSP |Δ| ≥ CSP_STOP_ABS_DELTA (debounced)


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
    # Skill 59. The bounce bull put is opened in a bearish regime on
    # purpose, so a regime check would close it at once — None.
    CALL_DEBIT_STRATEGY:      Regime.BULLISH,
    PUT_DEBIT_STRATEGY:       Regime.BEARISH,
    CALENDAR_STRATEGY:        Regime.SIDEWAYS,
    BOUNCE_BULL_PUT_STRATEGY: None,
}

# Regimes that reverse a directional debit thesis (skill 59).
_OPPOSITE_TREND = {
    Regime.BULLISH: (Regime.BEARISH,),
    Regime.BEARISH: (Regime.BULLISH,),
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
    # "broker" = Alpaca's last-trade mark; "mid" = re-marked from the
    # live bid/ask by ``remark_positions_at_mid`` (skill 44 §4).
    mark_source: str = "broker"
    # P&L if the leg were closed at its natural price right now (short →
    # buy back at the ask, long → sell at the bid). Set by the re-mark;
    # None when no usable quote. Drives the profit target (backlog §6.1).
    natural_unrealized_pl: Optional[float] = None


def remark_positions_at_mid(positions: List[PositionSnapshot],
                            quotes: Dict[str, Dict]) -> List[PositionSnapshot]:
    """Re-value each leg at its bid/ask mid instead of Alpaca's mark.

    Alpaca's ``current_price`` is often the last trade — minutes stale
    and on one side of a wide option market. Summed over 4 legs × N
    contracts that noise alone showed a -$150 "loss" on a flat Iron
    Condor (2026-09-29) and can trip stops. Legs without a usable quote
    (missing, bid ≤ 0, crossed) keep the broker mark.
    """
    out: List[PositionSnapshot] = []
    for p in positions:
        q = quotes.get(p.symbol) if quotes else None
        bid = float((q or {}).get("bid", 0) or 0)
        ask = float((q or {}).get("ask", 0) or 0)
        if bid <= 0 or ask < bid:
            out.append(p)
            continue
        mid = (bid + ask) / 2
        pl = round((mid - p.avg_entry_price) * p.qty * 100, 2)
        out.append(replace(
            p,
            current_price=round(mid, 4),
            market_value=round(mid * p.qty * 100, 2),
            unrealized_pl=pl,
            unrealized_plpc=(pl / abs(p.cost_basis)) if p.cost_basis else 0.0,
            mark_source="mid",
            natural_unrealized_pl=round(
                ((ask if p.qty < 0 else bid) - p.avg_entry_price) * p.qty * 100, 2),
        ))
    return out


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
    # Contract count across the spread. Derived as the min |qty| across
    # legs (all legs of a properly-filled spread carry the same qty; the
    # min guards against a partial fill). Used to scale hard_stop,
    # stop_loss, and profit_target thresholds — see skill 44.
    contracts_open: int = 1
    # ISO-8601 UTC timestamp of the trade plan's submit event. Used by
    # the post-fill grace period gate in ``_check_exit`` so the monitor
    # doesn't evaluate P&L on a stale mark in the first N seconds after
    # entry. Empty string means "no known submit time" (inferred
    # spreads, legacy positions) — grace gate is skipped in that case.
    opened_at: str = ""
    # Append-only (2026-09-29): live |Δ| of the short leg for Wheel positions,
    # filled by the agent from the option chain. None = unknown → the delta
    # stop cannot fire (profit target still works).
    short_delta: Optional[float] = None
    # Sum of legs' natural_unrealized_pl (cost to close at natural); None
    # when any leg lacks a usable quote. Profit target basis (§6.1).
    net_natural_pl: Optional[float] = None


def _sum_natural(legs) -> Optional[float]:
    vals = [getattr(leg, "natural_unrealized_pl", None) for leg in legs]
    return None if not vals or any(v is None for v in vals) else round(sum(vals), 2)


def attach_wheel_short_deltas(spreads: List[SpreadPosition],
                              fetch_chain: Callable[[str, str, str], Optional[List[Dict]]]
                              ) -> None:
    """Fill ``short_delta`` on single-leg Wheel positions from the option
    chain (quotes carry no greeks). ``fetch_chain(underlying, expiration,
    option_type)`` is the market-data provider's ``fetch_option_chain``.
    A failed or missing lookup leaves ``short_delta=None`` (sentinel) so
    the CSP delta stop simply cannot fire that cycle.
    """
    for s in spreads:
        if s.strategy_name not in WHEEL_STRATEGIES or len(s.legs) != 1:
            continue
        occ = PositionMonitor._parse_occ(s.legs[0].symbol)
        if not occ:
            continue
        try:
            chain = fetch_chain(occ["underlying"], occ["expiration"], occ["type"]) or []
        except Exception as exc:                                  # noqa: BLE001, skill-34-exempt — delta unknown this cycle
            logger.warning("[%s] Wheel delta lookup failed (%s)", s.underlying, exc)
            continue
        for c in chain:
            if c.get("symbol") == s.legs[0].symbol and c.get("delta") not in (None, 0):
                s.short_delta = float(c["delta"])
                break


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
                 strike_proximity_pct: float = 0.01,
                 post_fill_grace_seconds: int = 60,
                 profit_target_basis: str = "natural",
                 debit_profit_target_pct: float = 0.50,
                 debit_stop_loss_pct: float = 0.50,
                 calendar_profit_target_pct: float = 0.25):
        """
        Additional parameter
        --------------------
        post_fill_grace_seconds : int
            Skill 44 — number of seconds after a position's ``opened_at``
            during which the exit-signal check returns HOLD regardless
            of the current mark. Prevents a stale bid-ask immediately
            post-fill from triggering a phantom hard_stop. Default 60s.
            Set to 0 in tests when you want to exercise the exit paths
            directly against a synthesised mark.
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.hard_stop_multiplier = hard_stop_multiplier
        self.post_fill_grace_seconds = int(post_fill_grace_seconds)
        self.strike_proximity_pct = strike_proximity_pct
        # "natural": judge the profit target on the cost to close at the
        # natural price (PresetConfig.fill_model, backlog §6.1); "mid":
        # legacy mid valuation. Stops always use the mid valuation.
        self.profit_target_basis = profit_target_basis
        # Skill 59 debit structures (PresetConfig.debit_* / calendar_*).
        self.debit_profit_target_pct = debit_profit_target_pct
        self.debit_stop_loss_pct = debit_stop_loss_pct
        self.calendar_profit_target_pct = calendar_profit_target_pct

    def _profit_pl(self, spread: "SpreadPosition") -> float:
        if self.profit_target_basis == "natural" and spread.net_natural_pl is not None:
            return spread.net_natural_pl
        return spread.net_unrealized_pl

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
        last_exc: Optional[Exception] = None
        for attempt in range(1, POSITION_FETCH_RETRY_ATTEMPTS + 1):
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
                if attempt > 1:
                    logger.info(
                        "Fetched positions on retry attempt %d/%d "
                        "(%d total, %d options)",
                        attempt, POSITION_FETCH_RETRY_ATTEMPTS,
                        len(positions), len(option_positions),
                    )
                else:
                    logger.debug("Fetched %d total positions, %d are options",
                                 len(positions), len(option_positions))
                return option_positions

            except requests.RequestException as exc:
                last_exc = exc
                if attempt < POSITION_FETCH_RETRY_ATTEMPTS:
                    logger.warning(
                        "fetch_open_positions transient failure "
                        "(attempt %d/%d): %s — retrying in %.1fs",
                        attempt, POSITION_FETCH_RETRY_ATTEMPTS, exc,
                        POSITION_FETCH_RETRY_BACKOFF_S,
                    )
                    time.sleep(POSITION_FETCH_RETRY_BACKOFF_S)
                    continue

        # All attempts exhausted.  Returning None so the cycle's dedup
        # gate can fail closed (see method docstring).
        logger.error(
            "Failed to fetch positions after %d attempt(s): %s — "
            "returning None so the cycle's dedup gate can fail closed.",
            POSITION_FETCH_RETRY_ATTEMPTS, last_exc,
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

            # ── Skip plans the agent never actually executed ──────────
            # ``state_history`` retains every plan the chain scanner
            # emitted that cycle, including rejections (``valid=False``)
            # and risk-vetoed plans (``risk_verdict.approved=False``).
            # A rejected plan whose leg symbols partially overlap the
            # actually-submitted plan would otherwise greedily claim
            # the shared legs first — splitting a single Iron Condor
            # into two display rows and double-counting against the
            # per-ticker position cap. See the 2026-05-15 XLF incident:
            # 4 rejected plans shared the call wing with the submitted
            # plan and stole those legs into a phantom spread row.
            #
            # Defaults are permissive — missing ``valid`` or missing
            # ``risk_verdict`` are treated as valid/approved so that
            # inner-shaped plans (which don't carry risk_verdict) and
            # older history files (pre-``valid`` field) aren't dropped.
            if tp.get("valid") is False:
                continue
            if isinstance(plan, dict):
                rv = plan.get("risk_verdict")
                if isinstance(rv, dict) and rv.get("approved") is False:
                    continue

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

            # Contract count derivation. All legs of a correctly-filled
            # spread carry the same |qty|; a partial fill has one leg at
            # a lower qty. Taking the MIN guards against the partial-
            # fill case (the position's exposure is bounded by the
            # smaller side). Skill 44 §4 pins this to a conformance test.
            contracts_open = max(1, min(
                abs(leg.qty) for leg in matched_legs if leg.qty
            )) if matched_legs else 1

            # Submit timestamp — pulled from the trade plan's own
            # ``timestamp`` field, populated by the executor at submit
            # time. Falls back to the outer state-history timestamp on
            # older plan shapes.
            plan_outer_ts = (
                plan.get("timestamp", "") if isinstance(plan, dict) else ""
            )
            opened_at = str(tp.get("timestamp", "") or plan_outer_ts)

            spread = SpreadPosition(
                underlying=tp.get("ticker", ""),
                strategy_name=tp.get("strategy", ""),
                legs=matched_legs,
                original_credit=tp.get("net_credit", 0),
                max_loss=tp.get("max_loss", 0),
                spread_width=tp.get("spread_width", 0),
                net_unrealized_pl=net_pl,
                net_natural_pl=_sum_natural(matched_legs),
                expiration=tp.get("expiration", ""),
                short_strikes=short_strikes,
                origin="trade_plan",
                contracts_open=contracts_open,
                opened_at=opened_at,
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
        calendars, unmatched = self._infer_calendars(unmatched)
        spreads.extend(calendars)
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
    def _infer_calendars(cls, legs: List[PositionSnapshot]):
        """Pair a short and a long leg with the same underlying, type and
        strike but different expirations into a Calendar Spread (skill 59)
        — the (underlying, expiration) buckets below would otherwise split
        it into a "Naked Short" plus an orphan long. Returns (calendars,
        remaining legs)."""
        decoded = [(leg, cls._parse_occ(leg.symbol)) for leg in legs]
        used: set = set()
        out: List[SpreadPosition] = []
        for s_leg, s_occ in decoded:
            if s_occ is None or s_leg.side != "short" or s_leg.symbol in used:
                continue
            for l_leg, l_occ in decoded:
                if (l_occ is None or l_leg.side != "long" or l_leg.symbol in used
                        or l_occ["underlying"] != s_occ["underlying"]
                        or l_occ["type"] != s_occ["type"]
                        or l_occ["strike"] != s_occ["strike"]
                        or l_occ["expiration"] <= s_occ["expiration"]):
                    continue
                credit = s_leg.avg_entry_price - l_leg.avg_entry_price
                pair = [s_leg, l_leg]
                out.append(SpreadPosition(
                    underlying=s_occ["underlying"], strategy_name=CALENDAR_STRATEGY,
                    legs=pair, original_credit=round(credit, 2),
                    max_loss=round(max(0.0, -credit * 100), 2), spread_width=0.0,
                    net_unrealized_pl=sum(p.unrealized_pl for p in pair),
                    net_natural_pl=_sum_natural(pair),
                    expiration=s_occ["expiration"], short_strikes=[s_occ["strike"]],
                    origin="inferred"))
                used.update({s_leg.symbol, l_leg.symbol})
                break
        return out, [leg for leg in legs if leg.symbol not in used]

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

            # Entry credit per share (negative → a debit structure).
            entry_net = (sum(d["leg"].avg_entry_price for d in shorts)
                         - sum(d["leg"].avg_entry_price for d in longs))

            # Classify
            if len(decoded) == 4 and puts and calls and shorts and longs:
                strategy = "Iron Condor"
            elif (len(decoded) == 2 and len(puts) == 2 and len(shorts) == 1):
                strategy = PUT_DEBIT_STRATEGY if entry_net < 0 else "Bull Put Spread"
            elif (len(decoded) == 2 and len(calls) == 2 and len(shorts) == 1):
                strategy = CALL_DEBIT_STRATEGY if entry_net < 0 else "Bear Call Spread"
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
            max_loss = (max(0.0, -credit * 100) if strategy in DEBIT_VERTICALS
                        else max(0.0, (spread_width - credit) * 100))

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
                net_natural_pl=_sum_natural([d["leg"] for d in decoded]),
                expiration=expiration,
                short_strikes=short_strikes,
                origin="inferred",
            ))

        return out

    # ------------------------------------------------------------------
    # Evaluate exit signals
    # ------------------------------------------------------------------

    def _check_wheel_exit(self, spread: SpreadPosition):
        """Profit target at TAKE_PROFIT_PCT_OF_CREDIT; CSP delta stop at
        CSP_STOP_ABS_DELTA; otherwise hold (assignment accepted)."""
        contracts = max(1, spread.contracts_open)
        credit_position = spread.original_credit * 100 * contracts
        target = credit_position * TAKE_PROFIT_PCT_OF_CREDIT
        profit_pl = self._profit_pl(spread)
        if profit_pl >= target > 0:
            return (ExitSignal.PROFIT_TARGET,
                    f"Wheel: profit ${profit_pl:.2f} ({self.profit_target_basis}) ≥ "
                    f"{TAKE_PROFIT_PCT_OF_CREDIT:.0%} of credit ${credit_position:.2f}")
        if (spread.strategy_name == CSP_STRATEGY and spread.short_delta is not None
                and abs(spread.short_delta) >= CSP_STOP_ABS_DELTA):
            return (ExitSignal.DELTA_STOP,
                    f"Wheel CSP: |Δ| {abs(spread.short_delta):.2f} ≥ {CSP_STOP_ABS_DELTA}")
        return (ExitSignal.HOLD, "Wheel: holding — assignment accepted")

    def _check_debit_exit(self, spread: SpreadPosition,
                          current_regimes: Dict[str, Regime]):
        """Stop at ``debit_stop_loss_pct`` of the debit; profit target at
        ``debit_profit_target_pct`` of max profit (verticals) or
        ``calendar_profit_target_pct`` of the debit (calendars); DTE
        safety on the (near) expiry; regime shift against the thesis."""
        contracts = max(1, spread.contracts_open)
        debit_position = -spread.original_credit * 100 * contracts
        if debit_position <= 0:
            return (ExitSignal.HOLD, "Debit: no recorded debit — holding")
        loss = -spread.net_unrealized_pl
        stop = debit_position * self.debit_stop_loss_pct
        if loss >= stop > 0:
            return (ExitSignal.STOP_LOSS,
                    f"Debit: loss ${loss:.2f} ≥ {self.debit_stop_loss_pct:.0%} of "
                    f"debit ${debit_position:.2f}")
        if spread.strategy_name in DEBIT_VERTICALS:
            max_profit = (spread.spread_width + spread.original_credit) * 100 * contracts
            target = max_profit * self.debit_profit_target_pct
            label = f"{self.debit_profit_target_pct:.0%} of max profit ${max_profit:.2f}"
        else:
            target = debit_position * self.calendar_profit_target_pct
            label = f"{self.calendar_profit_target_pct:.0%} of debit ${debit_position:.2f}"
        profit_pl = self._profit_pl(spread)
        if profit_pl >= target > 0:
            return (ExitSignal.PROFIT_TARGET,
                    f"Debit: profit ${profit_pl:.2f} ({self.profit_target_basis}) ≥ {label}")
        dte_signal = self._check_dte_safety(spread.expiration)
        if dte_signal:
            return (ExitSignal.DTE_SAFETY, dte_signal)
        expected = STRATEGY_REGIME_MAP.get(spread.strategy_name)
        current = current_regimes.get(spread.underlying)
        if expected and current is not None and current != expected:
            # A debit vertical's thesis breaks only when the trend
            # REVERSES; a drift to sideways is not a contradiction.
            # 2026-10-05: IWM flickered bearish → sideways one cycle after
            # a put debit filled (price between its 50- and 200-day) and
            # the old rule voted to close it at the bid/ask cost.
            if (spread.strategy_name in DEBIT_VERTICALS
                    and current not in _OPPOSITE_TREND.get(expected, ())):
                return (ExitSignal.HOLD, "")
            return (ExitSignal.REGIME_SHIFT,
                    f"Regime shifted to {current.value} but holding "
                    f"{spread.strategy_name} (expects {expected.value})")
        return (ExitSignal.HOLD, "")

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
        """Return (ExitSignal, reason) for a single spread.

        Skill 44 (2026-07-02) — two invariants critical to correctness:

        1. **Contract-count scaling.** ``spread.net_unrealized_pl`` is
           the POSITION-scale total across all contracts (sum of every
           leg's unrealized_pl). ``spread.original_credit`` and
           ``spread.max_loss`` are PER-CONTRACT economics from the
           trade plan. All three thresholds below multiply by
           ``spread.contracts_open`` so a 12-contract position doesn't
           trip the hard-stop line at 1/12 of the intended loss.

        2. **Post-fill grace period.** Immediately after fill (within
           the first ``post_fill_grace_seconds``) Alpaca's spread quote
           can lag reality, producing phantom unrealized losses that
           don't reflect the actual market. Skip exit evaluation
           during that window so a stale mark doesn't liquidate a
           just-opened position.
        """

        # --- 0. Post-fill grace period (skill 44) -----------------------
        # If we know when the position opened AND it's less than
        # ``post_fill_grace_seconds`` old, hold — the mark may still be
        # settling. ``opened_at`` empty ("") means inferred / legacy
        # spread with no known submit time; skip the gate rather than
        # block indefinitely.
        if spread.opened_at:
            try:
                from datetime import datetime as _dt
                from datetime import timezone as _tz
                open_ts = _dt.fromisoformat(
                    spread.opened_at.replace("Z", "+00:00")
                )
                age = (_dt.now(_tz.utc) - open_ts).total_seconds()
                if 0 <= age < self.post_fill_grace_seconds:
                    return (
                        ExitSignal.HOLD,
                        f"Post-fill grace ({age:.0f}s < "
                        f"{self.post_fill_grace_seconds}s)"
                    )
            except (ValueError, TypeError):
                # Malformed timestamp — fall through, don't crash.
                pass

        # --- Wheel legs (CSP / covered call) — skill 40 §2.9 ------------
        # Assignment is part of the plan, so the spread rules that close
        # near the strike or before expiry (strike proximity, DTE safety,
        # regime shift, credit-multiple hard stop) must not run here.
        if spread.strategy_name in WHEEL_STRATEGIES:
            return self._check_wheel_exit(spread)

        # --- Debit structures (skill 59) ------------------------------
        # The credit rules are keyed to a credit received; a long spread
        # or calendar risks its debit instead. Strike proximity does not
        # apply (a debit vertical WANTS price through the short strike).
        if spread.strategy_name in DEBIT_STRATEGIES:
            return self._check_debit_exit(spread, current_regimes)

        # ---------------------------------------------------------------
        # Per-position economics (contract-count-scaled).
        # ---------------------------------------------------------------
        contracts = max(1, spread.contracts_open)
        credit_per_contract = spread.original_credit * 100
        credit_position = credit_per_contract * contracts    # dollars for the whole position
        max_loss_position = spread.max_loss * contracts

        # --- 1. Hard stop: position has lost 3× total credit (IMMEDIATE) ---
        hard_stop_threshold = credit_position * self.hard_stop_multiplier
        loss = -spread.net_unrealized_pl   # positive when losing
        if loss >= hard_stop_threshold > 0:
            return (
                ExitSignal.HARD_STOP,
                f"Loss ${loss:.2f} ≥ {self.hard_stop_multiplier:.0f}× credit "
                f"${credit_position:.2f} ({contracts}×${credit_per_contract:.2f}) "
                f"threshold=${hard_stop_threshold:.2f}"
            )

        # --- 2. Legacy stop-loss: loss ≥ 50% of defined max-loss ---
        loss_threshold = max_loss_position * self.stop_loss_pct
        if loss >= loss_threshold > 0:
            return (
                ExitSignal.STOP_LOSS,
                f"Loss ${loss:.2f} ≥ {self.stop_loss_pct*100:.0f}% of "
                f"max loss ${max_loss_position:.2f} ({contracts}×${spread.max_loss:.2f})"
            )

        # --- 3. Profit target: 50% of credit captured ---
        profit_threshold = credit_position * self.profit_target_pct
        profit_pl = self._profit_pl(spread)
        if profit_pl >= profit_threshold > 0:
            return (
                ExitSignal.PROFIT_TARGET,
                f"Profit ${profit_pl:.2f} ({self.profit_target_basis}) ≥ "
                f"{self.profit_target_pct*100:.0f}% of credit "
                f"${credit_position:.2f} ({contracts}×${credit_per_contract:.2f})"
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
        except Exception:  # noqa: skill-34-exempt — best-effort; returns empty list on broker outage
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
                    # Per-contract max loss ($) and contracts — the
                    # agent's total-risk cap sums these (2026-10-05).
                    "max_loss": s.max_loss,
                    "contracts": s.contracts_open,
                }
                for s in spreads
            ],
        }

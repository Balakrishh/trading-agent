"""market_state.py — market risk state + regime → playbook table (skill 58).

Backlog §1 (market risk-state overlay) and §6.2 (playbook table), 2026-10-05.

Two deterministic layers, both pure functions so the same inputs always
give the same answer and every rule is unit-testable:

1. **Market risk state** for the whole market, once per cycle, from SPY's
   trend (20/50/200-day averages, RSI), the VIX level, the VIX/VIX3M term
   structure and breadth (share of watchlist tickers above their 50-day):

   ========== ========================================================
   State      Meaning / effect on new entries
   ========== ========================================================
   NORMAL     everything allowed, full size
   CAUTION    half size; no new bull puts or cash-secured puts
   DEFENSIVE  quarter size; bear calls only; no new cash-secured puts
   CAPITULATN no new entries at all
   RECOVERY   half size; bullish premium (bull puts, CSPs) re-enabled
   ========== ========================================================

   Missing SPY data fails safe to CAUTION. Missing VIX / VIX3M / breadth
   inputs are skipped (sentinel ``None``), never read as zero.

2. **Playbook** per ticker: trend (the ticker's regime) × volatility bucket
   × RSI extreme → the strategy family that fits. Playbooks the agent
   cannot trade yet are flagged ``implemented=False`` so the journal shows
   how often the market asked for a missing tool (backlog §6.3–6.5).

Only ``compute_inputs`` / ``read_state`` / ``write_state`` do I/O.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

NORMAL, CAUTION, DEFENSIVE, CAPITULATION, RECOVERY = (
    "NORMAL", "CAUTION", "DEFENSIVE", "CAPITULATION", "RECOVERY")
STATES = (NORMAL, CAUTION, DEFENSIVE, CAPITULATION, RECOVERY)

BULL_PUT, BEAR_CALL, IRON_CONDOR, IRON_BUTTERFLY, MEAN_REVERSION = (
    "Bull Put Spread", "Bear Call Spread", "Iron Condor", "Iron Butterfly",
    "Mean Reversion Spread")
ALL_SPREADS = frozenset({BULL_PUT, BEAR_CALL, IRON_CONDOR, IRON_BUTTERFLY})

# Bonds, metals and commodities often rise in an equity sell-off — counting
# them would make breadth look healthy exactly when it is not.
NON_EQUITY_TICKERS = frozenset({"TLT", "IEF", "SHY", "AGG", "BND", "GLD", "IAU",
                                "SLV", "GDX", "GDXJ", "USO", "UNG", "UUP", "DBC"})


def breadth_universe(tickers: Iterable[str]) -> List[str]:
    """Watchlist tickers that count toward breadth (equities, not SPY)."""
    return [t for t in tickers if t != "SPY" and t not in NON_EQUITY_TICKERS]


STATE_PATH = Path(os.environ.get("TRADING_AGENT_MARKET_STATE",
                                 "trade_journal/market_state.json"))


@dataclass(frozen=True)
class MarketStateConfig:
    """Thresholds. Validated against SPY/VIX history 2020–2025 by
    ``scripts/research/validate_market_state.py`` (skill 58 §4)."""
    capitulation_vix: float = 35.0
    capitulation_rsi: float = 25.0
    capitulation_rsi_vix: float = 28.0
    capitulation_term_ratio: float = 1.10      # VIX / VIX3M — steep inversion
    defensive_vix: float = 22.0
    defensive_term_ratio: float = 1.00
    defensive_breadth: float = 0.40
    caution_vix: float = 20.0
    caution_term_ratio: float = 0.95
    caution_breadth: float = 0.50
    recovery_exit_vix: float = 20.0            # RECOVERY → NORMAL once VIX < this and SPY > SMA-50
    # Hysteresis: leaving CAUTION for NORMAL needs every condition clear by
    # a margin (SPY ≥ 1 % above SMA-50, VIX ≥ 1 pt below, term ≤ 0.93,
    # breadth ≥ 55 %). Without it the 2019–2026 replay flipped
    # NORMAL↔CAUTION 159 times (median CAUTION run: 2 days).
    caution_exit_sma_margin: float = 0.01
    caution_exit_vix_margin: float = 1.0
    caution_exit_term_ratio: float = 0.93
    caution_exit_breadth: float = 0.55


@dataclass(frozen=True)
class StateGate:
    size_multiplier: float
    allowed_strategies: FrozenSet[str]
    allow_new_csp: bool


GATES: Dict[str, StateGate] = {
    NORMAL:       StateGate(1.0, ALL_SPREADS, True),
    CAUTION:      StateGate(0.5, frozenset({BEAR_CALL, IRON_CONDOR, IRON_BUTTERFLY}), False),
    DEFENSIVE:    StateGate(0.25, frozenset({BEAR_CALL}), False),
    CAPITULATION: StateGate(0.0, frozenset(), False),
    RECOVERY:     StateGate(0.5, frozenset({BULL_PUT, IRON_CONDOR, IRON_BUTTERFLY}), True),
}


def effective_strategy(strategy_name: str, sold_option_types: Iterable[str] = ()) -> str:
    """A Mean Reversion Spread is a bull put (sold puts) or a bear call
    (sold calls); gate it by that direction, not by its label."""
    if strategy_name == MEAN_REVERSION:
        types = {t.lower() for t in sold_option_types}
        if types == {"put"}:
            return BULL_PUT
        if types == {"call"}:
            return BEAR_CALL
    return strategy_name


def gate_failure(state: Optional["MarketStateResult"], strategy_name: str,
                 sold_option_types: Iterable[str] = ()) -> Optional[str]:
    """``None`` when the market state allows a new *strategy_name* entry,
    otherwise a ``market_state_<STATE>_blocks_<strategy>`` reason.
    ``state=None`` (overlay disabled) allows everything."""
    if state is None:
        return None
    eff = effective_strategy(strategy_name, sold_option_types)
    if eff in state.gate.allowed_strategies:
        return None
    return f"market_state_{state.state}_blocks_{eff.lower().replace(' ', '_')}"


@dataclass(frozen=True)
class MarketInputs:
    spy_price: Optional[float] = None
    spy_sma20: Optional[float] = None
    spy_sma50: Optional[float] = None
    spy_sma200: Optional[float] = None
    spy_rsi: Optional[float] = None
    vix: Optional[float] = None
    vix3m: Optional[float] = None
    breadth: Optional[float] = None            # fraction of tickers above SMA-50
    breadth_n: int = 0

    @property
    def term_ratio(self) -> Optional[float]:
        if self.vix and self.vix3m and self.vix3m > 0:
            return self.vix / self.vix3m
        return None


@dataclass(frozen=True)
class MarketStateResult:
    state: str
    reasons: Tuple[str, ...]
    inputs: MarketInputs
    gate: StateGate
    as_of_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state, "reasons": list(self.reasons),
            "inputs": {**asdict(self.inputs), "term_ratio": self.inputs.term_ratio},
            "gate": {"size_multiplier": self.gate.size_multiplier,
                     "allowed_strategies": sorted(self.gate.allowed_strategies),
                     "allow_new_csp": self.gate.allow_new_csp},
            "as_of_utc": self.as_of_utc,
        }


def classify_market_state(inp: MarketInputs, prior_state: Optional[str] = None,
                          cfg: MarketStateConfig = MarketStateConfig()) -> MarketStateResult:
    """Deterministic rules, checked in order; the first that matches wins."""
    def result(state: str, reasons: List[str]) -> MarketStateResult:
        return MarketStateResult(state, tuple(reasons), inp, GATES[state])

    p, s20, s50, s200 = inp.spy_price, inp.spy_sma20, inp.spy_sma50, inp.spy_sma200
    if p is None or s50 is None or s200 is None:
        return result(CAUTION, ["spy_data_unavailable — failing safe to CAUTION"])
    vix, tr, br, rsi = inp.vix, inp.term_ratio, inp.breadth, inp.spy_rsi

    cap = []
    if vix is not None and vix >= cfg.capitulation_vix:
        cap.append(f"VIX {vix:.1f} ≥ {cfg.capitulation_vix:g}")
    if tr is not None and tr >= cfg.capitulation_term_ratio:
        cap.append(f"VIX/VIX3M {tr:.2f} ≥ {cfg.capitulation_term_ratio:g}")
    if (rsi is not None and vix is not None and rsi <= cfg.capitulation_rsi
            and vix >= cfg.capitulation_rsi_vix):
        cap.append(f"SPY RSI {rsi:.1f} ≤ {cfg.capitulation_rsi:g} with VIX {vix:.1f}")
    if cap:
        return result(CAPITULATION, cap)

    defn = []
    if p < s200:
        if vix is not None and vix >= cfg.defensive_vix:
            defn.append(f"SPY < SMA-200 and VIX {vix:.1f} ≥ {cfg.defensive_vix:g}")
        if tr is not None and tr >= cfg.defensive_term_ratio:
            defn.append(f"SPY < SMA-200 and VIX/VIX3M {tr:.2f} ≥ {cfg.defensive_term_ratio:g}")
        if br is not None and br < cfg.defensive_breadth:
            defn.append(f"SPY < SMA-200 and breadth {br:.0%} < {cfg.defensive_breadth:.0%}")
    if defn:
        return result(DEFENSIVE, defn)

    if prior_state in (DEFENSIVE, CAPITULATION, RECOVERY):
        if p > s50 and (vix is None or vix < cfg.recovery_exit_vix):
            return result(NORMAL, [f"recovered: SPY > SMA-50 and VIX "
                                   f"{'n/a' if vix is None else f'{vix:.1f}'} < {cfg.recovery_exit_vix:g}"])
        if s20 is not None and p > s20:
            return result(RECOVERY, [f"after {prior_state}: SPY reclaimed SMA-20 "
                                     f"({p:.2f} > {s20:.2f})"])

    caution = []
    if p < s50:
        caution.append(f"SPY {p:.2f} < SMA-50 {s50:.2f}")
    if vix is not None and vix >= cfg.caution_vix:
        caution.append(f"VIX {vix:.1f} ≥ {cfg.caution_vix:g}")
    if tr is not None and tr >= cfg.caution_term_ratio:
        caution.append(f"VIX/VIX3M {tr:.2f} ≥ {cfg.caution_term_ratio:g}")
    if br is not None and br < cfg.caution_breadth:
        caution.append(f"breadth {br:.0%} < {cfg.caution_breadth:.0%}")
    if caution:
        return result(CAUTION, caution)
    if prior_state == CAUTION:
        hold = []
        if p < s50 * (1 + cfg.caution_exit_sma_margin):
            hold.append(f"SPY within {cfg.caution_exit_sma_margin:.0%} of SMA-50")
        if vix is not None and vix >= cfg.caution_vix - cfg.caution_exit_vix_margin:
            hold.append(f"VIX {vix:.1f} not yet < {cfg.caution_vix - cfg.caution_exit_vix_margin:g}")
        if tr is not None and tr > cfg.caution_exit_term_ratio:
            hold.append(f"VIX/VIX3M {tr:.2f} not yet ≤ {cfg.caution_exit_term_ratio:g}")
        if br is not None and br < cfg.caution_exit_breadth:
            hold.append(f"breadth {br:.0%} not yet ≥ {cfg.caution_exit_breadth:.0%}")
        if hold:
            return result(CAUTION, ["hysteresis: " + "; ".join(hold)])
    return result(NORMAL, ["no risk condition met"])


# ---------------------------------------------------------------------------
# Playbook table (backlog §6.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Playbook:
    name: str
    implemented: bool
    reason: str


VOL_HIGH, VOL_LOW = 50.0, 30.0      # volatility-rank buckets (0–100)
RSI_OVERSOLD, RSI_OVERBOUGHT = 30.0, 70.0
_IMPLEMENTED = frozenset({"bull_put", "bear_call", "iron_condor", "mean_reversion"})


def playbook_for(regime: str, vol_rank: Optional[float], rsi: Optional[float]) -> Playbook:
    """Trend × volatility × RSI extreme → playbook. ``vol_rank`` is the
    regime classifier's ``iv_rank`` (a realized-volatility percentile)."""
    r = (regime or "").lower()
    v = vol_rank if vol_rank is not None and not math.isnan(vol_rank) else None
    rich = v is not None and v >= VOL_LOW         # premium worth selling
    vb = ("unknown" if v is None else "high" if v >= VOL_HIGH
          else "mid" if v >= VOL_LOW else "low")

    def pb(name: str) -> Playbook:
        return Playbook(name, name in _IMPLEMENTED,
                        f"{r or 'unknown'} trend, {vb} volatility"
                        + (f", RSI {rsi:.0f}" if rsi is not None else ""))

    if r == "mean_reversion":
        return pb("mean_reversion")
    if r == "bearish" and rsi is not None and rsi < RSI_OVERSOLD:
        return pb("bounce_bull_put" if rich else "wait_for_stabilization")
    if r == "bullish":
        return pb("bull_put" if rich else "call_debit")
    if r == "bearish":
        return pb("bear_call" if rich else "put_debit")
    if r == "sideways":
        return pb("iron_condor" if rich else "calendar")
    return pb("none")


# ---------------------------------------------------------------------------
# I/O — inputs and the per-cycle state snapshot
# ---------------------------------------------------------------------------

def _sma(closes: List[float], n: int) -> Optional[float]:
    return sum(closes[-n:]) / n if len(closes) >= n else None


def _rsi(closes: List[float], n: int = 14) -> Optional[float]:
    if len(closes) <= n:
        return None
    gains = losses = 0.0
    for a, b in zip(closes[-n - 1:-1], closes[-n:]):
        d = b - a
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    if losses == 0:
        return 100.0
    rs = (gains / n) / (losses / n)
    return 100.0 - 100.0 / (1.0 + rs)


def _closes(df: Any) -> List[float]:
    try:
        col = "Close" if "Close" in df.columns else "close"
        return [float(x) for x in df[col].dropna().tolist()]
    except Exception:                                           # noqa: BLE001 — unknown frame shape → no data
        return []


def compute_inputs(fetch_history: Callable[[str], Any], tickers: Iterable[str],
                   fetch_level: Callable[[str], Optional[float]]) -> MarketInputs:
    """Each input fetched in its own try/except (one failed RPC must not
    blank the others). ``fetch_history(ticker)`` returns a daily OHLCV
    frame; ``fetch_level(symbol)`` returns the latest index level."""
    spy: List[float] = []
    try:
        spy = _closes(fetch_history("SPY"))
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("market_state: SPY history unavailable (%s)", exc)
    vix = vix3m = None
    try:
        vix = fetch_level("^VIX")
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("market_state: VIX unavailable (%s)", exc)
    try:
        vix3m = fetch_level("^VIX3M")
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("market_state: VIX3M unavailable (%s)", exc)
    above = n = 0
    for t in tickers:
        try:
            c = _closes(fetch_history(t))
            s = _sma(c, 50)
            if s is not None:
                n += 1
                above += c[-1] > s
        except Exception as exc:                                # noqa: BLE001
            logger.debug("market_state: breadth skip %s (%s)", t, exc)
    return MarketInputs(
        spy_price=spy[-1] if spy else None, spy_sma20=_sma(spy, 20),
        spy_sma50=_sma(spy, 50), spy_sma200=_sma(spy, 200), spy_rsi=_rsi(spy),
        vix=vix, vix3m=vix3m, breadth=(above / n) if n else None, breadth_n=n,
    )


def yfinance_level(symbol: str) -> Optional[float]:
    """Latest level of an index (e.g. ^VIX, ^VIX3M) via yfinance."""
    import yfinance as yf  # type: ignore
    df = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


def write_state(result: MarketStateResult, path: Optional[Path] = None,
                extra: Optional[Dict[str, Any]] = None) -> None:
    path = path or STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({**result.to_dict(), **(extra or {})}, indent=2, default=str))
    tmp.replace(path)


def read_state(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads((path or STATE_PATH).read_text())
        as_of = datetime.fromisoformat(d["as_of_utc"])
        d["age_seconds"] = round((datetime.now(timezone.utc) - as_of).total_seconds())
        return d
    except (OSError, ValueError, KeyError, TypeError):
        return None


# A weekend plus a holiday: Friday's snapshot still gates Tuesday's first
# CSP. Older than this, the agent has not run and the snapshot is ignored.
CSP_PAUSE_MAX_AGE_S = 4 * 24 * 3600


def csp_pause_reason(snapshot: Optional[Dict[str, Any]],
                     max_age_s: float = CSP_PAUSE_MAX_AGE_S) -> Optional[str]:
    """Reason a new cash-secured put is paused by the last market-state
    snapshot (``read_state()``), or ``None``. A missing or stale snapshot
    does not pause (the overlay may be disabled or the agent not running)."""
    if not snapshot:
        return None
    age = snapshot.get("age_seconds")
    if age is not None and age > max_age_s:
        return None
    if (snapshot.get("gate") or {}).get("allow_new_csp", True):
        return None
    return f"market_state_{snapshot.get('state')}_pauses_new_csp"


def hedge_suggestion(beta_notional: float, spy_price: float,
                     coverage: float = 0.5) -> Optional[Dict[str, Any]]:
    """SPY put-spread hedge sized to ``coverage`` × beta-weighted notional:
    long ≈ 3 % OTM, short ≈ 10 % OTM, 30–45 DTE. A suggestion only —
    staged through /propose, never placed automatically."""
    if beta_notional <= 0 or spy_price <= 0:
        return None
    contracts = max(1, math.ceil(coverage * beta_notional / (spy_price * 100)))
    return {
        "instrument": "SPY put debit spread",
        "long_strike": round(spy_price * 0.97),
        "short_strike": round(spy_price * 0.90),
        "dte": "30–45",
        "contracts": contracts,
        "covers": f"{coverage:.0%} of ${beta_notional:,.0f} beta-weighted notional",
    }

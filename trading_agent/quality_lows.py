"""quality_lows.py — quality-at-lows sleeve, shadow scanner (skill 67).

Added 2026-10-26 (backlog §12). Finds strong companies the market sold
on sentiment rather than on a broken business, follows them through a
2–3 week base, and — when the price turns — records which long-option
structure the sleeve *would* buy and at what real chain price. It never
places an order: 4–8 weeks of these shadow rows come before any paper
trade (§12.6).

Once a day, for every ticker in the watchlist plus ``EXTRA_TICKERS``:

1. **Price features** from ~15 months of daily bars (Yahoo): 52-week
   range, where the recent base sits in it, sessions since the 52-week
   low, SMA-20 / SMA-50, RSI-14, ATR now vs a month ago, a higher low,
   30-day realized volatility.
2. **Quality** — the Wheel screen (``fundamentals_screen``) with the
   P/E and beta limits relaxed: a growth name at a low has a high P/E
   and a high beta by construction.
3. **Growth intact** — Schwab revenue growth and margin direction
   (latest quarter vs trailing year) plus analyst estimates
   (``analyst_estimates``): consensus EPS not cut, revisions not
   one-sided down, forward revenue growth positive.
4. **Drop cause** — ``fundamental`` when growth is not intact;
   ``sector`` when the sector ETF fell nearly as far; ``sentiment``
   otherwise; ``unclear`` when inputs are missing. Names near a low that
   pass quality also get ``llm_cause`` — Claude's reading of 30 days of
   news (``drop_cause_llm``), journalled beside the rule, not used by it.
5. **Stage** — ``not_near_low`` → ``watch`` → ``basing`` → ``turn`` →
   ``confirmed`` (or ``excluded``).
6. **Structure** for ``turn`` / ``confirmed``, priced from the data
   server's chains: deep ITM LEAPS call by default; a call debit spread
   up to the 52-week high when implied volatility is rich vs realized;
   a cash-secured put at the base as the patient alternative; nothing
   in the two weeks before earnings. Contracts sized to
   ``max_premium_pct`` of account equity.

Rows go to ``trade_journal/quality_lows.jsonl`` (one per ticker per day).

    python -m trading_agent.quality_lows [--dry-run] [--tickers NFLX,ORCL] [--no-chains] [--no-llm]
    python -m trading_agent.quality_lows --report        # latest day's table
"""
from __future__ import annotations

import argparse
import calendar
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.environ.get("TRADING_AGENT_QUALITY_LOWS",
                                 "trade_journal/quality_lows.jsonl"))
MARKET_STATE_PATH = Path(os.environ.get("TRADING_AGENT_MARKET_STATE",
                                        "trade_journal/market_state.json"))
# Non-tech names screened alongside the (tech-heavy) watchlist, so the
# sleeve can diversify. Override with QUALITY_LOWS_EXTRA_TICKERS.
EXTRA_TICKERS = ("JNJ", "LLY", "PG", "KO", "WMT", "COST", "XOM", "CAT", "HD", "UNH")
REQUEST_PAUSE_S = 0.3            # between data-server calls (Schwab rate limit)

STAGES = ("excluded", "not_near_low", "watch", "basing", "turn", "confirmed")


@dataclass(frozen=True)
class QualityLowsConfig:
    """Sleeve thresholds. Kept out of ``PresetConfig`` on purpose: the
    sleeve is not part of the spread agent's decision path (the same
    precedent as ``WheelScreenConfig`` / ``EvaluatorConfig``)."""
    # Near the low: the recent base must sit in the bottom of the range…
    base_window: int = 30                 # sessions that define "the base"
    base_max_range_pos: float = 0.10      # base low within the bottom 10 % of the 52-week range
    max_range_pos: float = 0.35           # …and the price not already recovered past this
    # Basing
    min_sessions_since_low: int = 10
    min_rsi: float = 40.0
    # Turn
    higher_low_window: int = 10           # recent-low window that must sit above the base low
    # Quality (fundamentals_screen, relaxed for growth names at lows)
    min_market_cap: float = 10e9
    max_pe_ratio: float = 120.0
    max_beta: float = 3.0
    min_net_margin_pct: float = 8.0
    min_roe_pct: float = 10.0
    min_avg_volume_10d: float = 1_000_000
    # Growth intact
    min_rev_growth_pct: float = 0.0       # trailing-12-month revenue growth
    max_margin_drop_pp: float = 3.0       # net margin, latest quarter vs TTM
    max_eps_cut_30d_pct: float = 3.0      # consensus EPS cut over 30 days (cur / next year)
    max_net_down_revisions: int = 5       # down − up analyst revisions, 30 days
    # Drop cause
    sector_share: float = 0.6             # sector ETF drawdown ≥ this × the stock's → "sector"
    # Structures
    iv_rich_ratio: float = 1.3            # iv / realized vol above this → debit spread
    earnings_blackout_days: int = 14
    leaps_dte: Tuple[int, int] = (270, 550)
    leaps_delta: float = 0.75
    spread_dte: Tuple[int, int] = (120, 270)
    spread_long_delta: float = 0.60
    csp_dte: Tuple[int, int] = (30, 60)
    # Sizing
    max_premium_pct: float = 0.02         # premium per position, share of equity (= max loss)
    max_csp_cash_pct: float = 0.15        # cash secured by one put, share of equity
    max_per_sector: int = 2
    # News-reading drop-cause label (drop_cause_llm, §2.4)
    llm_max_names: int = 10               # claude CLI calls per run, most advanced stage first


DEFAULT_CONFIG = QualityLowsConfig()


# ── Price features (pure) ─────────────────────────────────────────────

def _rsi(closes: List[float], n: int = 14) -> Optional[float]:
    """Wilder RSI of the last close; None with too little data."""
    if len(closes) < n + 1:
        return None
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    ag, al = gains / n, losses / n
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def _atr(highs, lows, closes, end: int, n: int = 14) -> Optional[float]:
    """Mean true range of the ``n`` sessions ending at index ``end``."""
    if end < n:
        return None
    trs = []
    for i in range(end - n + 1, end + 1):
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    return sum(trs) / n


def price_features(highs: List[float], lows: List[float], closes: List[float],
                   cfg: QualityLowsConfig = DEFAULT_CONFIG) -> Optional[Dict[str, Any]]:
    """Features of the last bar from parallel daily high / low / close
    lists (oldest first, ≥ 252 sessions). None with too little data."""
    n = len(closes)
    if n < 252 or len(highs) != n or len(lows) != n:
        return None
    hi52, lo52 = max(highs[-252:]), min(lows[-252:])
    if hi52 <= lo52:
        return None
    close = closes[-1]
    span = hi52 - lo52
    window = lows[-252:]
    # The most recent session that set the 52-week low.
    low_idx = max(i for i, v in enumerate(window) if v == lo52)
    sessions_since_low = 251 - low_idx
    base_low = min(lows[-cfg.base_window:])
    recent_low = min(lows[-cfg.higher_low_window:])
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50
    atr_now = _atr(highs, lows, closes, n - 1)
    atr_then = _atr(highs, lows, closes, n - 21)
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(n - 30, n) if closes[i - 1] > 0]
    mean = sum(rets) / len(rets)
    rv30 = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1) * 252)
    return {
        "close": round(close, 4), "high_52w": round(hi52, 4), "low_52w": round(lo52, 4),
        "range_pos": round((close - lo52) / span, 4),
        "base_low": round(base_low, 4),
        "base_range_pos": round((base_low - lo52) / span, 4),
        "pct_above_low": round((close / lo52 - 1) * 100, 2),
        "drawdown_pct": round((close / hi52 - 1) * 100, 2),
        "sessions_since_low": sessions_since_low,
        "higher_low": recent_low > lo52 and sessions_since_low >= cfg.higher_low_window,
        "sma20": round(sma20, 4), "sma50": round(sma50, 4),
        "rsi14": None if _rsi(closes) is None else round(_rsi(closes), 2),
        "atr_narrowing": (atr_now is not None and atr_then is not None and atr_now < atr_then),
        "rv30": round(rv30, 4),
    }


# ── Quality, growth, cause, stage (pure) ──────────────────────────────

def quality_check(fund: Optional[Dict[str, Any]], cfg: QualityLowsConfig = DEFAULT_CONFIG
                  ) -> Tuple[bool, List[str]]:
    """The Wheel quality screen with the long-term limits."""
    from trading_agent.fundamentals_screen import WheelScreenConfig, screen_fundamentals
    wcfg = WheelScreenConfig(min_market_cap=cfg.min_market_cap, max_pe_ratio=cfg.max_pe_ratio,
                             min_net_margin_pct=cfg.min_net_margin_pct, min_roe_pct=cfg.min_roe_pct,
                             max_beta=cfg.max_beta, min_avg_volume_10d=cfg.min_avg_volume_10d)
    r = screen_fundamentals("", fund, wcfg)
    return r.passed, list(r.reasons)


def growth_check(fund: Optional[Dict[str, Any]], est: Optional[Dict[str, Any]],
                 cfg: QualityLowsConfig = DEFAULT_CONFIG) -> Tuple[Optional[bool], List[str]]:
    """(intact, reasons). ``intact`` is None when neither Schwab growth
    nor analyst data exists; reasons lists every failed or missing check."""
    fund, est = fund or {}, est or {}
    reasons: List[str] = []
    known = 0
    rev = fund.get("rev_change_ttm_pct")
    if rev is not None:
        known += 1
        if rev < cfg.min_rev_growth_pct:
            reasons.append(f"revenue_ttm {rev:+.1f}%")
    else:
        reasons.append("missing:rev_change_ttm_pct")
    mrq, ttm = fund.get("net_profit_margin_mrq"), fund.get("net_profit_margin_ttm")
    if mrq is not None and ttm is not None:
        known += 1
        if mrq - ttm < -cfg.max_margin_drop_pp:
            reasons.append(f"net_margin_falling {mrq - ttm:+.1f}pp")
    cut_known = False
    for tag in ("cur_year", "next_year"):
        v = est.get(f"eps_{tag}_rev_30d_pct")
        if v is not None:
            known += 1
            cut_known = True
            if v < -cfg.max_eps_cut_30d_pct:
                reasons.append(f"eps_{tag}_cut {v:+.1f}%/30d")
    # The size of the consensus change decides; the count of analysts
    # revising is only the fallback when the size is unknown (NFLX on
    # 2026-10-25: 3 up / 10 down but next-year EPS −1.2 % after a 44 %
    # fall — that is sentiment, not a business turning).
    up, down = est.get("revisions_up_30d"), est.get("revisions_down_30d")
    if not cut_known and up is not None and down is not None:
        known += 1
        if down - up > cfg.max_net_down_revisions:
            reasons.append(f"revisions {up}up/{down}down")
    fwd = est.get("rev_growth_next_year_pct")
    if fwd is not None:
        known += 1
        if fwd <= 0:
            reasons.append(f"rev_growth_next_year {fwd:+.1f}%")
    if known == 0:
        return None, reasons
    failed = [r for r in reasons if not r.startswith("missing:")]
    return not failed, reasons


def drop_cause(intact: Optional[bool], stock_drawdown_pct: Optional[float],
               sector_drawdown_pct: Optional[float], cfg: QualityLowsConfig = DEFAULT_CONFIG) -> str:
    """``fundamental`` | ``sector`` | ``sentiment`` | ``unclear``. Drawdowns
    are negative percents from the 52-week high."""
    if intact is False:
        return "fundamental"
    if intact is None or stock_drawdown_pct is None:
        return "unclear"
    if (sector_drawdown_pct is not None and stock_drawdown_pct < 0
            and sector_drawdown_pct <= cfg.sector_share * stock_drawdown_pct):
        return "sector"
    return "sentiment"


def stage(pf: Optional[Dict[str, Any]], quality_ok: bool, cause: str,
          cfg: QualityLowsConfig = DEFAULT_CONFIG) -> str:
    """Where the ticker is in watch → basing → turn → confirmed."""
    if pf is None:
        return "excluded"
    near = pf["base_range_pos"] <= cfg.base_max_range_pos and pf["range_pos"] <= cfg.max_range_pos
    if not near:
        return "not_near_low"
    if not quality_ok or cause == "fundamental":
        return "excluded"
    rsi = pf.get("rsi14")
    basing = (pf["sessions_since_low"] >= cfg.min_sessions_since_low
              and rsi is not None and rsi > cfg.min_rsi)
    if not basing:
        return "watch"
    if not (pf["close"] > pf["sma20"] and pf["higher_low"]):
        return "basing"
    return "confirmed" if pf["close"] > pf["sma50"] else "turn"


# ── Structure choice and pricing ──────────────────────────────────────

def choose_structure(iv: Optional[float], rv: Optional[float], days_to_earnings: Optional[int],
                     cfg: QualityLowsConfig = DEFAULT_CONFIG) -> str:
    """``wait_earnings`` | ``call_debit_spread`` | ``leaps_call``."""
    if days_to_earnings is not None and 0 <= days_to_earnings <= cfg.earnings_blackout_days:
        return "wait_earnings"
    if iv and rv and rv > 0 and iv / rv > cfg.iv_rich_ratio:
        return "call_debit_spread"
    return "leaps_call"


def monthly_expirations(today: date, dte_range: Tuple[int, int]) -> List[date]:
    """Third Fridays inside ``dte_range`` days (where LEAPS and long-dated
    monthlies list), nearest the middle of the range first."""
    lo, hi = dte_range
    out = []
    y, m = today.year, today.month
    for _ in range(24):
        c = calendar.Calendar()
        fridays = [d for d in c.itermonthdates(y, m) if d.month == m and d.weekday() == 4]
        third = fridays[2]
        if lo <= (third - today).days <= hi:
            out.append(third)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    mid = (lo + hi) / 2
    return sorted(out, key=lambda d: abs((d - today).days - mid))


def _quote(c: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        bid, ask = float(c.get("bid") or 0), float(c.get("ask") or 0)
    except (TypeError, ValueError):
        return None
    return (bid, ask) if bid > 0 and ask >= bid else None


def _nearest_delta(contracts: Iterable[Dict[str, Any]], target: float) -> Optional[Dict[str, Any]]:
    best, gap = None, None
    for c in contracts:
        if c.get("delta") is None or _quote(c) is None:
            continue
        g = abs(abs(float(c["delta"])) - target)
        if gap is None or g < gap:
            best, gap = c, g
    return best


def _leg(c: Dict[str, Any]) -> Dict[str, Any]:
    bid, ask = _quote(c)
    return {"symbol": c.get("symbol"), "strike": float(c.get("strike")), "bid": bid, "ask": ask,
            "delta": round(float(c.get("delta")), 3),
            "iv": None if c.get("iv") is None else round(float(c["iv"]), 4)}


def price_leaps(calls: List[Dict[str, Any]], spot: float, dte: int,
                cfg: QualityLowsConfig = DEFAULT_CONFIG) -> Optional[Dict[str, Any]]:
    """Deep ITM call nearest ``leaps_delta``; cost at the ask (natural)."""
    c = _nearest_delta(calls, cfg.leaps_delta)
    if c is None:
        return None
    leg = _leg(c)
    cost = leg["ask"] * 100
    intrinsic = max(spot - leg["strike"], 0.0)
    extrinsic = max(leg["ask"] - intrinsic, 0.0)
    return {"legs": [leg], "cost_per_contract": round(cost, 2),
            "breakeven": round(leg["strike"] + leg["ask"], 2),
            "extrinsic_pct_of_spot": round(extrinsic / spot * 100, 2) if spot else None,
            "extrinsic_per_month": round(extrinsic / max(dte / 30.4, 1.0), 3),
            "spread_pct": round((leg["ask"] - leg["bid"]) / ((leg["ask"] + leg["bid"]) / 2) * 100, 2)}


def price_debit_spread(calls: List[Dict[str, Any]], target_high: float,
                       cfg: QualityLowsConfig = DEFAULT_CONFIG) -> Optional[Dict[str, Any]]:
    """Long call nearest ``spread_long_delta``; short call at the listed
    strike nearest the 52-week high above the long strike. Debit at
    natural (long ask − short bid)."""
    long_c = _nearest_delta(calls, cfg.spread_long_delta)
    if long_c is None:
        return None
    k_long = float(long_c["strike"])
    shorts = [c for c in calls if _quote(c) is not None and float(c.get("strike") or 0) > k_long]
    if not shorts:
        return None
    short_c = min(shorts, key=lambda c: abs(float(c["strike"]) - target_high))
    lg, sh = _leg(long_c), _leg(short_c)
    debit = lg["ask"] - sh["bid"]
    width = sh["strike"] - lg["strike"]
    if debit <= 0 or width <= 0:
        return None
    return {"legs": [lg, sh], "cost_per_contract": round(debit * 100, 2),
            "width": round(width, 2), "max_value": round(width * 100, 2),
            "reward_risk": round((width - debit) / debit, 2),
            "breakeven": round(lg["strike"] + debit, 2)}


def price_fitting_spread(calls: List[Dict[str, Any]], budget: Optional[float],
                         cfg: QualityLowsConfig = DEFAULT_CONFIG) -> Optional[Dict[str, Any]]:
    """The widest call debit spread (same long leg as ``price_debit_spread``)
    whose natural debit fits ``budget`` per contract with reward/risk ≥ 1.
    On a ~$29k account the spread up to the 52-week high rarely fits the
    2 % cap (NFLX $717, ORCL $1,993 on 2026-10-25); this is the one that does."""
    if not budget:
        return None
    long_c = _nearest_delta(calls, cfg.spread_long_delta)
    if long_c is None:
        return None
    lg = _leg(long_c)
    best = None
    for c in calls:
        if _quote(c) is None or float(c.get("strike") or 0) <= lg["strike"]:
            continue
        sh = _leg(c)
        debit, width = lg["ask"] - sh["bid"], sh["strike"] - lg["strike"]
        if debit <= 0 or debit * 100 > budget or (width - debit) / debit < 1.0:
            continue
        if best is None or width > best["width"]:
            best = {"legs": [lg, sh], "cost_per_contract": round(debit * 100, 2),
                    "width": round(width, 2), "max_value": round(width * 100, 2),
                    "reward_risk": round((width - debit) / debit, 2),
                    "breakeven": round(lg["strike"] + debit, 2)}
    return best


def price_csp(puts: List[Dict[str, Any]], base_low: float) -> Optional[Dict[str, Any]]:
    """Put at the listed strike nearest the base low (at or below spot's
    base); credit at the bid. Cash secured = strike × 100."""
    cands = [c for c in puts if _quote(c) is not None and float(c.get("strike") or 0) > 0]
    if not cands:
        return None
    c = min(cands, key=lambda c: abs(float(c["strike"]) - base_low))
    leg = _leg(c)
    cash = leg["strike"] * 100
    return {"legs": [leg], "credit_per_contract": round(leg["bid"] * 100, 2),
            "cash_secured": round(cash, 2),
            "yield_pct": round(leg["bid"] * 100 / cash * 100, 2) if cash else None,
            "effective_entry": round(leg["strike"] - leg["bid"], 2)}


def size_contracts(cost_per_contract: Optional[float], equity: Optional[float],
                   cfg: QualityLowsConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Contracts that fit ``max_premium_pct`` of equity (premium = max loss)."""
    if not cost_per_contract or not equity or cost_per_contract <= 0:
        return {"budget": None, "contracts": 0, "fits": False}
    budget = equity * cfg.max_premium_pct
    n = int(budget // cost_per_contract)
    return {"budget": round(budget, 2), "contracts": n, "fits": n >= 1}


def _chain(get, ticker: str, exp: date, kind: str, pause_s: float) -> List[Dict[str, Any]]:
    body = get(f"/chain/{ticker}?expiration={exp.isoformat()}&option_type={kind}") or {}
    time.sleep(pause_s)
    return body.get("contracts") or []


def _first_chain(get, ticker: str, today: date, dte_range: Tuple[int, int], kind: str,
                 pause_s: float, max_tries: int = 6) -> Tuple[Optional[date], List[Dict[str, Any]]]:
    for exp in monthly_expirations(today, dte_range)[:max_tries]:
        contracts = _chain(get, ticker, exp, kind, pause_s)
        if contracts:
            return exp, contracts
    return None, []


def structure_quotes(ticker: str, pf: Dict[str, Any], choice: str, today: date, get,
                     equity: Optional[float], cfg: QualityLowsConfig = DEFAULT_CONFIG,
                     pause_s: float = REQUEST_PAUSE_S) -> Dict[str, Any]:
    """Price the chosen structure and the CSP alternative from real chains."""
    out: Dict[str, Any] = {}
    spot = pf["close"]
    if choice in ("leaps_call", "call_debit_spread"):
        # Both are always priced so the shadow log can compare them.
        exp, calls = _first_chain(get, ticker, today, cfg.leaps_dte, "call", pause_s)
        if exp:
            q = price_leaps(calls, spot, (exp - today).days, cfg)
            if q:
                out["leaps_call"] = {"expiration": exp.isoformat(), "dte": (exp - today).days, **q,
                                     **size_contracts(q["cost_per_contract"], equity, cfg)}
        exp, calls = _first_chain(get, ticker, today, cfg.spread_dte, "call", pause_s)
        if exp:
            q = price_debit_spread(calls, pf["high_52w"], cfg)
            if q:
                out["call_debit_spread"] = {"expiration": exp.isoformat(), "dte": (exp - today).days,
                                            **q, **size_contracts(q["cost_per_contract"], equity, cfg)}
            budget = size_contracts(1.0, equity, cfg)["budget"]
            q = price_fitting_spread(calls, budget, cfg)
            if q:
                out["call_debit_spread_fit"] = {"expiration": exp.isoformat(),
                                                "dte": (exp - today).days, **q,
                                                **size_contracts(q["cost_per_contract"], equity, cfg)}
    exp, puts = _first_chain(get, ticker, today, cfg.csp_dte, "put", pause_s)
    if exp:
        q = price_csp(puts, pf["base_low"])
        if q:
            cap = equity * cfg.max_csp_cash_pct if equity else None
            out["cash_secured_put"] = {"expiration": exp.isoformat(), "dte": (exp - today).days, **q,
                                       "cash_cap": None if cap is None else round(cap, 2),
                                       "fits": bool(cap and q["cash_secured"] <= cap)}
    return out


# ── Collection ────────────────────────────────────────────────────────

def _bars(ticker: str, sessions: int = 320) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """Daily high / low / close from Yahoo, oldest first."""
    import yfinance as yf
    df = yf.Ticker(ticker).history(period="2y", auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df.tail(sessions)
    return list(map(float, df["High"])), list(map(float, df["Low"])), list(map(float, df["Close"]))


def _account_equity(path: Path = MARKET_STATE_PATH) -> Optional[float]:
    try:
        v = json.loads(path.read_text()).get("account_balance")
        return float(v) if v else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def sector_etf_for(ticker: str) -> Optional[str]:
    """The Select Sector SPDR for ``ticker``'s sector (``sector_map``)."""
    from trading_agent.sector_map import TICKER_SECTOR_MAP, sector_for
    sec = sector_for(ticker)
    for etf in ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"):
        if TICKER_SECTOR_MAP.get(etf) == sec:
            return etf
    return None


@dataclass
class Deps:
    """I/O seams; tests replace them."""
    bars: Callable[[str], Optional[Tuple[List[float], List[float], List[float]]]]
    fundamentals: Callable[[str], Optional[Dict[str, Any]]]
    estimates: Callable[[str, Optional[float]], Optional[Dict[str, Any]]]
    days_to_earnings: Callable[[str, date], Optional[int]]
    get: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None   # data-server GET; None = skip chains
    equity: Optional[float] = None
    classify: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None  # drop_cause_llm; None = skip


def evaluate_ticker(ticker: str, today: date, deps: Deps,
                    sector_drawdowns: Dict[str, Optional[float]],
                    cfg: QualityLowsConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """One ticker's shadow row."""
    row: Dict[str, Any] = {"date": today.isoformat(), "ticker": ticker,
                           "as_of_utc": datetime.now(timezone.utc).isoformat()}
    bars = deps.bars(ticker)
    pf = price_features(*bars, cfg=cfg) if bars else None
    row["price"] = pf
    if pf is None:
        row.update(stage="excluded", reasons=["missing:price_history"])
        return row
    if pf["base_range_pos"] > cfg.base_max_range_pos or pf["range_pos"] > cfg.max_range_pos:
        # Not near a low: skip the network calls for fundamentals / chains.
        row.update(stage="not_near_low", reasons=[])
        return row
    fund = deps.fundamentals(ticker) or {}
    est = deps.estimates(ticker, pf["close"])
    q_ok, q_reasons = quality_check(fund, cfg)
    intact, g_reasons = growth_check(fund, est, cfg)
    etf = sector_etf_for(ticker)
    cause = drop_cause(intact, pf["drawdown_pct"], sector_drawdowns.get(etf) if etf else None, cfg)
    st = stage(pf, q_ok, cause, cfg)
    row.update({
        "stage": st, "quality_ok": q_ok, "growth_intact": intact, "cause": cause,
        "sector_etf": etf, "sector_drawdown_pct": sector_drawdowns.get(etf) if etf else None,
        "reasons": q_reasons + g_reasons,
        "fundamentals": {k: fund.get(k) for k in (
            "market_cap", "pe_ratio", "beta", "rev_change_ttm_pct", "rev_change_year_pct",
            "net_profit_margin_ttm", "net_profit_margin_mrq", "gross_margin_ttm",
            "gross_margin_mrq", "roe", "eps_change_pct_ttm")},
        "estimates": est,
    })
    if st in ("turn", "confirmed"):
        dte_e = deps.days_to_earnings(ticker, today)
        iv = None
        choice = None
        quotes: Dict[str, Any] = {}
        if deps.get is not None:
            quotes = structure_quotes(ticker, pf, "leaps_call", today, deps.get, deps.equity, cfg)
            sp = quotes.get("call_debit_spread") or {}
            long_iv = (sp.get("legs") or [{}])[0].get("iv") if sp else None
            iv = long_iv
        choice = choose_structure(iv, pf["rv30"], dte_e, cfg)
        row.update({"days_to_earnings": dte_e, "iv": iv, "iv_rv": (round(iv / pf["rv30"], 2)
                    if iv and pf["rv30"] else None), "structure": choice, "quotes": quotes,
                    "affordable": affordable_structure(choice, quotes)})
    return row


def affordable_structure(choice: str, quotes: Dict[str, Any]) -> Optional[str]:
    """The first priced structure that fits its size cap, preferring the
    chosen one, then the budget-fitting spread, then the cash-secured put.
    None when nothing fits (or for ``wait_earnings``)."""
    if choice == "wait_earnings":
        return None
    for k in (choice, "call_debit_spread_fit", "cash_secured_put"):
        if (quotes.get(k) or {}).get("fits"):
            return k
    return None


def apply_sector_cap(rows: List[Dict[str, Any]], cfg: QualityLowsConfig = DEFAULT_CONFIG) -> None:
    """Mark ``turn`` / ``confirmed`` rows beyond ``max_per_sector`` per
    sector (strongest revisions first) with ``sector_capped``."""
    from trading_agent.sector_map import sector_for
    ready = [r for r in rows if r.get("stage") in ("turn", "confirmed")]
    def key(r):
        e = r.get("estimates") or {}
        return -(e.get("eps_next_year_rev_30d_pct") or 0.0)
    seen: Dict[str, int] = {}
    for r in sorted(ready, key=key):
        s = sector_for(r["ticker"])
        seen[s] = seen.get(s, 0) + 1
        r["sector_capped"] = seen[s] > cfg.max_per_sector


def wants_llm_label(row: Dict[str, Any]) -> bool:
    """Near a low and passing quality — whatever the rule-based cause, so
    a ``fundamental`` call by the numbers gets a second reading too."""
    return row.get("stage") not in (None, "not_near_low") and bool(row.get("quality_ok"))


def add_llm_labels(rows: List[Dict[str, Any]], classify: Callable[[Dict[str, Any]], Dict[str, Any]],
                   cfg: QualityLowsConfig = DEFAULT_CONFIG) -> None:
    """Set ``llm_cause`` on up to ``llm_max_names`` rows, most advanced
    stage first. Journal-only: the stage stays rule-based."""
    order = {s: i for i, s in enumerate(reversed(STAGES))}
    todo = sorted((r for r in rows if wants_llm_label(r)),
                  key=lambda r: (order.get(r.get("stage"), 9), r["ticker"]))
    for r in todo[:cfg.llm_max_names]:
        try:
            r["llm_cause"] = classify(r)
        except Exception as exc:                       # noqa: BLE001 — a label must not stop the scan
            logger.warning("quality lows: drop-cause label for %s failed (%s)", r["ticker"], exc)
            r["llm_cause"] = {"label": None, "error": str(exc)[:300]}
    for r in todo[cfg.llm_max_names:]:
        r["llm_cause"] = {"label": None, "error": "skipped: llm_max_names"}


def recorded_today(path: Path, day: str) -> set:
    out = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("date") == day:
                    out.add(r.get("ticker"))
    except OSError:
        pass
    return out


def collect(tickers: Iterable[str], deps: Deps, today: Optional[date] = None,
            path: Optional[Path] = None, dry_run: bool = False,
            cfg: QualityLowsConfig = DEFAULT_CONFIG) -> List[Dict[str, Any]]:
    """Evaluate every ticker not yet recorded today; append the rows
    (unless ``dry_run``) and return them."""
    today = today or date.today()
    path = Path(path or STORE_PATH)
    done = set() if dry_run else recorded_today(path, today.isoformat())
    tickers = [t for t in dict.fromkeys(tickers) if t not in done]
    sector_dd: Dict[str, Optional[float]] = {}
    for etf in sorted({e for e in (sector_etf_for(t) for t in tickers) if e}):
        try:
            b = deps.bars(etf)
            pf = price_features(*b, cfg=cfg) if b else None
            sector_dd[etf] = pf["drawdown_pct"] if pf else None
        except Exception as exc:                       # noqa: BLE001 — a missing ETF only blanks the sector comparison
            logger.warning("quality lows: sector %s failed (%s)", etf, exc)
            sector_dd[etf] = None
    rows = []
    for t in tickers:
        try:
            rows.append(evaluate_ticker(t, today, deps, sector_dd, cfg))
        except Exception as exc:                       # noqa: BLE001 — one ticker must not stop the rest
            logger.warning("quality lows: %s failed (%s)", t, exc)
    apply_sector_cap(rows, cfg)
    if deps.classify is not None:
        add_llm_labels(rows, deps.classify, cfg)
    if not dry_run and rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
    return rows


def universe() -> List[str]:
    """Watchlist + ``QUALITY_LOWS_EXTRA_TICKERS`` (default ``EXTRA_TICKERS``);
    broad-market ETFs dropped (no fundamentals, no "story")."""
    from trading_agent.watchlist_store import load_watchlist
    try:
        wl = load_watchlist().symbols()
    except Exception as exc:                           # noqa: BLE001 — no watchlist → extras only
        logger.warning("quality lows: watchlist unavailable (%s)", exc)
        wl = []
    extra_env = os.environ.get("QUALITY_LOWS_EXTRA_TICKERS")
    extra = ([t.strip().upper() for t in extra_env.split(",") if t.strip()]
             if extra_env is not None else list(EXTRA_TICKERS))
    from trading_agent.sector_map import sector_for
    return [t for t in dict.fromkeys(wl + extra) if sector_for(t) != "Broad Market"]


def report(path: Path = STORE_PATH) -> List[str]:
    """Lines for the latest recorded day: everything past ``not_near_low``."""
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except OSError:
        return ["no quality-lows rows yet"]
    if not rows:
        return ["no quality-lows rows yet"]
    day = max(r["date"] for r in rows)
    latest = {r["ticker"]: r for r in rows if r["date"] == day}
    order = {s: i for i, s in enumerate(reversed(STAGES))}
    lines = [f"quality-at-lows {day}: {len(latest)} tickers"]
    for r in sorted(latest.values(), key=lambda r: (order.get(r.get("stage"), 9), r["ticker"])):
        if r.get("stage") == "not_near_low":
            continue
        p = r.get("price") or {}
        e = r.get("estimates") or {}
        f = r.get("fundamentals") or {}
        line = (f"{r['ticker']:6} {r.get('stage', '?'):10} cause={r.get('cause', '-'):11} "
                f"pos={p.get('range_pos')} low+{p.get('pct_above_low')}% "
                f"since_low={p.get('sessions_since_low')} rsi={p.get('rsi14')} "
                f"rev_ttm={f.get('rev_change_ttm_pct')} eps_rev30={e.get('eps_next_year_rev_30d_pct')} "
                f"revs={e.get('revisions_up_30d')}/{e.get('revisions_down_30d')}")
        llm = r.get("llm_cause") or {}
        if llm.get("label"):
            conf = llm.get("confidence")
            line += f" llm={llm['label']}" + (f"({conf:.2f})" if isinstance(conf, (int, float)) else "")
        elif llm.get("error"):
            line += " llm=error"
        if r.get("structure"):
            q = (r.get("quotes") or {}).get(r["structure"]) or {}
            line += (f" → {r['structure']} cost={q.get('cost_per_contract')} "
                     f"contracts={q.get('contracts')} affordable={r.get('affordable')}"
                     f"{' CAPPED' if r.get('sector_capped') else ''}")
        if r.get("reasons"):
            line += "  [" + "; ".join(x for x in r["reasons"] if not x.startswith("missing:")) + "]"
        lines.append(line)
        if llm.get("summary"):
            lines.append(f"{'':6} why: {llm['summary']}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Quality-at-lows sleeve shadow scan (skill 67)")
    ap.add_argument("--tickers", default="", help="comma list; default watchlist + extras")
    ap.add_argument("--dry-run", action="store_true", help="print, do not append")
    ap.add_argument("--no-chains", action="store_true", help="skip option-chain pricing")
    ap.add_argument("--report", action="store_true", help="print the latest day and exit")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the claude-CLI drop-cause label (also QUALITY_LOWS_LLM_CAUSE=false)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.report:
        print("\n".join(report()))
        return 0
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    from trading_agent.analyst_estimates import get_estimates
    from trading_agent.calendar_utils import is_trading_day
    from trading_agent.earnings_calendar import next_earnings_date
    from trading_agent.mcp.tools.market import _data_server_get
    today = date.today()
    if not is_trading_day(today) and not args.dry_run:
        logger.info("quality lows: %s is not a trading day — nothing to record", today)
        return 0
    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               if args.tickers else universe())

    def fundamentals(t):
        # One retry: a cold Schwab instruments call can outlast the
        # client's 5 s timeout (NFLX on 2026-10-25), the second is cached.
        for _ in range(2):
            body = _data_server_get(f"/fundamentals/{t}") or {}
            if body.get("fundamentals"):
                return body["fundamentals"]
            time.sleep(1.0)
        return None

    def days_to_earnings(t, d):
        nd = next_earnings_date(t, today=d)
        return None if nd is None else (nd - d).days

    deps = Deps(bars=_bars, fundamentals=fundamentals,
                estimates=lambda t, spot: get_estimates(t, spot=spot),
                days_to_earnings=days_to_earnings,
                get=None if args.no_chains else _data_server_get,
                equity=_account_equity(), classify=None)
    llm_on = os.environ.get("QUALITY_LOWS_LLM_CAUSE", "true").strip().lower() not in ("0", "false", "no", "off")
    if llm_on and not args.no_llm:
        from trading_agent.drop_cause_llm import classify
        deps.classify = classify
    rows = collect(tickers, deps, today, dry_run=args.dry_run)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.get("stage", "?")] = counts.get(r.get("stage", "?"), 0) + 1
    logger.info("quality lows: %d tickers %s%s", len(rows), counts, " (dry run)" if args.dry_run else "")
    if args.dry_run:
        for r in rows:
            if r.get("stage") not in ("not_near_low",):
                logger.info("%s", json.dumps(r, default=str)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

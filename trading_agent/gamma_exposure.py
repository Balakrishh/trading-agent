"""gamma_exposure.py — daily dealer gamma estimate for the index ETFs (skill 68).

Added 2026-10-27, record-only. Options dealers hedge: when they are net
long gamma they sell rallies and buy dips (moves get damped); net short,
they chase moves (moves get amplified). That bears directly on condors,
butterflies and short strangles, but no free history of it exists — so,
as with the vol surface (skill 66), the agent records its own and the
rows are judged against next-day ranges after a few months.

Convention (the usual "naive GEX"): dealers are long the calls and short
the puts customers trade, so per contract

    GEX = Γ × open interest × 100 × S² × 0.01      (calls +, puts −)

— dollars of delta change per 1 % move. Summed over every expiry within
``MAX_DTE`` calendar days. ``flip`` is the spot level where the net
re-computed (Black–Scholes Γ at that spot, each contract's own IV) changes
sign — under it, dealers are short gamma. ``call_wall`` / ``put_wall``
are the strikes with the largest call / put GEX.

Chains come from the data server (``/chain``, with ``open_interest``
since 2026-10-27) so the collector never opens a Schwab session. Runs at
16:05 ET (``trading-agent-gamma.timer``): the agent has stopped for the
day, so Schwab's rate budget is free, and SPY / QQQ / IWM options trade
until 16:15, so quotes and IVs are still live. Open interest is the
prior close's (OCC publishes it overnight). The same run refreshes the
macro calendar weekly.

Row → ``trade_journal/gamma_exposure.jsonl`` (one per ticker per day)::

    {"date", "ticker", "spot", "as_of_utc", "expiries", "contracts",
     "net_gex", "call_gex", "put_gex", "net_gex_pct_spot_oi",
     "flip", "flip_pct", "call_wall", "put_wall", "regime"}

    python -m trading_agent.gamma_exposure [--dry-run] [--tickers SPY,QQQ,IWM]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.environ.get("TRADING_AGENT_GAMMA",
                                 "trade_journal/gamma_exposure.jsonl"))
DEFAULT_TICKERS = ("SPY", "QQQ", "IWM")
MAX_DTE = 35
REQUEST_PAUSE_S = 0.4
FLIP_RANGE = 0.08                # search the flip within ±8 % of spot
FLIP_STEP = 0.0025
MACRO_REFRESH_DAYS = 7


def contract_gex(c: Dict[str, Any], spot: float) -> Optional[float]:
    """Signed GEX of one contract ($ delta change per 1 % move), None when
    gamma or open interest is missing."""
    try:
        g, oi = float(c.get("gamma") or 0), float(c.get("open_interest") or 0)
    except (TypeError, ValueError):
        return None
    if g <= 0 or oi <= 0 or spot <= 0:
        return None
    sign = 1.0 if str(c.get("type", "")).lower() == "call" else -1.0
    return sign * g * oi * 100 * spot * spot * 0.01


def bs_gamma(spot: float, strike: float, t_years: float, iv: float) -> float:
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * t_years) / (iv * math.sqrt(t_years))
    return math.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * spot * iv * math.sqrt(t_years))


def net_gex_at(contracts: List[Dict[str, Any]], spot: float, today: date) -> float:
    """Net GEX if the underlying were at ``spot`` (Γ re-computed)."""
    total = 0.0
    for c in contracts:
        try:
            k, iv = float(c["strike"]), float(c.get("iv") or 0)
            oi = float(c.get("open_interest") or 0)
            t = max((date.fromisoformat(str(c["expiration"])[:10]) - today).days, 0.5) / 365.0
        except (KeyError, TypeError, ValueError):
            continue
        if oi <= 0 or iv <= 0:
            continue
        sign = 1.0 if str(c.get("type", "")).lower() == "call" else -1.0
        total += sign * bs_gamma(spot, k, t, iv) * oi * 100 * spot * spot * 0.01
    return total


def flip_level(contracts: List[Dict[str, Any]], spot: float, today: date,
               rng: float = FLIP_RANGE, step: float = FLIP_STEP) -> Optional[float]:
    """Spot level nearest the current one where net GEX changes sign."""
    n = int(rng / step)
    grid = [spot * (1 + i * step) for i in range(-n, n + 1)]
    vals = [net_gex_at(contracts, s, today) for s in grid]
    best = None
    for (s0, v0), (s1, v1) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
        if v0 == 0 or (v0 < 0) != (v1 < 0):
            x = s0 if v1 == v0 else s0 + (s1 - s0) * (-v0) / (v1 - v0)
            if best is None or abs(x - spot) < abs(best - spot):
                best = x
    return round(best, 2) if best is not None else None


def summarize(ticker: str, contracts: List[Dict[str, Any]], spot: float,
              today: date) -> Optional[Dict[str, Any]]:
    call = put = 0.0
    by_strike: Dict[tuple, float] = {}
    n = 0
    oi_total = 0.0
    for c in contracts:
        g = contract_gex(c, spot)
        if g is None:
            continue
        n += 1
        oi_total += float(c.get("open_interest") or 0)
        if g > 0:
            call += g
        else:
            put += g
        key = (str(c.get("type", "")).lower(), float(c.get("strike") or 0))
        by_strike[key] = by_strike.get(key, 0.0) + g
    if n == 0:
        return None
    net = call + put
    flip = flip_level(contracts, spot, today)
    calls = {k[1]: v for k, v in by_strike.items() if k[0] == "call"}
    puts = {k[1]: v for k, v in by_strike.items() if k[0] == "put"}
    return {
        "date": today.isoformat(), "ticker": ticker, "spot": round(spot, 2),
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "expiries": len({str(c.get("expiration"))[:10] for c in contracts}),
        "contracts": n, "open_interest": int(oi_total),
        "net_gex": round(net), "call_gex": round(call), "put_gex": round(put),
        "flip": flip,
        "flip_pct": round((flip / spot - 1) * 100, 2) if flip else None,
        "call_wall": max(calls, key=calls.get) if calls else None,
        "put_wall": min(puts, key=puts.get) if puts else None,
        "regime": "long_gamma" if net > 0 else "short_gamma",
    }


def expirations(today: date, max_dte: int = MAX_DTE) -> List[date]:
    from trading_agent.calendar_utils import is_trading_day
    return [d for d in (today + timedelta(days=i) for i in range(0, max_dte + 1)) if is_trading_day(d)]


def snapshot_ticker(ticker: str, today: date, get: Callable[[str], Optional[Dict[str, Any]]],
                    pause_s: float = REQUEST_PAUSE_S, max_dte: int = MAX_DTE) -> Optional[Dict[str, Any]]:
    spot = float((get(f"/price/{ticker}") or {}).get("price") or 0)
    if spot <= 0:
        return None
    contracts: List[Dict[str, Any]] = []
    for exp in expirations(today, max_dte):
        for side in ("call", "put"):
            body = get(f"/chain/{ticker}?expiration={exp.isoformat()}&option_type={side}") or {}
            for c in body.get("contracts") or []:
                contracts.append({**c, "type": c.get("type") or side})
            time.sleep(pause_s)
    if contracts and all(c.get("open_interest") is None for c in contracts):
        logger.warning("gamma: %s chains carry no open_interest — data server needs a restart", ticker)
        return None
    return summarize(ticker, contracts, spot, today)


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


def latest_rows(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """ticker → newest row (for the data server / MCP)."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with open(Path(path or STORE_PATH), encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("ticker")
                if t and str(r.get("as_of_utc", "")) >= str(out.get(t, {}).get("as_of_utc", "")):
                    out[t] = r
    except OSError:
        pass
    return out


def collect(tickers: Iterable[str], get, today: Optional[date] = None,
            path: Optional[Path] = None, dry_run: bool = False,
            pause_s: float = REQUEST_PAUSE_S) -> List[Dict[str, Any]]:
    today = today or date.today()
    path = Path(path or STORE_PATH)
    done = recorded_today(path, today.isoformat())
    rows = []
    for t in tickers:
        if t in done:
            continue
        try:
            row = snapshot_ticker(t, today, get, pause_s)
        except Exception as exc:                       # noqa: BLE001 — one ticker must not stop the rest
            logger.warning("gamma: %s failed (%s)", t, exc)
            continue
        if row is None:
            logger.warning("gamma: %s — no usable chain", t)
            continue
        rows.append(row)
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    return rows


def _refresh_macro_if_stale(today: date) -> None:
    from trading_agent import macro_calendar as mc
    try:
        age = (datetime.now().timestamp() - mc.STORE_PATH.stat().st_mtime) / 86400
    except OSError:
        age = None
    if age is not None and age < MACRO_REFRESH_DAYS:
        return
    try:
        logger.info("macro calendar refreshed: %s", mc.refresh(today))
    except Exception as exc:                           # noqa: BLE001 — built-in table still serves
        logger.warning("macro calendar refresh failed: %s", exc)
    for w in mc.coverage_warnings(today):
        logger.warning("macro calendar: %s", w)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Daily dealer gamma estimate (skill 68)")
    ap.add_argument("--tickers", default="", help=f"comma list; default {','.join(DEFAULT_TICKERS)}")
    ap.add_argument("--dry-run", action="store_true", help="print, do not append")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    from trading_agent.mcp.tools.market import _data_server_get
    from trading_agent.calendar_utils import is_trading_day
    today = date.today()
    if not args.dry_run:
        _refresh_macro_if_stale(today)
    if not is_trading_day(today) and not args.dry_run:
        logger.info("gamma: %s is not a trading day — nothing to record", today)
        return 0
    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               or list(DEFAULT_TICKERS))
    rows = collect(tickers, _data_server_get, today, dry_run=args.dry_run)
    for r in rows:
        logger.info("gamma: %s spot %.2f net %+.2fB/1%% flip %s (%s%%) walls C%s P%s — %s",
                    r["ticker"], r["spot"], r["net_gex"] / 1e9, r["flip"], r["flip_pct"],
                    r["call_wall"], r["put_wall"], r["regime"])
    logger.info("gamma: %d of %d tickers recorded%s", len(rows), len(tickers),
                " (dry run)" if args.dry_run else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

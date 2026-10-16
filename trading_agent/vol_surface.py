"""vol_surface.py — daily per-ticker skew and term-structure snapshot (skill 66).

Added 2026-10-16 (calibrated probability model, phase 2). There is no free
history of per-ticker option skew, so the agent starts recording its own:
once a day after the close, for every ticker in ``TICKERS``, the implied
volatility at the money and at the 25-delta put and call, at expiries
near 7, 30 and 60 days. After a few months the calibrated probability
model (``prob_model``) can test skew and term structure as features on
this data alone.

Chains come from the read-only data server (``/chain``, skill 47) — the
process that already holds the Schwab token — so the collector never
opens a Schwab session of its own (one token holder, skill 16 §4).

Row appended to ``trade_journal/vol_surface.jsonl`` (one per ticker per
day; a ticker already recorded today is skipped)::

    {"date": "2026-10-16", "ticker": "SPY", "spot": 769.6, "as_of_utc": "…",
     "expiries": [{"target_dte": 30, "expiration": "2026-11-13", "dte": 28,
                   "atm_iv": 0.117, "put25_iv": 0.151, "call25_iv": 0.098,
                   "rr25": -0.053, "fly25": 0.0075, "put_skew": 0.034,
                   "atm_spread_pct": 0.006, "n_contracts": 334}, …],
     "term": {"iv7": …, "iv30": …, "iv60": …, "ratio_7_30": …, "ratio_30_60": …}}

    python -m trading_agent.vol_surface [--dry-run] [--tickers SPY,QQQ]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.environ.get("TRADING_AGENT_VOL_SURFACE",
                                 "trade_journal/vol_surface.jsonl"))
TARGET_DTES = (7, 30, 60)
# Expiry search window around each target: (min, max) calendar days.
DTE_WINDOWS = {7: (4, 12), 30: (24, 38), 60: (50, 75)}
REQUEST_PAUSE_S = 0.3            # between data-server calls (Schwab rate limit)


# ── Pure summaries ────────────────────────────────────────────────────

def _usable(c: Dict[str, Any]) -> bool:
    try:
        return float(c.get("iv") or 0) > 0 and c.get("delta") is not None
    except (TypeError, ValueError):
        return False


def nearest_by_delta(contracts: Iterable[Dict[str, Any]], target: float) -> Optional[Dict[str, Any]]:
    """Contract whose |delta| is closest to ``target`` (with an IV), else None."""
    best, gap = None, None
    for c in contracts:
        if not _usable(c):
            continue
        g = abs(abs(float(c["delta"])) - target)
        if gap is None or g < gap:
            best, gap = c, g
    return best


def summarize_expiry(calls: List[Dict[str, Any]], puts: List[Dict[str, Any]]
                     ) -> Optional[Dict[str, Any]]:
    """ATM / 25-delta implied vols for one expiry. ATM = mean of the call
    and put nearest |Δ| 0.50. ``rr25`` = call25 − put25 (negative = puts
    dearer), ``fly25`` = (put25 + call25)/2 − ATM, ``put_skew`` = put25 −
    ATM. None when either side has no usable contract."""
    c50, p50 = nearest_by_delta(calls, 0.50), nearest_by_delta(puts, 0.50)
    c25, p25 = nearest_by_delta(calls, 0.25), nearest_by_delta(puts, 0.25)
    if not (c50 and p50 and c25 and p25):
        return None
    atm = (float(c50["iv"]) + float(p50["iv"])) / 2.0
    put25, call25 = float(p25["iv"]), float(c25["iv"])
    spreads = []
    for c in (c50, p50):
        bid, ask = float(c.get("bid") or 0), float(c.get("ask") or 0)
        if bid > 0 and ask >= bid:
            spreads.append((ask - bid) / ((ask + bid) / 2.0))
    return {"atm_iv": round(atm, 5), "put25_iv": round(put25, 5), "call25_iv": round(call25, 5),
            "rr25": round(call25 - put25, 5), "fly25": round((put25 + call25) / 2.0 - atm, 5),
            "put_skew": round(put25 - atm, 5),
            "atm_spread_pct": round(sum(spreads) / len(spreads), 4) if spreads else None,
            "n_contracts": len(calls) + len(puts)}


def term_structure(expiries: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """ATM IV by target DTE and the 7/30 and 30/60 ratios (> 1 = inverted)."""
    iv = {e["target_dte"]: e.get("atm_iv") for e in expiries}
    def ratio(a, b):
        return round(iv[a] / iv[b], 4) if iv.get(a) and iv.get(b) else None
    return {"iv7": iv.get(7), "iv30": iv.get(30), "iv60": iv.get(60),
            "ratio_7_30": ratio(7, 30), "ratio_30_60": ratio(30, 60)}


# ── Collection ────────────────────────────────────────────────────────

def candidate_expirations(today: date, target: int) -> List[date]:
    """Weekly expiry nearest ``target`` first, then the neighbouring weeks
    (not every name lists weeklies 60 days out)."""
    from datetime import timedelta
    from trading_agent.calendar_utils import next_weekly_expiration
    lo, hi = DTE_WINDOWS.get(target, (max(1, target - 7), target + 7))
    first = next_weekly_expiration(today, target, lo, hi)
    out = [first]
    for shift in (7, -7, 14, -14):
        d = first + timedelta(days=shift)
        if lo <= (d - today).days <= hi + 14 and d > today:
            out.append(d)
    return out


def snapshot_ticker(ticker: str, today: date,
                    get: Callable[[str], Optional[Dict[str, Any]]],
                    pause_s: float = REQUEST_PAUSE_S) -> Optional[Dict[str, Any]]:
    """One ticker's row, or None when no expiry produced a summary.
    ``get(path)`` is a data-server GET returning parsed JSON or None."""
    price = get(f"/price/{ticker}") or {}
    expiries = []
    for target in TARGET_DTES:
        for exp in candidate_expirations(today, target):
            iso = exp.isoformat()
            calls = (get(f"/chain/{ticker}?expiration={iso}&option_type=call") or {}).get("contracts") or []
            time.sleep(pause_s)
            puts = (get(f"/chain/{ticker}?expiration={iso}&option_type=put") or {}).get("contracts") or []
            time.sleep(pause_s)
            s = summarize_expiry(calls, puts)
            if s is not None:
                expiries.append({"target_dte": target, "expiration": iso,
                                 "dte": (exp - today).days, **s})
                break
    if not expiries:
        return None
    return {"date": today.isoformat(), "ticker": ticker, "spot": price.get("price"),
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "expiries": expiries, "term": term_structure(expiries)}


def recorded_today(path: Path, day: str) -> set:
    """Tickers already in the store for ``day``."""
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


def collect(tickers: Iterable[str], get, today: Optional[date] = None,
            path: Optional[Path] = None, dry_run: bool = False,
            pause_s: float = REQUEST_PAUSE_S) -> List[Dict[str, Any]]:
    """Snapshot every ticker not yet recorded today; append the rows
    (unless ``dry_run``) and return them."""
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
            logger.warning("vol surface: %s failed (%s)", t, exc)
            continue
        if row is None:
            logger.warning("vol surface: %s — no usable chain", t)
            continue
        rows.append(row)
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Daily skew / term-structure snapshot (skill 66)")
    ap.add_argument("--tickers", default="", help="comma list; default TICKERS from .env")
    ap.add_argument("--dry-run", action="store_true", help="print, do not append")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    from trading_agent.mcp.tools.market import _data_server_get
    from trading_agent.calendar_utils import is_trading_day
    today = date.today()
    if not is_trading_day(today) and not args.dry_run:
        logger.info("vol surface: %s is not a trading day — nothing to record", today)
        return 0
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        from trading_agent.config import load_config
        tickers = list(load_config(os.path.join(os.getcwd(), ".env")).trading.tickers)
    rows = collect(tickers, _data_server_get, today, dry_run=args.dry_run)
    for r in rows:
        e30 = next((e for e in r["expiries"] if e["target_dte"] == 30), None)
        logger.info("vol surface: %s ATM30 %s rr25 %s term7/30 %s", r["ticker"],
                    e30 and e30["atm_iv"], e30 and e30["rr25"], r["term"]["ratio_7_30"])
    logger.info("vol surface: %d of %d tickers recorded%s", len(rows), len(tickers),
                " (dry run)" if args.dry_run else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

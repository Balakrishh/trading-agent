"""Intraday option-chain logger (skill 69) — research data, no orders.

The free historical chains are end-of-day only, so same-day (0DTE /
1DTE) ideas cannot be tested on them (docs/plans/research_credit_vol.md).
While ``intraday_chain_log_mode == "on"`` the agent appends one snapshot
per ticker per cycle — the next ``intraday_chain_log_max_dte + 1``
expirations, strikes within ``intraday_chain_log_band_pct`` of spot,
bid / ask / delta / IV per contract — to a gzipped JSON-lines file per
ticker and day under ``INTRADAY_CHAIN_DIR`` (default
``~/data/intraday_chains``, outside the repo). After a few weeks the
files support honest same-day backtests.

Snapshots are taken in the regular session only (09:30–16:00 ET), no
more often than ``intraday_chain_log_interval_s`` (the last write time
lives in a sentinel file, because each cycle is a fresh process).
Best-effort throughout: any failure is logged at warning and the cycle
carries on.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Iterable, List, Optional

from trading_agent.calendar_utils import next_sessions

logger = logging.getLogger(__name__)

DEFAULT_DIR = "~/data/intraday_chains"
SESSION_OPEN, SESSION_CLOSE = dtime(9, 30), dtime(16, 0)


def log_dir() -> Path:
    return Path(os.path.expanduser(os.getenv("INTRADAY_CHAIN_DIR", DEFAULT_DIR)))


def compact(contracts: Iterable[dict], spot: float, band: float) -> List[list]:
    """[type, strike, bid, ask, delta, iv] for strikes within ``band`` of spot."""
    out = []
    for c in contracts or []:
        k = c.get("strike")
        if not k or abs(k - spot) / spot > band:
            continue
        out.append([c.get("type", "")[:1].upper(), k, c.get("bid"), c.get("ask"),
                    c.get("delta"), c.get("iv")])
    return sorted(out, key=lambda r: (r[0], r[1]))


def due(sentinel: Path, interval_s: float, now: float) -> bool:
    try:
        return now - float(sentinel.read_text().strip()) >= interval_s
    except (OSError, ValueError):
        return True


def maybe_log(preset, data_provider, now_et: datetime, out_dir: Optional[Path] = None) -> int:
    """Append one snapshot per configured ticker when due. Returns the
    number of snapshots written."""
    if getattr(preset, "intraday_chain_log_mode", "off") != "on":
        return 0
    if not (SESSION_OPEN <= now_et.time() < SESSION_CLOSE):
        return 0
    out_dir = out_dir or log_dir()
    sentinel = out_dir / ".last_snapshot"
    now = time.time()
    if not due(sentinel, preset.intraday_chain_log_interval_s, now):
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    today = now_et.date()
    exps = next_sessions(today, preset.intraday_chain_log_max_dte + 1)
    written = 0
    for tk in preset.intraday_chain_log_tickers:
        try:
            spot = data_provider.get_cached_price(tk) or data_provider.get_current_price(tk)
            if not spot:
                continue
            chains = {}
            for exp in exps:
                rows = []
                for side in ("put", "call"):
                    rows += compact(data_provider.fetch_option_chain(tk, exp.isoformat(), side),
                                    spot, preset.intraday_chain_log_band_pct)
                if rows:
                    chains[exp.isoformat()] = rows
            if not chains:
                continue
            row = {"ts": now_et.isoformat(timespec="seconds"), "ticker": tk,
                   "spot": spot, "chains": chains}
            with gzip.open(out_dir / f"{tk}_{today.isoformat()}.jsonl.gz", "at") as fh:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1
        except Exception as exc:                        # noqa: BLE001, skill-34-exempt — research logging must never break a cycle
            logger.warning("[%s] intraday chain snapshot failed: %s", tk, exc)
    tmp = sentinel.with_suffix(".tmp")
    tmp.write_text(str(now))
    tmp.replace(sentinel)
    if written:
        logger.info("Intraday chain log: %d snapshot(s) → %s", written, out_dir)
    return written

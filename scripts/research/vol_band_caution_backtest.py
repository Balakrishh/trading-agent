"""Vol-band bull put under CAUTION — would allowing it help?

Rows: upday_put_study's simulated 21d 0.25Δ 2%-wide put spreads on SPY/QQQ/IWM
(real EOD quotes, fills halfway mid→natural, agent exits without the proximity
exit). CAUTION proxy = SPY close < SPY SMA50 (the main live trigger; VIX-term
and breadth triggers are not in this data). HighIV (RV rank > 95) always blocks.
One position per ticker at a time; account at 2 % risk per trade.

    python scripts/research/upday_put_study.py      # writes the rows (~1 h on the Pi)
    python scripts/research/vol_band_caution_backtest.py
"""
import pickle
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from put_spread_playbook_backtest import report  # noqa: E402
from upday_put_study_rows import ROWS  # noqa: E402

rows = pickle.load(open(ROWS, "rb"))
spy_caution = {r["day"]: r["below50"] for r in rows if r["tk"] == "SPY"}
for r in rows:
    r["caution"] = spy_caution.get(r["day"])
rows = [r for r in rows if r["caution"] is not None]


def walk(keep, half_in_caution=False):
    busy, out = defaultdict(lambda: date.min), []
    for r in sorted(rows, key=lambda r: (r["day"], r["tk"])):
        if r["day"] <= busy[r["tk"]] or r["rank"] > 0.95 or not keep(r):
            continue
        pnl = r["pnl"] * (0.5 if half_in_caution and r["caution"] else 1.0)
        out.append((r["tk"], r["day"], r["exit_day"], pnl, r["why"]))
        busy[r["tk"]] = r["exit_day"]
    return out


band = lambda r: 0.50 <= r["rank"] <= 0.80   # noqa: E731
VARIANTS = {
    "live: band, not CAUTION":                     walk(lambda r: band(r) and not r["caution"]),
    "proposed: band, CAUTION ok if >SMA200, ½ size": walk(lambda r: band(r) and (not r["caution"] or r["above200"]), True),
    "proposed, full size in CAUTION":              walk(lambda r: band(r) and (not r["caution"] or r["above200"])),
    "CAUTION-only trades, >SMA200 (full size)":    walk(lambda r: band(r) and r["caution"] and r["above200"]),
    "CAUTION-only trades, <SMA200 (full size)":    walk(lambda r: band(r) and r["caution"] and not r["above200"]),
    "band, any state (no gate)":                   walk(band),
}
report(VARIANTS)

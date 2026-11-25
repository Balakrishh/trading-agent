"""same_day_replay.py — replay same-day (0DTE) SPY spreads on the logged intraday chains.

Question (2026-11-24): would same-day trades, opened and closed within the
session, have made money in the November sell-off and the 11-23 rebound?
Data: the agent's own intraday chain log (skill 69, ~90 s snapshots of the
0DTE / 1DTE SPY chain, bid / ask / delta). Only a few weeks exist, so this
is a replay of specific days, not a validated backtest.

Rules, each opened at fixed entry times on the 0DTE expiry, one per
rule per entry time:
  put credit / call credit / condor — short |Δ| ≈ 0.10, $5 wide;
  momentum credit — sell the side away from the move since 09:35;
  momentum debit  — buy a ≈ 0.50Δ $5 vertical in the direction of the move.
Fills: entry and exit halfway mid → natural. Exits on mid: credit target 50 %
of credit, stop when the loss reaches 2× credit; debit target +50 %, stop
−50 %; otherwise close at the last snapshot ≤ 15:50.

    python scripts/research/same_day_replay.py [--dir ~/data/intraday_chains]
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path

ENTRY_TIMES = (dtime(10, 0), dtime(11, 0), dtime(12, 30), dtime(14, 0))
LAST_EXIT = dtime(15, 50)
SHORT_DELTA, WIDTH = 0.10, 5.0
FRAC = 0.5


def load_day(fp: Path):
    snaps = []
    with gzip.open(fp, "rt") as f:
        for line in f:
            r = json.loads(line)
            ts = datetime.fromisoformat(r["ts"])
            exp = ts.date().isoformat()
            if ts.time() < dtime(9, 35) or exp not in r["chains"]:
                continue
            q = {(o[0], float(o[1])): (o[2], o[3], o[4]) for o in r["chains"][exp]}
            spot = implied_spot(q)
            if spot is not None:
                snaps.append((ts.time(), spot, q))
    return snaps


def implied_spot(q):
    """Median K + C − P over near-the-money strikes. The logged ``spot`` can be
    the prior close in the first snapshot after a gap (seen 2026-11-23 09:32)."""
    xs = sorted(k + (c[0] + c[1]) / 2 - (q[("P", k)][0] + q[("P", k)][1]) / 2
                for (o, k), c in q.items()
                if o == "C" and ("P", k) in q and c[2] is not None and 0.1 < abs(c[2]) < 0.9)
    return xs[len(xs) // 2] if xs else None


def leg_px(q, key, side):
    """side +1 = we buy, −1 = we sell; returns (mid, fill) or None."""
    v = q.get(key)
    if v is None or v[0] is None or v[1] is None or v[1] <= 0:
        return None
    bid, ask = v[0], v[1]
    mid = (bid + ask) / 2
    nat = ask if side > 0 else bid
    return mid, mid + FRAC * (nat - mid)


def spread_value(q, legs, closing=False):
    """legs: [(key, +1 long / −1 short)]. Value to the holder (mid, fill)."""
    mid = fill = 0.0
    for key, sign in legs:
        p = leg_px(q, key, -sign if closing else sign)
        if p is None:
            return None
        mid += sign * p[0]
        fill += sign * p[1]
    return mid, fill


def pick_short(q, opt, spot, target=SHORT_DELTA):
    cands = [(abs(abs(v[2]) - target), k) for (o, k), v in q.items()
             if o == opt and v[2] is not None and ((k < spot) if opt == "P" else (k > spot))]
    return min(cands)[1] if cands else None


def pick_atm(q, opt, spot):
    cands = [(abs(k - spot), k) for (o, k) in q if o == opt]
    return min(cands)[1] if cands else None


def build(rule, q, spot, move):
    if rule == "put credit":
        k = pick_short(q, "P", spot)
        return None if k is None else ("credit", [(("P", k), -1), (("P", k - WIDTH), +1)])
    if rule == "call credit":
        k = pick_short(q, "C", spot)
        return None if k is None else ("credit", [(("C", k), -1), (("C", k + WIDTH), +1)])
    if rule == "condor":
        a, b = build("put credit", q, spot, move), build("call credit", q, spot, move)
        return None if a is None or b is None else ("credit", a[1] + b[1])
    if rule == "momentum credit":
        return build("put credit" if move > 0 else "call credit", q, spot, move)
    if rule == "momentum debit":
        opt = "C" if move > 0 else "P"
        k = pick_atm(q, opt, spot)
        if k is None:
            return None
        far = k + WIDTH if opt == "C" else k - WIDTH
        return ("debit", [((opt, k), +1), ((opt, far), -1)])
    raise ValueError(rule)


RULES = ("put credit", "call credit", "condor", "momentum credit", "momentum debit")


def trade(snaps, i0, rule, ref_spot):
    t0, spot, q = snaps[i0]
    plan = build(rule, q, spot, spot - ref_spot)
    if plan is None:
        return None
    kind, legs = plan
    if any(k not in q for k, _ in legs):
        return None
    open_v = spread_value(q, legs)
    if open_v is None:
        return None
    if kind == "credit":
        credit = -open_v[1]
        risk = WIDTH - credit          # a condor can lose on one side only
        if credit <= 0 or risk <= 0:
            return None
    else:
        debit = open_v[1]
        risk = debit
        if debit <= 0 or debit >= WIDTH:
            return None
    for t, s, qq in snaps[i0 + 1:]:
        v = spread_value(qq, legs, closing=True)
        if v is None:
            continue
        last = t >= LAST_EXIT
        if kind == "credit":
            pl_mid, pl_fill = credit + v[0], credit + v[1]
            hit = pl_mid >= 0.5 * credit or -pl_mid >= 2 * credit or last
        else:
            pl_mid, pl_fill = v[0] - debit, v[1] - debit
            hit = pl_mid >= 0.5 * debit or -pl_mid >= 0.5 * debit or last
        if hit:
            why = "end" if last else ("target" if pl_mid > 0 else "stop")
            return dict(pnl=max(-risk, pl_fill) / risk, why=why, t0=t0, t1=t, usd=100 * pl_fill)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(Path.home() / "data/intraday_chains"))
    args = ap.parse_args()
    files = sorted(Path(args.dir).expanduser().glob("SPY_*.jsonl.gz"))
    per_rule = defaultdict(list)
    print(f"{'day':10} {'SPY 09:35→15:50':>16} " + " ".join(f"{r:>16}" for r in RULES))
    for fp in files:
        day = fp.stem.split("_")[1].split(".")[0]
        snaps = load_day(fp)
        if len(snaps) < 50:
            continue
        ref = snaps[0][1]
        end = [s for s in snaps if s[0] <= LAST_EXIT][-1][1]
        cells = []
        for rule in RULES:
            usd = []
            for et in ENTRY_TIMES:
                i0 = next((i for i, s in enumerate(snaps) if s[0] >= et), None)
                if i0 is None:
                    continue
                r = trade(snaps, i0, rule, ref)
                if r:
                    r["day"] = day
                    per_rule[rule].append(r)
                    usd.append(r["usd"])
            cells.append(f"{sum(usd):+8.0f} ({len(usd)})")
        print(f"{day:10} {ref:7.2f}→{end:7.2f} " + " ".join(f"{c:>16}" for c in cells))
    print("\nper rule (1 contract per trade):")
    print(f"{'rule':16} {'n':>3} {'win':>4} {'avg/$':>7} {'total $':>8} {'worst $':>8} {'best $':>7}  exits")
    for rule in RULES:
        xs = per_rule[rule]
        if not xs:
            continue
        wins = sum(x["pnl"] > 0 for x in xs) / len(xs)
        whys = defaultdict(int)
        for x in xs:
            whys[x["why"]] += 1
        print(f"{rule:16} {len(xs):3d} {wins:4.0%} {sum(x['pnl'] for x in xs) / len(xs):+7.3f} "
              f"{sum(x['usd'] for x in xs):+8.0f} {min(x['usd'] for x in xs):+8.0f} {max(x['usd'] for x in xs):+7.0f}  "
              + " ".join(f"{k}={v}" for k, v in sorted(whys.items())))


if __name__ == "__main__":
    main()

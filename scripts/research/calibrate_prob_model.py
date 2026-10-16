"""Walk-forward evaluation of the calibrated probability model (skill 66, phase 1).

Daily history from Yahoo (split/dividend-adjusted OHLC, 2009 →) for the
live watchlist plus ^VIX / ^VIX3M / ^SKEW and the implied-vol indices
^VIX (SPY), ^VXN (QQQ), ^GVZ (GLD). Features are rebuilt with the live
agent's own code, as of each decision day's close:

* direction — ``RegimeClassifier._determine_regime`` on SMA-50/200, the
  5-day SMA-50 slope and 20-day Bollinger bands (2σ width, 3σ touch);
* market regime — ``market_state.classify_market_state`` with SPY
  SMA-20/50/200 and RSI, VIX, VIX3M and breadth over the watchlist's
  equity tickers, hysteresis carried day to day;
* volatility regime — trailing-252-day percentile of 20-day realized vol;
* term structure — VIX / VIX3M; skew — trailing-252-day percentile of SKEW.

Samples: every ticker × trading day × horizon (7/14/21/30/45 calendar
days), outcome = the first close on or after the horizon. Single stocks
skip windows that contain an earnings date (the live agent's expiries end
before earnings since 2026-10-15). Walk-forward from 2013-01: each month is
predicted by a model fitted only on samples whose outcome date is before
that month. Baselines: ``pop_rv`` (zero-drift lognormal, realized vol) and,
for SPY / QQQ / GLD, the implied lognormal (≈ the delta POP).

Research only — never imported by the live agent. Writes the report to
``docs/plans/research_prob_model.md`` (``--report``) and, with ``--save``,
a model fitted on all data to the given JSON path.

    python scripts/research/calibrate_prob_model.py [--quick] [--report PATH] [--save PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trading_agent import prob_model as pm  # noqa: E402
from trading_agent.market_state import (  # noqa: E402
    MarketInputs, _rsi, breadth_universe, classify_market_state)
from trading_agent.regime import RegimeClassifier  # noqa: E402

WATCHLIST = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "TLT", "GLD", "EEM", "GDX",
             "META", "AMZN", "XLK", "XLV", "XLY", "XLI", "SMH", "XLU"]
IMPLIED = {"SPY": "^VIX", "QQQ": "^VXN", "GLD": "^GVZ"}
SINGLE = {"META", "AMZN"}
FETCH_FROM = "2009-01-01"
# Text after this marker in the report is written by hand and kept on regeneration.
FINDINGS_MARK = "<!-- findings: hand-written, kept when the report is regenerated -->"
SAMPLE_FROM = "2010-03-01"
TEST_FROM = "2013-01-01"

# (name, kind, z) — kind: above / below / between / touch_down / touch_up
EVENTS = [
    ("put_survive_0.50", "above", -0.50), ("put_survive_0.75", "above", -0.75),
    ("put_survive_1.00", "above", -1.00),
    ("call_survive_0.50", "below", 0.50), ("call_survive_0.75", "below", 0.75),
    ("call_survive_1.00", "below", 1.00),
    ("call_debit_win", "above", 0.25), ("put_debit_win", "below", -0.25),
    ("condor_1.00", "between", 1.00),
    ("put_touch_1.00", "touch_down", -1.00), ("call_touch_1.00", "touch_up", 1.00),
]
STRATEGY_OF = {
    "put_survive": "bull put / CSP short strike finishes OTM",
    "call_survive": "bear call short strike finishes OTM",
    "call_debit_win": "call debit: finish ≥ 0.25σ up",
    "put_debit_win": "put debit: finish ≥ 0.25σ down",
    "condor_1.00": "condor ±1σ: finish inside",
    "put_touch_1.00": "price touches −1σ before expiry (stop risk, puts)",
    "call_touch_1.00": "price touches +1σ before expiry (stop risk, calls)",
}


# ── Data ──────────────────────────────────────────────────────────────

def ohlc(symbol: str) -> pd.DataFrame:
    import yfinance as yf  # type: ignore
    df = yf.Ticker(symbol).history(start=FETCH_FROM, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[["Open", "High", "Low", "Close"]].astype(float)


def earnings_dates(ticker: str) -> List[pd.Timestamp]:
    try:
        import yfinance as yf  # type: ignore
        d = yf.Ticker(ticker).get_earnings_dates(limit=100)
        return sorted(pd.to_datetime(d.index).tz_localize(None).normalize().unique())
    except Exception as exc:                                    # noqa: BLE001
        print(f"  {ticker}: no earnings dates ({exc}) — windows kept")
        return []


def trailing_rank(s: pd.Series, n: int = 252) -> pd.Series:
    """Percentile of today's value within the trailing n values (inclusive)."""
    return s.rolling(n, min_periods=n // 2).rank(pct=True)


def ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day direction + realized vol + vol rank, from closes up to that day."""
    close = df["Close"]
    rv = np.log(close).diff().rolling(20).std() * math.sqrt(252)
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    slope = (sma50 - sma50.shift(5)) / 5
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    bbw = (4 * sd) / mid
    up3, lo3 = mid + 3 * sd, mid - 3 * sd
    rc = RegimeClassifier.__new__(RegimeClassifier)
    regime = []
    for p, s50, s200, sl, w, u, l in zip(close.values, sma50.values, sma200.values,
                                         slope.values, bbw.values, up3.values, lo3.values):
        if any(map(lambda v: v != v, (s50, s200, sl, w, u, l))):
            regime.append(None)
            continue
        mr_u, mr_l = p >= u, p <= l
        r, _ = rc._determine_regime(p, s50, s200, sl, w, bool(mr_u or mr_l),
                                    "upper" if mr_u else "lower" if mr_l else "", u, l)
        regime.append(r.value)
    return pd.DataFrame({"rv": rv, "vol_rank": trailing_rank(rv), "regime": regime},
                        index=df.index)


def market_states(spy: pd.Series, vix: pd.Series, vix3m: pd.Series,
                  breadth: pd.Series) -> pd.Series:
    """Daily market risk state with the live classifier and hysteresis."""
    closes = spy.values.tolist()
    out, prior = {}, None
    for i, d in enumerate(spy.index):
        c = closes[: i + 1]
        if len(c) < 200:
            continue
        inp = MarketInputs(spy_price=c[-1], spy_sma20=sum(c[-20:]) / 20,
                           spy_sma50=sum(c[-50:]) / 50, spy_sma200=sum(c[-200:]) / 200,
                           spy_rsi=_rsi(c), vix=_at(vix, d), vix3m=_at(vix3m, d),
                           breadth=_at(breadth, d))
        prior = classify_market_state(inp, prior).state
        out[d] = prior
    return pd.Series(out)


def _at(s: pd.Series, d) -> Optional[float]:
    v = s.get(d)
    return None if v is None or v != v else float(v)


def build_samples(data: Dict[str, pd.DataFrame], ctx: pd.DataFrame,
                  earnings: Dict[str, List[pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    for t, df in data.items():
        f = ticker_features(df)
        iv = ctx.get(f"iv_{t}")
        close, low, high = df["Close"].values, df["Low"].values, df["High"].values
        dates = df.index
        earn = np.array(earnings.get(t, []), dtype="datetime64[ns]")
        for i in range(len(dates)):
            d = dates[i]
            if d < pd.Timestamp(SAMPLE_FROM):
                continue
            rv, rk, reg = f["rv"].iat[i], f["vol_rank"].iat[i], f["regime"].iat[i]
            if not rv or rv != rv or reg is None:
                continue
            c0 = close[i]
            for h in pm.HORIZONS:
                j = dates.searchsorted(d + pd.Timedelta(days=h))
                if j >= len(dates):
                    break
                if t in SINGLE and len(earn) and ((earn > d.to_datetime64()) &
                                                  (earn <= dates[j].to_datetime64())).any():
                    continue
                scale = rv * math.sqrt(h / 365.0)
                ivv = None if iv is None else _at(iv, d)
                rows.append((t, d, dates[j], h, math.log(close[j] / c0) / scale,
                             math.log(low[i + 1: j + 1].min() / c0) / scale,
                             math.log(high[i + 1: j + 1].max() / c0) / scale,
                             rv, rk, reg, (j - i), None if ivv is None else ivv / 100.0))
    s = pd.DataFrame(rows, columns=["ticker", "decision", "outcome", "horizon", "x", "x_min",
                                    "x_max", "rv", "vol_rank", "regime", "overlap", "iv"])
    s = s.join(ctx[["market", "term_ratio", "skew_rank"]], on="decision")
    s["asset"] = s["ticker"].map(pm.asset_group)
    s["direction"] = s["regime"].map(pm.direction_bucket)
    s["vol"] = s["vol_rank"].map(pm.vol_bucket)
    s["market_b"] = s["market"].map(pm.market_bucket)
    s["term"] = s["term_ratio"].map(pm.term_bucket)
    s["skew"] = s["skew_rank"].map(pm.skew_bucket)
    s["weight"] = 1.0 / s["overlap"].clip(lower=1)
    return s


# ── Events ────────────────────────────────────────────────────────────

def outcomes(s: pd.DataFrame, kind: str, z: float) -> np.ndarray:
    x, lo, hi = s["x"].values, s["x_min"].values, s["x_max"].values
    return {"above": x > z, "below": x < z, "between": (x > -z) & (x < z),
            "touch_down": lo <= z, "touch_up": hi > z}[kind].astype(float)


def model_prob(f_term, f_min, f_max, kind: str, z: float) -> np.ndarray:
    n = len(f_term)
    zz = np.full(n, z)
    if kind == "above":
        return 1 - pm.cdf_at(f_term, zz)
    if kind == "below":
        return pm.cdf_at(f_term, zz)
    if kind == "between":
        return pm.cdf_at(f_term, zz) - pm.cdf_at(f_term, -zz)
    if kind == "touch_down":
        return pm.cdf_at(f_min, zz)
    return 1 - pm.cdf_at(f_max, zz)


def lognormal_prob(s: pd.DataFrame, kind: str, z: float, implied: bool = False) -> np.ndarray:
    """Baseline: zero-drift lognormal at realized vol (pop_rv) or implied vol."""
    rv, h = s["rv"].values, s["horizon"].values
    sig = s["iv"].values.astype(float) if implied else rv
    ratio = rv / sig
    vst = sig * np.sqrt(h / 365.0)
    above = lambda zz: pm.lognormal_prob_above(zz, ratio, vst)      # noqa: E731
    if kind == "above":
        return above(z)
    if kind == "below":
        return 1 - above(z)
    if kind == "between":
        return above(-z) - above(z)
    if kind == "touch_down":
        return np.minimum(1.0, 2 * (1 - above(z)))
    return np.minimum(1.0, 2 * above(z))


# ── Walk-forward driver ───────────────────────────────────────────────

def run_variant(s: pd.DataFrame, dims: Tuple[str, ...], k: float, refit: int,
                with_iso: bool = True) -> pd.DataFrame:
    feats = {d: s["market_b" if d == "market" else d].values for d in dims}
    feats["horizon"] = s["horizon"].astype(str).values
    enc = pm.Encoded(dims, feats, s["x"].values, s["x_min"].values, s["x_max"].values,
                     s["weight"].values)
    preds = {name: np.full(len(s), np.nan) for name, _, _ in EVENTS}

    def on_fold(start, model, rows):
        ft, fm, fx = pm.predict_cells(model, enc, rows)
        for name, kind, z in EVENTS:
            preds[name][rows] = model_prob(ft, fm, fx, kind, z)

    folds = pm.walk_forward(enc, s["decision"].values, s["outcome"].values,
                            np.datetime64(TEST_FROM), k, on_fold, refit_months=refit)
    out = pd.DataFrame({"row": np.arange(len(s))})
    month = s["decision"].values.astype("datetime64[M]")
    out_month = s["outcome"].values.astype("datetime64[D]")
    for name, kind, z in EVENTS:
        p = preds[name]
        out[name] = p
        if not with_iso:
            continue
        # Recalibrate each test month with an isotonic map fitted only on
        # earlier rows whose outcome was known before that month began.
        iso = np.full(len(s), np.nan)
        y = outcomes(s, kind, z)
        test_months = np.unique(month[~np.isnan(p)])
        for m in test_months:
            start = m.astype("datetime64[D]")
            hist = (~np.isnan(p)) & (out_month < start)
            cal = pm.isotonic_fit(p[hist], y[hist]) if hist.sum() >= 200 else None
            rows = (month == m) & ~np.isnan(p)
            iso[rows] = pm.isotonic_apply(cal, p[rows])
        out[name + "__iso"] = iso
    out.attrs["folds"] = folds
    return out


def score_table(s: pd.DataFrame, pr: pd.DataFrame, mask: np.ndarray) -> List[Dict]:
    rows = []
    month = s["decision"].values.astype("datetime64[M]")
    for name, kind, z in EVENTS:
        p = pr[name].values
        m = mask & ~np.isnan(p)
        if m.sum() == 0:
            continue
        y = outcomes(s, kind, z)[m]
        prv = lognormal_prob(s[m], kind, z)
        r = {"event": name, "n": int(m.sum()), "realized": y.mean(),
             "model": p[m].mean(), "rv": prv.mean(),
             "brier_model": pm.brier(p[m], y), "brier_rv": pm.brier(prv, y),
             "ll_model": pm.log_loss(p[m], y), "ll_rv": pm.log_loss(prv, y)}
        if name + "__iso" in pr:
            pi = pr[name + "__iso"].values[m]
            r.update(brier_iso=pm.brier(pi, y), ll_iso=pm.log_loss(pi, y))
        loss_m = (p[m] - y) ** 2
        loss_r = (prv - y) ** 2
        r["dbrier"], r["dbrier_lo"], r["dbrier_hi"] = pm.block_bootstrap_diff(loss_m, loss_r, month[m])
        rows.append(r)
    return rows


def implied_table(s: pd.DataFrame, pr: pd.DataFrame, mask: np.ndarray) -> List[Dict]:
    rows = []
    m0 = mask & s["iv"].notna().values
    for name, kind, z in EVENTS:
        p = pr[name].values
        m = m0 & ~np.isnan(p)
        if m.sum() == 0:
            continue
        y = outcomes(s, kind, z)[m]
        piv = lognormal_prob(s[m], kind, z, implied=True)
        prv = lognormal_prob(s[m], kind, z)
        rows.append({"event": name, "n": int(m.sum()), "realized": y.mean(), "implied": piv.mean(),
                     "rv": prv.mean(), "model": p[m].mean(),
                     "brier_implied": pm.brier(piv, y), "brier_rv": pm.brier(prv, y),
                     "brier_model": pm.brier(p[m], y)})
    return rows


def breakdown(s: pd.DataFrame, pr: pd.DataFrame, mask: np.ndarray, event: str, by: str) -> List[Dict]:
    name, kind, z = next(e for e in EVENTS if e[0] == event)
    p = pr[name].values
    y = outcomes(s, kind, z)
    prv = lognormal_prob(s, kind, z)
    out = []
    for val, idx in s[mask & ~np.isnan(p)].groupby(by).groups.items():
        ii = s.index.get_indexer(idx)
        out.append({"bucket": val, "n": len(ii), "realized": y[ii].mean(), "model": p[ii].mean(),
                    "rv": prv[ii].mean(), "brier_model": pm.brier(p[ii], y[ii]),
                    "brier_rv": pm.brier(prv[ii], y[ii])})
    return sorted(out, key=lambda r: str(r["bucket"]))


def pct(v):
    return "—" if v is None or v != v else f"{100 * v:.1f}%"


def write_report(path: Path, s: pd.DataFrame, results: Dict, best: str, meta: Dict) -> None:
    L = []
    L.append("# Calibrated probability model — walk-forward results (skill 66, phase 1)\n")
    L.append(f"*Generated {meta['generated']} by `scripts/research/calibrate_prob_model.py` on the Pi. "
             f"Yahoo daily OHLC (adjusted) {FETCH_FROM} → {meta['last_date']}, {meta['n_tickers']} tickers, "
             f"{len(s):,} samples (ticker × day × horizon {pm.HORIZONS}), test months {TEST_FROM[:7]} → "
             f"{meta['last_date'][:7]} ({meta['folds']} folds). Samples overlap: n overstates independent "
             f"evidence, so intervals come from resampling whole months.*\n")
    L.append("## Variants (out-of-sample log loss, all events pooled; lower is better)\n")
    L.append("Chosen on the first half of the test period, confirmed on the second.\n")
    L.append("| Variant | dims | K | refit | log loss 2013–19 | log loss 2020→ | Brier 2020→ |")
    L.append("|---|---|---|---|---|---|---|")
    for v in results["variants"]:
        L.append(f"| {v['name']}{' **(chosen)**' if v['name'] == best else ''} | {', '.join(v['dims'])} | "
                 f"{v['k']:g} | {v['refit']}m | {v['ll_first']:.4f} | {v['ll_second']:.4f} | {v['brier_second']:.4f} |")
    L.append(f"| pop_rv baseline | — | — | — | {results['rv_first']:.4f} | {results['rv_second']:.4f} | "
             f"{results['rv_brier_second']:.4f} |\n")
    L.append(f"## Chosen variant `{best}` vs pop_rv, by event (all test months)\n")
    L.append("ΔBrier = model − pop_rv (negative = model better), 90 % interval from a month-block bootstrap.\n")
    L.append("| Event | what it means | n | realized | model | +isotonic Brier | pop_rv | Brier model | Brier rv | ΔBrier [90 %] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results["by_event"]:
        what = STRATEGY_OF.get(r["event"].rsplit("_", 1)[0] if "survive" in r["event"] else r["event"], "")
        L.append(f"| {r['event']} | {what} | {r['n']:,} | {pct(r['realized'])} | {pct(r['model'])} | "
                 f"{r.get('brier_iso', float('nan')):.4f} | {pct(r['rv'])} | {r['brier_model']:.4f} | "
                 f"{r['brier_rv']:.4f} | {r['dbrier']:+.4f} [{r['dbrier_lo']:+.4f}, {r['dbrier_hi']:+.4f}] |")
    L.append("\n## Against implied odds (SPY ^VIX, QQQ ^VXN, GLD ^GVZ)\n")
    L.append("| Event | n | realized | implied | pop_rv | model | Brier implied | Brier rv | Brier model |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results["implied"]:
        L.append(f"| {r['event']} | {r['n']:,} | {pct(r['realized'])} | {pct(r['implied'])} | {pct(r['rv'])} | "
                 f"{pct(r['model'])} | {r['brier_implied']:.4f} | {r['brier_rv']:.4f} | {r['brier_model']:.4f} |")
    for ev in ("put_survive_0.75", "call_survive_0.75", "put_debit_win", "call_debit_win"):
        L.append(f"\n## `{ev}` by situation (all test months)\n")
        for by, label in (("direction", "Direction"), ("market_b", "Market regime"), ("vol", "Vol regime"),
                          ("term", "Term structure"), ("horizon", "DTE"), ("asset", "Asset group")):
            L.append(f"**{label}** — bucket: n · realized · model · pop_rv · Brier model / rv\n")
            for r in results["breakdowns"][ev][by]:
                L.append(f"- `{r['bucket']}`: {r['n']:,} · {pct(r['realized'])} · {pct(r['model'])} · "
                         f"{pct(r['rv'])} · {r['brier_model']:.4f} / {r['brier_rv']:.4f}")
            L.append("")
    L.append("## Caveats\n")
    L.append("- Outcomes are expiry / touch events on the underlying, not P&L with the agent's exits; "
             "the live shadow (phase 3) is the check on that.\n"
             "- 2010–2026 is mostly a rising market; drift shows up as put-side edge. Walk-forward shows "
             "how stable it was, not that it continues.\n"
             "- Per-ticker skew has no free history; ^SKEW (S&P 500) is the only skew input here. The "
             "daily `vol_surface.jsonl` snapshot (phase 2) collects per-ticker skew from 2026-10-16.\n"
             "- Earnings windows are excluded for META / AMZN only where Yahoo lists the dates (2012 →).\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = ""
    if path.exists() and FINDINGS_MARK in path.read_text():
        keep = path.read_text().split(FINDINGS_MARK, 1)[1]
    path.write_text("\n".join(L) + "\n" + FINDINGS_MARK + keep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="5 tickers, one variant (smoke test)")
    ap.add_argument("--report", default="docs/plans/research_prob_model.md")
    ap.add_argument("--save", default="", help="write a model fitted on all data to this JSON path")
    args = ap.parse_args(argv)
    t0 = time.time()
    tickers = ["SPY", "QQQ", "GLD", "XLF", "AMZN"] if args.quick else WATCHLIST
    print(f"Downloading {len(tickers)} tickers + indices …")
    data = {t: ohlc(t) for t in tickers}
    data = {t: d for t, d in data.items() if not d.empty}
    idx = {s: ohlc(s)["Close"] for s in ("^VIX", "^VIX3M", "^SKEW", "^VXN", "^GVZ")}
    spy = data["SPY"]["Close"] if "SPY" in data else ohlc("SPY")["Close"]
    eq = breadth_universe(tickers)
    above = pd.DataFrame({t: (data[t]["Close"] > data[t]["Close"].rolling(50).mean()).where(
        data[t]["Close"].rolling(50).count() >= 50) for t in eq if t in data})
    breadth = above.mean(axis=1, skipna=True)
    print("Replaying market states …")
    ctx = pd.DataFrame(index=spy.index)
    ctx["market"] = market_states(spy, idx["^VIX"], idx["^VIX3M"], breadth)
    ctx["term_ratio"] = (idx["^VIX"] / idx["^VIX3M"]).reindex(spy.index)
    ctx["skew_rank"] = trailing_rank(idx["^SKEW"]).reindex(spy.index)
    for t, sym in IMPLIED.items():
        ctx[f"iv_{t}"] = idx[sym].reindex(spy.index)
    earnings = {t: earnings_dates(t) for t in tickers if t in SINGLE}
    print("Building samples …")
    s = build_samples(data, ctx, earnings)
    s = s.dropna(subset=["x", "x_min", "x_max"]).reset_index(drop=True)
    print(f"  {len(s):,} samples, {time.time() - t0:.0f}s")

    core = ("horizon", "asset", "direction", "vol", "market")
    # "horizon only" = drift/fat-tail correction with no situation: the
    # other variants must beat it for the conditioning to be worth having.
    variants = [("horizon only", ("horizon",), 30.0, 1), ("core", core, 30.0, 1), ("core+term", core + ("term",), 30.0, 1),
                ("core+term+skew", core + ("term", "skew"), 30.0, 1),
                ("core K=10", core, 10.0, 1), ("core K=100", core, 100.0, 1),
                ("core+term K=100", core + ("term",), 100.0, 1)]
    if args.quick:
        variants = [v for v in variants if v[0] in ("horizon only", "core+term")]
    dec = s["decision"].values.astype("datetime64[D]")
    first = (dec >= np.datetime64(TEST_FROM)) & (dec < np.datetime64("2020-01-01"))
    second = dec >= np.datetime64("2020-01-01")

    def pooled_ll(pr, mask, col_suffix=""):
        ll, n = 0.0, 0
        for name, kind, z in EVENTS:
            p = pr[name + col_suffix].values
            m = mask & ~np.isnan(p)
            if m.sum():
                ll += pm.log_loss(p[m], outcomes(s, kind, z)[m]) * m.sum()
                n += m.sum()
        return ll / max(n, 1)

    def pooled_brier(pr, mask):
        b, n = 0.0, 0
        for name, kind, z in EVENTS:
            p = pr[name].values
            m = mask & ~np.isnan(p)
            if m.sum():
                b += pm.brier(p[m], outcomes(s, kind, z)[m]) * m.sum()
                n += m.sum()
        return b / max(n, 1)

    results = {"variants": []}
    preds = {}
    folds = 0
    for name, dims, k, refit in variants:
        t1 = time.time()
        pr = run_variant(s, dims, k, refit, with_iso=(name == "core+term" or args.quick))
        folds = max(folds, pr.attrs["folds"])
        preds[name] = pr
        v = {"name": name, "dims": dims, "k": k, "refit": refit,
             "ll_first": pooled_ll(pr, first), "ll_second": pooled_ll(pr, second),
             "brier_second": pooled_brier(pr, second)}
        results["variants"].append(v)
        print(f"  {name}: log loss {v['ll_first']:.4f} / {v['ll_second']:.4f}  ({time.time() - t1:.0f}s)")
    rv_pr = pd.DataFrame({name: lognormal_prob(s, kind, z) for name, kind, z in EVENTS})
    valid = ~np.isnan(preds[variants[0][0]][EVENTS[0][0]].values)
    results["rv_first"] = pooled_ll(rv_pr, first & valid)
    results["rv_second"] = pooled_ll(rv_pr, second & valid)
    results["rv_brier_second"] = pooled_brier(rv_pr, second & valid)
    best = min(results["variants"], key=lambda v: v["ll_first"])["name"]
    if "__iso" not in "".join(preds[best].columns):
        # isotonic column is computed for core+term; recompute for the chosen variant
        v = next(v for v in variants if v[0] == best)
        preds[best] = run_variant(s, v[1], v[2], v[3], with_iso=True)
    pr = preds[best]
    test = dec >= np.datetime64(TEST_FROM)
    results["by_event"] = score_table(s, pr, test)
    results["implied"] = implied_table(s, pr, test)
    results["breakdowns"] = {ev: {by: breakdown(s, pr, test, ev, by)
                                  for by in ("direction", "market_b", "vol", "term", "horizon", "asset")}
                             for ev in ("put_survive_0.75", "call_survive_0.75", "put_debit_win", "call_debit_win")}
    meta = {"generated": time.strftime("%Y-%m-%d"), "last_date": str(s["outcome"].max().date()),
            "n_tickers": len(data), "folds": folds}
    write_report(Path(args.report), s, results, best, meta)
    print(f"Report: {args.report}  (chosen: {best}, {time.time() - t0:.0f}s total)")
    if args.save:
        v = next(v for v in variants if v[0] == best)
        feats = {d: s["market_b" if d == "market" else d].values for d in v[1]}
        feats["horizon"] = s["horizon"].astype(str).values
        enc = pm.Encoded(v[1], feats, s["x"].values, s["x_min"].values, s["x_max"].values,
                         s["weight"].values)
        model = pm.fit(enc, None, v[2], fit_through=str(s["outcome"].max().date()))
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(model.to_dict()))
        print(f"Model saved: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

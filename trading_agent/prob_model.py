"""prob_model.py — calibrated probability model, walk-forward (skill 66).

Added 2026-10-16 (phase 1: research). The agent's POP comes from option
delta (the market's implied odds) or ``shadow_pop.pop_rv`` (zero-drift
lognormal on 20-day realized vol). The 2026-10-13 study showed both are
biased in a stable way — short puts survived ~10 points more often than
implied, short calls 2–6 points less. This module estimates the odds
*conditionally* and only from data available at decision time.

**What is modelled.** Not one table per strategy (too sparse), but the
distribution of the standardized move

    x = ln(S_T / S_0) / (σ_rv · √(h/365))

where σ_rv is the 20-day realized vol known on the decision day and h the
calendar days to expiry — plus the standardized lowest low and highest
high on the way (for touch / stop odds). Any strategy's probability is
read off these at its own strikes: a short put at strike K survives when
x > ln(K/S_0)/(σ_rv√t), a put debit pays when x < its breakeven's z, and
so on (``prob_above`` / ``prob_below`` / ``prob_touch_*``).

**Conditioning.** Samples are grouped by an ordered list of dimensions
(default: horizon, asset group, direction, volatility regime, market
regime — term structure and SKEW were tested and dropped). Each level's empirical CDF on ``Z_GRID`` is shrunk
toward its parent level's estimate:

    F_k = (n_eff · F_raw + K · F_parent) / (n_eff + K)

with ``n_eff`` = Σ 1/overlap (h daily samples share one h-day path) and K
the shrinkage strength — a thin cell borrows from its parent instead of
returning a wild number. A cell never seen in training gets its parent's
estimate.

**Walk-forward.** ``walk_forward`` fits on samples whose *outcome* date is
before the test month starts and predicts that month; every reported
score is out-of-sample. ``isotonic_fit`` / ``isotonic_apply`` recalibrate
a family of predictions using only earlier months' out-of-sample results.

Pure numpy; no I/O, no network. The research driver is
``scripts/research/calibrate_prob_model.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Z_GRID = np.round(np.arange(-3.0, 3.0001, 0.1), 2)
HORIZONS = (7, 14, 21, 30, 45)
# Chosen walk-forward 2026-10-16 (docs/plans/research_prob_model.md): term
# structure and SKEW did not improve out-of-sample log loss; K = 100 beat 10 / 30.
DEFAULT_DIMS = ("horizon", "asset", "direction", "vol", "market")
DEFAULT_K = 100.0
TRADING_DAYS_PER_CAL_DAY = 252.0 / 365.0

ASSET_GROUP = {
    **{t: "broad" for t in ("SPY", "QQQ", "IWM", "DIA", "EEM")},
    **{t: "sector" for t in ("XLF", "XLE", "XLK", "XLV", "XLY", "XLI", "XLU", "SMH")},
    **{t: "alt" for t in ("TLT", "GLD", "GDX")},
}


def asset_group(ticker: str) -> str:
    """broad / sector / alt (bonds, gold, miners) / single (stocks)."""
    return ASSET_GROUP.get(ticker.upper(), "single")


# ── Feature buckets (all inputs known on the decision day) ─────────────

def vol_bucket(rank: Optional[float]) -> str:
    """Trailing-year percentile of 20-day realized vol → low / mid / high."""
    if rank is None or not math.isfinite(rank):
        return "na"
    return "low" if rank < 1 / 3 else "high" if rank > 2 / 3 else "mid"


def term_bucket(ratio: Optional[float]) -> str:
    """VIX / VIX3M → contango (< 0.90) / flat (< 1.00) / inverted."""
    if ratio is None or not math.isfinite(ratio):
        return "na"
    return "contango" if ratio < 0.90 else "flat" if ratio < 1.00 else "inverted"


def skew_bucket(rank: Optional[float]) -> str:
    """Trailing-year percentile of the CBOE SKEW index → low / mid / high."""
    return vol_bucket(rank)


def market_bucket(state: Optional[str]) -> str:
    """Market risk state (skill 58) → normal / caution / stress / recovery."""
    s = state.upper() if isinstance(state, str) else ""
    if s in ("DEFENSIVE", "CAPITULATION"):
        return "stress"
    return {"NORMAL": "normal", "CAUTION": "caution", "RECOVERY": "recovery"}.get(s, "na")


def direction_bucket(regime: Optional[str]) -> str:
    """Regime label (bullish / bearish / sideways / mean_reversion) → bull / bear / side / mr."""
    r = regime.lower() if isinstance(regime, str) else ""      # NaN from pandas → "na"
    return {"bullish": "bull", "bearish": "bear", "sideways": "side",
            "mean_reversion": "mr"}.get(r, "na")


def standardized(spot: float, level: float, sigma: float, dte: float) -> float:
    """z of ``level`` relative to ``spot`` in realized-vol units over ``dte`` days."""
    return math.log(level / spot) / (sigma * math.sqrt(max(dte, 1) / 365.0))


# ── Baselines ─────────────────────────────────────────────────────────

def _phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def lognormal_prob_above(z, sigma_ratio=1.0, vol_sqrt_t=0.0):
    """P(x > z) when the true move is zero-drift lognormal with vol σ*
    (``sigma_ratio`` = σ_rv / σ*, ``vol_sqrt_t`` = σ*·√t). With
    sigma_ratio = 1 this is ``shadow_pop.pop_rv``; with σ* = implied vol it
    is the delta-style implied probability."""
    zs = np.asarray(z, dtype=float) * np.asarray(sigma_ratio, dtype=float)
    return 1.0 - _phi(zs + 0.5 * np.asarray(vol_sqrt_t, dtype=float))


# ── Model ─────────────────────────────────────────────────────────────

@dataclass
class LevelTable:
    """Shrunk CDFs for every cell of one hierarchy level."""
    keys: List[Tuple[str, ...]]
    n_eff: np.ndarray                      # (cells,)
    f_term: np.ndarray                     # (cells, G)  P(x ≤ z)
    f_min: np.ndarray                      # (cells, G)  P(x_min ≤ z)
    f_max: np.ndarray                      # (cells, G)  P(x_max ≤ z)


@dataclass
class ProbModel:
    """Fitted hierarchy. ``levels[k]`` conditions on ``dims[:k]``."""
    dims: Tuple[str, ...]
    k: float
    levels: List[LevelTable] = field(default_factory=list)
    fit_through: str = ""                  # last outcome date used (ISO)

    def _lookup(self, values: Dict[str, str]) -> Tuple[int, int]:
        """(level, cell) of the deepest cell present for these feature values."""
        best = (0, 0)
        for k in range(1, len(self.dims) + 1):
            key = tuple(str(values.get(d, "na")) for d in self.dims[:k])
            idx = self._index[k].get(key)
            if idx is None:
                break
            best = (k, idx)
        return best

    def __post_init__(self):
        self._index: List[Dict[Tuple[str, ...], int]] = []

    def finalize(self) -> "ProbModel":
        self._index = [{key: i for i, key in enumerate(t.keys)} for t in self.levels]
        return self

    def cdfs(self, values: Dict[str, str]) -> Dict[str, Any]:
        k, i = self._lookup(values)
        t = self.levels[k]
        return {"level": k, "n_eff": float(t.n_eff[i]), "f_term": t.f_term[i],
                "f_min": t.f_min[i], "f_max": t.f_max[i]}

    def prob_above(self, values: Dict[str, str], z: float) -> float:
        """P(x_T > z) — e.g. a short put at z < 0 finishing out of the money."""
        return float(1.0 - _interp(self.cdfs(values)["f_term"], z))

    def prob_below(self, values: Dict[str, str], z: float) -> float:
        return float(_interp(self.cdfs(values)["f_term"], z))

    def prob_touch_down(self, values: Dict[str, str], z: float) -> float:
        """P(lowest low ≤ z) before expiry (z < 0)."""
        return float(_interp(self.cdfs(values)["f_min"], z))

    def prob_touch_up(self, values: Dict[str, str], z: float) -> float:
        """P(highest high > z) before expiry (z > 0)."""
        return float(1.0 - _interp(self.cdfs(values)["f_max"], z))

    def to_dict(self) -> Dict[str, Any]:
        return {"dims": list(self.dims), "k": self.k, "fit_through": self.fit_through,
                "z_grid": Z_GRID.tolist(),
                "levels": [{"keys": [list(k) for k in t.keys],
                            "n_eff": np.round(t.n_eff, 2).tolist(),
                            "f_term": np.round(t.f_term, 5).tolist(),
                            "f_min": np.round(t.f_min, 5).tolist(),
                            "f_max": np.round(t.f_max, 5).tolist()} for t in self.levels]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProbModel":
        m = cls(tuple(d["dims"]), float(d["k"]), fit_through=d.get("fit_through", ""))
        m.levels = [LevelTable([tuple(k) for k in t["keys"]], np.array(t["n_eff"]),
                               np.array(t["f_term"]), np.array(t["f_min"]), np.array(t["f_max"]))
                    for t in d["levels"]]
        return m.finalize()


def _interp(f: np.ndarray, z: float) -> float:
    """CDF at z by linear interpolation on Z_GRID (clamped at the ends)."""
    return float(np.interp(z, Z_GRID, f))


def _bins(x: np.ndarray) -> np.ndarray:
    """Index b with x ≤ Z_GRID[j] ⇔ b ≤ j (G = beyond the grid)."""
    return np.searchsorted(Z_GRID, x, side="left")


class Encoded:
    """Samples pre-encoded once for fast repeated fits (walk-forward).

    ``features`` maps each dimension to an array of string labels;
    ``x_term`` / ``x_min`` / ``x_max`` are standardized moves; ``weight``
    is 1 / overlap (trading days in the window)."""

    def __init__(self, dims: Sequence[str], features: Dict[str, np.ndarray],
                 x_term: np.ndarray, x_min: np.ndarray, x_max: np.ndarray,
                 weight: np.ndarray):
        self.dims = tuple(dims)
        self.n = len(x_term)
        self.b_term, self.b_min, self.b_max = _bins(x_term), _bins(x_min), _bins(x_max)
        self.weight = np.asarray(weight, dtype=float)
        self.codes: List[np.ndarray] = [np.zeros(self.n, dtype=np.int64)]
        self.keys: List[List[Tuple[str, ...]]] = [[()]]
        self.parent: List[np.ndarray] = [np.zeros(1, dtype=np.int64)]
        cols = [np.asarray(features[d]).astype(str) for d in self.dims]
        for k in range(1, len(self.dims) + 1):
            joined = np.array(["|".join(t) for t in zip(*cols[:k])]) if k > 1 else cols[0]
            uniq, code = np.unique(joined, return_inverse=True)
            self.codes.append(code.astype(np.int64))
            self.keys.append([tuple(u.split("|")) for u in uniq])
            # parent cell (level k-1) of each level-k cell
            par = np.zeros(len(uniq), dtype=np.int64)
            par[code] = self.codes[k - 1]
            self.parent.append(par)


def _cdf_counts(code: np.ndarray, bins: np.ndarray, w: np.ndarray, n_cells: int) -> np.ndarray:
    g1 = len(Z_GRID) + 1
    c = np.bincount(code * g1 + bins, weights=w, minlength=n_cells * g1).reshape(n_cells, g1)
    return np.cumsum(c, axis=1)[:, :-1]          # (cells, G): weight with x ≤ z_j


def fit(enc: Encoded, mask: Optional[np.ndarray] = None, k: float = DEFAULT_K,
        fit_through: str = "") -> ProbModel:
    """Fit the shrunk hierarchy on the rows in ``mask`` (all when None)."""
    m = np.ones(enc.n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    w = enc.weight * m
    model = ProbModel(enc.dims, float(k), fit_through=fit_through)
    prev: Optional[LevelTable] = None
    for lvl in range(len(enc.dims) + 1):
        code, n_cells = enc.codes[lvl], len(enc.keys[lvl])
        n_eff = np.bincount(code, weights=w, minlength=n_cells)
        safe = np.where(n_eff > 0, n_eff, 1.0)[:, None]
        raw = [_cdf_counts(code, b, w, n_cells) / safe for b in (enc.b_term, enc.b_min, enc.b_max)]
        if prev is None:
            shrunk = raw
        else:
            par = enc.parent[lvl]
            pf = [prev.f_term[par], prev.f_min[par], prev.f_max[par]]
            a = (n_eff / (n_eff + k))[:, None]
            shrunk = [a * r + (1 - a) * p for r, p in zip(raw, pf)]
        prev = LevelTable(enc.keys[lvl], n_eff, *shrunk)
        model.levels.append(prev)
    return model.finalize()


def predict_cells(model: ProbModel, enc: Encoded, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f_term, f_min, f_max) CDF rows for the given sample indices, from
    the deepest level (cells unseen in training already equal their parent)."""
    t = model.levels[-1]
    c = enc.codes[-1][rows]
    return t.f_term[c], t.f_min[c], t.f_max[c]


def cdf_at(f_rows: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Row-wise CDF at per-row z (linear on Z_GRID, clamped)."""
    z = np.clip(np.asarray(z, dtype=float), Z_GRID[0], Z_GRID[-1])
    pos = (z - Z_GRID[0]) / (Z_GRID[1] - Z_GRID[0])
    lo = np.floor(pos).astype(int).clip(0, len(Z_GRID) - 1)
    hi = (lo + 1).clip(0, len(Z_GRID) - 1)
    frac = pos - lo
    r = np.arange(len(z))
    return f_rows[r, lo] * (1 - frac) + f_rows[r, hi] * frac


# ── Walk-forward ──────────────────────────────────────────────────────

def month_starts(first: np.datetime64, last: np.datetime64) -> List[np.datetime64]:
    m = first.astype("datetime64[M]")
    out = []
    while m <= last.astype("datetime64[M]"):
        out.append(m.astype("datetime64[D]"))
        m = m + np.timedelta64(1, "M")
    return out


def walk_forward(enc: Encoded, decision: np.ndarray, outcome: np.ndarray,
                 test_from: np.datetime64, k: float = DEFAULT_K,
                 on_fold: Optional[Callable[[np.datetime64, ProbModel, np.ndarray], None]] = None,
                 refit_months: int = 1) -> int:
    """For each month from ``test_from``: fit on samples whose outcome
    date < month start, call ``on_fold(month_start, model, test_rows)``
    with the rows decided in that month. Returns the number of folds."""
    decision = decision.astype("datetime64[D]")
    outcome = outcome.astype("datetime64[D]")
    months = month_starts(test_from, decision.max())
    model = None
    folds = 0
    for i, start in enumerate(months):
        end = (start.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
        rows = np.nonzero((decision >= start) & (decision < end))[0]
        if not len(rows):
            continue
        if model is None or i % refit_months == 0:
            train = outcome < start
            if not train.any():
                continue
            model = fit(enc, train, k, fit_through=str(outcome[train].max()))
        on_fold(start, model, rows)
        folds += 1
    return folds


# ── Isotonic recalibration (pool-adjacent-violators) ──────────────────

def isotonic_fit(pred: np.ndarray, outcome: np.ndarray, bins: int = 50
                 ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Monotone map from predicted to realized frequency, fitted on
    quantile bins of earlier out-of-sample predictions. None below 200 rows."""
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if len(pred) < 200:
        return None
    order = np.argsort(pred)
    groups = np.array_split(order, min(bins, len(pred)))
    x = np.array([pred[g].mean() for g in groups])
    v = np.array([y[g].mean() for g in groups])
    w = np.array([len(g) for g in groups], dtype=float)
    # PAV
    vals, wts, xs = list(v), list(w), [[xi] for xi in x]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            tot = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            wts[i] = tot
            xs[i] = xs[i] + xs[i + 1]
            del vals[i + 1], wts[i + 1], xs[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    knots = np.array([np.mean(g) for g in xs])
    return knots, np.array(vals)


def isotonic_apply(cal: Optional[Tuple[np.ndarray, np.ndarray]], pred: np.ndarray) -> np.ndarray:
    if cal is None:
        return np.asarray(pred, dtype=float)
    knots, vals = cal
    if len(knots) == 1:
        return np.full(len(pred), vals[0])
    return np.interp(pred, knots, vals)


# ── Scores ────────────────────────────────────────────────────────────

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def block_bootstrap_diff(a_loss: np.ndarray, b_loss: np.ndarray, blocks: np.ndarray,
                         n_boot: int = 500, seed: int = 7) -> Tuple[float, float, float]:
    """Mean of (a − b) per-sample losses with a 90 % interval from
    resampling whole blocks (months) — overlapping samples share paths."""
    d = np.asarray(a_loss) - np.asarray(b_loss)
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ub), len(ub))
        stats.append(sums[pick].sum() / cnts[pick].sum())
    lo, hi = np.percentile(stats, [5, 95])
    return float(d.mean()), float(lo), float(hi)

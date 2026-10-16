"""Skill 66 phase 1 — calibrated probability model core."""
from __future__ import annotations

import json

import numpy as np
import pytest

from trading_agent import prob_model as pm


def _enc(x, dims=("horizon", "direction"), feats=None, xmin=None, xmax=None, w=None):
    n = len(x)
    feats = feats or {"horizon": np.array(["30"] * n), "direction": np.array(["bull"] * n)}
    x = np.asarray(x, dtype=float)
    return pm.Encoded(dims, feats, x, x if xmin is None else np.asarray(xmin, float),
                      x if xmax is None else np.asarray(xmax, float),
                      np.ones(n) if w is None else np.asarray(w, float))


def test_buckets():
    assert [pm.vol_bucket(v) for v in (0.1, 0.5, 0.9, None)] == ["low", "mid", "high", "na"]
    assert [pm.term_bucket(v) for v in (0.85, 0.95, 1.05)] == ["contango", "flat", "inverted"]
    assert pm.market_bucket("CAPITULATION") == "stress" and pm.market_bucket("NORMAL") == "normal"
    assert pm.direction_bucket("mean_reversion") == "mr" and pm.asset_group("META") == "single"
    assert pm.asset_group("XLU") == "sector" and pm.asset_group("TLT") == "alt"
    assert pm.direction_bucket(float("nan")) == "na" and pm.market_bucket(float("nan")) == "na"


def test_standardized_and_rv_baseline_match_shadow_pop():
    from trading_agent.shadow_pop import prob_above
    spot, k, sig, dte = 100.0, 95.0, 0.2, 30
    z = pm.standardized(spot, k, sig, dte)
    base = float(pm.lognormal_prob_above(z, 1.0, sig * np.sqrt(dte / 365)))
    assert base == pytest.approx(prob_above(spot, k, dte, sig), abs=1e-9)


def test_empirical_cdf_on_the_grid():
    m = pm.fit(_enc([-1.0, 0.0, 0.0, 1.0]), k=0.0)
    vals = {"horizon": "30", "direction": "bull"}
    assert m.prob_below(vals, 0.0) == pytest.approx(0.75)        # x ≤ 0
    assert m.prob_above(vals, 0.5) == pytest.approx(0.25)
    assert m.prob_above(vals, -2.0) == pytest.approx(1.0)


def test_thin_cell_shrinks_to_parent():
    feats = {"horizon": np.array(["30"] * 101), "direction": np.array(["bull"] * 100 + ["bear"])}
    x = np.r_[np.zeros(100) + 1.0, -2.0]                        # parent: 100 at +1, one bear at −2
    m = pm.fit(_enc(x, feats=feats), k=30.0)
    bear = {"horizon": "30", "direction": "bear"}
    # raw bear cell says P(x > 0) = 0; with n_eff 1 and K 30 it stays near the parent's ≈ 0.99
    assert 0.9 < m.prob_above(bear, 0.0) < 1.0
    # the 100-sample bull cell barely moves (parent includes the bear)
    assert 0.99 < m.prob_above({"horizon": "30", "direction": "bull"}, 0.0) < 1.0


def test_unseen_cell_uses_parent():
    m = pm.fit(_enc([0.5, -0.5]), k=30.0)
    assert m.cdfs({"horizon": "30", "direction": "side"})["level"] == 1
    assert m.prob_above({"horizon": "30", "direction": "side"}, 0.0) == pytest.approx(0.5)


def test_weights_count_as_effective_samples():
    m = pm.fit(_enc([1.0, -1.0], w=[3.0, 1.0]), k=0.0)
    assert m.prob_above({"horizon": "30", "direction": "bull"}, 0.0) == pytest.approx(0.75)
    assert m.levels[0].n_eff[0] == pytest.approx(4.0)


def test_touch_probabilities_use_path_extremes():
    enc = _enc([0.0, 0.0], xmin=[-1.5, -0.2], xmax=[0.3, 1.2])
    m = pm.fit(enc, k=0.0)
    v = {"horizon": "30", "direction": "bull"}
    assert m.prob_touch_down(v, -1.0) == pytest.approx(0.5)
    assert m.prob_touch_up(v, 1.0) == pytest.approx(0.5)


def test_mask_excludes_rows():
    enc = _enc([1.0, 1.0, -1.0])
    m = pm.fit(enc, np.array([True, True, False]), k=0.0)
    assert m.prob_above({"horizon": "30", "direction": "bull"}, 0.0) == pytest.approx(1.0)


def test_walk_forward_trains_only_on_known_outcomes():
    """Fold for month M sees only samples whose outcome date < M's first day."""
    dec = np.array(["2020-01-10", "2020-01-25", "2020-02-10", "2020-03-05"], dtype="datetime64[D]")
    out = np.array(["2020-02-09", "2020-02-24", "2020-03-11", "2020-04-04"], dtype="datetime64[D]")
    enc = _enc([1.0, 1.0, -1.0, -1.0])
    seen = {}

    def on_fold(start, model, rows):
        seen[str(start)] = (model.levels[0].n_eff[0], model.fit_through, list(rows))

    pm.walk_forward(enc, dec, out, np.datetime64("2020-02-01"), k=0.0, on_fold=on_fold)
    assert "2020-02-01" not in seen                             # nothing known before February
    assert seen["2020-03-01"][0] == pytest.approx(2.0)          # both January outcomes (Feb 9, Feb 24)
    assert seen["2020-03-01"][1] == "2020-02-24" and seen["2020-03-01"][2] == [3]


def test_cdf_at_rowwise_and_clamped():
    f = np.vstack([np.linspace(0, 1, len(pm.Z_GRID))] * 2)
    v = pm.cdf_at(f, np.array([0.0, 10.0]))
    assert v[0] == pytest.approx(0.5) and v[1] == pytest.approx(1.0)


def test_isotonic_recalibrates_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 5000)
    y = (rng.uniform(size=5000) < (0.5 + (p - 0.5) * 0.5)).astype(float)   # truth flatter than p
    cal = pm.isotonic_fit(p, y)
    adj = pm.isotonic_apply(cal, np.array([0.95]))[0]
    assert 0.65 < adj < 0.8
    assert np.all(np.diff(cal[1]) >= -1e-12)                    # monotone
    assert pm.isotonic_fit(p[:50], y[:50]) is None


def test_scores_and_bootstrap():
    y = np.array([1, 0, 1, 1.0])
    assert pm.brier(np.array([1, 0, 1, 1.0]), y) == 0.0
    assert pm.log_loss(np.full(4, 0.5), y) == pytest.approx(np.log(2))
    mean, lo, hi = pm.block_bootstrap_diff(np.zeros(4), np.ones(4), np.array([1, 1, 2, 2]))
    assert mean == -1.0 and lo == -1.0 and hi == -1.0


def test_model_round_trips_through_json():
    m = pm.fit(_enc([-1.0, 0.0, 1.0]), k=10.0, fit_through="2026-10-15")
    m2 = pm.ProbModel.from_dict(json.loads(json.dumps(m.to_dict())))
    v = {"horizon": "30", "direction": "bull"}
    assert m2.prob_above(v, 0.0) == pytest.approx(m.prob_above(v, 0.0), abs=1e-5)
    assert m2.fit_through == "2026-10-15" and m2.dims == ("horizon", "direction")

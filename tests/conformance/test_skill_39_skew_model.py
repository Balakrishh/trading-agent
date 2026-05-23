"""Conformance test: skill 39 — volatility skew in the backtester.

Pins SkewModel.sigma_for_strike math + the threading of the skew
model through BacktestRunner → run_one_cycle → build_chain_slice.

SkewModel is a pure dataclass with no project dependencies, so it
can be exercised in the sandbox without scipy/pandas.

Failure modes caught:
- Someone removes the skew_model param from build_chain_slice →
  per-strike IV reverts to flat, backtest under-prices OTM puts again.
- Someone reorders the put_skew/call_skew args → defaults to
  call-side bias, wrong shape entirely.
- Someone changes the moneyness formula (e.g., log instead of
  linear) → numerical drift on every backtest.
"""

from __future__ import annotations
import sys
import importlib.util


def _load_skew_module():
    """Load skew_model.py without triggering trading_agent/__init__
    (which pulls in pandas_market_calendars via position_monitor).
    Test sandboxes can therefore exercise the pure math even when
    runtime deps are missing."""
    spec = importlib.util.spec_from_file_location(
        "_skew_under_test",
        "trading_agent/backtest/skew_model.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_skew_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_skill_39_flat_skew_returns_atm_sigma() -> None:
    """Skill 39 §1: FLAT_SKEW preserves legacy flat-vol behavior —
    every strike gets atm_sigma unchanged. Default for backward
    compat."""
    m = _load_skew_module()
    sigma = 0.20
    flat = m.FLAT_SKEW
    # ATM
    assert flat.sigma_for_strike(100, 100, sigma) == sigma
    # 10% OTM put — flat returns atm_sigma
    assert flat.sigma_for_strike(90, 100, sigma) == sigma
    # 10% OTM call — flat returns atm_sigma
    assert flat.sigma_for_strike(110, 100, sigma) == sigma


def test_skill_39_otm_put_sigma_rises_with_distance() -> None:
    """Skill 39 §2: OTM-put σ is lifted proportional to put_skew × |m|."""
    m = _load_skew_module()
    sigma_atm = 0.20
    skew = m.SkewModel(put_skew=0.5, call_skew=0.1)
    # 5% OTM put: m = -0.05 → lift = 0.5 × 0.05 = 0.025
    s = skew.sigma_for_strike(strike=95, spot=100, atm_sigma=sigma_atm)
    expected = sigma_atm * (1 + 0.5 * 0.05)
    assert abs(s - expected) < 1e-9, (
        f"Skill 39 §2: 5%-OTM put with put_skew=0.5 must yield "
        f"σ = atm × 1.025 = {expected}. Got {s}."
    )
    # 10% OTM put: lift = 0.5 × 0.10 = 0.05
    s2 = skew.sigma_for_strike(strike=90, spot=100, atm_sigma=sigma_atm)
    expected2 = sigma_atm * (1 + 0.5 * 0.10)
    assert abs(s2 - expected2) < 1e-9


def test_skill_39_otm_call_sigma_rises_with_distance() -> None:
    """Skill 39 §2: OTM-call side uses call_skew which is typically
    much smaller than put_skew for index ETFs."""
    m = _load_skew_module()
    sigma_atm = 0.20
    skew = m.SkewModel(put_skew=0.5, call_skew=0.1)
    # 5% OTM call: m = +0.05 → lift = 0.1 × 0.05 = 0.005
    s = skew.sigma_for_strike(strike=105, spot=100, atm_sigma=sigma_atm)
    expected = sigma_atm * (1 + 0.1 * 0.05)
    assert abs(s - expected) < 1e-9


def test_skill_39_skew_clips_to_safe_band() -> None:
    """Skill 39 §2: even adversarially-deep OTM strikes must not
    produce σ outside [0.01, 5.0]. Crashes the BS pricer otherwise."""
    m = _load_skew_module()
    # Massive put_skew + far-OTM put — would extrapolate to absurd σ
    extreme = m.SkewModel(put_skew=100.0, call_skew=0.0)
    s = extreme.sigma_for_strike(strike=50, spot=100, atm_sigma=0.20)
    assert s <= 5.0, (
        f"Skill 39 §2: σ must clip at 5.0 to keep BS pricer sane. "
        f"Got {s}."
    )
    # Negative atm_sigma input — return safely
    s2 = extreme.sigma_for_strike(strike=100, spot=100, atm_sigma=0.0)
    assert s2 == 0.0


def test_skill_39_put_skew_only_affects_otm_puts() -> None:
    """Skill 39 §2: a model with put_skew=1.0 and call_skew=0.0 must
    leave ATM and OTM-call strikes at atm_sigma."""
    m = _load_skew_module()
    sigma_atm = 0.20
    skew = m.SkewModel(put_skew=1.0, call_skew=0.0)
    # ATM
    assert abs(skew.sigma_for_strike(100, 100, sigma_atm)
               - sigma_atm) < 1e-9
    # 10% OTM call — call_skew=0 → no lift
    assert abs(skew.sigma_for_strike(110, 100, sigma_atm)
               - sigma_atm) < 1e-9
    # 10% OTM put — put_skew=1.0 → lift = 0.10
    assert abs(skew.sigma_for_strike(90, 100, sigma_atm)
               - sigma_atm * 1.10) < 1e-9


def test_skill_39_skew_model_is_frozen_dataclass() -> None:
    """Skill 39 §3: SkewModel is a frozen dataclass — operators can't
    accidentally mutate the model mid-backtest."""
    import dataclasses
    m = _load_skew_module()
    skew = m.SkewModel(put_skew=0.5, call_skew=0.1)
    # Frozen → assignment must raise FrozenInstanceError
    try:
        skew.put_skew = 0.9
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SkewModel must be frozen (skill 39 §3)")


def test_skill_39_canonical_presets_have_reasonable_shape() -> None:
    """Skill 39 §3: built-in presets must order correctly —
    FLAT < INDEX_ETF < SINGLE_STOCK in put-side steepness, and all
    have call_skew < put_skew (equity skew bias)."""
    m = _load_skew_module()
    assert m.FLAT_SKEW.put_skew == 0.0
    assert m.INDEX_ETF_SKEW.put_skew > 0
    assert m.SINGLE_STOCK_SKEW.put_skew > m.INDEX_ETF_SKEW.put_skew
    # Equity-skew bias: put-side always steeper
    for preset in (m.INDEX_ETF_SKEW, m.SINGLE_STOCK_SKEW):
        assert preset.put_skew > preset.call_skew, (
            f"{preset}: equity skew has put_skew > call_skew always."
        )


def test_skill_39_chain_slice_accepts_skew_model() -> None:
    """Skill 39 §3: build_chain_slice must accept skew_model kwarg
    and (when None) preserve legacy flat-vol behavior."""
    import inspect
    # Source-level pin (avoids needing scipy to exercise the function)
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[2]
        / "trading_agent" / "backtest" / "synthetic_chain.py"
    ).read_text(encoding="utf-8")
    assert "skew_model=None" in src, (
        "Skill 39 §3: build_chain_slice must accept skew_model=None "
        "kwarg. Without it, BacktestRunner cannot opt operators "
        "into the skew model."
    )
    assert "skew_model.sigma_for_strike(" in src, (
        "Skill 39 §3: build_chain_slice must call "
        "skew_model.sigma_for_strike(...) when the model is "
        "supplied. Otherwise the kwarg is dead code."
    )


def test_skill_39_runner_exposes_skew_model_kwarg() -> None:
    """Skill 39 §3: BacktestRunner exposes skew_model so operators
    can sweep flat vs INDEX_ETF_SKEW vs custom calibrations."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[2]
        / "trading_agent" / "backtest" / "runner.py"
    ).read_text(encoding="utf-8")
    assert "skew_model=None" in src, (
        "Skill 39 §3: BacktestRunner.__init__ must accept "
        "skew_model=None kwarg."
    )
    assert "self.skew_model = skew_model" in src, (
        "Skill 39 §3: runner must store skew_model on self so "
        "_handle_intraday_decision can thread it through."
    )
    assert "skew_model=self.skew_model" in src, (
        "Skill 39 §3: runner must pass self.skew_model to "
        "run_one_cycle. Otherwise the kwarg never reaches "
        "build_chain_slice."
    )

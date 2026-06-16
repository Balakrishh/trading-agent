"""Conformance tests for skill 40 — long-term options evaluator.

Walking-skeleton scope = covered-call (income overlay) only. CSP / LEAPS /
PMCC / debit-spread tests land in this same file next session.

Pinned behaviors:

- _score_covered_call accepts the §2.1 sample math — §2.1, §3.2
- Per-leg gates fire in the documented order — §4
- Score = annualised_return × pop (skill 01 reuse) — §2.1
- LongTermEvaluator.recommend skips qty < 100 holdings — §4
- LongTermEvaluator does NOT write or submit orders — §4 (read-only contract)
- Recommendation TP < entry_limit; stop_trigger sanity — §4
- _score_covered_call lives in decision_engine.py — CI invariant 2

Quote the same fields and shapes the skill cites; if those change, the test
fails and forces a skill update.
"""

from __future__ import annotations

import pytest

from trading_agent.decision_engine import (
    LT_REJECT_CREDIT_NON_POSITIVE_LT,
    LT_REJECT_DTE_OUT_OF_BAND,
    LT_REJECT_IV_RANK_TOO_LOW,
    LT_REJECT_SHORT_DELTA_TOO_HIGH,
    LT_REJECT_STRIKE_BELOW_COST_BASIS,
    _score_covered_call,
    _score_covered_call_with_reason,
)
from trading_agent.long_term_evaluator import (
    EvaluatorConfig,
    LongTermEvaluator,
    Recommendation,
    RecommendationLeg,
)
from trading_agent.positions_provider import (
    ManualPositionsProvider,
    Position,
)


# ---------------------------------------------------------------------------
# §2.1, §3.2 — _score_covered_call scoring math
# ---------------------------------------------------------------------------

class _StubPreset:
    """Duck-typed PresetConfig stand-in. Skill 40 §3.3 fields."""
    cc_max_short_delta = 0.30
    cc_dte_band = (30, 60)
    cc_min_iv_rank = 0.25


def _short_call(*, strike=220.0, delta=0.20, bid=1.20, ask=1.30,
                dte=45, iv_rank=0.35, symbol="AAPL  271015C00220000"):
    return {
        "strike": strike, "delta": delta,
        "bid": bid, "ask": ask, "dte": dte,
        "iv_rank": iv_rank, "symbol": symbol,
    }


def test_score_covered_call_happy_path():
    """§2.1 — score = annualised_return × pop, gates pass."""
    out = _score_covered_call(
        short_call=_short_call(),
        cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert out is not None
    score, metrics, _ = out
    # credit = mid(1.20, 1.30) − $0.02 haircut = $1.23 → × 100 = $123
    assert metrics["credit"] == pytest.approx(123.0, abs=0.01)
    # capital_at_risk = $200 × 100 − $123 = $19,877
    assert metrics["capital_at_risk"] == pytest.approx(19_877.0, abs=0.01)
    assert metrics["pop"] == pytest.approx(0.80, abs=1e-6)  # 1 − |0.20|
    assert metrics["dte"] == 45.0
    # score = annualised_return × pop > 0
    assert score > 0


def test_strike_below_cost_basis_rejects():
    """§4 — never write a CC strike below cost basis × 1.01."""
    out = _score_covered_call(
        short_call=_short_call(strike=199.0),    # < 200 × 1.01 = 202
        cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert out is None
    verbose = _score_covered_call_with_reason(
        short_call=_short_call(strike=199.0),
        cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert verbose["status"] == "rejected"
    assert verbose["reason"] == LT_REJECT_STRIKE_BELOW_COST_BASIS


def test_short_delta_above_cap_rejects():
    """§2.1 gate: |Δ_short| ≤ cc_max_short_delta."""
    verbose = _score_covered_call_with_reason(
        short_call=_short_call(delta=0.40),
        cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert verbose["status"] == "rejected"
    assert verbose["reason"] == LT_REJECT_SHORT_DELTA_TOO_HIGH


def test_dte_outside_band_rejects():
    """§2.1 gate: dte ∈ cc_dte_band (default [30, 60])."""
    assert _score_covered_call(
        short_call=_short_call(dte=14), cost_basis=200.0,
        preset=_StubPreset(),
    ) is None
    verbose = _score_covered_call_with_reason(
        short_call=_short_call(dte=14), cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert verbose["reason"] == LT_REJECT_DTE_OUT_OF_BAND


def test_credit_non_positive_rejects():
    """§2.1 gate: credit > 0 (zero-bid / zero-ask = no credit)."""
    verbose = _score_covered_call_with_reason(
        short_call=_short_call(bid=0.0, ask=0.0), cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert verbose["reason"] == LT_REJECT_CREDIT_NON_POSITIVE_LT


def test_iv_rank_gate_only_fires_when_supplied():
    """§4 — fail-open when iv_rank missing; fail-closed when supplied below cap."""
    # iv_rank present but below cap → reject
    verbose = _score_covered_call_with_reason(
        short_call=_short_call(iv_rank=0.10), cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert verbose["reason"] == LT_REJECT_IV_RANK_TOO_LOW

    # iv_rank absent → pass (fail-open)
    sc = _short_call()
    del sc["iv_rank"]
    verbose2 = _score_covered_call_with_reason(
        short_call=sc, cost_basis=200.0, preset=_StubPreset(),
    )
    assert verbose2["status"] == "accepted"


def test_score_uses_pop_from_delta_skill_01():
    """§2.1 — pop = 1 − |Δ|, reused from skill 01."""
    out = _score_covered_call(
        short_call=_short_call(delta=-0.20),     # signed Δ for puts/calls
        cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert out is not None
    _, metrics, _ = out
    assert metrics["pop"] == pytest.approx(0.80, abs=1e-6)


# ---------------------------------------------------------------------------
# §4 — LongTermEvaluator behaviour
# ---------------------------------------------------------------------------

def _fixture_provider():
    return ManualPositionsProvider.from_dicts([
        {"ticker": "AAPL", "qty": 200, "avg_cost": 200.0, "kind": "stock"},
        {"ticker": "TSLA", "qty": 50,  "avg_cost": 250.0, "kind": "stock"},  # < 100, should be skipped
        {"ticker": "MSFT", "qty": 100, "avg_cost": 400.0, "kind": "stock"},  # not on watchlist
    ])


def _fixture_chain(ticker):
    if ticker == "AAPL":
        # Two valid candidates + one strike-below-basis reject.
        return [
            _short_call(strike=220.0, delta=0.20, bid=1.20, ask=1.30, dte=45),
            _short_call(strike=210.0, delta=0.28, bid=2.10, ask=2.20, dte=35),
            _short_call(strike=190.0, delta=0.55, bid=4.00, ask=4.20, dte=45),  # below cost basis × 1.01
        ]
    return []


def test_evaluator_skips_holdings_below_100_shares():
    """§4 — covered calls require 100 shares per contract."""
    ev = LongTermEvaluator(
        positions_provider=_fixture_provider(),
        call_chain_fetcher=_fixture_chain,
        preset=_StubPreset(),
    )
    recs = ev.recommend(["AAPL", "TSLA"])
    # TSLA has only 50 shares → no CC suggestion regardless of chain
    assert {r.ticker for r in recs} == {"AAPL"}


def test_evaluator_skips_unwatched_tickers():
    """§4 — recommendations only for tickers on the watchlist."""
    ev = LongTermEvaluator(
        positions_provider=_fixture_provider(),
        call_chain_fetcher=_fixture_chain,
        preset=_StubPreset(),
    )
    recs = ev.recommend(["AAPL"])
    # MSFT is held but not on watchlist → no CC
    assert {r.ticker for r in recs} == {"AAPL"}


def test_evaluator_produces_at_most_max_recommendations_per_ticker():
    cfg = EvaluatorConfig(cc_max_recommendations_per_ticker=1)
    ev = LongTermEvaluator(
        positions_provider=_fixture_provider(),
        call_chain_fetcher=_fixture_chain,
        preset=_StubPreset(),
        config=cfg,
    )
    recs = ev.recommend(["AAPL"])
    assert len([r for r in recs if r.ticker == "AAPL"]) == 1


def test_recommendation_qty_matches_contracts_per_lot():
    """200-share lot → 2 covered-call contracts."""
    ev = LongTermEvaluator(
        positions_provider=_fixture_provider(),
        call_chain_fetcher=_fixture_chain,
        preset=_StubPreset(),
    )
    recs = ev.recommend(["AAPL"])
    rec = recs[0]
    assert rec.legs[0].qty == 2


def test_recommendation_tp_below_entry_limit():
    """§4 — TP must be < entry_limit for a credit strategy."""
    ev = LongTermEvaluator(
        positions_provider=_fixture_provider(),
        call_chain_fetcher=_fixture_chain,
        preset=_StubPreset(),
    )
    recs = ev.recommend(["AAPL"])
    rec = recs[0]
    assert rec.entry_kind == "credit"
    assert 0 < rec.take_profit_limit < rec.entry_limit


def test_recommendation_stop_trigger_is_underlying_price_based():
    """§2.6 — covered-call stop uses underlying-price anchor."""
    ev = LongTermEvaluator(
        positions_provider=_fixture_provider(),
        call_chain_fetcher=_fixture_chain,
        preset=_StubPreset(),
    )
    rec = ev.recommend(["AAPL"])[0]
    assert rec.stop_kind == "underlying_price"
    # default cc_stop_underlying_pct = 0.92 × $200 = $184
    assert rec.stop_trigger == pytest.approx(184.0, abs=0.01)


def test_recommendation_post_init_rejects_invalid_anchors():
    """§4 — TP > entry on a credit strategy is a structural error."""
    with pytest.raises(ValueError, match="tp < entry_limit"):
        Recommendation(
            ticker="AAPL",
            strategy="covered_call",
            legs=[RecommendationLeg(
                action="STO", occ_symbol="X" * 21,
                qty=1, side="short", limit_price=1.0,
            )],
            entry_limit=1.0,
            entry_kind="credit",
            take_profit_limit=1.5,     # ❌ > entry on credit
            stop_trigger=190.0,
            stop_kind="underlying_price",
            stop_limit_offset_pct=0.05,
            score=1.0,
            rationale="bad",
        )


def test_evaluator_recommend_is_read_only():
    """§4 — recommend() never writes or submits an order.

    Conformance: the evaluator imports decision_engine helpers and the
    positions_provider snapshot — it does NOT import any executor /
    order-placement module. CI invariant scanning will pick up any new
    import of executor.* from this module.
    """
    import trading_agent.long_term_evaluator as mod
    src = open(mod.__file__).read()
    # Hard rule: no executor / order / Schwab Trader API imports here.
    assert "from trading_agent.executor" not in src
    assert "submit_order" not in src
    assert "place_order" not in src
    assert "/trader/v1/accounts" not in src


def test_score_covered_call_lives_in_decision_engine():
    """CI invariant 2 — scoring helpers may only be DEFINED in
    chain_scanner.py / decision_engine.py. Confirm `_score_covered_call`
    is sourced from decision_engine.py and not redefined elsewhere.
    """
    from trading_agent import decision_engine
    assert _score_covered_call.__module__ == decision_engine.__name__


def test_empty_holdings_yields_empty_recommendations():
    ev = LongTermEvaluator(
        positions_provider=ManualPositionsProvider(positions=[]),
        call_chain_fetcher=_fixture_chain,
        preset=_StubPreset(),
    )
    assert ev.recommend(["AAPL", "TSLA"]) == []


def test_chain_fetch_failure_falls_back_to_empty():
    """§4 — fail-open on chain fetch errors (don't crash the panel)."""
    def boom(_ticker):
        raise RuntimeError("vendor down")

    ev = LongTermEvaluator(
        positions_provider=_fixture_provider(),
        call_chain_fetcher=boom,
        preset=_StubPreset(),
    )
    assert ev.recommend(["AAPL"]) == []


def test_rationale_includes_strategy_signature():
    """Operator-facing rationale should name strike, DTE, Δ, ann yield, POP."""
    ev = LongTermEvaluator(
        positions_provider=_fixture_provider(),
        call_chain_fetcher=_fixture_chain,
        preset=_StubPreset(),
    )
    rec = ev.recommend(["AAPL"])[0]
    for token in ("CC @", "d)", "|Δ|=", "ann. yield", "POP"):
        assert token in rec.rationale

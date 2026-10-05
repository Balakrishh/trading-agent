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
    # credit at natural (fill_model default, backlog §6.1) = bid $1.20 → $120
    assert metrics["credit"] == pytest.approx(120.0, abs=0.01)
    # capital_at_risk = $200 × 100 − $120 = $19,880
    assert metrics["capital_at_risk"] == pytest.approx(19_880.0, abs=0.01)
    assert metrics["pop"] == pytest.approx(0.80, abs=1e-6)  # 1 − |0.20|
    assert metrics["dte"] == 45.0
    # score = annualised_return × pop > 0
    assert score > 0


def test_score_covered_call_mid_fill_model():
    """Legacy fill_model="mid": mid(1.20, 1.30) − $0.02 haircut = $1.23."""
    class _Mid(_StubPreset):
        fill_model = "mid"
    _, metrics, _ = _score_covered_call(short_call=_short_call(), cost_basis=200.0,
                                        preset=_Mid())
    assert metrics["credit"] == pytest.approx(123.0, abs=0.01)


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


# ---------------------------------------------------------------------------
# §2.2 — cash-secured put scorer (Wheel entry, added 2026-09-29)
# ---------------------------------------------------------------------------

from trading_agent.decision_engine import (   # noqa: E402
    LT_REJECT_COLLATERAL_OVER_BUDGET,
    LT_REJECT_STRIKE_OUT_OF_BAND,
    _score_cash_secured_put,
    _score_cash_secured_put_with_reason,
)
from trading_agent.fundamentals_screen import (   # noqa: E402
    WheelScreenConfig,
    screen_fundamentals,
)


def _short_put(*, strike=95.0, delta=-0.25, bid=1.95, ask=2.05, dte=35,
               iv_rank=None, symbol="KO    261106P00095000"):
    d = {"strike": strike, "delta": delta, "bid": bid, "ask": ask,
         "dte": dte, "symbol": symbol, "type": "put"}
    if iv_rank is not None:
        d["iv_rank"] = iv_rank
    return d


def test_csp_happy_path_math():
    res = _score_cash_secured_put_with_reason(short_put=_short_put(), spot=100.0)
    assert res["status"] == "accepted"
    m = res["metrics"]
    assert m["collateral"] == 9500.0
    assert m["capital_at_risk"] == pytest.approx(9500.0 - m["credit"])
    assert m["pop"] == pytest.approx(0.75)
    assert m["effective_entry"] == pytest.approx(95.0 - m["credit"] / 100.0)
    assert res["score"] == pytest.approx(m["annualised_return"] * m["pop"])
    assert _score_cash_secured_put(short_put=_short_put(), spot=100.0) is not None


@pytest.mark.parametrize("put,spot,cap,reason", [
    (_short_put(strike=98.0), 100.0, None, LT_REJECT_STRIKE_OUT_OF_BAND),   # < 3% OTM
    (_short_put(strike=80.0), 100.0, None, LT_REJECT_STRIKE_OUT_OF_BAND),   # > 15% OTM
    (_short_put(delta=-0.35), 100.0, None, LT_REJECT_SHORT_DELTA_TOO_HIGH),
    (_short_put(dte=10), 100.0, None, LT_REJECT_DTE_OUT_OF_BAND),
    (_short_put(), 100.0, 9000.0, LT_REJECT_COLLATERAL_OVER_BUDGET),
    (_short_put(bid=0.0, ask=0.0), 100.0, None, LT_REJECT_CREDIT_NON_POSITIVE_LT),
    (_short_put(iv_rank=0.10), 100.0, None, LT_REJECT_IV_RANK_TOO_LOW),
])
def test_csp_gates(put, spot, cap, reason):
    res = _score_cash_secured_put_with_reason(short_put=put, spot=spot, max_collateral=cap)
    assert res == {"status": "rejected", "reason": reason}


def test_csp_missing_iv_rank_fails_open():
    assert _score_cash_secured_put_with_reason(
        short_put=_short_put(iv_rank=None), spot=100.0)["status"] == "accepted"


def test_csp_scorer_lives_in_decision_engine():
    import trading_agent.decision_engine as de
    assert de._score_cash_secured_put.__module__ == "trading_agent.decision_engine"


# ---------------------------------------------------------------------------
# §2.7 — fundamentals quality screen
# ---------------------------------------------------------------------------

_GOOD = {"market_cap": 2.5e11, "pe_ratio": 22.0, "eps_ttm": 3.1,
         "net_profit_margin_ttm": 22.0, "roe": 38.0, "beta": 0.6,
         "vol_avg_10d": 14_000_000.0}


def test_screen_passes_quality_large_cap():
    assert screen_fundamentals("KO", _GOOD).passed


def test_screen_reports_every_failure():
    bad = {**_GOOD, "pe_ratio": -5.0, "beta": 2.3, "roe": 4.0}
    r = screen_fundamentals("X", bad)
    assert not r.passed and len(r.reasons) == 3


@pytest.mark.parametrize("patch,reason", [
    ({"roe": None}, "missing:roe"),
    ({"vol_avg_10d": 0.0}, "missing:vol_avg_10d"),      # 0.0 volume = unmapped
    ({"market_cap": "n/a"}, "missing:market_cap"),
])
def test_screen_missing_data_fails_closed(patch, reason):
    r = screen_fundamentals("X", {**_GOOD, **patch})
    assert not r.passed and reason in r.reasons


def test_screen_empty_block():
    assert screen_fundamentals("X", None).reasons == ["missing:fundamentals"]


# ---------------------------------------------------------------------------
# §3 — LongTermEvaluator CSP path
# ---------------------------------------------------------------------------

def _csp_evaluator(*, fundamentals=None, chain=None, spot=100.0, holdings=(),
                   config=None):
    fundamentals = fundamentals if fundamentals is not None else {"KO": _GOOD, "BAD": {**_GOOD, "pe_ratio": 90.0}}
    chain = chain if chain is not None else [
        _short_put(strike=95.0, delta=-0.25, bid=1.95, ask=2.05),
        _short_put(strike=92.0, delta=-0.18, bid=1.10, ask=1.20, symbol="KO    261106P00092000"),
        _short_put(strike=99.0, delta=-0.45, bid=3.50, ask=3.70),   # out of band
    ]
    return LongTermEvaluator(
        positions_provider=ManualPositionsProvider.from_dicts(list(holdings)),
        call_chain_fetcher=lambda _t: [],
        config=config,
        put_chain_fetcher=lambda t: chain,
        fundamentals_fetcher=lambda t: fundamentals.get(t),
        spot_fetcher=lambda t: spot,
    )


def test_evaluator_csp_ranks_and_anchors_exits():
    ev = _csp_evaluator()
    recs = ev.recommend(["KO", "BAD"])
    assert [r.ticker for r in recs] == ["KO", "KO"]
    assert recs[0].score >= recs[1].score
    r = recs[0]
    assert r.strategy == "cash_secured_put" and r.entry_kind == "credit"
    assert 0 < r.take_profit_limit < r.entry_limit
    assert r.stop_kind == "delta_threshold" and r.stop_trigger == 0.45
    assert ev.last_diagnostics["BAD"] == ["pe_ratio∉(0,40]"]


def test_evaluator_csp_skips_tickers_held_100_plus():
    ev = _csp_evaluator(holdings=[{"ticker": "KO", "qty": 100, "avg_cost": 60.0, "kind": "stock"}])
    assert [r for r in ev.recommend(["KO"]) if r.strategy == "cash_secured_put"] == []


def test_evaluator_csp_collateral_cap_reported():
    ev = _csp_evaluator(config=EvaluatorConfig(csp_max_collateral=5_000.0))
    assert ev.recommend(["KO"]) == []
    assert "collateral_over_budget" in ev.last_diagnostics["KO"][0]


def test_evaluator_csp_tiny_credit_skipped_not_raised():
    ev = _csp_evaluator(chain=[_short_put(bid=0.01, ask=0.01)])
    assert ev.recommend(["KO"]) == []


def test_evaluator_csp_fetcher_failure_is_contained():
    ev = _csp_evaluator()
    ev.put_chain_fetcher = lambda t: (_ for _ in ()).throw(RuntimeError("boom"))
    assert ev.recommend(["KO"]) == []
    assert ev.last_diagnostics["KO"] == ["data_unavailable"]


def test_evaluator_csp_disabled_without_fetchers():
    ev = LongTermEvaluator(positions_provider=ManualPositionsProvider.from_dicts([]),
                           call_chain_fetcher=lambda _t: [])
    assert ev.recommend(["KO"]) == [] and ev.last_diagnostics == {}


def test_wheel_screen_mcp_tool(monkeypatch):
    """Read-only MCP surface: string watchlist, string numbers, output shape."""
    import trading_agent.mcp.tools.strategy as st
    monkeypatch.setattr(st, "_positions_provider", lambda: ManualPositionsProvider.from_dicts([]))
    monkeypatch.setattr(st, "_earnings_days", lambda t: None)
    monkeypatch.setattr(st._market, "get_quote", lambda t: {"price": 100.0})
    monkeypatch.setattr(st._market, "get_fundamentals",
                        lambda t: {"fundamentals": _GOOD if t == "KO" else {}})
    monkeypatch.setattr(st, "_chain_from_dataserver",
                        lambda t, e, o: [_short_put()] if t == "KO" else None)
    out = st.wheel_screen("KO, ZZZ", target_dte="35", max_collateral="12000")
    assert out["watchlist"] == ["KO", "ZZZ"] and out["max_collateral"] == 12000.0
    assert len(out["recommendations"]) == 1
    rec = out["recommendations"][0]
    assert rec["strike"] == 95.0 and rec["strategy"] == "cash_secured_put"
    assert out["diagnostics"]["ZZZ"] == ["missing:fundamentals"]


def test_wheel_screen_pauses_csp_in_market_state(monkeypatch):
    """Skill 58: a CAUTION snapshot removes CSP rows and says why."""
    import trading_agent.mcp.tools.strategy as st
    from trading_agent import market_state
    monkeypatch.setattr(st, "_positions_provider", lambda: ManualPositionsProvider.from_dicts([]))
    monkeypatch.setattr(st, "_earnings_days", lambda t: None)
    monkeypatch.setattr(st._market, "get_quote", lambda t: {"price": 100.0})
    monkeypatch.setattr(st._market, "get_fundamentals", lambda t: {"fundamentals": _GOOD})
    monkeypatch.setattr(st, "_chain_from_dataserver", lambda t, e, o: [_short_put()])
    caution = market_state.classify_market_state(market_state.MarketInputs(
        spy_price=100.0, spy_sma20=101.0, spy_sma50=102.0, spy_sma200=90.0))
    market_state.write_state(caution)
    out = st.wheel_screen(["KO"], max_collateral=12000)
    assert out["recommendations"] == []
    assert out["market_state"] == "CAUTION"
    assert out["csp_paused"] == "market_state_CAUTION_pauses_new_csp"
    assert out["diagnostics"]["KO"] == ["market_state_CAUTION_pauses_new_csp"]


def test_wheel_screen_falls_back_to_monthly_expiration(monkeypatch):
    """Regression 2026-09-29: the weekly (11/13) had no KO chain; the 11/20
    monthly did. The tool must try the next candidate, not report no_chain."""
    import trading_agent.mcp.tools.strategy as st
    monkeypatch.setattr(st, "_positions_provider", lambda: ManualPositionsProvider.from_dicts([]))
    monkeypatch.setattr(st, "_earnings_days", lambda t: None)
    monkeypatch.setattr(st, "_wheel_expiration_candidates",
                        lambda today, dte, **kw: ["2026-11-13", "2026-11-20"])
    monkeypatch.setattr(st._market, "get_quote", lambda t: {"price": 100.0})
    monkeypatch.setattr(st._market, "get_fundamentals", lambda t: {"fundamentals": _GOOD})
    monkeypatch.setattr(st, "_chain_from_dataserver",
                        lambda t, e, o: [_short_put()] if e == "2026-11-20" else None)
    out = st.wheel_screen(["KO"], max_collateral=12000)
    assert out["expiration_by_ticker"] == {"KO": "2026-11-20"}
    assert len(out["recommendations"]) == 1


def test_wheel_expiration_candidates_include_monthlies():
    from datetime import date
    from trading_agent.mcp.tools.strategy import _wheel_expiration_candidates
    got = _wheel_expiration_candidates(date(2026, 9, 29), 35)
    assert "2026-11-20" in got                      # third Friday of November
    assert all(21 <= (date.fromisoformat(d) - date(2026, 9, 29)).days <= 60 for d in got)


# ---------------------------------------------------------------------------
# §2.8 — earnings gate (added 2026-09-29)
# ---------------------------------------------------------------------------

def _earnings_tool(monkeypatch, *, earnings_days, chains):
    """Wire wheel_screen to fixtures. ``chains`` maps expiration → contracts."""
    from datetime import date, timedelta
    import trading_agent.mcp.tools.strategy as st
    today = date.today()
    exps = {k: (today + timedelta(days=k)).isoformat() for k in chains}
    monkeypatch.setattr(st, "_wheel_expiration_candidates",
                        lambda _d, _t, **kw: [exps[k] for k in sorted(chains, key=lambda k: abs(k - 35))])
    monkeypatch.setattr(st, "_positions_provider", lambda: ManualPositionsProvider.from_dicts([]))
    monkeypatch.setattr(st, "_earnings_days", lambda t: earnings_days)
    monkeypatch.setattr(st._market, "get_quote", lambda t: {"price": 100.0})
    monkeypatch.setattr(st._market, "get_fundamentals", lambda t: {"fundamentals": _GOOD})
    by_exp = {exps[k]: v for k, v in chains.items()}
    monkeypatch.setattr(st, "_chain_from_dataserver", lambda t, e, o: by_exp.get(e))
    return st, exps


def test_earnings_avoid_picks_expiration_before_report(monkeypatch):
    st, exps = _earnings_tool(monkeypatch, earnings_days=30,
                              chains={35: [_short_put()], 24: [_short_put()]})
    out = st.wheel_screen(["KO"])
    assert out["expiration_by_ticker"]["KO"] == exps[24]          # 35d would straddle the report
    rec = out["recommendations"][0]
    assert rec["earnings_in_days"] == 30 and rec["earnings_before_expiry"] is False


def test_earnings_avoid_blocks_when_no_expiration_fits(monkeypatch):
    st, _ = _earnings_tool(monkeypatch, earnings_days=9, chains={35: [_short_put()]})
    out = st.wheel_screen(["KO"])
    assert out["recommendations"] == []
    assert out["diagnostics"]["KO"][0].startswith("earnings_in_9d")


def test_earnings_allow_keeps_and_flags(monkeypatch):
    st, exps = _earnings_tool(monkeypatch, earnings_days=9, chains={35: [_short_put()]})
    out = st.wheel_screen(["KO"], earnings_policy="allow")
    rec = out["recommendations"][0]
    assert rec["earnings_before_expiry"] is True and rec["expiration"] == exps[35]


def test_earnings_unknown_is_flagged_not_excluded(monkeypatch):
    st, _ = _earnings_tool(monkeypatch, earnings_days=None, chains={35: [_short_put()]})
    rec = st.wheel_screen(["KO"])["recommendations"][0]
    assert rec["earnings_known"] is False and rec["earnings_before_expiry"] is False


def test_earnings_policy_validated(monkeypatch):
    st, _ = _earnings_tool(monkeypatch, earnings_days=None, chains={35: [_short_put()]})
    with pytest.raises(ValueError, match="earnings_policy"):
        st.wheel_screen(["KO"], earnings_policy="yolo")


# ---------------------------------------------------------------------------
# Skill 29 liquidity gate on Wheel legs (added 2026-09-30)
# ---------------------------------------------------------------------------

from trading_agent.chain_scanner import REJECT_LEG_SPREAD_WIDE   # noqa: E402


def test_csp_rejects_premarket_width_quote():
    """Regression: BMY $60P 0.40/1.11 pre-market ranked #1 at a mid-based
    21.8 % yield. 71¢ wide = 94 % of mid → fails both caps."""
    res = _score_cash_secured_put_with_reason(
        short_put=_short_put(strike=60.0, bid=0.40, ask=1.11, delta=-0.25, dte=23),
        spot=63.17)
    assert res == {"status": "rejected", "reason": REJECT_LEG_SPREAD_WIDE}


@pytest.mark.parametrize("bid,ask", [
    (0.98, 1.04),     # KO-style regular-hours quote: 6¢ ≤ 15¢
    (0.05, 0.10),     # penny option: 100 % of mid but 5¢ ≤ 15¢ absolute
    (4.00, 4.18),     # 18¢ > 15¢ but 4.4 % ≤ 5 % of mid
])
def test_csp_liquidity_gate_passes_tradeable_quotes(bid, ask):
    res = _score_cash_secured_put_with_reason(
        short_put=_short_put(bid=bid, ask=ask), spot=100.0)
    assert res.get("reason") != REJECT_LEG_SPREAD_WIDE


def test_liquidity_gate_reads_preset_thresholds():
    class Loose:
        max_leg_spread_cents = 1.00
        max_leg_spread_pct_mid = 1.00
    res = _score_cash_secured_put_with_reason(
        short_put=_short_put(bid=0.40, ask=1.11), spot=100.0, preset=Loose())
    assert res.get("reason") != REJECT_LEG_SPREAD_WIDE


def test_covered_call_rejects_wide_quote():
    res = _score_covered_call_with_reason(
        short_call=_short_call(bid=1.00, ask=1.60), cost_basis=200.0,
        preset=_StubPreset())
    assert res == {"status": "rejected", "reason": REJECT_LEG_SPREAD_WIDE}


def test_evaluator_reports_liquidity_rejects():
    ev = _csp_evaluator(chain=[_short_put(bid=0.40, ask=1.11)])
    assert ev.recommend(["KO"]) == []
    assert "leg_spread_wide" in ev.last_diagnostics["KO"][0]


@pytest.mark.parametrize("wl", ['["KO"]', "['KO']", "KO", " KO , ", ["KO"]])
def test_wheel_screen_accepts_list_shapes(monkeypatch, wl):
    """Regression 2026-09-30: '["VZ"]' (a JSON list sent as text) was
    treated as a single ticker → missing:fundamentals."""
    import trading_agent.mcp.tools.strategy as st
    monkeypatch.setattr(st, "_positions_provider", lambda: ManualPositionsProvider.from_dicts([]))
    monkeypatch.setattr(st, "_earnings_days", lambda t: None)
    monkeypatch.setattr(st._market, "get_quote", lambda t: {"price": 100.0})
    monkeypatch.setattr(st._market, "get_fundamentals", lambda t: {"fundamentals": _GOOD})
    monkeypatch.setattr(st, "_chain_from_dataserver", lambda t, e, o: [_short_put()])
    assert st.wheel_screen(wl)["watchlist"] == ["KO"]

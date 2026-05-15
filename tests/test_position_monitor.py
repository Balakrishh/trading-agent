"""
test_position_monitor.py — guards the schema contract of
``PositionMonitor.group_into_spreads``.

Before this test, ``group_into_spreads`` accepted the **envelope** shape
(``{trade_plan: {...}}``) used by the agent's per-cycle ``_load_trade_plans``,
but silently returned an empty list when the Streamlit UI's pre-unwrapped
**inner** shape (``{ticker, legs, ...}``) was passed. The dashboard's
Open-Positions panel feeds inner-shaped dicts, so filled spreads on the
broker never showed up in the UI.

The fix added ``_extract_inner_plan`` which tolerates both shapes; this
test pins down that contract so a future "simplification" can't quietly
re-break the live tab.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading_agent.position_monitor import (
    PositionMonitor,
    PositionSnapshot,
)


# --- Fixtures -----------------------------------------------------------


def _snap(symbol: str, qty: int = -1, unreal_pl: float = 0.0) -> PositionSnapshot:
    """Build a PositionSnapshot the legs-list expects.

    The real PositionSnapshot has more fields than ``group_into_spreads``
    actually reads; we fill them all with sensible defaults so the
    dataclass constructs cleanly without having to pull live broker data
    into the test.
    """
    avg_entry = 1.50
    current = 1.20
    return PositionSnapshot(
        symbol=symbol,
        qty=qty,
        side="short" if qty < 0 else "long",
        avg_entry_price=avg_entry,
        current_price=current,
        market_value=current * qty * 100,
        cost_basis=avg_entry * qty * 100,
        unrealized_pl=unreal_pl,
        unrealized_plpc=0.0,
        asset_class="us_option",
    )


_INNER_PLAN = {
    "ticker":       "DIA",
    "strategy":     "Iron Condor",
    "expiration":   "2026-05-29",
    "spread_width": 5.0,
    "net_credit":   4.54,
    "max_loss":     0.46,
    "legs": [
        {"symbol": "DIA260529P00483000", "strike": 483.0, "action": "sell"},
        {"symbol": "DIA260529P00478000", "strike": 478.0, "action": "buy"},
        {"symbol": "DIA260529C00500000", "strike": 500.0, "action": "sell"},
        {"symbol": "DIA260529C00505000", "strike": 505.0, "action": "buy"},
    ],
}

_ENVELOPE_PLAN = {
    "run_id":       "20260504_162722",
    "timestamp":    "2026-05-04T16:27:22Z",
    "trade_plan":   _INNER_PLAN,
    "risk_verdict": {"approved": True},
    "mode":         "live",
}

_OPEN_LEGS = [
    _snap("DIA260529P00483000", qty=-1, unreal_pl=10.0),
    _snap("DIA260529P00478000", qty=+1, unreal_pl=-3.0),
    _snap("DIA260529C00500000", qty=-1, unreal_pl=8.0),
    _snap("DIA260529C00505000", qty=+1, unreal_pl=-2.0),
]


def _monitor() -> PositionMonitor:
    """Build a PositionMonitor without touching the broker."""
    return PositionMonitor(
        api_key="dummy",
        secret_key="dummy",
        base_url="https://paper-api.alpaca.markets/v2",
    )


# --- Tests --------------------------------------------------------------


def test_envelope_shape_groups_correctly():
    """Agent path: trade_plans = [{trade_plan: {...}}, ...] should match."""
    spreads = _monitor().group_into_spreads(_OPEN_LEGS, [_ENVELOPE_PLAN])
    assert len(spreads) == 1
    s = spreads[0]
    assert s.underlying == "DIA"
    assert s.strategy_name == "Iron Condor"
    assert len(s.legs) == 4
    # Net P&L = 10 - 3 + 8 - 2 = 13
    assert s.net_unrealized_pl == pytest.approx(13.0)


def test_inner_shape_groups_correctly():
    """UI path: trade_plans = [{ticker, legs, ...}, ...] (pre-unwrapped).

    This is the regression that caused filled spreads to vanish from the
    Open-Positions panel. The pre-fix code did
    ``plan.get("trade_plan", {})`` and got ``{}`` here → empty result.
    """
    spreads = _monitor().group_into_spreads(_OPEN_LEGS, [_INNER_PLAN])
    assert len(spreads) == 1, (
        "Inner-shaped plans must match too — this is the live-monitor path "
        "that was silently producing zero spreads before the fix."
    )
    s = spreads[0]
    assert s.underlying == "DIA"
    assert s.strategy_name == "Iron Condor"
    assert len(s.legs) == 4


def test_mixed_shapes_in_one_call():
    """Defensive: caller mixes envelopes and inners — both should match."""
    # Two distinct plans on different tickers so we can tell them apart.
    second_inner = {
        **_INNER_PLAN,
        "ticker": "XLF",
        "legs": [
            {"symbol": "XLF260529P00038000", "strike": 38.0, "action": "sell"},
            {"symbol": "XLF260529P00037000", "strike": 37.0, "action": "buy"},
        ],
    }
    second_envelope = {"trade_plan": second_inner, "run_id": "x"}
    legs = _OPEN_LEGS + [
        _snap("XLF260529P00038000", qty=-1, unreal_pl=2.0),
        _snap("XLF260529P00037000", qty=+1, unreal_pl=-1.0),
    ]
    spreads = _monitor().group_into_spreads(legs, [_ENVELOPE_PLAN, second_envelope])
    underlyings = {s.underlying for s in spreads}
    assert underlyings == {"DIA", "XLF"}

    # Same input, but feed one as inner and one as envelope.
    spreads = _monitor().group_into_spreads(legs, [_INNER_PLAN, second_envelope])
    assert {s.underlying for s in spreads} == {"DIA", "XLF"}


def test_empty_legs_returns_empty():
    spreads = _monitor().group_into_spreads([], [_INNER_PLAN])
    # No matched legs → the plan is dropped, so spreads list is empty.
    assert spreads == []


def test_no_matching_symbols_falls_through_to_inference():
    """
    A leg that doesn't match any plan still surfaces in the dashboard
    via the Task #53 inference fallback.  Single short puts are
    classified as "Naked Short" so the user always sees the position
    even when the plan is rotated out of state_history or the leg was
    opened manually.
    """
    foreign_legs = [_snap("AAPL260529P00150000", qty=-1)]
    spreads = _monitor().group_into_spreads(foreign_legs, [_INNER_PLAN])
    assert len(spreads) == 1
    assert spreads[0].underlying == "AAPL"
    assert spreads[0].origin == "inferred"
    assert spreads[0].strategy_name == "Naked Short"


def test_extract_inner_plan_static_helper():
    """Direct contract on the helper — both shapes round-trip correctly."""
    extract = PositionMonitor._extract_inner_plan
    assert extract(_INNER_PLAN) is _INNER_PLAN
    assert extract(_ENVELOPE_PLAN) is _INNER_PLAN
    # Defensive: dict with non-dict 'trade_plan' value falls back to the
    # outer dict (treated as inner).
    weird = {"ticker": "X", "trade_plan": "not-a-dict", "legs": []}
    assert extract(weird) is weird


# --- Inference fallback (legs with no matching trade plan) --------------


def test_parse_occ_helper():
    """OCC symbol decoder handles real-world option symbols."""
    parse = PositionMonitor._parse_occ
    out = parse("DIA260529P00483000")
    assert out == {
        "underlying": "DIA",
        "expiration": "2026-05-29",
        "type":       "put",
        "strike":     483.0,
    }
    out = parse("AAPL260515C00275000")
    assert out == {
        "underlying": "AAPL",
        "expiration": "2026-05-15",
        "type":       "call",
        "strike":     275.0,
    }
    # Fractional strike (penny pilot)
    out = parse("GOOG260508P00337500")
    assert out["strike"] == 337.5
    # Malformed — returns None instead of raising
    assert parse("not-an-occ-symbol") is None
    assert parse("DIA") is None


def test_infer_iron_condor_from_4_legs():
    """4-leg short-call/long-call/short-put/long-put → Iron Condor."""
    legs = [
        _snap("DIA260529C00499000", qty=-2, unreal_pl=-60.0),  # short call
        _snap("DIA260529C00510000", qty=+2, unreal_pl=-60.0),  # long call
        _snap("DIA260529P00470000", qty=+2, unreal_pl=-88.0),  # long put
        _snap("DIA260529P00483000", qty=-2, unreal_pl=-30.0),  # short put
    ]
    spreads = PositionMonitor._infer_spreads_from_legs(legs)
    assert len(spreads) == 1
    s = spreads[0]
    assert s.underlying == "DIA"
    assert s.expiration == "2026-05-29"
    assert s.strategy_name == "Iron Condor"
    assert s.origin == "inferred"
    assert len(s.legs) == 4
    # Net P&L = -60 - 60 - 88 - 30 = -238
    assert s.net_unrealized_pl == pytest.approx(-238.0)
    # Short strikes: 499 (call) + 483 (put)
    assert sorted(s.short_strikes) == [483.0, 499.0]
    # Spread width = max(call_wing 11, put_wing 13) = 13
    assert s.spread_width == 13.0


def test_infer_bull_put_from_2_legs():
    """2 puts (1 short, 1 long, short closer to money) → Bull Put Spread."""
    legs = [
        _snap("SPY260529P00700000", qty=-1, unreal_pl=5.0),   # short put
        _snap("SPY260529P00695000", qty=+1, unreal_pl=-2.0),  # long put
    ]
    spreads = PositionMonitor._infer_spreads_from_legs(legs)
    assert len(spreads) == 1
    s = spreads[0]
    assert s.strategy_name == "Bull Put Spread"
    assert s.origin == "inferred"
    assert s.spread_width == 5.0


def test_infer_bear_call_from_2_legs():
    """2 calls (1 short, 1 long) → Bear Call Spread."""
    legs = [
        _snap("XLF260605C00053500", qty=-2, unreal_pl=10.0),
        _snap("XLF260605C00054000", qty=+2, unreal_pl=-3.0),
    ]
    spreads = PositionMonitor._infer_spreads_from_legs(legs)
    assert len(spreads) == 1
    assert spreads[0].strategy_name == "Bear Call Spread"


def test_infer_groups_by_expiration():
    """Same underlying, different expirations → separate spreads."""
    legs = [
        # IC expiring 2026-05-29
        _snap("DIA260529C00499000", qty=-1),
        _snap("DIA260529C00510000", qty=+1),
        _snap("DIA260529P00470000", qty=+1),
        _snap("DIA260529P00483000", qty=-1),
        # Bull Put expiring 2026-06-05 (different expiration)
        _snap("DIA260605P00465000", qty=-1),
        _snap("DIA260605P00460000", qty=+1),
    ]
    spreads = PositionMonitor._infer_spreads_from_legs(legs)
    assert len(spreads) == 2
    by_expiry = {s.expiration: s for s in spreads}
    assert by_expiry["2026-05-29"].strategy_name == "Iron Condor"
    assert by_expiry["2026-06-05"].strategy_name == "Bull Put Spread"


def test_inference_kicks_in_when_plan_missing():
    """Full pipeline: legs with no matching plan → spreads via inference."""
    # Broker has these 4 DIA legs but trade_plans list is empty.
    legs = [
        _snap("DIA260529C00499000", qty=-2, unreal_pl=-60.0),
        _snap("DIA260529C00510000", qty=+2, unreal_pl=-60.0),
        _snap("DIA260529P00470000", qty=+2, unreal_pl=-88.0),
        _snap("DIA260529P00483000", qty=-2, unreal_pl=-30.0),
    ]
    spreads = _monitor().group_into_spreads(legs, [])  # NO plans
    assert len(spreads) == 1, (
        "When no trade_plan matches, inference must still produce a "
        "spread row so the user sees the position in the dashboard."
    )
    s = spreads[0]
    assert s.origin == "inferred"
    assert s.strategy_name == "Iron Condor"
    assert s.underlying == "DIA"


def test_matched_and_inferred_coexist():
    """Mix: one ticker matched via plan, another inferred from leg structure.

    DIA legs use the same OCC symbols as ``_INNER_PLAN.legs`` so they
    match via the plan path (origin='trade_plan').  GLD legs share no
    plan, so they fall through to inference (origin='inferred').
    """
    # Symbols here MUST match _INNER_PLAN.legs exactly.
    dia_legs = list(_OPEN_LEGS)
    # GLD legs that no plan covers — fall through to inference.
    gld_legs = [
        _snap("GLD260529C00427000", qty=-2),
        _snap("GLD260529C00442000", qty=+2),
        _snap("GLD260529P00390000", qty=+2),
        _snap("GLD260529P00405000", qty=-2),
    ]
    spreads = _monitor().group_into_spreads(dia_legs + gld_legs, [_INNER_PLAN])
    origins = {s.underlying: s.origin for s in spreads}
    assert origins == {"DIA": "trade_plan", "GLD": "inferred"}, (
        f"Expected DIA matched + GLD inferred, got {origins}"
    )


# --- Reject-plan leg-claim regression (2026-05-15 XLF incident) ---------
#
# State history can retain plans the chain scanner emitted but never
# submitted (``valid=False``) or the risk manager vetoed
# (``risk_verdict.approved=False``).  Before the fix, those rejected
# plans iterated through ``group_into_spreads`` like any other and could
# greedily claim broker leg symbols they happened to share with the
# actually-submitted plan — splitting one Iron Condor into two display
# rows and double-counting against the per-ticker position cap.
#
# Concretely on 2026-05-15: four rejected XLF plans all carried the
# call wing C52.5/C54.0 and an invalid put wing P49.0/P50.5; the actual
# submission carried the same calls plus P48.5/P50.0.  The rejected
# plan won the iteration race, claimed the two call legs, and the
# put legs got attributed to a second SpreadPosition row.

_XLF_REJECTED_INNER = {
    "ticker":       "XLF",
    "strategy":     "Iron Condor",
    "expiration":   "2026-06-05",
    "spread_width": 1.5,
    "net_credit":   0.44,
    "max_loss":     106.0,
    "valid":        False,
    "rejection_reason": "Credit/Width ratio 0.2933 < 0.3",
    "legs": [
        {"symbol": "XLF260605C00052500", "strike": 52.5, "action": "sell"},
        {"symbol": "XLF260605C00054000", "strike": 54.0, "action": "buy"},
        {"symbol": "XLF260605P00049000", "strike": 49.0, "action": "sell"},
        {"symbol": "XLF260605P00050500", "strike": 50.5, "action": "buy"},
    ],
}

_XLF_SUBMITTED_INNER = {
    "ticker":       "XLF",
    "strategy":     "Iron Condor",
    "expiration":   "2026-06-05",
    "spread_width": 1.5,
    "net_credit":   0.47,
    "max_loss":     103.0,
    "valid":        True,
    "legs": [
        {"symbol": "XLF260605C00052500", "strike": 52.5, "action": "sell"},
        {"symbol": "XLF260605C00054000", "strike": 54.0, "action": "buy"},
        {"symbol": "XLF260605P00048500", "strike": 48.5, "action": "sell"},
        {"symbol": "XLF260605P00050000", "strike": 50.0, "action": "buy"},
    ],
}

# Broker reports the four legs of the SUBMITTED plan only — the rejected
# plan never resulted in any fills.
_XLF_BROKER_LEGS = [
    _snap("XLF260605C00052500", qty=-1, unreal_pl=-4.0),
    _snap("XLF260605C00054000", qty=+1, unreal_pl=2.0),
    _snap("XLF260605P00048500", qty=+1, unreal_pl=-3.0),
    _snap("XLF260605P00050000", qty=-1, unreal_pl=-5.0),
]


def test_rejected_plan_does_not_claim_legs_from_submitted_plan():
    """State history has rejected plans sharing the call wing with the
    actually-submitted plan.  Result must be exactly ONE spread row
    using the SUBMITTED plan's economics — not two halves.

    Pre-fix behaviour: greedy iteration let the rejected plan claim
    C52.5/C54.0 first, then the put wing was attributed to a second
    inferred-style row using the rejected plan's net_credit/max_loss.
    The cap-counting layer in agent.py would then count this as
    ``positions_per_ticker['XLF'] = 2`` and block any further entry
    (which happened by accident, not by design — see runbook 04).
    """
    # Order matters: rejected plans listed FIRST so they'd win greedy
    # iteration without the filter. This mirrors how state_history is
    # written — chronologically, with rejections always preceding the
    # final submission of the cycle.
    plans = [
        _XLF_REJECTED_INNER,
        _XLF_REJECTED_INNER,
        _XLF_REJECTED_INNER,
        _XLF_REJECTED_INNER,
        _XLF_SUBMITTED_INNER,
    ]
    spreads = _monitor().group_into_spreads(_XLF_BROKER_LEGS, plans)

    assert len(spreads) == 1, (
        f"Expected ONE spread (the submitted Iron Condor) — got "
        f"{len(spreads)}: {[(s.underlying, s.strategy_name, s.original_credit) for s in spreads]}. "
        "This is the 2026-05-15 XLF regression — a rejected plan with "
        "shared legs split the IC into two rows."
    )
    s = spreads[0]
    assert s.strategy_name == "Iron Condor"
    assert len(s.legs) == 4
    assert s.original_credit == pytest.approx(0.47), (
        "Economics must come from the SUBMITTED plan (cr=0.47), not "
        "the rejected plan (cr=0.44). Pre-fix the row showed cr=0.44 "
        "for the call-wing half because the rejected plan won iteration."
    )
    assert s.max_loss == pytest.approx(103.0)


def test_multiple_risk_rejected_envelopes_with_identical_legs():
    """The 2026-05-15 GLD reproduction.

    State history retains EVERY plan the chain scanner emitted that
    cycle, with the risk manager flagging each as approved=True or
    False. For a single ticker that the planner re-emits 80x in a day,
    you can easily have 3+ risk-rejected envelopes whose legs are
    identical to the eventually-submitted one (the legs converge as
    the chain matures even when the credit drifts above/below the CW
    floor). Without proper envelope handling those rejected entries
    win the iteration race and the dashboard shows the WRONG plan's
    economics — concretely on 2026-05-15 GLD displayed cr=$1.75 / 60%
    loss when the real fill was cr=$1.95 / 54% loss.
    """
    INNER = {
        "ticker": "GLD", "strategy": "Iron Condor",
        "expiration": "2026-06-05", "spread_width": 5.0, "valid": True,
        "legs": [
            {"symbol": "GLD260605P00408000", "strike": 408.0, "action": "sell"},
            {"symbol": "GLD260605P00403000", "strike": 403.0, "action": "buy"},
            {"symbol": "GLD260605C00428000", "strike": 428.0, "action": "sell"},
            {"symbol": "GLD260605C00432000", "strike": 432.0, "action": "buy"},
        ],
    }
    # Three rejected envelopes with different cr/ml but SAME legs as
    # the submitted one, then the actual submission.
    envelopes = [
        {"trade_plan": {**INNER, "net_credit": 1.75, "max_loss": 325.0},
         "risk_verdict": {"approved": False,
                          "checks_failed": ["Credit/Width ratio 0.35 < 0.4"]}},
        {"trade_plan": {**INNER, "net_credit": 1.80, "max_loss": 320.0},
         "risk_verdict": {"approved": False,
                          "checks_failed": ["Credit/Width ratio 0.36 < 0.4"]}},
        {"trade_plan": {**INNER, "net_credit": 1.95, "max_loss": 305.0},
         "risk_verdict": {"approved": False,
                          "checks_failed": ["Max loss > 5% equity"]}},
        {"trade_plan": {**INNER, "net_credit": 1.95, "max_loss": 305.0},
         "risk_verdict": {"approved": True}},
    ]
    broker = [
        _snap("GLD260605P00408000", qty=-1),
        _snap("GLD260605P00403000", qty=+1),
        _snap("GLD260605C00428000", qty=-1),
        _snap("GLD260605C00432000", qty=+1),
    ]
    spreads = _monitor().group_into_spreads(broker, envelopes)
    assert len(spreads) == 1
    s = spreads[0]
    assert s.original_credit == pytest.approx(1.95), (
        f"Expected SUBMITTED cr=1.95, got {s.original_credit}. "
        "If this assertion fails, a rejected plan is winning the "
        "iteration race against the submitted plan — the 2026-05-15 "
        "GLD regression has re-opened."
    )
    assert s.max_loss == pytest.approx(305.0)
    assert len(s.legs) == 4


def test_envelope_with_risk_verdict_rejected_is_filtered():
    """Envelope shape carrying ``risk_verdict.approved=False`` must
    also be filtered, even if the inner plan happens to have
    ``valid=True``.  Defensive: belt + braces against any drift where
    the chain scanner says "valid" but the risk manager vetoes."""
    inner = {**_XLF_SUBMITTED_INNER, "net_credit": 0.99, "max_loss": 51.0}
    envelope_rejected = {
        "trade_plan":   inner,
        "risk_verdict": {"approved": False,
                         "checks_failed": ["Max loss exceeds cap"]},
        "run_id":       "rejected",
    }
    envelope_approved = {
        "trade_plan":   _XLF_SUBMITTED_INNER,
        "risk_verdict": {"approved": True},
        "run_id":       "approved",
    }
    spreads = _monitor().group_into_spreads(
        _XLF_BROKER_LEGS, [envelope_rejected, envelope_approved],
    )
    assert len(spreads) == 1
    assert spreads[0].original_credit == pytest.approx(0.47), (
        "Risk-vetoed envelope must not claim legs; the approved "
        "envelope's economics (cr=0.47) must surface."
    )


def test_plan_missing_valid_field_is_treated_as_valid():
    """Backwards-compat: older trade_plan history files predate the
    ``valid`` field.  A plan with no ``valid`` key must still match
    broker legs — otherwise upgrading the agent against an existing
    state_history archive would blank out the Open-Positions panel.
    """
    plan_no_valid = {k: v for k, v in _XLF_SUBMITTED_INNER.items()
                     if k != "valid"}
    spreads = _monitor().group_into_spreads(_XLF_BROKER_LEGS, [plan_no_valid])
    assert len(spreads) == 1
    assert spreads[0].strategy_name == "Iron Condor"


# --- fetch_open_positions: distinguish None from [] ---------------------
#
# 2026-05-05 regression: a TCP connection-reset during Stage 1 returned
# `[]` from `fetch_open_positions`, which the agent's dedup gate
# couldn't distinguish from "broker says zero positions" — leading to a
# duplicate DIA Iron Condor.  The contract is now:
#   * RPC failure         → None
#   * Genuine empty book  → []
# The cycle's fail-closed branch in `agent.py` keys off this distinction.


def test_fetch_open_positions_returns_none_on_request_exception(monkeypatch):
    """A RequestException must surface as None, not [].

    Pre-fix this returned [] which fail-opened the dedup gate.
    """
    import requests as _requests

    def boom(*_a, **_kw):
        raise _requests.exceptions.ConnectionError(
            ("Connection aborted.",
             ConnectionResetError(54, "Connection reset by peer")),
        )

    monkeypatch.setattr(
        "trading_agent.position_monitor.requests.get", boom
    )
    result = _monitor().fetch_open_positions()
    assert result is None, (
        "fetch_open_positions must return None on RPC failure so the "
        "dedup gate can fail closed; pre-2026-05-05 it returned [] "
        "which was indistinguishable from a clean slate."
    )


def test_fetch_open_positions_returns_empty_list_when_broker_has_none(monkeypatch):
    """A successful response with zero positions returns [], not None."""

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return []

    monkeypatch.setattr(
        "trading_agent.position_monitor.requests.get",
        lambda *_a, **_kw: _Resp(),
    )
    result = _monitor().fetch_open_positions()
    assert result == [], (
        "Genuinely-empty broker response must still return [] (not None) "
        "so the cycle proceeds normally on a clean slate."
    )


def test_fetch_open_positions_filters_to_us_options_only(monkeypatch):
    """Non-option positions must be stripped out, but the result is
    still [] (not None) when only equity positions exist."""

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"symbol": "AAPL", "qty": 100, "side": "long",
                 "avg_entry_price": 168.0, "current_price": 170.0,
                 "market_value": 17000.0, "cost_basis": 16800.0,
                 "unrealized_pl": 200.0, "unrealized_plpc": 0.012,
                 "asset_class": "us_equity"},
            ]

    monkeypatch.setattr(
        "trading_agent.position_monitor.requests.get",
        lambda *_a, **_kw: _Resp(),
    )
    result = _monitor().fetch_open_positions()
    assert result == [], "Equity positions must be filtered out"
    assert result is not None, "Filtering to zero options is a clean slate, not an RPC failure"

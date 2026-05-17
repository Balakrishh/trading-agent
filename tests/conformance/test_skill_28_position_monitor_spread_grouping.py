"""Conformance test: skill 28 — Position-monitor spread grouping.

Skill 28 §4 documents two specific failure modes that this test
pins down:

  1. **Rejected-plan leg-claim (the 2026-05-15 XLF bug)** — a plan
     in state_history with ``valid=False`` must not claim broker
     legs that the actually-submitted plan also covers.
  2. **Caller must pass envelopes** — the ``risk_verdict.approved``
     filter only fires when the envelope (not inner plan) is
     passed in. Specifically, an envelope carrying
     ``risk_verdict.approved=False`` with ``valid=True`` inside
     must still be skipped.

These are the GLD + XLF bugs that motivated skill 28 in the first
place. The conformance test asserts the gate semantics directly
rather than going through the integration path; the module-level
tests in test_position_monitor.py cover the broader edge-case
surface.

Failure modes caught:
- Someone refactors group_into_spreads and drops the valid filter
- Someone "simplifies" the envelope check to inner-plan only —
  re-opens the GLD bug class
"""

from __future__ import annotations

from trading_agent.position_monitor import PositionMonitor, PositionSnapshot


def _snap(symbol: str, qty: int = -1) -> PositionSnapshot:
    """Minimal PositionSnapshot for the gate's leg-symbol matching."""
    return PositionSnapshot(
        symbol=symbol, qty=qty,
        side="short" if qty < 0 else "long",
        avg_entry_price=1.0, current_price=1.0,
        market_value=qty * 100, cost_basis=qty * 100,
        unrealized_pl=0.0, unrealized_plpc=0.0,
        asset_class="us_option",
    )


def _monitor() -> PositionMonitor:
    return PositionMonitor(
        api_key="conformance", secret_key="conformance",
        base_url="https://paper-api.alpaca.markets/v2",
    )


def test_skill_28_invalid_plan_does_not_claim_broker_legs() -> None:
    """Skill 28 §4 — XLF 2026-05-15 incident: a plan with ``valid=False``
    must not show up as a spread row even when broker legs match
    its symbols. The submitted (valid=True) plan must own them."""
    common_legs = [
        {"symbol": "TEST260605P00100000", "strike": 100.0, "action": "sell"},
        {"symbol": "TEST260605P00095000", "strike": 95.0, "action": "buy"},
    ]
    rejected = {
        "ticker": "TEST", "strategy": "Bull Put", "valid": False,
        "net_credit": 0.20, "max_loss": 480, "spread_width": 5.0,
        "legs": common_legs,
    }
    submitted = {
        "ticker": "TEST", "strategy": "Bull Put", "valid": True,
        "net_credit": 0.40, "max_loss": 460, "spread_width": 5.0,
        "legs": common_legs,
    }
    broker = [_snap(common_legs[0]["symbol"], qty=-1),
              _snap(common_legs[1]["symbol"], qty=+1)]

    spreads = _monitor().group_into_spreads(broker, [rejected, submitted])
    assert len(spreads) == 1, (
        f"Skill 28 §4: rejected plan should not produce its own row; "
        f"the submitted plan should own the legs. Got {len(spreads)} "
        f"spread rows."
    )
    assert spreads[0].original_credit == 0.40, (
        f"Skill 28 §4: the surviving row should use the SUBMITTED "
        f"plan's economics (0.40), not the rejected plan's (0.20). "
        f"Got {spreads[0].original_credit}."
    )


def test_skill_28_risk_vetoed_envelope_is_filtered() -> None:
    """Skill 28 §4 — GLD 2026-05-15 incident: a plan with
    ``valid=True`` but ``risk_verdict.approved=False`` (chain
    scanner accepted, risk manager vetoed) must NOT claim legs.

    This is the case the inner-only shape silently neutered before
    the 2026-05-15 envelope-passing fix."""
    common_legs = [
        {"symbol": "TEST260605C00100000", "strike": 100.0, "action": "sell"},
        {"symbol": "TEST260605C00105000", "strike": 105.0, "action": "buy"},
    ]
    inner_passes_valid = {
        "ticker": "TEST", "strategy": "Bear Call", "valid": True,
        "net_credit": 0.50, "max_loss": 450, "spread_width": 5.0,
        "legs": common_legs,
    }
    rejected_envelope = {
        "trade_plan": inner_passes_valid,
        "risk_verdict": {"approved": False, "checks_failed": ["test"]},
    }
    approved_envelope = {
        "trade_plan": {**inner_passes_valid, "net_credit": 0.75},
        "risk_verdict": {"approved": True},
    }
    broker = [_snap(common_legs[0]["symbol"], qty=-1),
              _snap(common_legs[1]["symbol"], qty=+1)]

    spreads = _monitor().group_into_spreads(
        broker, [rejected_envelope, approved_envelope],
    )
    assert len(spreads) == 1
    assert spreads[0].original_credit == 0.75, (
        f"Skill 28 §4: when both a risk-vetoed envelope AND an "
        f"approved envelope cite the same broker legs, the approved "
        f"envelope's economics must win. Got "
        f"{spreads[0].original_credit} (expected 0.75)."
    )

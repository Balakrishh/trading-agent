"""Live mid-first debit ladder (backlog C1, skill 59 §3)."""
from types import SimpleNamespace

import pytest

from trading_agent import executor as ex
from trading_agent.strategy_presets import _make_custom


def _plan():
    legs = [SimpleNamespace(symbol="IWM261120P00273000", action="buy"),
            SimpleNamespace(symbol="IWM261120P00270000", action="sell")]
    return SimpleNamespace(ticker="IWM", strategy_name="Put Debit Spread", legs=legs)


class FakeBroker:
    """Scripted per-probe outcomes: list of (status, filled_qty, avg)."""

    def __init__(self, outcomes, submit_ok=True):
        self.outcomes = list(outcomes)
        self.submitted = []
        self.submit_ok = submit_ok

    def submit(self, plan, plan_path, run_id, order_payload, client_order_id):
        self.submitted.append(order_payload)
        if not self.submit_ok:
            return {"status": "error", "error": "422"}
        return {"status": "submitted", "order_id": f"o{len(self.submitted)}",
                "qty": int(order_payload["qty"]), "limit_price": float(order_payload["limit_price"])}

    def fill(self, order_id):
        return self.outcomes.pop(0)


def _executor(broker, mid_quotes=((1.30, 1.40), (0.20, 0.30))):
    e = ex.OrderExecutor.__new__(ex.OrderExecutor)
    e.debit_ladder_live, e.debit_ladder_steps = True, 4
    syms = ["IWM261120P00273000", "IWM261120P00270000"]
    e.data_provider = SimpleNamespace(fetch_option_quotes=lambda s: {
        syms[0]: {"bid": mid_quotes[0][0], "ask": mid_quotes[0][1]},
        syms[1]: {"bid": mid_quotes[1][0], "ask": mid_quotes[1][1]}})
    e._submit_order_with_idempotency = broker.submit
    e._await_fill = lambda oid, wait_s=0: "canceled"
    e._order_fill = broker.fill
    return e


# mid = 1.35 − 0.25 = 1.10; natural = 1.40 − 0.20 = 1.20 → probes 1.10, 1.13, 1.17 (1.20 excluded)

def test_probe_prices_stop_below_natural():
    b = FakeBroker([("canceled", 0, None)] * 3)
    out = _executor(b)._debit_ladder_probes(_plan(), "p", "run12345", [], 2, natural=1.20)
    assert [float(p["limit_price"]) for p in b.submitted] == [1.10, 1.13, 1.17]
    assert out["remaining"] == 2 and out["final"] is None and out["summary"]["saving_usd"] == 0.0


def test_full_fill_on_second_probe_returns_final_with_saving():
    b = FakeBroker([("canceled", 0, None), ("filled", 2, 1.13)])
    out = _executor(b)._debit_ladder_probes(_plan(), "p", "run12345", [], 2, natural=1.20)
    assert out["final"]["qty"] == 2 and out["final"]["limit_price"] == 1.13
    assert out["summary"]["saving_usd"] == pytest.approx((1.20 - 1.13) * 2 * 100)
    assert len(b.submitted) == 2


def test_partial_fill_sends_only_the_rest():
    b = FakeBroker([("canceled", 1, 1.10), ("canceled", 0, None), ("canceled", 0, None)])
    out = _executor(b)._debit_ladder_probes(_plan(), "p", "run12345", [], 3, natural=1.20)
    assert out["remaining"] == 2 and out["final"] is None
    assert [p["qty"] for p in b.submitted] == ["3", "2", "2"]
    assert out["summary"]["filled_qty"] == 1 and out["summary"]["avg_fill"] == 1.10


def test_unconfirmed_cancel_stops_the_ladder():
    b = FakeBroker([("pending_cancel", 0, None)])
    out = _executor(b)._debit_ladder_probes(_plan(), "p", "run12345", [], 2, natural=1.20)
    assert out["final"] is not None and len(b.submitted) == 1          # nothing more sent


def test_no_mid_or_mid_at_natural_skips_the_ladder():
    b = FakeBroker([])
    e = _executor(b, mid_quotes=((1.20, 1.20), (0.20, 0.20)))
    out = e._debit_ladder_probes(_plan(), "p", "run12345", [], 2, natural=1.00)
    assert b.submitted == [] and out["remaining"] == 2


def test_broker_refusal_falls_back_to_natural():
    b = FakeBroker([], submit_ok=False)
    out = _executor(b)._debit_ladder_probes(_plan(), "p", "run12345", [], 2, natural=1.20)
    assert out["final"] is None and out["remaining"] == 2 and len(b.submitted) == 1


def test_preset_accepts_live():
    assert _make_custom({"debit_ladder_mode": "live"}).debit_ladder_mode == "live"
    assert _make_custom({"debit_ladder_mode": "turbo"}).debit_ladder_mode == "shadow"

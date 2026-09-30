"""Wheel trade lifecycle — skill 40 §2.9 (2026-09-29).

Covers: single-leg plan builder, promote-time checks, single-leg open
(fill confirmation), live-delta attach, expiry reconciliation.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import trading_agent.executor as executor_mod
from trading_agent.executor import OrderExecutor
from trading_agent.executor_promote import check_wheel_order
from trading_agent.position_monitor import SpreadPosition, attach_wheel_short_deltas
from trading_agent.positions_provider import Position
from trading_agent.wheel_lifecycle import reconcile, resolve_expired_wheel_trades
from trading_agent.wheel_policy import (
    CC_STRATEGY, CSP_STRATEGY, EXIT_ASSIGNED, EXIT_CALLED_AWAY,
    EXIT_EXPIRED_WORTHLESS, build_single_leg_plan,
)

SYM = "BMY261120P00057500"


def _csp_plan(strike=57.5, bid=1.25, ask=1.29):
    return build_single_leg_plan(ticker="bmy", strategy_name=CSP_STRATEGY, symbol=SYM,
                                 strike=strike, option_type="put", delta=-0.25,
                                 bid=bid, ask=ask, expiration="2026-11-20")


def _cc_plan():
    return build_single_leg_plan(ticker="BMY", strategy_name=CC_STRATEGY,
                                 symbol="BMY261218C00062500", strike=62.5,
                                 option_type="call", delta=0.25, bid=0.9, ask=1.0,
                                 expiration="2026-12-18")


# ── plan builder ──────────────────────────────────────────────────────────

def test_build_single_leg_plan_csp_economics():
    p = _csp_plan()
    assert p.ticker == "BMY" and len(p.legs) == 1 and p.legs[0].action == "sell"
    assert p.spread_width == 57.5                           # collateral per share
    assert p.max_loss == pytest.approx((57.5 - p.net_credit) * 100)
    assert p.credit_to_width_ratio == pytest.approx(p.net_credit / 57.5, abs=1e-4)


def test_build_single_leg_plan_cc_has_no_option_max_loss():
    assert _cc_plan().max_loss == 0.0


def test_build_single_leg_plan_rejects_spread_strategy():
    with pytest.raises(ValueError):
        build_single_leg_plan(ticker="X", strategy_name="Iron Condor", symbol=SYM,
                              strike=1, option_type="put", delta=0, bid=1, ask=1,
                              expiration="2026-11-20")


# ── promote-time checks ───────────────────────────────────────────────────

def test_csp_check_passes_when_cash_secured():
    assert check_wheel_order(_csp_plan(), qty=1, equity=30_000,
                             options_buying_power=30_000, shares_held=0) == []


@pytest.mark.parametrize("kw,needle", [
    ({"options_buying_power": None}, "buying power unknown"),
    ({"options_buying_power": 5_000}, "> options buying power"),
    ({"equity": 10_000}, "of equity"),                         # 5,750 > 40% × 10k
    ({"qty": 0}, "qty 0"),
])
def test_csp_check_failures(kw, needle):
    args = {"qty": 1, "equity": 30_000, "options_buying_power": 30_000, "shares_held": 0}
    args.update(kw)
    fails = check_wheel_order(_csp_plan(), **args)
    assert any(needle in f for f in fails), fails


def test_covered_call_requires_shares():
    ok = check_wheel_order(_cc_plan(), qty=1, equity=30_000,
                           options_buying_power=0, shares_held=100)
    naked = check_wheel_order(_cc_plan(), qty=1, equity=30_000,
                              options_buying_power=0, shares_held=99)
    assert ok == [] and any("not covered" in f for f in naked)


# ── single-leg open ───────────────────────────────────────────────────────

def _resp(body):
    r = MagicMock()
    r.json.return_value = body
    r.raise_for_status.return_value = None
    r.content = b"x"
    return r


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(executor_mod, "CLOSE_FILL_WAIT_S", 0.0)
    monkeypatch.setattr(executor_mod.time, "sleep", lambda s: None)


def _open(tmp_path, statuses, quotes=None, fill_avg=None):
    """``statuses`` feed the fill polls; when the last one is "filled" the
    executor makes one more GET for ``filled_avg_price`` (``fill_avg``)."""
    dp = MagicMock()
    dp.fetch_option_quotes.return_value = quotes or {SYM: {"bid": 1.20, "ask": 1.30}}
    ex = OrderExecutor("k", "s", trade_plan_dir=str(tmp_path), dry_run=False, data_provider=dp)
    posts = iter([_resp({"id": "o1"}), _resp({"id": "o2"}), _resp({"id": "o3"})])
    bodies = [{"status": s} for s in statuses]
    if statuses and statuses[-1] == "filled":
        bodies.append({"status": "filled", "filled_avg_price": fill_avg})
    gets = iter([_resp(b) for b in bodies])
    with patch.object(executor_mod.requests, "post", side_effect=lambda *a, **k: next(posts)) as post, \
         patch.object(executor_mod.requests, "get", side_effect=lambda *a, **k: next(gets)), \
         patch.object(executor_mod.requests, "delete", return_value=_resp({})):
        return ex.execute_single_leg(_csp_plan(), qty=1, account_balance=30_000), post


def test_single_leg_fills_at_mid(tmp_path, fast):
    res, post = _open(tmp_path, ["filled"])
    body = post.call_args.kwargs["json"]
    assert res["status"] == "filled" and res["limit_price"] == 1.25
    assert body["position_intent"] == "sell_to_open" and body["side"] == "sell"
    assert "order_class" not in body and body["symbol"] == SYM


def test_single_leg_concedes_to_bid_then_reports_unfilled(tmp_path, fast):
    res, post = _open(tmp_path, ["new", "canceled"] * 3)
    prices = [float(c.kwargs["json"]["limit_price"]) for c in post.call_args_list]
    # bid 1.20 / ask 1.30: mid 1.25, halfway mid→bid (1.225 → cent), then the bid
    assert prices[0] == 1.25 and 1.20 < prices[1] < 1.25 and prices[2] == 1.20
    assert res["status"] == "unfilled"


def _plan_entry(tmp_path):
    import json
    doc = json.loads((tmp_path / "trade_plan_BMY.json").read_text())
    return doc["state_history"][-1]["trade_plan"]


def test_fill_price_rewrites_trade_plan_credit(tmp_path, fast):
    """Regression 2026-09-30: VZ filled at 0.21 but the plan kept the 0.26
    estimate, so the monitor's 50 % target used the wrong credit."""
    res, _ = _open(tmp_path, ["filled"], fill_avg="1.23")
    tp = _plan_entry(tmp_path)
    assert res["fill_price"] == 1.23
    assert tp["net_credit"] == 1.23 and tp["estimated_net_credit"] == _csp_plan().net_credit
    assert tp["max_loss"] == pytest.approx((57.5 - 1.23) * 100)
    assert tp["credit_to_width_ratio"] == pytest.approx(1.23 / 57.5, abs=1e-4)


def test_fill_price_falls_back_to_limit_when_unreported(tmp_path, fast):
    res, _ = _open(tmp_path, ["filled"], fill_avg=None)
    assert res["fill_price"] == res["limit_price"] == 1.25
    assert _plan_entry(tmp_path)["net_credit"] == 1.25


def test_single_leg_fills_at_bid_on_final_attempt(tmp_path, fast):
    res, post = _open(tmp_path, ["new", "canceled", "new", "canceled", "filled"])
    assert res["status"] == "filled" and res["limit_price"] == 1.20
    assert post.call_count == 3


def test_single_leg_skips_bid_attempt_on_wide_live_quote(tmp_path, fast):
    """Quote widened after screening (0.21/0.60: 39¢, 96 % of mid) — never
    concede to the bid; stop after mid and halfway."""
    res, post = _open(tmp_path, ["new", "canceled"] * 2,
                      quotes={SYM: {"bid": 0.21, "ask": 0.60}})
    prices = [float(c.kwargs["json"]["limit_price"]) for c in post.call_args_list]
    assert post.call_count == 2 and 0.21 not in prices
    assert res["status"] == "unfilled"


def test_single_leg_unresolved_cancel_is_not_filled(tmp_path, fast):
    res, _ = _open(tmp_path, ["new", "pending_cancel"])
    assert res["status"] == "unresolved"


def test_single_leg_dry_run_submits_nothing(tmp_path):
    ex = OrderExecutor("k", "s", trade_plan_dir=str(tmp_path), dry_run=True)
    with patch.object(executor_mod.requests, "post") as post:
        assert ex.execute_single_leg(_csp_plan())["status"] == "dry_run"
    post.assert_not_called()


def test_single_leg_rejects_multi_leg_plan(tmp_path):
    ex = OrderExecutor("k", "s", trade_plan_dir=str(tmp_path), dry_run=True)
    p = _csp_plan()
    p.legs = p.legs * 2
    with pytest.raises(ValueError):
        ex.execute_single_leg(p)


def test_single_leg_close_uses_plain_limit_order(tmp_path, fast):
    """Buy-to-close a Wheel leg: limit order, not a market DELETE."""
    dp = MagicMock()
    dp.fetch_option_quotes.return_value = {SYM: {"bid": 0.60, "ask": 0.66}}
    ex = OrderExecutor("k", "s", trade_plan_dir=str(tmp_path), dry_run=False, data_provider=dp)
    from trading_agent.position_monitor import ExitSignal
    spread = SimpleNamespace(underlying="BMY", strategy_name=CSP_STRATEGY,
                             exit_signal=ExitSignal.PROFIT_TARGET, exit_reason="t",
                             legs=[SimpleNamespace(symbol=SYM, qty=-1)])
    with patch.object(executor_mod.requests, "post", return_value=_resp({"id": "c1"})) as post, \
         patch.object(executor_mod.requests, "get", return_value=_resp({"status": "filled"})), \
         patch.object(executor_mod.requests, "delete") as delete:
        res = ex.close_spread(spread)
    body = post.call_args.kwargs["json"]
    assert res["all_closed"] and body["position_intent"] == "buy_to_close"
    assert "order_class" not in body
    delete.assert_not_called()


# ── live delta attach ─────────────────────────────────────────────────────

def _wheel_spread(strategy=CSP_STRATEGY, symbol=SYM):
    return SpreadPosition(underlying="BMY", strategy_name=strategy,
                          legs=[SimpleNamespace(symbol=symbol)], original_credit=1.27,
                          max_loss=0, spread_width=57.5, net_unrealized_pl=0,
                          expiration="2026-11-20", short_strikes=[57.5])


def test_attach_delta_from_chain():
    s = _wheel_spread()
    calls = []

    def fetch(u, e, o):
        calls.append((u, e, o))
        return [{"symbol": SYM, "delta": -0.47}, {"symbol": "OTHER", "delta": -0.1}]

    attach_wheel_short_deltas([s], fetch)
    assert s.short_delta == -0.47 and calls == [("BMY", "2026-11-20", "put")]


def test_attach_delta_skips_spreads_and_survives_failure():
    spread = _wheel_spread(strategy="Iron Condor")
    csp = _wheel_spread()
    attach_wheel_short_deltas([spread, csp], lambda *a: (_ for _ in ()).throw(RuntimeError()))
    assert spread.short_delta is None and csp.short_delta is None


# ── expiry reconciliation ─────────────────────────────────────────────────

def _open_trade(strategy, expiration="2026-11-20", contracts=1, ticker="BMY"):
    return SimpleNamespace(ticker=ticker, strategy=strategy, expiration=expiration,
                           credit=1.27, contracts=contracts)


@pytest.mark.parametrize("strategy,shares,signal", [
    (CSP_STRATEGY, 100, EXIT_ASSIGNED),
    (CSP_STRATEGY, 0, EXIT_EXPIRED_WORTHLESS),
    (CC_STRATEGY, 0, EXIT_CALLED_AWAY),
    (CC_STRATEGY, 100, EXIT_EXPIRED_WORTHLESS),
])
def test_resolve_outcomes(strategy, shares, signal):
    [r] = resolve_expired_wheel_trades([_open_trade(strategy)], {"BMY": shares}, date(2026, 11, 21))
    assert r.exit_signal == signal and r.realized_pl == 127.0


def test_resolve_waits_until_after_expiration_and_ignores_spreads():
    trades = [_open_trade(CSP_STRATEGY), _open_trade("Iron Condor", "2026-01-01")]
    assert resolve_expired_wheel_trades(trades, {}, date(2026, 11, 20)) == []


def test_resolve_scales_by_contracts():
    [r] = resolve_expired_wheel_trades([_open_trade(CSP_STRATEGY, contracts=2)],
                                       {"BMY": 100}, date(2026, 11, 21))
    assert r.exit_signal == EXIT_EXPIRED_WORTHLESS    # 100 < 200 shares for 2 contracts
    assert r.realized_pl == 254.0


def _reconcile(tmp_path, *, fetch_ok=True, holdings=()):
    provider = SimpleNamespace(snapshot=lambda: list(holdings), last_fetch_ok=fetch_ok)
    reader = SimpleNamespace(open_trades=lambda: [_open_trade(CSP_STRATEGY)])
    kb = MagicMock()
    sentinel = tmp_path / ".wheel_reconcile_date"
    out = reconcile(journal_reader=reader, positions_provider=provider, journal_kb=kb,
                    today=date(2026, 11, 23), sentinel=sentinel)
    return out, kb, sentinel


def test_reconcile_journals_and_runs_once_per_day(tmp_path):
    held = [Position(ticker="BMY", qty=100, avg_cost=56.23, kind="stock")]
    out, kb, sentinel = _reconcile(tmp_path, holdings=held)
    assert [r.exit_signal for r in out] == [EXIT_ASSIGNED]
    raw = kb.log_signal.call_args.kwargs["raw_signal"]
    assert kb.log_signal.call_args.kwargs["action"] == "closed"
    assert raw["strategy"] == CSP_STRATEGY and raw["fill_status"] == "complete"
    assert sentinel.read_text() == "2026-11-23"
    out2, kb2, _ = _reconcile(tmp_path, holdings=held)            # same day → no-op
    assert out2 == [] and not kb2.log_signal.called


def test_reconcile_skips_when_holdings_unknown(tmp_path):
    out, kb, sentinel = _reconcile(tmp_path, fetch_ok=False)
    assert out == [] and not kb.log_signal.called and not sentinel.exists()


def test_unfilled_run_is_invalidated_so_it_cannot_shadow_a_later_fill(tmp_path, fast):
    """Regression 2026-09-30: VZ's unfilled 10:00 run (estimate 0.26) was
    matched to the position before the filled 10:14 run (fill 0.21)."""
    _open(tmp_path, ["new", "canceled"] * 3)                 # run 1: unfilled
    tp = _plan_entry(tmp_path)
    assert tp["valid"] is False and tp["rejection_reason"].startswith("unfilled")


def test_unresolved_run_is_not_invalidated(tmp_path, fast):
    """A cancel that wasn't confirmed may still fill — keep the entry valid."""
    _open(tmp_path, ["new", "pending_cancel"])
    assert _plan_entry(tmp_path).get("valid", True) is True

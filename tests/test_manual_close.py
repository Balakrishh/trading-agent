"""Skill 63 — operator close CLI (trading_agent/manual_close.py)."""
import json
from types import SimpleNamespace

import pytest

from trading_agent import manual_close as mc
from trading_agent.executor import OrderExecutor, close_order_prices
from trading_agent.position_monitor import (ExitSignal, PositionSnapshot, SpreadPosition,
                                            load_trade_plans)

SHORT, LONG = "IWM261106P00278000", "IWM261106P00283000"
QUOTES = {SHORT: {"bid": 6.30, "ask": 6.48}, LONG: {"bid": 9.34, "ask": 9.46}}


def leg(symbol, qty, entry):
    return PositionSnapshot(symbol=symbol, qty=qty, side="short" if qty < 0 else "long",
                            avg_entry_price=entry, current_price=entry, market_value=0.0,
                            cost_basis=0.0, unrealized_pl=0.0, unrealized_plpc=0.0,
                            asset_class="us_option")


def spread(ticker="IWM", strategy="Put Debit Spread", legs=None, expiration="2026-11-06"):
    legs = legs if legs is not None else [leg(SHORT, -4, 4.00), leg(LONG, 4, 5.92)]
    return SpreadPosition(underlying=ticker, strategy_name=strategy, legs=legs,
                          original_credit=-1.92, max_loss=768.0, spread_width=5.0,
                          net_unrealized_pl=386.0, expiration=expiration,
                          short_strikes=[278.0], contracts_open=4)


# ── select_spread ──────────────────────────────────────────────────────

def test_select_finds_the_one_position_case_insensitively():
    s, err = mc.select_spread([spread(), spread("QQQ", "Call Debit Spread")], "iwm", None)
    assert err is None and s.underlying == "IWM"


def test_select_reports_none_with_what_is_open():
    s, err = mc.select_spread([spread("QQQ", "Call Debit Spread")], "IWM", None)
    assert s is None and "No open IWM" in err and "QQQ Call Debit Spread" in err


def test_select_never_picks_between_two_positions():
    two = [spread(), spread(strategy="Calendar Spread", expiration="2026-10-30")]
    s, err = mc.select_spread(two, "IWM", None)
    assert s is None and "--strategy" in err
    s, err = mc.select_spread(two, "IWM", "calendar spread")
    assert err is None and s.strategy_name == "Calendar Spread"


# ── refusal ────────────────────────────────────────────────────────────

def test_refusal_allows_a_normal_spread_in_session():
    assert mc.refusal(spread(), market_open=True, dry_run=False) is None


@pytest.mark.parametrize("legs, market_open, dry_run, needle", [
    ([leg("VZ261023P00043000", -1, 0.21)], True, False, "single leg"),
    ([leg(SHORT, -4, 4.0), leg(LONG, 3, 5.92)], True, False, "Unequal"),
    (None, True, True, "DRY_RUN"),
    (None, False, False, "market is closed"),
])
def test_refusals(legs, market_open, dry_run, needle):
    assert needle in mc.refusal(spread(legs=legs), market_open=market_open, dry_run=dry_run)


# ── pricing / preview ──────────────────────────────────────────────────

def test_close_order_prices_signs():
    mid, natural, payload = close_order_prices(spread().legs, QUOTES)
    assert mid == pytest.approx(6.39 - 9.40)          # credit → negative
    assert natural == pytest.approx(6.48 - 9.34)
    assert [p["side"] for p in payload] == ["buy", "sell"]


def test_close_order_prices_none_without_two_sided_quotes():
    assert close_order_prices(spread().legs, {SHORT: QUOTES[SHORT], LONG: {"bid": 0, "ask": 9.4}}) is None


def test_preview_prices_and_estimates_pl():
    p = mc.preview(spread(), QUOTES)
    assert p["priced"] and p["first_limit"] == pytest.approx(-2.93)
    # entry debit 1.92/share; closing for a 2.93 credit on 4 contracts = +$404
    assert p["pl_at_first_limit"] == pytest.approx(404.0)
    assert p["pl_at_natural"] == pytest.approx(376.0)


def test_preview_unpriced_when_a_quote_is_missing():
    p = mc.preview(spread(), {SHORT: QUOTES[SHORT]})
    assert p["priced"] is False and "first_limit" not in p


# ── journal payload ────────────────────────────────────────────────────

def test_close_context_is_a_manual_close_row():
    ctx = mc.close_context(spread(), "taking profit")
    assert ctx["exit_signal"] == ExitSignal.MANUAL.value == "manual"
    assert ctx["exit_reason"] == "taking profit" and ctx["expiration"] == "2026-11-06"


def test_apply_fill_prefers_the_fill_pl():
    ctx, status = mc.apply_fill(mc.close_context(spread(), "r"),
                                {"realized_pl": 288.0, "fill_debit": -2.64, "all_closed": True})
    assert status == "complete" and ctx["net_unrealized_pl"] == 288.0
    assert ctx["pl_source"] == "fill" and ctx["signal_mark_pl"] == 386.0


def test_apply_fill_unresolved_is_partial_and_keeps_the_mark():
    ctx, status = mc.apply_fill(mc.close_context(spread(), "r"), {"all_closed": False})
    assert status == "partial" and ctx["pl_source"] == "signal_mark"
    assert ctx["net_unrealized_pl"] == 386.0


# ── executor: atomic only ──────────────────────────────────────────────

def test_close_spread_atomic_never_falls_back_to_per_leg(monkeypatch):
    ex = OrderExecutor.__new__(OrderExecutor)
    monkeypatch.setattr(ex, "_close_spread_mleg", lambda s: None)
    monkeypatch.setattr(ex, "_close_single_leg",
                        lambda sym: pytest.fail("per-leg close must not run"))
    assert ex.close_spread_atomic(spread()) is None


# ── shared trade-plan loader ───────────────────────────────────────────

def test_load_trade_plans_both_formats(tmp_path):
    (tmp_path / "trade_plan_IWM.json").write_text(json.dumps({"state_history": [{"a": 1}, {"b": 2}]}))
    (tmp_path / "trade_plan_SPY_20260101.json").write_text(json.dumps({"c": 3}))
    (tmp_path / "trade_plan_BAD.json").write_text("{not json")
    assert load_trade_plans(str(tmp_path)) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_load_trade_plans_missing_dir(tmp_path):
    assert load_trade_plans(str(tmp_path / "nope")) == []


# ── main(): preview sends nothing; submit journals ─────────────────────

class _Fake:
    def __init__(self, result=None):
        self.closed, self.written, self.result = [], [], result


def _wire(monkeypatch, *, market_open=True, dry_run=False, result=None):
    fake = _Fake(result)
    cfg = SimpleNamespace(alpaca=SimpleNamespace(base_url="https://paper-api.alpaca.markets/v2"),
                          logging=SimpleNamespace(trade_plan_dir="/nonexistent"),
                          trading=SimpleNamespace(dry_run=dry_run))
    data = SimpleNamespace(fetch_option_quotes=lambda syms: QUOTES)
    monitor = SimpleNamespace(fetch_open_positions=lambda: ["x"],
                              group_into_spreads=lambda pos, plans: [spread()])

    def close(s):
        fake.closed.append(s)
        return fake.result
    executor = SimpleNamespace(close_spread_atomic=close)
    writer = SimpleNamespace(write=lambda s, ctx, **kw: fake.written.append((ctx, kw)))
    monkeypatch.setattr("trading_agent.config.load_config", lambda: cfg)
    monkeypatch.setattr("trading_agent.market_hours.is_within_market_hours", lambda: market_open)
    monkeypatch.setattr(mc, "_build", lambda c: (data, monitor, executor, writer))
    return fake


def test_main_preview_sends_no_order(monkeypatch):
    fake = _wire(monkeypatch)
    assert mc.main(["--ticker", "IWM"]) == mc.EXIT_OK
    assert fake.closed == [] and fake.written == []


def test_main_submit_refused_when_market_closed(monkeypatch):
    fake = _wire(monkeypatch, market_open=False)
    assert mc.main(["--ticker", "IWM", "--submit"]) == mc.EXIT_REFUSED
    assert fake.closed == []


def test_main_submit_not_filled_journals_nothing(monkeypatch):
    fake = _wire(monkeypatch, result=None)
    assert mc.main(["--ticker", "IWM", "--submit"]) == mc.EXIT_NOT_FILLED
    assert len(fake.closed) == 1 and fake.written == []


def test_main_submit_fill_journals_a_manual_close(monkeypatch):
    fill = {"realized_pl": 288.0, "fill_debit": -2.64, "all_closed": True,
            "close_method": "mleg_improved", "leg_results": [{"symbol": SHORT, "status": "closed"}]}
    fake = _wire(monkeypatch, result=fill)
    assert mc.main(["--ticker", "IWM", "--submit", "--reason", "early"]) == mc.EXIT_OK
    s = fake.closed[0]
    assert s.exit_signal is ExitSignal.MANUAL and s.exit_reason == "early"
    ctx, kw = fake.written[0]
    assert ctx["net_unrealized_pl"] == 288.0 and kw["fill_status"] == "complete"


def test_main_unknown_ticker(monkeypatch):
    _wire(monkeypatch)
    assert mc.main(["--ticker", "TSLA"]) == mc.EXIT_NOT_FOUND

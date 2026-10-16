"""Daily reviewer inputs (skill 56, 2026-10-16): closes carry P&L and
reason, the agent's real ticker list, the market-state macro snapshot
with a VIX zone, and the shadow trend-filter tally. Before the fix the
LLM saw pnl=null, a 31-symbol dashboard watchlist and vix_zone=null."""
import json
from datetime import datetime, timezone

import pytest

from trading_agent import daily_reviewer as dr
from trading_agent import market_state
from trading_agent.journal_reader import JournalReader


def _row(ts, tk, action, **rs):
    return json.dumps({"timestamp": ts, "ticker": tk, "action": action, "raw_signal": rs})


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trade_journal").mkdir()
    (tmp_path / "trade_journal" / "signals_live.jsonl").write_text("\n".join([
        _row("2026-10-15T13:46:10+00:00", "TLT", "closed", strategy="Put Debit Spread",
             exit_signal="stop_loss", exit_reason="Debit: loss $168.00 ≥ 50% of debit $276.00",
             net_unrealized_pl=-186.0, expiration="2026-11-13"),
        _row("2026-10-15T13:50:38+00:00", "TLT", "submitted", strategy="Put Debit Spread",
             net_credit=-0.55, max_loss=55.0, expiration="2026-11-13"),
        _row("2026-10-15T13:50:00+00:00", "TLT", "rejected", strategy="Put Debit Spread",
             trend_filters={"direction": "bear", "blocked": False, "checks": [
                 {"name": "relative_strength", "ok": True},
                 {"name": "pullback", "ok": False}]}),
    ]) + "\n")
    state = tmp_path / "market_state.json"
    state.write_text(json.dumps({
        "state": "CAUTION", "reasons": ["breadth 36% < 50%"],
        "inputs": {"vix": 16.41, "vix3m": 18.47, "term_ratio": 0.89, "breadth": 0.357,
                   "breadth_n": 14, "spy_price": 769.6, "spy_sma200": 724.2},
        "gate": {"size_multiplier": 0.5, "allow_new_csp": False},
        "as_of_utc": datetime.now(timezone.utc).isoformat(), "account_balance": 29018.13}))
    monkeypatch.setattr(market_state, "STATE_PATH", state)
    monkeypatch.setattr(dr, "REVIEWS_DIR", tmp_path / "daily_reviews")
    monkeypatch.setenv("TICKERS", "SPY,QQQ,TLT")
    return dr.assemble_context("2026-10-15")


def test_close_carries_pnl_signal_and_reason(ctx):
    c = ctx.closes[0]
    assert c["pnl"] == -186.0
    assert c["exit_signal"] == "stop_loss"
    assert c["reason"].startswith("Debit: loss")
    assert c["closed_at_utc"].startswith("2026-10-15T13:46")


def test_open_carries_risk_and_time(ctx):
    o = ctx.opens[0]
    assert o["ticker"] == "TLT" and o["opened_at_utc"].startswith("2026-10-15T13:50")


def test_trading_universe_is_tickers(ctx):
    assert ctx.trading_universe == ["SPY", "QQQ", "TLT"]


def test_macro_from_market_state(ctx):
    m = ctx.macro
    assert m["vix_zone"] == "Normal"
    assert m["market_state"] == "CAUTION" and m["size_multiplier"] == 0.5
    assert m["allow_new_csp"] is False and m["account_balance"] == 29018.13


def test_trend_filter_tally(ctx):
    assert ctx.trend_filters == {"plans_checked": 1, "would_block": {"pullback": 1},
                                 "would_block_tickers": {"pullback": ["TLT"]}}


def test_prompt_carries_new_fields_and_explains_lists(ctx):
    body = dr._build_user_prompt(ctx)
    assert '"trading_universe"' in body and '"trend_filters"' in body
    assert "trading_universe is the list of tickers" in dr._SYSTEM_PROMPT


def test_missing_market_state_leaves_vix_zone_null(tmp_path, monkeypatch):
    monkeypatch.setattr(market_state, "STATE_PATH", tmp_path / "nope.json")
    assert dr._load_macro_snapshot() == {"vix_zone": None}


def test_reject_reason_keeps_the_cause_past_120_chars(tmp_path):
    long = ("bear calls off (bear_call_enabled=False) — put debit spread instead; "
            "fallback Put Debit Spread: no acceptable candidate  (leg_spread_wide)")
    assert len(long) > 120
    fp = tmp_path / "j.jsonl"
    fp.write_text(_row("2026-10-15T15:00:00+00:00", "XLF", "rejected",
                       rejection_reason=long) + "\n")
    from datetime import date
    top = JournalReader(str(fp), as_of=date(2026, 10, 15)).reject_reasons_today()
    assert top[0][0].endswith("(leg_spread_wide)")

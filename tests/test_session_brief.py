"""2026-11-09 — pre-market brief and plan for tomorrow (skill 71)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime

import pytest

from trading_agent import session_brief as sb
from trading_agent.market_hours import EASTERN
from trading_agent.strategy_presets import PRESETS

PRE = replace(PRESETS["balanced"], debit_profit_target_pct_bear=0.4, debit_stop_loss_pct_bull=0.8,
              vol_band_put_enabled=True, bear_call_enabled=False)
MON = date(2026, 11, 9)
STATE = {"state": "CAUTION", "reasons": ["SPY < SMA-50", "VIX 24.8 ≥ 20"],
         "gate": {"size_multiplier": 0.5, "allow_new_csp": False,
                  "allowed_strategies": ["Bear Call Spread", "Put Debit Spread"]},
         "as_of_utc": "2026-11-06T21:04:22+00:00"}


def _pos(**kw):
    p = dict(ticker="GLD", strategy="Put Debit Spread", credit=-2.65, contracts=1, mid_pl=-50.0,
             max_loss=265.0, total_max_loss=265.0, expiration="2026-11-20")
    p.update(kw)
    return p


# ── exit view ───────────────────────────────────────────────────────────

def test_debit_targets_follow_direction_rules():
    v = sb.exit_view(_pos(), PRE, MON)
    assert v["target"] == pytest.approx(106.0) and v["stop"] == pytest.approx(-132.5)
    c = sb.exit_view(_pos(strategy="Call Debit Spread"), PRE, MON)
    assert c["target"] == pytest.approx(132.5) and c["stop"] == pytest.approx(-212.0)


def test_credit_targets_use_credit_and_max_loss():
    v = sb.exit_view(_pos(strategy="Bull Put Spread", credit=0.60, total_max_loss=140.0, mid_pl=10.0), PRE, MON)
    assert v["target"] == pytest.approx(30.0) and v["stop"] == pytest.approx(-70.0)


@pytest.mark.parametrize("kw, today, flag", [
    ({"mid_pl": 100.0}, MON, "near profit target"),
    ({"mid_pl": -120.0}, MON, "near stop"),
    ({}, date(2026, 11, 19), "DTE-safety close this session"),     # last session before 11-20
    ({}, date(2026, 11, 18), "2 session(s) to expiry"),
    ({}, date(2026, 11, 13), "≤7 DTE: trail no longer holds"),
])
def test_flags(kw, today, flag):
    assert flag in sb.exit_view(_pos(**kw), PRE, today)["flags"]


def test_quiet_position_has_no_flags():
    assert sb.exit_view(_pos(), PRE, MON)["flags"] == []


# ── builders ────────────────────────────────────────────────────────────

def _ctx(**kw):
    ctx = dict(today=MON, next_day=date(2026, 11, 10), preset=PRE, market_state=STATE,
               quotes={"SPY": {"close": 735.82, "extended": 740.0, "last": 735.82, "change_pct": 740 / 735.82 - 1},
                       "META": {"close": 600.0, "extended": 585.0, "last": 600.0, "change_pct": -0.025},
                       "XLU": {"close": 80.0, "extended": 80.4, "last": 80.0, "change_pct": 0.005}},
               events=[{"kind": "CPI", "date": "2026-11-10", "release_et": "08:30"}],
               earnings={"META": date(2026, 11, 18), "AMZN": date(2027, 1, 30), "SPY": None},
               views=[sb.exit_view(_pos(), PRE, MON)], held={"GLD"},
               journal={"opens_today": 1, "closes_today": 2, "realized_pl_today": 131.0,
                        "reject_reasons_top5": [["cw_below_floor", 40], ["leg_spread_wide", 12]]})
    ctx.update(kw)
    return ctx


def test_premarket_brief_sections():
    t = sb.build_premarket(_ctx())
    assert "CAUTION" in t and "×0.5" in t and "new CSPs paused" in t
    assert "SPY 735.82→740.00 (+0.6%)" in t
    assert "META -2.5%" in t and "XLU" not in t.split("Movers")[1].split("\n")[0]
    assert "no FOMC / CPI / jobs report" in t and "Next macro: CPI 2026-11-10" in t
    assert "META 11-18" in t and "AMZN" not in t                     # earnings ≤ 14 d only
    assert "GLD</b> Put Debit Spread" in t and "Held" in t
    assert "Bear Call Spread (off)" in t
    assert "Vol-band bull put" in t and "blocked by the market state" in t
    assert "&lt;" not in t or "<" not in t.replace("<b>", "").replace("</b>", "").replace(
        "<i>", "").replace("</i>", "")                                # free text is escaped


def test_html_is_escaped():
    t = sb.build_premarket(_ctx(market_state={**STATE, "reasons": ["SPY 735 < SMA-50 765"]}))
    assert "SPY 735 &lt; SMA-50 765" in t


def test_tomorrow_brief_lists_only_flagged_positions():
    views = [sb.exit_view(_pos(), PRE, date(2026, 11, 10)),
             sb.exit_view(_pos(ticker="XLE", mid_pl=100.0), PRE, date(2026, 11, 10))]
    t = sb.build_tomorrow(_ctx(views=views))
    assert "Plan for tomorrow — Tue 2026-11-10" in t
    assert "1 open(s), 2 close(s), realized +$131" in t and "cw_below_floor ×40" in t
    assert "Tomorrow</b>: CPI 08:30 ET" in t
    assert "(1 of 2)" in t and "XLE" in t and "GLD" not in t.split("Positions to watch")[1].split("Market state")[0]


def test_missing_inputs_still_render():
    t = sb.build_premarket(dict(today=MON, preset=PRESETS["balanced"]))
    assert "unavailable" in t and "no pre-market quotes" in t and "none" in t


# ── gather / schedule ───────────────────────────────────────────────────

def test_gather_survives_failing_sources():
    def boom(*a, **k):
        raise RuntimeError("down")
    ctx = sb.gather("premarket", MON, ["SPY", "META"], PRE, get=boom, positions=boom,
                    market_state=boom, journal=boom, earnings=boom)
    assert ctx["views"] == [] and "quotes" not in ctx
    assert sb.build_premarket(ctx)


def test_gather_wires_sources_and_judges_tomorrow_on_next_session():
    seen = []

    def get(path):
        seen.append(path)
        return {"events": []} if path == "macro-events" else {"quotes": {"SPY": {"change_pct": 0.01}}}
    ctx = sb.gather("tomorrow", date(2026, 11, 25), ["SPY"], PRE, get=get,
                    positions=lambda: {"open_positions": [_pos()]}, market_state=lambda: STATE,
                    journal=lambda: {}, earnings=lambda t: None)
    assert ctx["next_day"] == date(2026, 11, 27)                     # skips Thanksgiving
    assert seen == ["macro-events"]                                  # no quotes for the evening brief
    assert ctx["held"] == {"GLD"}


@pytest.mark.parametrize("kind, when, reason", [
    ("premarket", datetime(2026, 11, 9, 9, 0), None),
    ("premarket", datetime(2026, 11, 9, 10, 0), "outside"),
    ("premarket", datetime(2026, 11, 26, 9, 0), "not a trading day"),
    ("tomorrow", datetime(2026, 11, 9, 16, 20), None),
    ("tomorrow", datetime(2026, 11, 9, 12, 0), "outside"),
])
def test_due_window_and_calendar(tmp_path, kind, when, reason):
    got = sb.due(kind, when.replace(tzinfo=EASTERN), tmp_path)
    assert (got is None) if reason is None else (reason in got)


def test_sent_once_per_day(tmp_path):
    when = datetime(2026, 11, 9, 9, 0, tzinfo=EASTERN)
    sb.mark_sent("premarket", when.date(), tmp_path)
    assert sb.due("premarket", when, tmp_path) == "already sent today"
    assert sb.due("tomorrow", when.replace(hour=16, minute=20), tmp_path) is None


def test_telegram_message_is_capped(monkeypatch):
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    sent = []
    monkeypatch.setattr(n, "_send", lambda text, channel="info": sent.append((text, channel)) or True)
    assert n.notify_session_brief("x" * 5000)
    assert len(sent[0][0]) < 4096 and sent[0][1] == "info"


# ── data server endpoint ────────────────────────────────────────────────

def test_session_quotes_parsing(monkeypatch):
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    md = object.__new__(SchwabMarketDataProvider)
    monkeypatch.setattr(md, "_get", lambda path, params=None, **k: {
        "SPY": {"quote": {"closePrice": 735.82, "lastPrice": 735.82}, "extended": {"lastPrice": 740.0}},
        "QQQ": {"quote": {"closePrice": 650.0, "lastPrice": 0}, "extended": {}},
        "BAD": "x"})
    out = md.fetch_session_quotes(["SPY", "QQQ", "BAD"])
    assert out["SPY"]["change_pct"] == pytest.approx(740 / 735.82 - 1)
    assert out["QQQ"]["change_pct"] is None and out["QQQ"]["last"] is None
    assert "BAD" not in out


def test_session_quotes_use_the_most_recent_trade(monkeypatch):
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    md = object.__new__(SchwabMarketDataProvider)
    monkeypatch.setattr(md, "_get", lambda path, params=None, **k: {
        "SPY": {"quote": {"closePrice": 700.0, "lastPrice": 707.0, "tradeTime": 2_000},
                "extended": {"lastPrice": 701.0, "tradeTime": 1_000}},      # Friday after-hours
        "QQQ": {"quote": {"closePrice": 600.0, "lastPrice": 600.0, "tradeTime": 1_000},
                "extended": {"lastPrice": 606.0, "tradeTime": 2_000}}})     # this morning
    out = md.fetch_session_quotes(["SPY", "QQQ"])
    assert out["SPY"]["change_pct"] == pytest.approx(0.01)
    assert out["QQQ"]["change_pct"] == pytest.approx(0.01)

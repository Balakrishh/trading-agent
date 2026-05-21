"""Conformance test: skill 32 — Telegram operator alerts.

Pins the public-contract surface of TelegramNotifier and the agent
dedup helper. Behavior tests use monkeypatched _send (no real Telegram
calls) so they're sandbox-runnable.

Failure modes caught:
- Someone renames TelegramNotifier or one of the notify_* methods
- Someone removes the env-gating (is_active) — making the notifier
  always-on would crash test/CI environments that don't set the env
- Someone removes the 5-second timeout — making a slow Telegram API
  capable of tanking the 5-min cycle
- Someone removes the dedup helper — the operator would receive 78
  alerts per day for the same stuck DIA position
"""

from __future__ import annotations

import os


def test_skill_32_telegram_notifier_class_exists() -> None:
    """Skill 32 §3.1: TelegramNotifier is the documented entry point."""
    from trading_agent.telegram_notifier import TelegramNotifier
    assert callable(TelegramNotifier)


def test_skill_32_is_active_requires_both_env_vars() -> None:
    """Skill 32 §1: notifier opt-in needs BOTH TELEGRAM_BOT_TOKEN and
    TELEGRAM_CHAT_ID. Missing either → is_active False → no-op.

    Without this, a test environment with only one var set would still
    try to call Telegram and either crash on bad creds or page someone.
    """
    from trading_agent.telegram_notifier import TelegramNotifier
    # All combinations: only-False should be when both are populated.
    assert TelegramNotifier(token="", chat_id="").is_active is False
    assert TelegramNotifier(token="abc", chat_id="").is_active is False
    assert TelegramNotifier(token="", chat_id="xyz").is_active is False
    assert TelegramNotifier(token="abc", chat_id="xyz").is_active is True


def test_skill_32_three_notify_methods_documented() -> None:
    """Skill 32 §3.2: the three operator-actionable alert methods exist
    with the documented names. Renaming any silently breaks the agent's
    wire-up and CI doesn't catch it without this pin."""
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    for method in ("notify_pdt_block", "notify_close_cooldown",
                   "notify_open_failed_after_close"):
        assert callable(getattr(n, method, None)), (
            f"Skill 32 §3.2: TelegramNotifier.{method} must exist. "
            f"Renaming silently breaks the agent's alert wire-up."
        )


def test_skill_32_send_is_bounded_by_timeout() -> None:
    """Skill 32 §3.3: every send must use the bounded timeout.
    A 5-second cap is the boundary between 'alert system' and 'thing
    that can crash the trade cycle'."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "telegram_notifier.py").read_text(
        encoding="utf-8"
    )
    assert "_TELEGRAM_TIMEOUT_SEC = 5" in src, (
        "Skill 32 §3.3: the timeout constant must be 5 seconds. "
        "Larger values risk a slow Telegram API tanking a 5-min cycle."
    )
    assert "timeout=_TELEGRAM_TIMEOUT_SEC" in src, (
        "Skill 32 §3.3: the requests.post call must use the constant. "
        "Inline numeric timeouts drift; the constant is the contract."
    )


def test_skill_32_send_swallows_exceptions() -> None:
    """Skill 32 §4: a flaky network / DNS / 502 must never propagate
    out of _send. The agent's primary job (trade execution) cannot
    fail because Telegram has a problem."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "telegram_notifier.py").read_text(
        encoding="utf-8"
    )
    # Pin the bare-Exception catch in _send. Specific-class catches
    # would let weird errors through (e.g. unhashable type from a JSON
    # serialisation bug). Skill 32 §4 explicitly documents bare-Exception.
    assert "except Exception" in src, (
        "Skill 32 §4: _send must catch a bare Exception. Narrower "
        "catches let a category of errors crash the agent cycle."
    )


def test_skill_32_agent_dedup_helper_exists() -> None:
    """Skill 32 §3.4: agent must have a journal-derived dedup helper."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "def _telegram_alert_already_sent_today" in src, (
        "Skill 32 §3.4: agent must define _telegram_alert_already_sent_today. "
        "Without dedup, every cycle re-fires the same alert — operator "
        "receives 78 messages per day per stuck position."
    )
    assert "def _send_telegram_alert" in src, (
        "Skill 32 §3.4: agent must define _send_telegram_alert wrapper. "
        "The wrapper combines dedup + send + journal-write into one "
        "call so individual call sites can't accidentally skip dedup."
    )


def test_skill_32_dedup_is_date_keyed_for_self_clear() -> None:
    """Skill 32 §3.4: dedup must match on (ticker, alert_type, UTC date)
    so the state self-clears at midnight."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    helper_start = src.find("def _telegram_alert_already_sent_today")
    helper_end = src.find("\n    def ", helper_start + 1)
    helper_body = src[helper_start:helper_end if helper_end > 0 else None]
    assert 'alert_type' in helper_body, (
        "Skill 32 §3.4: dedup must distinguish alert_type so a PDT "
        "block and a cooldown engagement on the same ticker don't "
        "collide into one alert."
    )
    assert 'alert_date' in helper_body, (
        "Skill 32 §3.4: dedup must compare alert_date for UTC-midnight "
        "self-clearing. Without it, yesterday's alerts would suppress "
        "today's repeats."
    )
    assert 'today_iso' in helper_body, (
        "Skill 32 §3.4: dedup must compute today's UTC date for "
        "comparison. Hardcoding a fixed date would freeze state."
    )


def test_skill_32_notify_messages_carry_manual_action() -> None:
    """Skill 32 §1: every alert tells the operator WHAT TO DO. Pin the
    message bodies so a refactor doesn't drop the manual-action half."""
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    captured = []
    n._send = lambda text, *, channel="info": (captured.append(text), True)[1]

    n.notify_pdt_block("DIA", "Iron Condor", "strike_proximity",
                       "$497 near $502", 4554.0)
    assert "manually" in captured[-1].lower() or "manual" in captured[-1].lower(), (
        "Skill 32 §1: pdt_block alert must include a manual-action step."
    )

    n.notify_close_cooldown("DIA", "Iron Condor", 3, 3,
                            "2026-05-20T16:34:00+00:00", "DIA260612C00502000")
    assert "manually" in captured[-1].lower() or "manual" in captured[-1].lower(), (
        "Skill 32 §1: close_cooldown alert must include a manual-action step."
    )

    n.notify_open_failed_after_close("DIA", "Iron Condor", "timeout")
    assert "Alpaca" in captured[-1] or "manually" in captured[-1].lower(), (
        "Skill 32 §1: open_failed_after_close alert must mention "
        "Alpaca UI verification or manual re-open."
    )


def test_skill_32_lifecycle_alerts_exist() -> None:
    """Skill 32 §3.6: position open + close alerts are a separate
    method pair from the operator-actionable stuck-position alerts."""
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    for method in ("notify_position_opened", "notify_position_closed"):
        assert callable(getattr(n, method, None)), (
            f"Skill 32 §3.6: TelegramNotifier.{method} must exist. "
            f"Renaming silently breaks the agent's lifecycle wire-up."
        )


def test_skill_32_open_alert_body_carries_essentials() -> None:
    """Skill 32 §3.6: position_opened alert must include ticker,
    strategy, credit, max-loss, and a 'why' (regime/RSI/thesis)."""
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    captured = []
    n._send = lambda text, *, channel="info": (captured.append(text), True)[1]
    n.notify_position_opened(
        ticker="XLF", strategy="Bull Put Spread", regime="sideways",
        net_credit=0.47, max_loss=4.53, spread_width=5.0,
        expiration="2026-06-05", short_strikes="$50",
        thesis="Sideways + RSI 47 + IV 11%",
    )
    msg = captured[-1]
    for piece in ("XLF", "Bull Put Spread", "sideways", "$0.47",
                  "$4.53", "2026-06-05", "$50"):
        assert piece in msg, (
            f"Skill 32 §3.6: position_opened must include {piece!r}. "
            f"Body: {msg[:300]!r}"
        )


def test_skill_32_close_alert_body_distinguishes_profit_loss() -> None:
    """Skill 32 §3.6: profit and loss close alerts must visually differ
    (emoji + % surface) so the operator knows at a glance."""
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    captured = []
    n._send = lambda text, *, channel="info": (captured.append(text), True)[1]
    # Profit close
    n.notify_position_closed(
        ticker="XLF", strategy="Bull Put Spread",
        exit_signal="profit_target", exit_reason="50%",
        realized_pl=0.24, original_credit=0.47, max_loss=4.53,
    )
    profit_msg = captured[-1]
    # Loss close
    n.notify_position_closed(
        ticker="DIA", strategy="Iron Condor",
        exit_signal="strike_proximity", exit_reason="near short strike",
        realized_pl=-108.00, original_credit=2.05, max_loss=297.95,
    )
    loss_msg = captured[-1]
    assert profit_msg != loss_msg, (
        "Skill 32 §3.6: profit vs loss messages must differ visually. "
        "Identical bodies hide the win/loss outcome at a glance."
    )
    assert "+$0.24" in profit_msg or "+0.24" in profit_msg
    assert "$108" in loss_msg or "108" in loss_msg


def test_skill_32_two_channel_routing() -> None:
    """Skill 32 §3.1 (2026-05-21): error-class alerts route to channel
    'error', lifecycle-class alerts route to channel 'info'. Pin the
    routing so a refactor can't silently re-merge the two channels
    (defeating the purpose of having a separate error bot)."""
    from trading_agent.telegram_notifier import TelegramNotifier
    # Construct with both channels distinct so we can observe routing
    n = TelegramNotifier(token="info_t", chat_id="info_c",
                        error_token="err_t", error_chat_id="err_c")
    routed = []
    n._send = lambda text, *, channel="info": (
        routed.append(channel), True
    )[1]

    # ERROR-class
    n.notify_pdt_block("DIA", "Iron Condor", "strike_proximity",
                       "?", 4554.0)
    n.notify_close_cooldown("DIA", "Iron Condor", 3, 3, "?", "?")
    n.notify_open_failed_after_close("DIA", "Iron Condor", "?")
    # INFO-class
    n.notify_position_opened("XLF", "Bull Put", "sideways",
                             0.47, 4.53, 5.0, "2026-06-05", "$50", "?")
    n.notify_position_closed("XLF", "Bull Put", "profit_target",
                             "?", 0.24, 0.47, 4.53)
    n.notify_eod_summary(
        date_label="?", account_balance=0.0, starting_balance=None,
        opens_today=[], closes_today=[], realized_pl_today=0.0,
        unrealized_pl_today=0.0, cycles_today=0, errors_today=0,
        stuck_tickers=[],
    )

    expected = ["error", "error", "error", "info", "info", "info"]
    assert routed == expected, (
        f"Skill 32 §3.1: channel routing drifted. Expected "
        f"{expected}, got {routed}. The three error-class methods "
        f"(notify_pdt_block, notify_close_cooldown, "
        f"notify_open_failed_after_close) must route to channel="
        f"'error'; the three info-class methods (notify_position_"
        f"opened/closed, notify_eod_summary) must route to "
        f"channel='info'."
    )


def test_skill_32_error_channel_falls_back_to_info() -> None:
    """Skill 32 §3.1: when error env vars are unset, the error channel
    inherits the info channel's credentials. Single-bot deployments
    must continue working unchanged."""
    from trading_agent.telegram_notifier import TelegramNotifier
    # Only the info channel is configured
    n = TelegramNotifier(token="info_t", chat_id="info_c")
    assert n.token == "info_t" and n.chat_id == "info_c"
    assert n.error_token == "info_t", (
        "Skill 32 §3.1: error_token must fall back to info token when "
        "no error env var is set. Without fallback, single-bot users "
        "would silently lose error alerts."
    )
    assert n.error_chat_id == "info_c", (
        "Skill 32 §3.1: error_chat_id must fall back to info chat_id."
    )
    assert n.error_channel_distinct is False, (
        "Skill 32 §3.1: error_channel_distinct must be False when "
        "credentials match — informs the dashboard whether two bots "
        "are actually wired."
    )


def test_skill_32_is_active_covers_either_channel() -> None:
    """Skill 32 §3.1: is_active = True when AT LEAST ONE channel is
    configured. Otherwise an error-only deployment couldn't fire alerts."""
    from trading_agent.telegram_notifier import TelegramNotifier
    # Error-only (no info)
    n = TelegramNotifier(token="", chat_id="",
                        error_token="err_t", error_chat_id="err_c")
    assert n.is_active is True, (
        "Skill 32 §3.1: is_active must be True even when only the "
        "error channel is configured. Otherwise _send_telegram_alert "
        "would short-circuit before any alert fires."
    )
    # Fully unconfigured
    n2 = TelegramNotifier(token="", chat_id="",
                         error_token="", error_chat_id="")
    assert n2.is_active is False


def test_skill_32_eod_builder_uses_et_date_and_skips_dry_run() -> None:
    """Skill 32 §3.8 (2026-05-21 hotfix): _build_eod_summary must filter
    rows by ET trading-session date AND skip rows where
    fill_status='dry_run'.

    Pi observation: Thursday afternoon's EOD recap showed 23 phantom
    DIA closes summing to -$2,976 because:

      1. The outer ``today_iso`` filter used UTC date, pulling in 22
         dry-run pseudo-closes written at 20:14-22:49 ET Wed (= UTC
         next-day 00:14-02:49) into Thursday's recap.
      2. Even with the ET-date fix, the 1 row from Thursday morning
         that's correctly tagged would be joined by any historical
         mislabeled action="closed" + fill_status="dry_run" rows
         that happen to fall on Thursday's ET date.

    Pin the source-level guards so these regressions can't recur.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    builder_start = src.find("def _build_eod_summary")
    builder_end = src.find("\n    def ", builder_start + 1)
    body = src[builder_start:builder_end if builder_end > 0 else None]

    # ET-date filter must be present
    assert "today_et_iso" in body, (
        "Skill 32 §3.8: _build_eod_summary must compute today_et_iso "
        "(ET calendar date) and use it as the outer row filter. "
        "Pure-UTC filter mistakenly includes Wed-evening rows in "
        "Thursday's recap because UTC midnight is 4 hours after ET "
        "trading-session boundary."
    )
    assert "datetime.now(EASTERN).date().isoformat()" in body, (
        "Skill 32 §3.8: today_et_iso must come from "
        "datetime.now(EASTERN).date().isoformat()."
    )
    # Skip dry-run mislabeled rows from realized-P&L sum
    assert 'rs.get("fill_status") != "dry_run"' in body, (
        "Skill 32 §3.8: the closes-list builder must skip rows "
        "where fill_status='dry_run' (defense against pre-2026-05-21 "
        "mislabeled action='closed' rows). Without this guard, "
        "historical phantom rows still pollute the recap."
    )


def test_skill_32_eod_dedup_keyed_by_et_trading_date() -> None:
    """Skill 32 §3.8 (2026-05-21 hotfix): _maybe_send_eod_summary must
    embed the ET trading session date into the dedup ``alert_type``
    string so Wed-evening and Thu-afternoon recaps use distinct keys
    even when they share a UTC date.

    Pi observation: Wed's recap fired 23:41 ET = 03:41 UTC Thu →
    journal row's alert_date=2026-05-21. Thu's recap at 16:43 ET =
    20:43 UTC also computes today_iso=2026-05-21. UTC dedup
    falsely matched → Thu's recap silently suppressed. ET-keyed
    alert_type breaks the collision."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    helper_start = src.find("def _maybe_send_eod_summary")
    helper_end = src.find("\n    def ", helper_start + 1)
    body = src[helper_start:helper_end if helper_end > 0 else None]
    # The trading-session date string must be computed and embedded
    # in the alert_type passed to _send_telegram_alert.
    assert "trading_session_date" in body, (
        "Skill 32 §3.8: _maybe_send_eod_summary must compute "
        "trading_session_date (ET) and embed it in the dedup key. "
        "Without this, Wed-evening and Thu-afternoon recaps "
        "collide on the same UTC date."
    )
    assert 'f"eod_summary:{trading_session_date}"' in body, (
        "Skill 32 §3.8: alert_type must be formatted as "
        "f'eod_summary:{trading_session_date}' so the dedup helper "
        "(which matches by alert_type) sees distinct keys per "
        "ET trading session."
    )


def test_skill_32_eod_summary_method_exists() -> None:
    """Skill 32 §3.8: notify_eod_summary is the end-of-day recap entry
    point. Pin the name + signature shape (keyword-only after *) so a
    refactor can't accidentally change the contract."""
    from trading_agent.telegram_notifier import TelegramNotifier
    import inspect
    n = TelegramNotifier(token="t", chat_id="c")
    assert callable(getattr(n, "notify_eod_summary", None)), (
        "Skill 32 §3.8: TelegramNotifier.notify_eod_summary must exist."
    )
    sig = inspect.signature(n.notify_eod_summary)
    for required in ("date_label", "account_balance", "starting_balance",
                     "opens_today", "closes_today", "realized_pl_today",
                     "unrealized_pl_today", "cycles_today",
                     "errors_today", "stuck_tickers"):
        assert required in sig.parameters, (
            f"Skill 32 §3.8: notify_eod_summary must accept "
            f"{required!r}. Removing it silently breaks the agent's "
            f"_maybe_send_eod_summary call site."
        )


def test_skill_32_eod_body_aggregates_essentials() -> None:
    """Skill 32 §3.8: the recap body must include balance + P&L + opens
    + closes + the manual-action note for stuck positions."""
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    captured = []
    n._send = lambda text, *, channel="info": (captured.append(text), True)[1]
    n.notify_eod_summary(
        date_label="Wednesday 2026-05-20",
        account_balance=4550.05,
        starting_balance=4700.33,
        opens_today=[{"ticker": "DIA", "strategy": "Iron Condor", "credit": 2.05}],
        closes_today=[{"ticker": "SPY", "strategy": "Bear Call Spread",
                       "exit_signal": "profit_target", "realized_pl": 202.00}],
        realized_pl_today=202.00,
        unrealized_pl_today=-108.00,
        cycles_today=78,
        errors_today=0,
        stuck_tickers=[{"ticker": "DIA",
                        "reason": "PDT block — manual close required"}],
    )
    body = captured[-1]
    for piece in ("$4,550.05", "$4,700.33", "End-of-Day Summary",
                  "DIA Iron Condor", "SPY Bear Call Spread",
                  "+$202.00", "PDT block"):
        assert piece in body, (
            f"Skill 32 §3.8: EOD body must include {piece!r}. "
            f"Operators rely on the at-a-glance summary."
        )


def test_skill_32_agent_send_helper_exists() -> None:
    """Skill 32 §3.8: agent must define _maybe_send_eod_summary and
    _build_eod_summary helpers."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "def _maybe_send_eod_summary" in src, (
        "Skill 32 §3.8: agent must define _maybe_send_eod_summary."
    )
    assert "def _build_eod_summary" in src, (
        "Skill 32 §3.8: agent must define _build_eod_summary."
    )
    # Pin the after-hours hook
    assert "self._maybe_send_eod_summary()" in src, (
        "Skill 32 §3.8: after-hours shutdown path must call the recap. "
        "Without this hook the alert never fires."
    )


def test_skill_32_eod_gated_on_post_market_or_weekend() -> None:
    """Skill 32 §3.8: pre-market shutdowns (weekday before 16:00) must
    NOT fire the recap. Otherwise a Monday-morning startup would send
    Friday's data wrapped in a Monday date label."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    helper_start = src.find("def _maybe_send_eod_summary")
    helper_end = src.find("\n    def ", helper_start + 1)
    body = src[helper_start:helper_end if helper_end > 0 else None]
    # Pin the post-16:00 weekday gate (or weekend pass-through)
    assert "is_weekday" in body and "now_et.hour < 16" in body, (
        "Skill 32 §3.8: pre-market weekday gate must be present. "
        "Otherwise the recap fires before any trades happen today."
    )


def test_skill_32_send_telegram_alert_forwards_ticker() -> None:
    """Skill 32 §3.4 (2026-05-21 hotfix): _send_telegram_alert must
    forward the ``ticker`` arg into ``send_fn`` when the underlying
    notify_* signature accepts it.

    Pi-deploy regression: notify_position_closed("DIA", ...) was
    failing with "TypeError: missing 1 required positional argument:
    'ticker'" because _send_telegram_alert consumed ticker as its
    own named param but didn't propagate it. Every real broker
    close silently lost its alert via the exception handler.

    Pin the introspection-based pass-through so the next refactor
    can't drop ticker for the position-closed path AND
    notify_eod_summary (which doesn't accept ticker) keeps working.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    helper_start = src.find("def _send_telegram_alert")
    helper_end = src.find("\n    def ", helper_start + 1)
    body = src[helper_start:helper_end if helper_end > 0 else None]
    # The helper must inspect send_fn's signature and conditionally
    # add ticker. Plain ``send_fn(**payload)`` (without ticker
    # propagation) drops the arg silently.
    assert "inspect.signature(send_fn)" in body, (
        "Skill 32 §3.4: _send_telegram_alert must introspect "
        "send_fn's signature to decide whether to pass ticker. "
        "Pre-hotfix the helper silently dropped ticker for every "
        "notify_* method that required it as positional arg."
    )
    assert '"ticker" in sig_params' in body, (
        "Skill 32 §3.4: ticker propagation must be conditional on "
        "the signature accepting it — notify_eod_summary doesn't "
        "have ticker in its kwargs-only signature."
    )
    assert 'call_kwargs["ticker"] = ticker' in body, (
        "Skill 32 §3.4: the helper must inject ticker into the "
        "kwargs passed to send_fn. Without this, position_closed "
        "alerts raise TypeError and the exception handler swallows "
        "them — observed on pi 2026-05-21 09:34:19."
    )


def test_skill_32_lifecycle_alerts_deduped_per_event() -> None:
    """Skill 32 §3.6 hotfix: position_opened and position_closed alerts
    MUST go through the dedup helper. Without dedup, dry-run mode (or
    any other source of repeat journal writes) spams the operator —
    pi-deploy 2026-05-20 saw 3 identical DIA close alerts every 5 min.

    Dedup keys:
      position_closed: alert_type = "position_closed:<expiration>:<exit_signal>"
      position_opened: alert_type = "position_opened:<expiration>"

    These keys let legitimate same-day re-trades (different expirations)
    fire fresh alerts while suppressing repeats on the same event."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    # The agent must thread both alerts through _send_telegram_alert
    # (which carries the dedup helper). Direct calls to
    # self.telegram.notify_position_* would bypass dedup.
    closed_block_start = src.find("Position-closed alert (skill 32")
    closed_block = src[closed_block_start:closed_block_start + 2000]
    assert "_send_telegram_alert(" in closed_block, (
        "Skill 32 §3.6 hotfix: position-closed alert must route through "
        "_send_telegram_alert (which carries the dedup gate). Direct "
        "calls to self.telegram.notify_position_closed bypass dedup."
    )
    assert "position_closed:" in closed_block, (
        "Skill 32 §3.6 hotfix: position_closed dedup_alert_type must "
        "embed expiration + exit_signal so legitimate same-day re-"
        "trades aren't accidentally deduped together."
    )

    opened_block_start = src.find("Position-opened alert (skill 32")
    opened_block = src[opened_block_start:opened_block_start + 2000]
    assert "_send_telegram_alert(" in opened_block, (
        "Skill 32 §3.6 hotfix: position-opened alert must route through "
        "_send_telegram_alert. Direct calls bypass dedup."
    )
    assert "position_opened:" in opened_block, (
        "Skill 32 §3.6 hotfix: position_opened dedup_alert_type must "
        "embed expiration so a real same-day re-open (different "
        "expiry) still alerts."
    )


def test_skill_32_open_positions_table_has_rolls_today_column() -> None:
    """Skill 32 §3.7: positions_table must add a Rolls Today column
    when journal_df is provided. Pins the dashboard hook for skill 31."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "streamlit" /
           "components.py").read_text(encoding="utf-8")
    assert '"Rolls Today"' in src, (
        "Skill 32 §3.7: positions_table must populate a 'Rolls Today' "
        "column. Without it, defensive-roll evaluator firings are "
        "buried in the journal and require grep to surface."
    )
    assert "_roll_summary_for" in src, (
        "Skill 32 §3.7: the column helper must be named _roll_summary_for "
        "so the format ('18× ⛔ credit-neg') stays stable across refactors."
    )

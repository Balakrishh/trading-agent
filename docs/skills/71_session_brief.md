# Pre-Market Brief and Plan for Tomorrow

> **One-line summary:** two read-only Telegram messages each trading day — a 09:00 ET pre-market brief (market state, overnight moves, today's events, positions vs their exits, what entries are allowed) and a 16:20 ET plan for tomorrow (today's activity, tomorrow's events, positions near an exit, the closing state).
> **Source of truth:** [`trading_agent/session_brief.py:51-96`](../../trading_agent/session_brief.py), [`trading_agent/session_brief.py:254-265`](../../trading_agent/session_brief.py)
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `58_market_state_playbook.md` (state file), `47_schwab_data_api.md` (data server), `68_macro_events_vrp_credit_gamma.md` (macro calendar), `65_earnings_expiry_and_trend_filters.md` (earnings dates), `32_telegram_operator_alerts.md` (Telegram, end-of-day recap)
> **Consumed by:** the operator only — no decision reads it.

---

## 1. Theory & Objective

The agent's own messages are reactive (opens, closes, the end-of-day recap). Before the open the operator had no single view of what the day would bring: whether the market state allows new entries, which ETFs gapped overnight, whether a CPI print or an earnings report is due, and which positions are close to a target, stop or forced close. The brief gathers what the agent already knows — the market-state snapshot, the trade-plan positions, the macro calendar, the earnings cache, the journal — plus one new read, pre-market quotes from the local data server, into one message at 09:00 ET; the evening message answers "what should I watch tomorrow". Both are timers, not part of the cycle, so they cannot slow or break trading.

## 2. Mathematical Formula

```text
change_pct      = (the more recently traded of extended-hours last / regular last) / previous close − 1
mover           ⇔ |change_pct| ≥ 1.5 %           (indexes SPY / QQQ / IWM / DIA always shown)
debit target    = tp × debit $      tp = debit_profit_target_pct (bear: …_bear when set)
debit stop      = −sl × debit $     sl = debit_stop_loss_pct (bull: …_bull when set)
credit target   = profit_target_pct × credit $
credit stop     = −50 % × max loss
near an exit    ⇔ P&L within 10 points of the basis of the target or the stop
flags (judged on the brief's day d): last session before expiry (DTE-safety close),
                  ≤ 2 sessions to expiry, ≤ trail_max_hold_dte days (trail stops holding)
```

## 3. Reference Python Implementation

```python
# trading_agent/session_brief.py:51-96
def exit_view(pos: Dict[str, Any], preset, today: date) -> Dict[str, Any]:
    """P&L, target and stop of one open position (``list_positions`` row),
    in dollars, plus flags for the day ``today``."""
    from trading_agent.calendar_utils import is_last_trading_day_before, sessions_before
    strat, n = pos.get("strategy", ""), max(1, int(pos.get("contracts") or 1))
    pl = pos.get("mid_pl")
    credit = float(pos.get("credit") or 0.0)
    if credit < 0:                                   # debit structure: risk is the debit
        basis = -credit * 100 * n
        tp = (preset.debit_profit_target_pct_bear if strat == "Put Debit Spread"
              and preset.debit_profit_target_pct_bear else preset.debit_profit_target_pct)
        sl = (preset.debit_stop_loss_pct_bull if strat == "Call Debit Spread"
              and preset.debit_stop_loss_pct_bull else preset.debit_stop_loss_pct)
        target, stop = tp * basis, -sl * basis
    else:                                            # credit: target on credit, stop on max loss
        basis = credit * 100 * n
        target = preset.profit_target_pct * basis
        stop = -0.5 * float(pos.get("total_max_loss") or pos.get("max_loss") or 0.0)
    flags: List[str] = []
    exp = None
    try:
        exp = date.fromisoformat(str(pos.get("expiration", "")))
    except ValueError:
        pass
    sessions = sessions_before(today, exp) if exp else None
    if exp and is_last_trading_day_before(today, exp):
        flags.append("DTE-safety close this session")
    elif sessions is not None and sessions <= 2:
        flags.append(f"{sessions} session(s) to expiry")
    if pl is not None and basis > 0:
        if target > 0 and pl >= target - NEAR_EXIT_PTS * basis:
            flags.append("near profit target")
        if stop < 0 and pl <= stop + NEAR_EXIT_PTS * basis:
            flags.append("near stop")
    if exp and (exp - today).days <= getattr(preset, "trail_max_hold_dte", 7):
        flags.append(f"≤{preset.trail_max_hold_dte} DTE: trail no longer holds")
    return dict(ticker=pos.get("ticker"), strategy=strat, pl=pl, basis=basis, target=target,
                stop=stop, expiration=pos.get("expiration"), sessions=sessions, flags=flags,
                playbook=pos.get("playbook", ""))


def _position_line(v: Dict[str, Any]) -> str:
    pct = f" ({v['pl'] / v['basis']:+.0%})" if v["pl"] is not None and v["basis"] else ""
    line = (f"• <b>{_e(v['ticker'])}</b> {_e(v['strategy'])} exp {_e(v['expiration'])}: "
            f"{_money(v['pl'])}{pct} · target {_money(v['target'])} · stop {_money(v['stop'])}")
    if v["flags"]:
```

```python
# trading_agent/session_brief.py:254-265
def due(kind: str, now_et: datetime, sentinel_dir: Path = SENTINEL_DIR) -> Optional[str]:
    """None when the brief should go out now, else the reason it should not."""
    from trading_agent.calendar_utils import is_trading_day
    if not is_trading_day(now_et.date()):
        return f"{now_et.date()} is not a trading day"
    lo, hi = WINDOWS[kind]
    if not (lo <= now_et.time() <= hi):
        return f"outside the {kind} window {lo:%H:%M}–{hi:%H:%M} ET"
    if (sentinel_dir / f".session_brief_{kind}_{now_et.date()}").exists():
        return "already sent today"
    return None
```

`gather()` collects each input in its own try/except (a failed source drops only its section); `build_premarket` / `build_tomorrow` are pure HTML builders (free text escaped). Quotes: `SchwabMarketDataProvider.fetch_session_quotes` (`/quotes?fields=quote,extended`, uncached) behind the data server's `GET /session-quotes?tickers=…` — the brief never opens a Schwab session of its own. Sending: `TelegramNotifier.notify_session_brief` (info channel, capped below Telegram's 4,096 characters). Units: `deploy/systemd/trading-agent-brief-premarket.{service,timer}` (Mon–Fri 09:00 America/New_York), `trading-agent-brief-tomorrow.{service,timer}` (16:20). CLI: `python -m trading_agent.session_brief premarket|tomorrow [--dry-run] [--force]`.

## 4. Edge Cases / Guardrails

- **Holidays and late runs** — `Persistent=true` can start a missed timer after a reboot; `due()` skips non-trading days and anything outside 06:00–09:30 (pre-market) / 16:00–23:59 (tomorrow) ET.
- **Twice in a day** — a `.session_brief_<kind>_<date>` sentinel in `trade_journal/` (atomic temp + rename), written only after Telegram accepted the message; a failed send retries on the next manual run.
- **Pre-market quotes** — thin and indicative (the agent ignores anything before 09:30 as a mark); a ticker with no close or no trade shows nothing rather than 0 %. `/price` returned 0.0 for SPY at 07:20 ET on 2026-11-09, which is why the brief uses the close / extended fields instead.
- **Stale extended block** — Schwab's `extended` fields can still hold the previous evening's after-hours trade (2026-11-09 07:30 ET: SPY extended 738.27 vs last 740.07); the price with the newer `tradeTime` wins.
- **Stale market state** — the state comes from the last cycle (Friday's close on a Monday); the header says when.
- **Allowed list vs preset** — the state's allowed list can name a structure the preset has switched off (bear calls); it is shown "(off)". A vol-band bull put (skill 70) is reported blocked when the state does not allow "Bull Put Spread".
- **Tomorrow's flags** — judged on the next NYSE session (`next_sessions`), so Wednesday-before-Thanksgiving looks at Friday.
- **No Telegram configured** — logs a warning, sends nothing, writes no sentinel.

## 5. Cross-References

- `58_market_state_playbook.md` — the risk state and allowed strategies.
- `47_schwab_data_api.md` — the data server and the `/session-quotes` endpoint.
- `70_vol_band_bull_put.md` — its eligibility line.
- `30_profit_target_management.md` / `59_debit_spreads_calendars.md` — the exit rules the targets and stops mirror.

---

*Last verified against repo HEAD on 2026-11-09.*

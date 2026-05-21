# Telegram operator alerts + stuck-position dashboard banner

> **One-line summary:** Opt-in Telegram bot alerts that fire on operator-actionable events (PDT block, close cooldown engaged, defensive-roll open-failed) — paired with a red dashboard banner above Open Positions that surfaces the same stuck-position state. Both read from the same journal markers the close-loop suppression gates use, so what the operator sees equals what the agent is actually doing.
> **Source of truth:** [`trading_agent/telegram_notifier.py`](../../trading_agent/telegram_notifier.py), [`trading_agent/agent.py:_send_telegram_alert`](../../trading_agent/agent.py), [`trading_agent/streamlit/live_monitor.py:_render_stuck_position_banner`](../../trading_agent/streamlit/live_monitor.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `17_close_failure_and_cooldown.md` (the markers this skill reads), `19_journal_schema.md` (new `telegram_alert_sent` action string), `31_defensive_roll.md` (the `roll_open_failed` outcome that fires the FLAT alert).
> **Consumed by:** the operator's phone (paging path), the dashboard banner (visual path); both are downstream of the same journal markers.

---

## 1. Theory & Objective

The agent already detects three classes of operator-actionable events and writes them to the journal: (a) Alpaca's `40310100` PDT-block response, (b) 3-strike partial-fill cooldown engagement, (c) defensive-roll close-then-open where the open half failed (leaving the account flat). Until 2026-05-20 these events surfaced only as log lines + journal rows — an operator who wasn't watching the dashboard had no way to know a position was stuck.

This skill closes the operator-paging gap with two outputs sharing one trigger source:

1. **Telegram alert** — opt-in via env, dedupd to one message per ticker per day per alert type. Tells the operator: what happened, what to do, where to do it.
2. **Dashboard banner** — red, prominently placed above Open Positions, lists every stuck ticker with cooldown timer + manual-action steps.

Both read the same journal markers (`pdt_blocked_today`, `close_cooldown_until`, `defensive_roll_open_failed`) the agent's suppression gates use, so banner-content drift from gate-behavior is structurally impossible.

**Why a separate notifier module.** The agent has 11 `journal_kb.log_signal` call sites; only some are operator-actionable. Routing every write through Telegram would flood the operator. A dedicated `TelegramNotifier` class with named `notify_pdt_block`, `notify_close_cooldown`, `notify_open_failed_after_close` methods keeps the alert vocabulary discoverable in one file and reviewable as a single contract.

**Why opt-in not always-on.** A new install with no `TELEGRAM_BOT_TOKEN` set must keep working. `is_active` reads both env vars and returns False if either is missing; every `notify_*` then short-circuits silently. No surprises for existing users.

## 2. Mathematical Formula

N/A — control flow only. The "logic" is two predicates:

```text
is_active = bool(TELEGRAM_BOT_TOKEN) AND bool(TELEGRAM_CHAT_ID)

dedup_match(ticker, alert_type) =
    ∃ journal_row where:
        action == "telegram_alert_sent" AND
        ticker  == <given ticker> AND
        raw_signal.alert_type == <given type> AND
        raw_signal.alert_date == <today UTC>
```

The agent calls `_send_telegram_alert` which combines (a) `is_active`, (b) `not dedup_match(...)`, (c) `notifier.send(...)`, (d) journal-write on success.

## 3. Reference Python Implementation

### 3.1 Notifier construction (env-gated)

```python
# trading_agent/telegram_notifier.py
class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
```

### 3.2 The three operator-actionable alerts

```python
# trading_agent/telegram_notifier.py
def notify_pdt_block(self, ticker: str, strategy: str,
                     exit_signal: str, exit_reason: str,
                     account_balance: float) -> bool:
```

```python
# trading_agent/telegram_notifier.py
def notify_close_cooldown(self, ticker: str, strategy: str,
                          streak: int, threshold: int,
                          cooldown_until_iso: str,
                          failed_legs: str) -> bool:
```

```python
# trading_agent/telegram_notifier.py
def notify_open_failed_after_close(self, ticker: str,
                                   strategy: str,
                                   reason: str) -> bool:
```

### 3.3 Bounded, defensive send

```python
# trading_agent/telegram_notifier.py
resp = requests.post(
    url, json=payload, timeout=_TELEGRAM_TIMEOUT_SEC,
)
```

A 5-second timeout caps the worst case so a slow Telegram API can't tank a 5-minute cycle.

### 3.4 Dedup helper (journal-derived, date-keyed)

```python
# trading_agent/agent.py:_telegram_alert_already_sent_today
def _telegram_alert_already_sent_today(self, ticker: str,
                                        alert_type: str) -> bool:
```

Scans `signals_live.jsonl` for rows where `action="telegram_alert_sent"` AND `ticker` matches AND `raw_signal.alert_type` matches AND `raw_signal.alert_date == today UTC`. Returns True if found — caller short-circuits the send. State self-clears at UTC midnight because the date is in the match key.

### 3.5 Dashboard banner

```python
# trading_agent/streamlit/live_monitor.py
def _render_stuck_position_banner(journal_df: pd.DataFrame,
                                  held_tickers: Optional[set] = None) -> None:
```

### 3.6 Position-lifecycle alerts (added 2026-05-20)

```python
# trading_agent/telegram_notifier.py
def notify_position_opened(self, ticker: str, strategy: str,
                           regime: str, net_credit: float,
                           max_loss: float, spread_width: float,
                           expiration: str, short_strikes: str,
                           thesis: str) -> bool:
```

```python
# trading_agent/telegram_notifier.py
def notify_position_closed(self, ticker: str, strategy: str,
                           exit_signal: str, exit_reason: str,
                           realized_pl: float, original_credit: float,
                           max_loss: float) -> bool:
```

Both fire inline with the corresponding journal write — `notify_position_opened` after `_log_signal` records `action="submitted"`, `notify_position_closed` after `_journal_close_event` records `action="closed"`. Both route through `_send_telegram_alert` with event-scoped dedup keys (`position_opened:<exp>` and `position_closed:<exp>:<exit_signal>`) so a repeat journal write of the same event — e.g. a synthetic dry-run "close" that re-fires every cycle because the underlying position is still open — does NOT spam the operator. A legitimate same-day re-trade at a different expiration produces a different dedup key and alerts normally.

### 3.7 Defensive-roll activity in Open Positions table

```python
# trading_agent/streamlit/components.py:positions_table
row["Rolls Today"] = _roll_summary_for(s.get("underlying", ""))
```

Groups today's `defensive_roll_evaluated` rows per ticker by their `decision` field. Renders a compact label per held position:

- `—` — no rolls fired
- `18× ⛔ credit-neg` — all 18 declined for the same reason
- `5× ✅ / 3× ⛔` — some rolled, some declined

### 3.8 End-of-day recap (added 2026-05-20)

```python
# trading_agent/telegram_notifier.py
def notify_eod_summary(self, *,
                       date_label: str,
                       account_balance: float,
                       starting_balance: Optional[float],
                       opens_today: list,
                       closes_today: list,
                       realized_pl_today: float,
                       unrealized_pl_today: float,
                       cycles_today: int,
                       errors_today: int,
                       stuck_tickers: list) -> bool:
```

Fired once per trading day from the agent's after-hours shutdown path. Only sends when:

- the agent is post-16:00 ET on a weekday (or any time on weekend so Friday's recap goes out)
- the dedup helper confirms no `eod_summary` alert has been journalled today (sentinel ticker `__eod__`)
- today's journal contains at least one `submitted` or `closed` row (empty day → no alert)

```python
# trading_agent/agent.py:_maybe_send_eod_summary
def _maybe_send_eod_summary(self) -> None:
```

Builds the recap by walking the journal once and aggregating: opens, closes (with realized P&L), distinct-minute count as an activity proxy, error count, balance start vs. end, stuck-ticker list (PDT-blocked or cooldown-active).

Filters today's `close_failed` rows, picks the latest per ticker, separates into:
- **PDT-blocked group:** `pdt_blocked_today == True` AND `pdt_blocked_date == today UTC`
- **Cooldown group:** `close_cooldown_until` parses as a future ISO timestamp

Renders one red-bordered HTML block listing every stuck ticker with time, strategy, exit-signal, and the recommended manual action. Returns early (renders nothing) when both groups are empty — clean state shows no banner.

## 4. Edge Cases / Guardrails

- **Notifier silently no-op when env unset.** `is_active` returns False, every `notify_*` returns False, `_send_telegram_alert` short-circuits before the network call. Existing installs that don't set `TELEGRAM_BOT_TOKEN` see zero behavior change.
- **5-second timeout protects the cycle.** A slow Telegram API can't make a 5-min cycle miss its tick. Hard cap; no retries.
- **Bare `Exception` catch in `_send`.** Network, JSON, DNS, timeout, schema — anything Telegram-side becomes a single WARNING log line and a `False` return. The agent's primary job (execute trades) must never fail because the paging system has a problem.
- **Dedup fail-open.** If the journal can't be read (file corruption, permission, etc.), `_telegram_alert_already_sent_today` returns False — preferring one extra alert over zero. Operator-side noise is recoverable; a missed stuck-position alert is not.
- **Journal write failure after successful send.** If the alert reaches Telegram but the dedup-journal write fails, the operator may receive a duplicate alert next cycle. Logged at WARNING; intentional fallback because re-attempting an already-delivered alert is harmless.
- **The `roll_open_failed` alert is always-on (no dedup).** Unlike the other two, a "close filled but open crashed → position flat" event is a rare emergency. The operator MUST see it every time it happens, even if hypothetically the same position rolled-and-failed twice in one day.
- **HTML formatting via `parse_mode=HTML`.** Telegram's MarkdownV2 has too many escape rules to be safe across arbitrary error strings. HTML uses `<b>`, `<code>`, `<pre>` — easier to construct, escape-free for the strings we send.
- **Dashboard banner is journal-derived, not agent-process-derived.** A dashboard restart, a stopped agent, or a fresh Streamlit reload still sees the banner if the journal carries today's markers. Single source of truth.
- **Banner suppressed on clean state.** When no `pdt_blocked_today` markers exist and no `close_cooldown_until` is in the future, `_render_stuck_position_banner` returns before emitting any HTML. No empty placeholder, no scrollbar churn.

## 5. Cross-References

- `17_close_failure_and_cooldown.md` — `pdt_blocked_today` marker (§4) + `close_cooldown_until` (§1) are written there; this skill consumes them.
- `19_journal_schema.md` — registers the new `telegram_alert_sent` action string in the canonical action vocabulary.
- `31_defensive_roll.md` — `defensive_roll_open_failed` is the trigger for the FLAT alert.

---

*Last verified against repo HEAD on 2026-05-20.*

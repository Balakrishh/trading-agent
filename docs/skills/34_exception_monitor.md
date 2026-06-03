# Exception monitor — operator visibility for silenced failures

> **One-line summary:** Every `except Exception` block in the agent's hot path calls `self._exception_monitor.record(...)`. The first occurrence per `(exc_class, source)` per UTC day pages the operator via the Telegram error channel; every subsequent occurrence is journalled but doesn't re-page. The EOD recap surfaces the count + top sources so an operator catching up at end of day sees what failed quietly during the session.
> **Source of truth:** [`trading_agent/exception_monitor.py:ExceptionMonitor`](../../trading_agent/exception_monitor.py), [`trading_agent/agent.py`](../../trading_agent/agent.py) call sites, [`trading_agent/telegram_notifier.py:notify_silenced_exception`](../../trading_agent/telegram_notifier.py), [`trading_agent/journal_reader.py:silenced_exceptions_today`](../../trading_agent/journal_reader.py), [`trading_agent/market_data_schwab.py`](../../trading_agent/market_data_schwab.py), [`trading_agent/executor.py`](../../trading_agent/executor.py), [`trading_agent/strategy.py`](../../trading_agent/strategy.py) global-registry call sites.
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `19_journal_schema.md` (new `silenced_exception` / `silenced_exception_paged` action vocab), `32_telegram_operator_alerts.md` (error channel, dedup pattern).
> **Consumed by:** the operator's phone (immediate paging), the EOD Telegram recap (catch-up), the dashboard (future surface).

---

## 1. Theory & Objective

Defensive `except Exception` blocks are everywhere in the agent — a flaky network shouldn't crash a cycle, a malformed journal row shouldn't blank the dashboard, a Telegram API hiccup shouldn't stop trading. Each one logs a WARNING and continues. That's correct for robustness — and disastrous for visibility.

Two recent multi-day-undetected bugs hid behind exactly this pattern:

1. **48-hour Telegram-ticker `TypeError`.** Every `notify_position_closed(...)` call raised because `ticker` was being dropped in the wrapper. The `except Exception` in `_send_telegram_alert` logged one WARNING per cycle. Nobody read the log; the operator only noticed via the absence of expected alerts.
2. **Schwab token expiration.** A multi-hour silent failure surfaced only when chain fetches returned None — eventually inferred from rejected-with-no-chain rows.

`ExceptionMonitor` turns silenced exceptions into a paging system:

- **Record** every occurrence as a `silenced_exception` journal row.
- **Dedup** per `(exc_class, source)` per UTC day — operator gets ONE Telegram message per distinct failure mode, not 78.
- **Surface** the count + top sources in the EOD recap so end-of-day catch-up sees the quiet failures.

The first occurrence pages immediately (so the operator catches a bug within ~5 minutes of it happening). Subsequent occurrences are journalled but silent. The next UTC day, the dedup resets — if the bug isn't fixed yet, the operator gets re-paged.

## 2. Mathematical Formula

Control flow only — no math.

```text
For each ExceptionMonitor.record(source=S, exc=E, …) call:

  1. Write a `silenced_exception` row to the journal.
       fields: source=S, exc_class=type(E).__name__, message=str(E)

  2. If telegram is active AND not _already_paged_today(S, exc_class):
       fire notify_silenced_exception(...)
       write `silenced_exception_paged` marker row to journal

  3. Return — never raise. A failure inside the monitor logs a
     single WARNING and continues. The original except handler
     must be allowed to complete its recovery.
```

The fast-path in-memory dedup (a `set[tuple]` per process) avoids a journal scan for repeat hits within one process's lifetime; cross-process dedup falls back to a journal scan.

## 3. Reference Python Implementation

### 3.1 ExceptionMonitor.record

```python
# trading_agent/exception_monitor.py
def record(self, *,
           source: str,
           exc: Optional[BaseException] = None,
           message: Optional[str] = None,
           ticker: str = "") -> None:
```

### 3.2 Telegram alert (error channel)

```python
# trading_agent/telegram_notifier.py
def notify_silenced_exception(self, *, source: str, exc_class: str,
                              message: str, ticker: str = "") -> bool:
```

Routes to channel="error" — same priority as PDT block / cooldown engagement. Operator gets one page per distinct failure mode per day.

### 3.3 Agent integration — instrument key except blocks

```python
# trading_agent/agent.py — _send_telegram_alert
ok = send_fn(**call_kwargs)
except Exception as exc:                                # noqa: BLE001
    self._exception_monitor.record(
        source=f"agent._send_telegram_alert/{alert_type}",
        exc=exc, ticker=ticker,
    )
    return
```

The instrumented call sites (more can be added incrementally):

  * `agent._send_telegram_alert/<alert_type>` — Telegram send failures
  * `agent._build_eod_summary` — journal read failures inside EOD recap
  * `agent._maybe_defensive_roll/plan` — planner crashes during roll
  * `agent._maybe_defensive_roll/risk_check` — risk-manager crashes during roll
  * `agent._run_cycle_impl` — top-level cycle crash (no trades this minute)
  * `agent._process_ticker` — per-ticker unhandled error (silent dropout)
  * `agent._tickers_with_open_orders` — order-tracker fetch failure (dedup risk)
  * `agent._cancel_stale_orders` — order-tracker fetch failure (stale-order accrual)
  * `executor.submit_order` — order POST exhausted retries / 4xx rejection
  * `executor.close_position` — broker close API failure (position remains open)
  * `executor.defensive_roll_open` — CLOSE FILLED but OPEN CRASHED (FLAT)
  * `strategy.scan/<side>` — adaptive scanner crashed for one ticker
  * `market_data_schwab.get` — Schwab token expiry / auth failure

### 3.4 Journal vocabulary

Two new action strings (also documented in skill 19 §2):

```python
# trading_agent/journal_kb.py
_DEDUP_BYPASS_ACTIONS = frozenset({
    ...,
    "silenced_exception",          # every occurrence recorded
    "silenced_exception_paged",    # one per (source, exc_class, UTC day)
})
```

Both bypass the per-ticker dedup so consumers see accurate counts.

### 3.5 EOD recap surfaces the breakdown

```python
# trading_agent/journal_reader.py
def silenced_exceptions_today(self) -> List["SilencedException"]:
```

Returns groups sorted by count desc. `_build_eod_summary` consumes the list; `notify_eod_summary` renders the top 5 groups with their counts + last message + source.

### 3.6 Global monitor registry — for pre-agent-construction modules

`market_data_schwab.SchwabMarketDataProvider`, the OAuth helper, the chain
scanner, and other low-level modules are built by `agent_factory.py` BEFORE
`TradingAgent.__init__` finishes, so they can't accept an
`ExceptionMonitor` via the constructor. They still need to page the
operator when a Schwab token expires or a chain fetch fails silently.

The fix: a module-level registry. `TradingAgent.__init__` calls
`set_global_monitor(self._exception_monitor)` right after constructing
its monitor; any pre-existing module fetches it lazily inside the
except block:

```python
# trading_agent/market_data_schwab.py
except RuntimeError as exc:
    if not self._auth_warned:
        logger.warning("Schwab GET %s skipped — no auth (%s)", path, exc)
        self._auth_warned = True
        try:
            from trading_agent.exception_monitor import get_global_monitor
            mon = get_global_monitor()
            if mon is not None:
                mon.record(
                    source="market_data_schwab.get",
                    exc=exc,
                    message=(
                        f"Schwab auth failed on {path}: {exc}. "
                        f"Run `python -m trading_agent.schwab_oauth login` "
                        f"to re-auth."
                    ),
                )
        except Exception:                              # noqa: BLE001
            pass   # monitor is best-effort
```

Callers MUST tolerate `get_global_monitor()` returning `None` — CLI
scripts and tests don't run `TradingAgent.__init__`, so no monitor is
registered. The convention is `if mon is not None: mon.record(...)`.

## 4. Edge Cases / Guardrails

- **Monitor never propagates.** A failure inside `record` (journal-write fail, Telegram timeout, etc.) catches a bare `Exception`, logs once, returns. The caller's except handler must complete its recovery; we don't add fragility.
- **Dedup is `(source, exc_class, UTC date)`.** Different sources of the same exception class (e.g. `ValueError` from Schwab AND `ValueError` from the chain scanner) get separate alerts. Different exception classes from the same source also get separate alerts.
- **In-memory + journal dual dedup.** Fast path: a `set` checked first. Slow path: journal scan for `silenced_exception_paged` rows. The journal path matters because the agent's cycle subprocess re-execs every cycle; the in-memory cache resets but the journal persists.
- **Fail-open on dedup-scan failure.** If the journal can't be read, we ASSUME not-paged (better to send a duplicate alert than miss one). Same principle as the existing `_telegram_alert_already_sent_today` helper.
- **Telegram inactive → still journal.** When `telegram.is_active` is False, the journal is still written; the page is silently skipped. Operator can later inspect via the dashboard or `grep action=silenced_exception signals_live.jsonl`.
- **Not for every WARNING.** Only call `record` from places where the silence actually matters operationally. Generic INFO/DEBUG logs from chain scanners aren't candidates.
- **Idempotent across process restarts.** A cron-style cycle that exits and re-launches doesn't re-page already-paged groups today because the dedup marker is in the journal.
- **No back-pressure.** `record` is fire-and-forget. A high-rate failure (e.g. retry loop) still pages once but journals every occurrence — the counter remains accurate for the EOD recap.

## 5. Cross-References

- `19_journal_schema.md` — registers the two new action strings.
- `32_telegram_operator_alerts.md` §3.6 — companion error-channel alerts share the same dedup pattern.
- `00_sdlc_and_conventions.md` — adding a new `except Exception` block in production code should be followed by an `ExceptionMonitor.record` call.

---

*Last verified against repo HEAD on 2026-06-03.*

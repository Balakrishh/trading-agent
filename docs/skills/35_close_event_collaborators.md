# Close-event collaborators — extracted from `_journal_close_event`

> **One-line summary:** The ~300-line `agent._journal_close_event` method was split into four constructor-injected collaborators (`PartialFillCooldown`, `PdtBlockDetector`, `CloseAlertNotifier`, `CloseJournalWriter`) so the close path is testable without `MagicMock(spec=TradingAgent)` and each concern (journal write, cooldown bookkeeping, PDT detection, Telegram dispatch) can be reasoned about in isolation.
> **Source of truth:** [`trading_agent/close_event_collaborators.py`](../../trading_agent/close_event_collaborators.py), [`trading_agent/agent.py`](../../trading_agent/agent.py) construction + delegation.
> **Phase:** 2  •  **Group:** architecture
> **Depends on:** `19_journal_schema.md` (action vocabulary), `17_pdt_reactive_block.md` (PDT detection semantics), `32_telegram_operator_alerts.md` (close-event alert dedup).
> **Consumed by:** the close loop in `_stage_monitor`, the dashboard cooldown banner, the close-loop PDT suppression check.

---

## 1. Theory & Objective

Pre-2026-05-22 `_journal_close_event` was a ~300-line method that did five distinct things:

1. Construct the close-event journal row (action + notes + payload).
2. Read the journal to count partial-fill streaks; embed cooldown deadline + reason when threshold engaged.
3. Scan leg-error strings for Alpaca's PDT response code; embed PDT markers.
4. Dispatch up to three Telegram alerts (cooldown engaged, PDT block, position closed).
5. Write the row to the journal.

That coupling caused two operational pain points:

- **Test fixtures needed `MagicMock(spec=TradingAgent)`.** Every collaborator was a private method, so any test exercising the close path had to mock around the entire agent. The fixtures in `tests/conformance/test_skill_30_profit_target_management.py` carried `agent.telegram = MagicMock()` workarounds because the spec-mock didn't expose instance-level attrs.
- **Cross-concern bugs hid in the method body.** The `-$2,976 phantom-loss recap` (commit 9cc5636), the 22-row dry-run pollution, the UTC-vs-ET dedup collision — all landed in or near this method because changing one concern silently affected another.

The extraction:

- **`PartialFillCooldown`** owns streak derivation + cooldown deadline math.
- **`PdtBlockDetector`** owns leg-error scanning + the cross-day block-set reader.
- **`CloseAlertNotifier`** owns the three Telegram dispatch sites (still routed through `_send_telegram_alert` for dedup, so behavior is identical).
- **`CloseJournalWriter`** orchestrates the four above and writes the row.

`agent._journal_close_event` shrinks to a single delegation line. The five other helper methods (`_close_failed_streak_within_window`, `_close_cooldown_minutes_remaining`, `_record_partial_close`, `_clear_close_cooldown`, `_pdt_blocked_today_tickers`) become one-line shims pointing at the collaborators.

## 2. Mathematical Formula

Control flow only — no math.

```text
agent._journal_close_event(spread, ctx, leg_results, fill_status, dry_run):
    self._close_writer.write(spread, ctx, leg_results=…, fill_status=…, dry_run=…)

CloseJournalWriter.write:
    1. Action + notes branch on fill_status:
         complete  → action="closed",         exec_status="closed_<exit_signal>"
         dry_run   → action="dry_run_close",  exec_status="dry_run_close_<exit_signal>"
         partial   → action="close_failed",   exec_status="close_failed_<exit_signal>"
    2. Construct payload (ctx + leg_results + fill_status + mode tag).
    3. IF action == "close_failed":
         streak, fields = cooldown.build_payload_fields(ticker)
         payload.update(fields)
         IF streak >= threshold:
             alerts.notify_cooldown_engaged(…)
         IF pdt_detector.detect(leg_results):
             payload.update(pdt_detector.build_markers())
             alerts.notify_pdt_block(…)
    4. journal_kb.log_signal(…)
    5. IF action == "closed":
         alerts.notify_position_closed(…)
    6. Any exception inside → log + return (never propagate).
```

## 3. Reference Python Implementation

### 3.1 PartialFillCooldown

```python
# trading_agent/close_event_collaborators.py
class PartialFillCooldown:
    def streak_within_window(self, ticker: str) -> Tuple[int, Optional[datetime]]:
        ...
    def minutes_remaining(self, ticker: str) -> int:
        ...
    def build_payload_fields(self, ticker: str) -> Tuple[int, Dict[str, Any]]:
        ...
```

### 3.2 PdtBlockDetector

```python
# trading_agent/close_event_collaborators.py
class PdtBlockDetector:
    PDT_SIGNALS = ("40310100", "pattern day trading")
    @classmethod
    def detect(cls, leg_results: List[Dict]) -> bool:
        ...
    def blocked_tickers_today(self) -> Set[str]:
        ...
```

### 3.3 Agent integration

```python
# trading_agent/agent.py — TradingAgent.__init__
self._cooldown = PartialFillCooldown(
    journal_kb=self.journal_kb,
    threshold=PARTIAL_CLOSE_COOLDOWN_THRESHOLD,
    window_min=CLOSE_COOLDOWN_MINUTES,
)
self._pdt_detector = PdtBlockDetector(journal_kb=self.journal_kb)
self._close_alerts = CloseAlertNotifier(
    send_alert=self._send_telegram_alert,
    telegram=self.telegram,
)
self._close_writer = CloseJournalWriter(
    journal_kb=self.journal_kb,
    cooldown=self._cooldown,
    pdt_detector=self._pdt_detector,
    alerts=self._close_alerts,
    price_lookup=self._cached_price,
)
```

### 3.4 CloseJournalWriter

```python
# trading_agent/close_event_collaborators.py
class CloseJournalWriter:
    def write(self, spread, ctx: Dict, *, leg_results: List[Dict],
              fill_status: str, dry_run: bool) -> None:
        ...
```

## 4. Edge Cases / Guardrails

- **Writer never propagates.** A journal-write failure or alert-dispatch crash inside `write()` catches a bare `Exception`, logs a single WARNING, returns. The close itself already happened at the broker; we don't pile a journal regression on top of a state-divergence event.
- **Cooldown is journal-derived only.** Pre-2026-05-13 the streak counter lived in an in-memory dict on TradingAgent. Cycle-restart blew it away in production (XLF/GLD post-mortem) and the cooldown never engaged. The journal is now the single source of truth — `streak_within_window` scans the file on every call.
- **PDT detection matches both signal forms.** Alpaca returns code `40310100` in structured responses but a `"pattern day trading"` phrase in some error bodies. `PDT_SIGNALS` matches either so detection doesn't silently fail when Alpaca changes the response shape.
- **PDT markers self-clear at UTC midnight.** `blocked_tickers_today` filters by `pdt_blocked_date == today_iso`. Yesterday's markers stop matching when the UTC date rolls over, so the suppression auto-lifts without operator intervention.
- **`CloseAlertNotifier` wraps `_send_telegram_alert`, not the raw notifier.** Dedup behavior is identical to pre-refactor — the alert dedup gate (per-ticker, per-alert-type, per-UTC-day) is unchanged. Only the call sites moved.
- **`dry_run_close` is a distinct action from `closed`.** The writer never tags a synthetic close as `action="closed"` — this is the bug from commit 9cc5636 (-$2,976 phantom-loss recap) hard-pinned by test_skill_35_writer_writes_dry_run_close_distinct_from_closed.
- **Public method names retained on TradingAgent.** `_close_failed_streak_within_window`, `_close_cooldown_minutes_remaining`, `_record_partial_close`, `_clear_close_cooldown`, `_pdt_blocked_today_tickers` are kept as one-line shims. External call sites in `_stage_monitor` and `streamlit/components.py` don't have to change.

## 5. Cross-References

- `17_pdt_reactive_block.md` — semantic origin of the `pdt_blocked_today` marker; this skill is the implementation host.
- `19_journal_schema.md` — defines the `closed` / `dry_run_close` / `close_failed` action vocabulary the writer emits.
- `32_telegram_operator_alerts.md` — the dedup pattern shared by `CloseAlertNotifier`.
- `00_sdlc_and_conventions.md` — adding a new close-event side effect should be a new method on a collaborator, NOT a new branch in `CloseJournalWriter.write`.

---

*Last verified against repo HEAD on 2026-05-23.*

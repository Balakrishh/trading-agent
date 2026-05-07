# Signal-Journal Schema (Action Enum + Dedup Bypass)

> **One-line summary:** Every cycle row in `trade_journal/signals_<mode>.jsonl` carries an `action` field from a closed enumeration; some action values pass through a per-ticker dedup gate (chain-scanner rejection spam) while material events bypass it. This skill is the canonical reference for what each value means and which fields a downstream UI / replay tool should expect to find inside `raw_signal`.
> **Source of truth:** [`trading_agent/journal_kb.py`](../../trading_agent/journal_kb.py), [`trading_agent/agent.py:_log_signal`](../../trading_agent/agent.py), [`trading_agent/agent.py:_journal_close_event`](../../trading_agent/agent.py)
> **Phase:** 2  •  **Group:** architecture
> **Depends on:** `00_sdlc_and_conventions.md` (run_mode split — live vs backtest)
> **Consumed by:** `streamlit/live_monitor.py` (Recent Journal Entries panel, guardrail grid, Closed Today, Close Failures Today, staleness beacon, Daily P&L tile), the LLM Extension RAG over `signals_live.jsonl`.

---

## 1. Theory & Objective

The journal is the only durable source of truth that survives Streamlit restarts, agent process crashes, and watchdog reloads. Every cycle writes at least one row per ticker — even when nothing tradeable happened, the journal records *why*. This makes the journal the substrate for:

- **The dashboard's grid + tiles.** Closed Today, Close Failures Today, the per-ticker × per-guardrail Strategy-Profile grid, the Daily P&L metric, and the staleness beacon all derive from `signals_live.jsonl`.
- **The LLM RAG corpus.** The Tier-3 sentiment verifier and the LLM analyst both retrieve historical journal rows by ticker as evidence.
- **Forensic replay.** Every `submitted` row carries enough context (plan, regime, raw_signal payload) to reconstruct a trade decision.
- **Post-mortem.** When something looks wrong, the journal is read first, the agent log is read second, the broker UI third.

The `action` field is the verbatim categorisation. Down-stream consumers rely on `action` being from a closed set — adding a new value without updating consumers means the dashboard silently drops rows.

## 2. Action enumeration

Stable string keys. Grouped by lifecycle stage.

| Action | Meaning | Bypass dedup? | Where written |
|---|---|---|---|
| `submitted` | Order POST succeeded at Alpaca (live or paper) | ✅ | `executor._submit_order_with_idempotency` success branch |
| `dry_run` | Order skipped because `dry_run=True`; plan is otherwise valid | ✅ | `executor.execute()` dry-run branch |
| `rejected` | Risk manager OR live-credit recheck OR sizing failed | ❌ | `agent._log_signal` for `RiskVerdict.approved=False` and post-approval failures |
| `error` | Cycle/orchestration failure — broker connectivity, plan corruption | ✅ | `agent` exception handlers; `journal_kb.log_error` |
| `warning` | Subsystem warning operator should see (retry exhausted, OAuth refresh failed, retries on position fetch exhausted) | ✅ | `journal_kb.log_warning` (added 2026-05-06) |
| `closed` | Position closed at the broker; **all** legs accepted | ✅ | `agent._journal_close_event(fill_status="complete")` |
| `dry_run_close` | Dry-run-mode close pseudo-event | ✅ | same, `dry_run=True` branch |
| `close_failed` | Position close was a partial fill; one or more legs rejected by Alpaca (added 2026-05-06) | ✅ | `agent._journal_close_event(fill_status="partial")` |
| `skipped_existing` | Cycle skipped because the ticker already has an open position or pending order (dedup gate) | ❌ | `agent._process_ticker` short-circuit |
| `skipped_bias` | Cycle skipped because the active preset's `directional_bias` filters out the classified regime | ❌ | `agent._process_ticker` after Phase II |
| `skipped_rsi_gate` | RSI gate filtered the cycle | ❌ | `agent._process_ticker` after Phase II |
| `skipped_defense_first` | Stage-1 closes had higher priority; Stage-2 short-circuited | ❌ | `agent.run_cycle` defense-first guard |
| `skipped_by_llm` | LLM Phase V vetoed an otherwise-approved plan | ❌ | `agent._process_ticker` Phase V branch |
| `skipped_liquidation_mode` | Account drawdown breaker is open; new entries suppressed | ❌ | `agent._stage_open` daily-state guard |
| `skipped` | Catch-all (legacy / pre-classify) | ❌ | rare, mostly covered by more specific skipped_* |
| `cycle_timeout` | 270s hard guard fired | ✅ | `agent` timeout handler |
| `daily_drawdown_circuit_breaker` | Drawdown breaker tripped this cycle | ✅ | `agent.run_cycle` |

`_DEDUP_BYPASS_ACTIONS` (defined in `journal_kb.py:95-101`) is the authoritative set of "always-write" actions. Any action NOT in this set is subject to the per-ticker signature dedup — successive identical rejection rows on the same ticker get suppressed except for periodic heartbeats (every `JOURNAL_DEDUP_HEARTBEAT_EVERY` cycles, default 12 ≈ 1 hour).

## 3. Reference Python Implementation

### `trading_agent/journal_kb.py:95-101` — dedup bypass enumeration

```python
# Actions that ALWAYS write a journal row regardless of dedup state.
# Submissions, closes, close-failures, and dry-run sentinels are
# material events; an operator must see each one at the exact moment
# it happened.  In particular, ``close_failed`` (added 2026-05-06)
# bypasses dedup so successive partial-fill attempts on a zombie
# position remain individually visible — that's the data the operator
# needs to recognise the zombie state and intervene.
_DEDUP_BYPASS_ACTIONS = frozenset({
    "submitted", "dry_run", "closed", "close_failed", "dry_run_close",
    "error",
    "warning",  # connectivity / retry-exhausted / OAuth failures
})
```

### `trading_agent/journal_kb.py:155-213` — `log_signal` schema

Every row is a single-line JSON object:

```jsonc
{
  "timestamp":   "2026-05-06T18:42:31.207024+00:00",  // ISO UTC
  "ticker":      "SPY",
  "action":      "submitted",                          // see enum above
  "price":       729.40,
  "exec_status": "submitted",                          // refined version of action
  "notes":       "Bull Put 21d width=$5.0 credit=$0.42",
  "raw_signal":  { /* full structured payload — varies by action */ },
  "dedup":       { "streak": 7, "kind": "heartbeat" }  // only on heartbeat rows
}
```

`raw_signal` is freeform per action. Common fields the dashboard expects:

| Field | Type | Present on actions |
|---|---|---|
| `regime` | str (lowercase) | classified rows (submitted, rejected, skipped_*) |
| `mode` | "live" \| "dry_run" | submitted, closed, close_failed |
| `checks_passed` / `checks_failed` | List[str] | submitted, rejected (risk-manager output) |
| `reason` | str | rejected, skipped_* |
| `account_balance` | float | every classified row |
| `scan_results` | dict | adaptive-mode plan() rows — see skill 14 |
| `strategy` | str | submitted, closed, close_failed |
| `exit_signal` / `exit_reason` | str | closed, close_failed |
| `net_unrealized_pl` | float | closed (= realized P&L) |
| `fill_status` | "complete" \| "partial" \| "dry_run" | closed, close_failed |
| `leg_close_results` | List[{symbol, status}] | closed, close_failed |
| `partial_close_streak` | int | close_failed |
| `partial_close_threshold` | int | close_failed |
| `close_cooldown_until` | ISO timestamp | close_failed (only when threshold crossed) |
| `close_cooldown_reason` | str | close_failed (paired with cooldown_until) |
| `client_order_id` | str | submitted (executor result), warning (executor source) |
| `retry_attempts` | int | submitted, warning |
| `source` | str | warning (e.g. `"executor"`, `"position_monitor"`, `"schwab_oauth"`) |
| `message` | str | warning |
| `context` | dict | warning |

### `trading_agent/journal_kb.py:355-432` — `log_warning` helper

```python
def log_warning(self, source, message, ticker=None, context=None):
    """Emit an action="warning" row. source tag groups by subsystem
    (executor, schwab_oauth, position_monitor); context carries
    structured triage info (HTTP status, attempt count, etc).
    Bypasses dedup so successive cycles all surface the issue."""
    raw = {"source": source,
           "message": message[:500],
           "context": context or {}}
    self.log_signal(
        ticker=ticker or "",
        action="warning",
        price=0.0,
        raw_signal=raw,
        exec_status=f"warning_{source}",
        notes=f"[{source}] {message[:180]}")
```

### Mode split — `signals_live.jsonl` vs `signals_backtest.jsonl`

Two files, identical schema, separated by `run_mode`:

```python
self.jsonl_path = os.path.join(journal_dir, f"signals_{run_mode}.jsonl")
```

Set via `JournalKB(journal_dir, run_mode="live")` or `"backtest"`. Live cycles always go to live; the backtester always goes to backtest. Dashboard reads both but defaults to live.

## 4. Edge Cases / Guardrails

- **Adding a new `action` value is a coordinated change.** New values must be added to the `_DEDUP_BYPASS_ACTIONS` set if they're "material" (operator must see every occurrence) AND surfaced in `live_monitor.py` somewhere (a panel, a column, the staleness check). The dashboard's verdict-cell renderer at `live_monitor.py:1500-1530` falls back to a checks-based heuristic for unknown actions but produces a vague "rejected" label.

- **`raw_signal` field additions are append-only.** A consumer reading `rs.get("partial_close_streak")` must tolerate the field being absent (legacy rows pre-2026-05-06). Removing or renaming a field requires updating every consumer in lockstep.

- **Notes are capped at 200 chars.** Raised from 120 on 2026-05-06 — chain-scanner rejection reasons regularly run 150-180 chars and were getting truncated mid-paren in the dashboard. The 200 cap fires deterministically (test in `test_journal_kb.py::test_notes_truncated_to_200_chars`).

- **`raw_signal` has NO size cap.** Adaptive scanner `scan_results` payloads can be ~2KB. The `notes` cell is the human-readable summary; `raw_signal` is the structured payload for replay / RAG.

- **Heartbeat dedup rows carry `dedup={"streak": N, "kind": "heartbeat"}`.** Downstream analytics can identify a heartbeat (suppression compaction) vs a fresh row by inspecting this field. UI panels that count "events per cycle" should filter out heartbeats to avoid double-counting.

- **`close_failed` rows are auto-counted as "still-open positions" by the dashboard.** The Close Failures Today panel groups by ticker and sums attempts. The Closed Today panel filters strictly to `action="closed"` so a partial-fill zombie is never counted as a closed position. See skill 17.

- **`warning` rows use ticker=""` when not ticker-scoped.** Schwab OAuth refresh failures or position-fetch exhaustion don't pertain to a single ticker. The Recent Journal Entries panel renders these with an empty ticker column; that's intentional. Future ticker-filtered panels should treat empty-ticker warnings as global.

- **Atomic write per row via `locked_append`.** Concurrent writers (a cron-launched cycle + a Streamlit-launched manual cycle + the backtester) won't interleave bytes mid-line and break the JSONL parser. The lock is a per-file `fcntl` exclusive lock.

- **Run-mode mistake.** A bug introduced 2025 wrote to `signals_lvie.jsonl` (typo) silently. The current code asserts `run_mode in {"live", "backtest"}` at construction so the typo can't recur (`journal_kb.py:118-122`).

## 5. Cross-References

- `00_sdlc_and_conventions.md` — run_mode split rationale, journal-as-source-of-truth principle.
- `13_preset_system_hot_reload.md` — mode tag (`raw_signal.mode`) is the pivot key the dashboard's Strategy-Profile grid uses.
- `14_adaptive_vs_static_scan_modes.md` — `scan_results` block schema (only present on adaptive-mode plan() rows).
- `17_close_failure_and_cooldown.md` — `close_failed` action + cooldown surface in `raw_signal`.
- `18_order_submission_idempotency.md` — `client_order_id` + `retry_attempts` in `submitted` and `warning` rows.
- Tests: `tests/test_journal_kb.py`, `tests/test_journal_dedup.py`, `tests/test_production_readiness.py`.

---

*Last verified against repo HEAD on 2026-05-06.*

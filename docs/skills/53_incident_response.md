# Incident Response — Playbook

> **One-line summary:** When the ExceptionMonitor (skill 34) pages, Claude Code reads the error + recent journal context and proposes a fix. The fix goes through the normal SDLC (write skill → failing test → code) — this playbook never bypasses the invariant scanner or lands a hot-fix directly.
> **Source of truth:** [`trading_agent/mcp/tools/market.py`](../../trading_agent/mcp/tools/market.py), [`trading_agent/exception_monitor.py`](../../trading_agent/exception_monitor.py), [`trading_agent/journal_reader.py`](../../trading_agent/journal_reader.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `48_claude_code_mcp_surface.md`, `34_exception_monitor.md`, `00_sdlc_and_conventions.md` (the SDLC every fix follows).
> **Consumed by:** `.claude/commands/incident.md`, invoked when the operator's Telegram channel pages.

---

## 1. Theory & Objective

ExceptionMonitor pages once per failure mode per day. When it fires, the operator wants: (a) the raw event, (b) the last N journal rows for context, (c) a candidate root cause + fix path — all in one place. Historically this meant SSH'ing to the agent host and grepping logs; now the operator invokes `/incident <error_id>` and this playbook runs.

The fix path is **strictly SDLC**: even for what looks like a one-line change, Claude drafts (or updates) the skill file first, adds a failing conformance test, and only then edits code. Skipping any step is a CI failure by design (skill 00 §3).

## 2. Mathematical Formula

N/A — orchestration + prompt structure only.

## 3. Reference tool sequence

```python
alerts   = get_recent_alerts(hours=24)
journal  = get_journal_summary()
recent   = list_recent_trades(days=1)
positions = list_positions()   # in case the error affects an open position
market   = get_market_status()
```

Claude then:

1. Identifies the matching alert row from `alerts["silenced_today"]`.
2. Correlates with `recent["closes_today"]` (did a close fail?), `positions["opens_today"]` (was an open in flight?).
3. Names the skill file whose invariant was violated (if any).
4. Proposes: (a) which skill to update or draft; (b) which conformance test to add; (c) what the minimal fix is.
5. Does NOT open a PR automatically — the operator confirms the skill/test/code sequence in chat first.

## 4. Edge Cases / Guardrails

- **No auto-fix.** The playbook produces a plan, not a commit. Even if the fix is one line, Claude Code writes the skill + failing test first per the SDLC.
- **No executor calls.** If the error is a hung order, the playbook tells the operator to run `python -m trading_agent.executor.cancel_stuck` (an existing operator tool) — Claude does NOT invoke it.
- **Silenced-alerts dedup.** `get_recent_alerts` already dedups per skill 34. The playbook renders each error once with its silence count.
- **Panic-open positions.** If `positions["opens_today"]` shows a position that opened after the alert time, flag it explicitly — the operator may want to close manually before diagnosing.

## 5. Cross-References

- `34_exception_monitor.md` — the paging surface this playbook consumes.
- `00_sdlc_and_conventions.md` — the SDLC every incident-response fix follows.
- `48_claude_code_mcp_surface.md` — the read tools.

---

*Last verified against repo HEAD on 2026-10-07.*

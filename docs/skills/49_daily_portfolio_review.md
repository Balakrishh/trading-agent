# Daily Portfolio Review — Playbook

> **One-line summary:** Playbook Claude Code follows to produce a morning brief over the trading agent's live state — open positions, PnL, expiring-soon, macro overlays, recent alerts. Uses only the read tools from skill 48; never touches the executor.
> **Source of truth:** [`trading_agent/mcp/tools/positions.py`](../../trading_agent/mcp/tools/positions.py), [`trading_agent/mcp/tools/strategy.py`](../../trading_agent/mcp/tools/strategy.py), [`trading_agent/mcp/tools/market.py`](../../trading_agent/mcp/tools/market.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `48_claude_code_mcp_surface.md` (the tools this playbook calls), `19_journal_schema.md` (position/trade shapes).
> **Consumed by:** `.claude/commands/portfolio.md` (`/portfolio` slash command), the `journal-analyst` subagent.

---

## 1. Theory & Objective

The operator wants a one-command morning brief that answers: "What am I holding, how did we do overnight, what's expiring this week, what fired an alert, and what does the macro backdrop look like today?" Before this skill existed the answer meant opening Streamlit and clicking through three tabs. Now the operator types `/portfolio` in Claude Code, or asks in natural language, and Claude follows this playbook.

The playbook is prescriptive on ordering (positions → PnL → expiring → alerts → market status) because the operator reads top-to-bottom and needs the most-actionable rows first.

## 2. Mathematical Formula

N/A — orchestration only. Every number rendered in the brief comes verbatim from a skill-48 tool response; the playbook does not compute new statistics.

## 3. Reference tool sequence

```python
# Executed by Claude Code, one tool call per line. Order matters —
# see §4 on why "market_status first" would waste a round-trip when
# the market is closed.

positions      = list_positions()
journal        = get_journal_summary()
recent         = list_recent_trades(days=7)
preset         = get_preset()
market         = get_market_status()
alerts         = get_recent_alerts(hours=24)
```

The brief is rendered as prose sections in this order:

1. **Positions.** From `positions["open_positions"]` (any open date) — one line per ticker + strategy + expiration + credit + width. Flag `status="expired_unrecorded"` rows prominently: their P&L is missing from the journal.
2. **Realized PnL today.** From `journal["realized_pl_today"]`.
3. **Recent closes.** From `recent["closes"]` (the `days` window) — closed trades with reason + PnL; total from `recent["realized_pl_window"]`.
4. **Expiring this week.** Filter `positions` by expiration ≤ 7 days out.
5. **Alerts.** From `alerts["silenced_today"]` + `alerts["error_count_today"]`.
6. **Macro.** `market["open"]`, `preset["preset"]["name"]`, `preset["preset"]["directional_bias"]`.

## 4. Edge Cases / Guardrails

- **Market closed → no live macro fetch.** If `get_market_status()` returns `{"open": false}`, skip any tool call that would touch Schwab quotes; render the "positions as of last close" caveat.
- **Empty positions.** Render an explicit "no open positions" line — the operator needs to see the absence confirmed, not infer it from silence.
- **Data server unreachable.** `get_quote`/`get_chain` may return `{"source": "unavailable"}`. Playbook skips those sections rather than failing the whole brief.
- **Alerts noise floor.** `get_recent_alerts(hours=24)` includes silenced dedups; render only rows the operator hasn't already seen (compare against yesterday's brief if the state file exists).
- **No writes.** This skill NEVER invokes `promote.py`, writes to `pending_orders/`, or calls a slash command that does. Read-only end-to-end.
- **`morning` skill interop.** The desktop-app `morning` skill renders a styled HTML brief; this playbook renders prose in the Claude Code terminal. They share the same tool calls but different output surfaces — do not conflate.
- **Wheel assignments (2026-09-29).** When `list_recent_trades` shows an `exit_signal` of `assigned`, call `wheel_screen(watchlist=[ticker])` — the shares now appear in Alpaca holdings, so the tool returns covered-call recommendations (never below cost basis, skill 40 §2.1). `called_away` / `expired_worthless` need no follow-up.

## 5. Cross-References

- `48_claude_code_mcp_surface.md` — the tool registry every step of this playbook comes from.
- `19_journal_schema.md` — the position/trade schemas this brief renders.
- `.claude/commands/portfolio.md` — the slash command that invokes this playbook.
- `50_position_triage.md` — the next step after the brief when a position needs attention.

---

*Last verified against repo HEAD on 2026-09-29.*

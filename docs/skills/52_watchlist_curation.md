# Watchlist Curation — Playbook

> **One-line summary:** Periodic pass Claude Code runs over recent journal data to propose adds/drops to the live watchlist. Reads only; the actual watchlist file is edited by the operator through the Streamlit UI (or by hand) — this skill produces the recommendation.
> **Source of truth:** [`trading_agent/mcp/tools/positions.py`](../../trading_agent/mcp/tools/positions.py), [`trading_agent/watchlist_store.py`](../../trading_agent/watchlist_store.py), [`trading_agent/journal_reader.py`](../../trading_agent/journal_reader.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `48_claude_code_mcp_surface.md`, `19_journal_schema.md`.
> **Consumed by:** the `journal-analyst` subagent, plus ad-hoc operator invocations.

---

## 1. Theory & Objective

The watchlist is the input to every scan cycle. Over weeks it accumulates tickers that never fire (chain too illiquid, per-leg spreads too wide) and misses tickers the operator has been trading manually. This playbook produces a proposal: keep, drop, add — with a hit-rate + reject-reason breakdown per ticker so the operator can approve or edit before applying.

## 2. Mathematical Formula

- **Hit rate per ticker** = closes with PnL > 0 in window ÷ opens in window. Window default: 30 days.
- **Reject saturation** = fraction of scan cycles a ticker was scanned but rejected for the same reason. Anything > 80% is a "chronic reject" candidate for drop.

## 3. Reference tool sequence

```python
journal = get_journal_summary()
recent  = list_recent_trades(days=30)   # currently today-only; see §4
alerts  = get_recent_alerts(hours=168)  # 7 days
preset  = get_preset()
```

Claude Code then:

1. Groups closes by ticker → hit rate.
2. Groups reject reasons (from `journal["reject_reasons_top5"]`) by ticker.
3. Renders a table: ticker | opens | wins | hit rate | dominant reject reason.
4. Proposes drops (hit rate < 30% AND opens ≥ 5) and adds (only if operator supplies a candidate list — this playbook does not auto-discover new tickers).

## 4. Edge Cases / Guardrails

- **Journal-window limitation.** `list_recent_trades` currently returns today-only until `JournalReader` grows a multi-day window (Phase 2 follow-up; see `docs/plans/claude_code_portfolio_integration_plan.md`). Until then the playbook explicitly labels its window as "today" not "30 days" when the reader can't extend.
- **No auto-write to the watchlist.** The playbook produces a proposal; edits go through `watchlist_store.save_watchlist()` invoked by the operator, not by Claude. This preserves the audit trail (the operator's Git commit is the record of watchlist changes).
- **Statistical significance.** Tickers with <5 opens in the window get "insufficient data" instead of a hit-rate number. Small-sample noise is the wrong reason to drop a ticker.

## 5. Cross-References

- `48_claude_code_mcp_surface.md` — read tools.
- `19_journal_schema.md` — the closes/opens schema this playbook aggregates.
- `docs/plans/claude_code_portfolio_integration_plan.md` — where the multi-day journal window is tracked as a follow-up.

---

*Last verified against repo HEAD on 2026-09-30.*

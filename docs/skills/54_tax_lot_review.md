# Tax-Lot Review — Playbook

> **One-line summary:** Quarterly (or on-demand) pass Claude Code runs to flag wash-sale risk and short-vs-long-term-holding-period considerations across recent closes. Read-only; produces a prose report the operator gives to their accountant. Never proposes trades.
> **Source of truth:** [`trading_agent/mcp/tools/positions.py`](../../trading_agent/mcp/tools/positions.py), [`trading_agent/journal_reader.py`](../../trading_agent/journal_reader.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `48_claude_code_mcp_surface.md`, `19_journal_schema.md`.
> **Consumed by:** `.claude/commands/taxreview.md`.

---

## 1. Theory & Objective

Credit-spread traders accumulate short-term realized gains fast; wash-sale rule (IRC §1091) triggers when a substantially-identical position is opened within 30 days of a realized loss. The operator wants Claude Code to scan the last 90 days of closes and open positions for wash-sale candidates and flag them BEFORE year-end so the accountant has time to plan.

This playbook produces a prose report. It does not compute cost basis (that's the broker's job), does not file amended returns, and does not propose closing trades to harvest losses. It's a review, not an action.

## 2. Mathematical Formula

- **Wash-sale window** = ±30 calendar days around a realized loss on the same underlying + strategy.
- **Short-term holding period** = position held < 366 days (per §1222).
- **Substantial identity** — the playbook uses ticker + strategy as a coarse proxy; the accountant refines.

No arithmetic beyond date arithmetic; all dollar figures come from the journal verbatim.

## 3. Reference tool sequence

```python
recent = list_recent_trades(days=90)   # window extended in Phase 2 follow-up
open   = list_positions()
```

Claude Code walks `recent["closes_today"]` (later broader window) for negative-PnL rows, then checks `open["opens_today"]` for any position on the same ticker + strategy opened within 30 days. Renders a table: date | ticker | strategy | realized loss | possible wash sale (Y/N).

## 4. Edge Cases / Guardrails

- **Coarse identity proxy.** "Same ticker + same strategy" is a starting point — a bear call vs. bull put on the same ticker isn't a wash sale, but two iron condors at different DTEs likely are. The report labels every flagged row "possible" and defers to the accountant.
- **Window limitation.** Until `JournalReader` grows a multi-day window, the playbook operates on today's data and labels the report accordingly.
- **No cost-basis math.** The playbook renders realized PnL from the journal verbatim and does not attempt to reconcile against the broker's 1099-B.
- **Read-only end-to-end.** No writes; no trade proposals; no invocations of skill 51 (`/propose`).

## 5. Cross-References

- `48_claude_code_mcp_surface.md` — the read tools.
- `19_journal_schema.md` — the closes/opens schema.
- `docs/plans/claude_code_portfolio_integration_plan.md` — the multi-day journal window is tracked as a follow-up.

---

*Last verified against repo HEAD on 2026-10-05.*

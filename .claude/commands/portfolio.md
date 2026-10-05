---
description: Morning portfolio brief — positions, PnL, expiring, alerts, macro.
---

Follow `docs/skills/49_daily_portfolio_review.md` step by step.

Call, in order, using the trading-agent MCP:

1. `list_positions()`
2. `get_journal_summary()`
3. `list_recent_trades(days=7)`
4. `get_preset()`
5. `get_market_status()`
6. `get_recent_alerts(hours=24)`
7. `get_market_state()`

Render the brief as prose in six sections in this exact order: Positions,
Realized PnL today, Recent closes, Expiring this week, Alerts, Macro. In
Macro, lead with the market risk state from `get_market_state()` (state,
top reasons, size multiplier, whether new CSPs are paused) and, if present,
its `hedge_suggestion` — as an idea for `/propose`, never an order. If
`get_market_status()` returns `{"open": false}`, add a "positions as of
last close" caveat and skip any tool that would touch live quotes. Do NOT
call `/propose`, `/triage`, or any tool that writes to `pending_orders/`.
This is a read-only report.

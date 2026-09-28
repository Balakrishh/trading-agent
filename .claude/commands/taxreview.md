---
description: Wash-sale + holding-period scan over recent closes (quarterly).
---

Follow `docs/skills/54_tax_lot_review.md`.

1. `list_recent_trades(days=90)` — window auto-narrows to today until the
   journal reader grows a multi-day window (see the follow-up in
   `docs/plans/claude_code_portfolio_integration_plan.md`).
2. `list_positions()` — check for candidates opened within 30 days of a
   realized loss on the same ticker + strategy.

Render a table: date | ticker | strategy | realized PnL | possible wash
sale (Y/N/insufficient data). Label the window explicitly ("today" vs.
"90 days") to match the reader's actual coverage. Explicit prose caveat:
this is a review to hand to the accountant, not tax advice or a trade
proposal. Do NOT compute cost basis, do NOT stage trades, do NOT invoke
`/propose`.

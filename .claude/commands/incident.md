---
description: Diagnose a recent ExceptionMonitor alert and propose a fix plan.
argument-hint: "[error_id_or_text]"
---

Follow `docs/skills/53_incident_response.md`.

Arguments: `$ARGUMENTS` — the error id, text from the Telegram alert, or
empty (use the most recent alert in that case).

1. `get_recent_alerts(hours=24)` — find the matching row.
2. `get_journal_summary()`
3. `list_recent_trades(days=1)`
4. `list_positions()` — flag any position that opened after the alert.
5. `get_market_status()`.

Render a short root-cause hypothesis + SDLC fix plan: which skill file
to update or draft, which conformance test to add, the minimal code
change. Do NOT open a PR. Do NOT run any executor command. Do NOT
cancel orders — if the incident involves a hung order, tell the operator
to run `python -m trading_agent.executor.cancel_stuck` themselves.

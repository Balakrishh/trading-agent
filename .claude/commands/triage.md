---
description: Triage one open position — hold / roll / close verdict with reasoning.
argument-hint: "<ticker> [strategy]"
---

Follow `docs/skills/50_position_triage.md`.

Arguments: `$ARGUMENTS` — ticker, optionally followed by strategy name
(e.g. `SPY iron_condor`). If the strategy is omitted and the ticker has
multiple open strategies, ASK before picking one — never guess.

Call:

1. `get_position(ticker=..., strategy=...)`
2. `get_preset()`
3. `get_quote(symbol=...)` (the underlying)
4. If proximity looks close: `score_candidate(underlying=..., strategy=..., params={target_dte: preset.dte_iron_condor})`

Apply the three predicates in priority order — profit target → stop loss
→ strike-proximity + defensive roll (only if `preset.defensive_roll_enabled`
is true). Render the verdict as prose with the numeric distance to each
threshold. Do NOT stage an order; if the verdict is "roll" or "close",
tell the operator to invoke `/propose` explicitly. This command is read-only.

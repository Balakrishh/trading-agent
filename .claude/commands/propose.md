---
description: Propose a trade — writes pending_orders/<uuid>.json, does NOT submit.
argument-hint: "<underlying> <strategy> [<param>=<value> ...]"
---

Follow `docs/skills/51_pre_trade_approval.md`.

Arguments: `$ARGUMENTS` — underlying ticker, strategy name, then any
`key=value` overrides (e.g. `AAPL iron_condor target_dte=35 width_pct=0.02`).

Steps:

1. `get_preset()` → note preset name + directional bias.
2. `get_chain(underlying=..., expiration=..., option_type=...)` — pick the
   expiration matching preset DTE window.
3. `score_candidate(underlying=..., strategy=..., params=...)`.
4. `get_risk_report()`.
5. **Invoke the `risk-reviewer` subagent** (`.claude/agents/risk-reviewer.md`)
   with the scored verdict + risk report. Wait for its approve/reject.
6. On approve: shell out to a Python one-liner that calls
   `trading_agent.pending_orders_writer.write(...)` with the collected
   fields. Record the returned UUID.
7. Tell the operator: *"Proposal `<uuid>` written to `pending_orders/`. Run
   `python -m trading_agent.executor_promote <uuid>` to review and submit."*

Do NOT call `trading_agent.executor.*`. Do NOT run the promote CLI
automatically. This command only stages; the operator (or auto-promote,
inside the promote CLI itself) owns the submission decision.

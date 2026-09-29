---
description: Propose a trade — writes pending_orders/<uuid>.json, does NOT submit.
argument-hint: "<underlying> <strategy> [<param>=<value> ...]"
---

Follow `docs/skills/51_pre_trade_approval.md`.

Arguments: `$ARGUMENTS` — underlying ticker, strategy name, then any
`key=value` overrides (e.g. `AAPL iron_condor target_dte=35 width_pct=0.02`).

Steps:

1. `get_preset()` → note preset name + directional bias.
2. `score_candidate(underlying=..., strategy=..., params=...)`.
   The tool calls `decision_engine.decide()` internally and returns
   `verdict.plan` (a `SpreadPlan`-shaped dict). If `verdict` is `null`,
   report the error to the operator and stop.
3. `get_risk_report()`.
4. **Invoke the `risk-reviewer` subagent** (`.claude/agents/risk-reviewer.md`)
   with the scored verdict + risk report. Wait for its approve/reject.
5. On approve: shell out to a Python one-liner that calls
   `trading_agent.pending_orders_writer.write(...)` passing the whole
   verdict dict from step 2. Record the returned UUID. Example:

   ```bash
   python -c "
   import trading_agent.pending_orders_writer as w
   import json, sys
   verdict = json.loads(sys.stdin.read())
   print(w.write(
       underlying='AAPL', strategy='bull_put', params={},
       verdict=verdict, risk_snapshot={}, preset_name='current',
       auto_promote_requested=False,
   ))
   " <<< '<json>'
   ```

6. Tell the operator: *"Proposal `<uuid>` written to `pending_orders/`. Run
   `python -m trading_agent.executor_promote <uuid>` to review and submit."*

Do NOT call `trading_agent.executor.*`. Do NOT run the promote CLI
automatically. This command only stages; the operator (or auto-promote,
inside the promote CLI itself) owns the submission decision.

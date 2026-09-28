# Position Triage — Playbook

> **One-line summary:** For one open position, Claude Code decides "hold / roll / close" using the same scoring primitive the live agent uses. Read-only: the recommendation is prose; any resulting order flows through skill 55 (`promote.py`), never from this playbook directly.
> **Source of truth:** [`trading_agent/mcp/tools/positions.py`](../../trading_agent/mcp/tools/positions.py), [`trading_agent/mcp/tools/strategy.py`](../../trading_agent/mcp/tools/strategy.py), [`trading_agent/defensive_roll_evaluator.py`](../../trading_agent/defensive_roll_evaluator.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `48_claude_code_mcp_surface.md`, `30_profit_target_management.md`, `31_defensive_roll_evaluator.md`, `17_close_failure_and_cooldown.md`.
> **Consumed by:** `.claude/commands/triage.md`, the `risk-reviewer` subagent.

---

## 1. Theory & Objective

The trading agent already runs profit-target and defensive-roll evaluators every cycle. But those decisions fire silently — the operator only sees them after the fact in the journal. When the operator wants to *ask* about a specific position ("should I close this SPY iron condor early?"), they need Claude Code to run the same predicates the live agent runs, explain them, and produce a recommendation with reasoning. This skill is that playbook.

The three verdicts:

- **Hold** — no exit predicate fires; brief the operator on how close the nearest predicate is to firing.
- **Close** — profit target reached OR stop-loss reached OR strike-proximity + no defensive-roll opportunity.
- **Roll** — strike-proximity fires AND `defensive_roll_evaluator.evaluate_defensive_roll()` returns a positive-EV roll (all six predicates pass).

## 2. Mathematical Formula

Reuses:

- **Profit target predicate** — skill 30: `unrealized_pnl >= profit_target_pct × initial_credit`.
- **Defensive-roll six predicates** — skill 31: proximity band, days-to-expiry floor, roll-credit floor, IV rank, portfolio-heat cap, cooldown.
- **Stop-loss predicate** — skill 17: `unrealized_loss >= stop_loss_multiplier × initial_credit`.

The playbook computes NONE of these; it calls `get_preset()` for the thresholds and `get_position()` for the position state, then re-uses the existing evaluators.

## 3. Reference tool sequence

```python
# Claude Code runs, one step per user command:

position = get_position(ticker, strategy)      # ticker + strategy from user prompt
preset   = get_preset()
market   = get_quote(position["positions"][0]["ticker"])
# Optional — if strike-proximity is close, request a scored candidate
# for the roll target:
candidate = score_candidate(
    underlying=ticker,
    strategy=position["positions"][0]["strategy"],
    params={"target_dte": preset["preset"]["dte_iron_condor"]},
)
```

Claude then applies the three predicates in priority order (profit target → stop loss → strike proximity + roll) and renders a decision with the numeric distance to each threshold.

## 4. Edge Cases / Guardrails

- **No matching position.** `get_position` returns `{"found": false}` → the playbook tells the operator explicitly rather than fabricating a position.
- **Multiple strategies on one ticker.** If `get_position(ticker)` returns >1 row and the user didn't specify a strategy, Claude Code asks which one — never picks silently.
- **Preset disables defensive rolls.** If `preset["preset"]["defensive_roll_enabled"] == False`, the "Roll" verdict is unreachable — the playbook renders only Hold / Close.
- **Recommendation is prose, not an order.** The output is a paragraph the operator reads. Producing an actual order means invoking skill 51 (`/propose`); this playbook does NOT emit a `pending_orders/*.json` file.
- **Backtest mode.** For "how would this have played historically", the operator invokes `/triage` with a `--backtest` flag; the playbook then routes `score_candidate` through `decide()` in backtest mode (invariant #3).

## 5. Cross-References

- `30_profit_target_management.md` — the profit-take predicate this playbook re-uses.
- `31_defensive_roll_evaluator.md` — the roll predicates.
- `48_claude_code_mcp_surface.md` — the tools called here.
- `51_pre_trade_approval.md` — the next step when the verdict is "roll" (produces a `pending_orders/*.json`).

---

*Last verified against repo HEAD on 2026-09-28.*

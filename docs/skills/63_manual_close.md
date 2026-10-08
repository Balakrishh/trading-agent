# Manual Close — the Operator Closes a Position Now

> **One-line summary:** one command closes one open spread through the agent's own atomic close order and journals it as an ordinary `closed` row with `exit_signal="manual"`; preview is the default, `--submit` sends.
> **Source of truth:** [`trading_agent/manual_close.py`](../../trading_agent/manual_close.py), [`trading_agent/executor.py:122-147`](../../trading_agent/executor.py) (`close_order_prices`), `executor.py` `close_spread_atomic`
> **Phase:** 2  •  **Group:** ops / risk
> **Depends on:** `17_close_failure_and_cooldown.md`, `19_journal_schema.md`, `35_close_event_collaborators.md`, `30_profit_target_management.md`
> **Consumed by:** the operator CLI; the human-in-the-loop roadmap (backlog §9)

---

## 1. Theory & Objective

The agent closes positions only when one of its exit rules fires. An operator who wants out earlier — taking a profit before the target, cutting a loser before the stop — had two bad options: the Alpaca UI (legs can fill separately, and the journal never learns of the close, so the position shows open until someone runs `journal_reconcile`) or a one-off script. This skill is the supported lever. It reuses the three things the agent already trusts: the atomic mleg close (every leg fills together or none does), the shared close pricing, and the agent's close writer (skill 35), so the journal row, Telegram close alert and realized-P&L readers see a manual close exactly like an automatic one. First used 2026-10-08 to take +$288 on the IWM put debit spread (as a script; this skill is that script made permanent).

## 2. Mathematical Formula

```text
per leg (quote bid, ask > 0):
  short leg (qty < 0):  natural += ask      mid += (bid + ask)/2
  long  leg (qty > 0):  natural −= bid      mid −= (bid + ask)/2
net close price (per share, Alpaca sign: + = pay a debit, − = receive a credit)

order attempts:  1. limit = round((mid + natural)/2, 2)   ("mleg_improved")
                 2. limit = round(natural, 2)             ("mleg_natural")
realized P&L  = (Σ short avg_entry − Σ long avg_entry − fill_price) × 100 × contracts
```

Example (2026-10-08 preview): IWM 283/278 put debit × 4, entry 1.92 debit; mid −3.01, natural −2.86 → first limit −2.93 → estimated +$404.

## 3. Reference Python Implementation

```python
# trading_agent/executor.py:122-147
def close_order_prices(legs, quotes: Dict) -> Optional[Tuple[float, float, List[Dict]]]:
    """Per-share net debit to close ``legs`` at mid and at natural, plus
    the mleg legs payload. Positive = pay a debit, negative = receive a
    credit (Alpaca's limit_price sign). Short legs buy back at the ask,
    long legs sell at the bid. None when any leg lacks a two-sided quote.
    Shared by the atomic close and the operator close preview (skill 63).
    """
    natural = mid = 0.0
    payload: List[Dict] = []
    for leg in legs:
        q = quotes.get(leg.symbol)
        if not q or float(q.get("bid", 0)) <= 0 or float(q.get("ask", 0)) <= 0:
            return None
        bid, ask = float(q["bid"]), float(q["ask"])
        if leg.qty < 0:      # short → buy to close, pay the ask
            natural += ask
            mid += (bid + ask) / 2
            payload.append({"symbol": leg.symbol, "ratio_qty": "1",
                            "side": "buy", "position_intent": "buy_to_close"})
        else:                # long → sell to close, receive the bid
            natural -= bid
            mid -= (bid + ask) / 2
            payload.append({"symbol": leg.symbol, "ratio_qty": "1",
                            "side": "sell", "position_intent": "sell_to_close"})
    return mid, natural, payload
```

```python
# trading_agent/manual_close.py:53-66
def refusal(spread: SpreadPosition, *, market_open: bool, dry_run: bool) -> Optional[str]:
    """Why ``spread`` cannot be closed with --submit right now, or None."""
    if len(spread.legs) < 2:
        return (f"{spread.underlying} {spread.strategy_name} is a single leg — "
                "close it in the Alpaca UI (the atomic close needs two or more legs).")
    qtys = {abs(int(leg.qty)) for leg in spread.legs}
    if len(qtys) != 1:
        return (f"Unequal leg quantities {sorted(qtys)} (a partial fill) — "
                "close it in the Alpaca UI.")
    if dry_run:
        return "DRY_RUN is on — the agent would not trade, so neither does this."
    if not market_open:
        return "The market is closed — options orders need the regular session."
    return None
```

Usage:

```bash
python -m trading_agent.manual_close --ticker IWM                    # preview: legs, quotes, prices, estimated P&L
python -m trading_agent.manual_close --ticker IWM --submit \
       [--strategy "Put Debit Spread"] [--reason "taking profit early"]
```

Exit codes: `0` closed / preview shown, `1` no such position, `2` refused (or broker fetch failed), `3` not filled or unresolved.

## 4. Edge Cases / Guardrails

- **Preview by default.** Without `--submit` nothing is sent and nothing is journalled.
- **Two positions on one ticker.** `select_spread` refuses and lists them; `--strategy` (case-insensitive) picks one. Never picks silently.
- **Single-leg positions (wheel CSP / CC).** Refused — the atomic close needs two or more legs; close in the Alpaca UI.
- **Unequal leg quantities (partial fill).** Refused — the mleg order needs one ratio; close in the Alpaca UI.
- **DRY_RUN on / market closed.** Refused before any order.
- **Atomic only.** `close_spread_atomic` never falls back to per-leg DELETEs (the 2026-07-02 leg-out, skill 17). Not filled at either price → the order is cancelled, the position is unchanged, nothing is journalled (exit 3).
- **Order unresolved after cancel** (`mleg_unresolved`) → journalled as `close_failed` via the close writer (counts toward the skill-17 cooldown); check the Alpaca UI.
- **Realized P&L** comes from the fill and the broker entry prices (`realized_pl_from_close`); when an entry price is missing the row keeps the mark (`pl_source="signal_mark"`).
- **Quotes from Alpaca, never Schwab.** The live agent holds the Schwab token, whose refresh token rotates on every use; a second process refreshing it can strand the agent.
- **Live agent running.** The close is one broker order; the agent's next cycle sees the position gone. No journal action is new (`closed` / `close_failed`); only the `exit_signal` value `manual` is new (`ExitSignal.MANUAL`, never set by `evaluate()`).
- **Price moves between preview and submit.** The order is priced from fresh quotes at submit time, not the preview (2026-10-08: preview +$404, fill +$288 a minute later).

## 5. Cross-References

- `17_close_failure_and_cooldown.md` — why the close is atomic; the cooldown a `close_failed` row feeds.
- `35_close_event_collaborators.md` — `CloseJournalWriter`, reused here unchanged.
- `19_journal_schema.md` — the `closed` row; `exit_signal="manual"`.
- `30_profit_target_management.md` — the automatic exit this overrides.
- `62_kill_switch_drawdown_governor.md` — the other operator lever (pauses entries; this closes exits).

---

*Last verified against repo HEAD on 2026-10-08.*

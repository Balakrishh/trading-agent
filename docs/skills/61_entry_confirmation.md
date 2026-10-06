# Entry Confirmation — Wait Before Opening

> **One-line summary:** an approved plan is submitted only after the same ticker produced the same strategy + expiration on N consecutive cycles, not before 09:45 ET, and within a per-hour entry limit.
> **Source of truth:** [`trading_agent/entry_confirmation.py`](../../trading_agent/entry_confirmation.py)
> **Phase:** 2  •  **Group:** risk
> **Depends on:** `37_position_caps.md` (the other entry gates), `58_market_state_playbook.md`, `57_agent_supervisor.md` (cycle cadence)
> **Consumed by:** `agent.py` (`_begin_entry_gates`, `_entry_rate_gate`, `_entry_confirmation_block`)

---

## 1. Theory & Objective

A cycle runs about every 75 seconds, and single-cycle signals are noisy: on 2026-10-05 IWM was bought as "bearish" (a put debit) and read "sideways" one cycle later, and four entries went out within 8 minutes, spending the whole risk budget. Requiring the same signal on several consecutive cycles filters one-cycle flickers at the cost of a few minutes' delay; skipping the first 15 minutes avoids the open's widest quotes; an hourly limit spreads entries across the session. Exits are not delayed by this skill (they have their own 3-cycle debounce, and stops act immediately).

## 2. Mathematical Formula

```text
signature(plan) = strategy | expiration | far_expiration
count_t = count_(t−1) + 1   if signature unchanged and the previous sighting is ≤ 180 s old
        = 1                  otherwise (different signature, no approved plan last cycle, or a gap)
submit only if  count_t ≥ entry_confirm_cycles  AND  ET clock ≥ no_entry_before_et
                AND  submissions in the last 60 min < max_new_entries_per_hour   (0 = no limit)
after a submission the ticker's count is cleared (the next entry starts over)
```

## 3. Reference Python Implementation

```python
# trading_agent/entry_confirmation.py:84-98
    def observe(self, ticker: str, plan: Any) -> Confirmation:
        sig = signature(plan)
        prev = self.previous.get(ticker) or {}
        count = 1
        try:
            last = datetime.fromisoformat(prev.get("last_seen", ""))
            fresh = (self.now - last).total_seconds() <= MAX_GAP_SECONDS
        except (TypeError, ValueError):
            fresh = False
        if prev.get("signature") == sig and fresh:
            count = int(prev.get("count", 0)) + 1
        self.current[ticker] = {"signature": sig, "count": count,
                                "first_seen": prev.get("first_seen") if count > 1 else self.now.isoformat(),
                                "last_seen": self.now.isoformat()}
        return Confirmation(count, self.required, sig)
```

```python
# trading_agent/agent.py:2347-2362
    def _entry_confirmation_block(self, ticker: str, plan) -> Optional[str]:
        """Failed-check text while ``plan`` is not yet confirmed on enough
        consecutive cycles, or before the ET entry window opens; None when
        the order may go out."""
        from trading_agent.entry_confirmation import before_entry_window
        tracker = getattr(self, "_entry_confirm", None)
        if tracker is None:
            return None
        conf = tracker.observe(ticker, plan)
        not_before = str(getattr(self.preset, "no_entry_before_et", "") or "")
        if not_before and before_entry_window(datetime.now(timezone.utc), not_before):
            return f"entry_window (no entries before {not_before} ET)"
        if not conf.confirmed:
            logger.info("[%s] %s", ticker, conf.reason)
            return f"entry_confirming ({conf.count}/{conf.required})"
        return None
```

## 4. Edge Cases / Guardrails

- **Where it sits.** The confirmation check runs last in the post-plan gate chain (RiskManager → market-state gate → ladder gate → confirmation), so only plans that would otherwise be submitted count. Not-yet-confirmed plans are journaled as `rejected` with `risk: entry_confirming (n/N)` (or `entry_window (…)`), so the reject histogram shows them and the shadow-POP fields are still logged.
- **Consecutive means consecutive.** Candidates are rebuilt every cycle: a ticker skipped for any reason (cap, filter, capitulation, no positive-EV plan) loses its count. A sighting older than 180 s also resets.
- **Strikes may drift.** Only strategy and expiry must match; the order uses the latest plan and still passes the executor's live-price recheck (credit floor / debit cap).
- **Entry window.** Counting continues before `no_entry_before_et`; at 09:45 a ticker with a full count can go immediately.
- **Hourly limit.** Counted from the journal's `submitted` rows in the last hour plus in-cycle submissions; a limited ticker is skipped before planning (`skipped_entry_rate`) and starts its count again afterwards.
- **Restarts.** State is `trade_journal/entry_candidates.json` (atomic temp + rename), because the agent process restarts every cycle.
- **`entry_confirm_cycles = 1`** restores immediate entry; the Wheel's operator-staged orders (`/propose` → promote) are not affected.

## 5. Cross-References

- `30_profit_target_management.md` — the exit-side counterpart (trailing profit).
- `37_position_caps.md` — per-ticker / sector / total-risk caps applied before these gates.
- `57_agent_supervisor.md` — the ~75 s cycle the counts are measured in.

---

*Last verified against repo HEAD on 2026-10-05.*

# Entry Confirmation — Wait Before Opening

> **One-line summary:** an approved plan is submitted only after the same ticker produced the same strategy + expiration on N consecutive cycles, not before 09:45 ET, within a per-hour entry limit — and (entry timing) at the best-priced cycle of a short window, never chasing a worsening price.
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

Entry timing (entry_timing_mode, 2026-10-06) — once confirmed:
score per cycle   credit: net ÷ width · debit vertical: (width − debit) ÷ debit · calendar: −debit ÷ mid debit
gap per cycle     mid net − natural net
enter  if score ≥ best_seen − tol × |best_seen|  AND  gap ≤ median(gap over the window)
else at n ≥ max_wait:  enter if score ≥ s_confirm − chase × |s_confirm|, else SKIP (start over)
else wait
defaults: max_wait 8 cycles, tol 1 %, chase 3 %
shadow: enter at confirmation; keep quoting the filled legs; journal entry_timing_shadow
        {decision, cycle, actual_net, timing_net, improvement_usd = (timing_net − actual_net) × 100 × qty}
```

## 3. Reference Python Implementation

```python
# trading_agent/entry_confirmation.py:196-215
    def observe(self, ticker: str, plan: Any) -> Confirmation:
        sig = signature(plan)
        prev = self.previous.get(ticker) or {}
        count = 1
        try:
            last = datetime.fromisoformat(prev.get("last_seen", ""))
            fresh = (self.now - last).total_seconds() <= MAX_GAP_SECONDS
        except (TypeError, ValueError):
            fresh = False
        history: List[Dict[str, float]] = []
        if prev.get("signature") == sig and fresh:
            count = int(prev.get("count", 0)) + 1
            history = list(prev.get("history") or [])
        scored = score_plan(plan)
        if scored is not None:
            history.append(scored)
        self.current[ticker] = {"signature": sig, "count": count,
                                "first_seen": prev.get("first_seen") if count > 1 else self.now.isoformat(),
                                "last_seen": self.now.isoformat(), "history": history}
        return Confirmation(count, self.required, sig, history)
```

```python
# trading_agent/entry_confirmation.py:84-116
def score_legs(legs: List[Dict[str, Any]], width: float) -> Optional[Dict[str, float]]:
    """Score one cycle's price for a structure. ``legs``: dicts with
    ``action`` ("sell"/"buy"), ``bid``, ``ask``. Returns ``score`` (higher
    = better entry), ``gap`` (mid − natural net, ≥ 0 normally) and ``net``
    (natural net credit; negative = debit), or None without usable quotes.

    credit:          score = net ÷ width
    debit vertical:  score = (width − debit) ÷ debit
    calendar:        score = −debit ÷ mid debit
    """
    nat = mid = 0.0
    for leg in legs:
        bid, ask = float(leg.get("bid") or 0), float(leg.get("ask") or 0)
        if ask <= 0:
            return None
        m = (bid + ask) / 2 if bid > 0 else ask
        if leg["action"] == "sell":
            nat += bid
            mid += m
        else:
            nat -= ask
            mid -= m
    gap = mid - nat
    if nat > 0 and width > 0:
        score = nat / width
    elif nat < 0 and width > 0:
        debit = -nat
        score = (width - debit) / debit
    elif nat < 0 and mid < 0:
        score = -(-nat) / (-mid)
    else:
        return None
    return {"score": round(score, 6), "gap": round(gap, 4), "net": round(nat, 4)}
```

```python
# trading_agent/entry_confirmation.py:128-151
def timing_decision(history: List[Dict[str, float]], required: int,
                    params: TimingParams) -> Tuple[str, str]:
    """``("enter" | "wait" | "skip", reason)`` for a confirmed candidate.
    ``history`` holds one score dict per consecutive cycle, oldest first;
    the confirmation point is ``history[required − 1]``."""
    n = len(history)
    if n < required:
        return "wait", f"confirming {n}/{required}"
    cur = history[-1]
    best = max(h["score"] for h in history)
    tol = params.best_tolerance_pct * abs(best)
    median_gap = statistics.median(h["gap"] for h in history)
    if cur["score"] >= best - tol and cur["gap"] <= median_gap + 1e-9:
        return "enter", (f"timing: best price in {n} cycles (score {cur['score']:.4f}, "
                         f"gap {cur['gap']:.2f} ≤ median {median_gap:.2f})")
    if n >= max(required, params.max_wait_cycles):
        ref = history[required - 1]["score"]
        if cur["score"] >= ref - params.chase_limit_pct * abs(ref):
            return "enter", (f"timing: max wait {n} cycles, score {cur['score']:.4f} within "
                             f"{params.chase_limit_pct:.0%} of confirmation {ref:.4f}")
        return "skip", (f"timing: price worsened — score {cur['score']:.4f} vs "
                        f"{ref:.4f} at confirmation after {n} cycles")
    return "wait", (f"timing: waiting {n}/{params.max_wait_cycles} (score "
                    f"{cur['score']:.4f}, best {best:.4f}, gap {cur['gap']:.2f})")
```

```python
# trading_agent/agent.py:2377-2403
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
        if str(getattr(self.preset, "entry_timing_mode", "off")) == "live":
            from trading_agent.entry_confirmation import TimingParams, timing_decision
            decision, reason = timing_decision(conf.history, conf.required,
                                               TimingParams.from_preset(self.preset))
            logger.info("[%s] %s", ticker, reason)
            if decision == "wait":
                return "entry_timing_wait"
            if decision == "skip":
                tracker.consumed(ticker)          # start over; never chase
                return "entry_timing_skip"
        self._entry_plans[ticker] = plan
        return None
```

## 4. Edge Cases / Guardrails

- **Where it sits.** The confirmation check runs last in the post-plan gate chain (RiskManager → market-state gate → ladder gate → confirmation), so only plans that would otherwise be submitted count. Not-yet-confirmed plans are journaled as `rejected` with `risk: entry_confirming (n/N)` (or `entry_window (…)`), so the reject histogram shows them and the shadow-POP fields are still logged.
- **Consecutive means consecutive.** Candidates are rebuilt every cycle: a ticker skipped for any reason (cap, filter, capitulation, no positive-EV plan) loses its count. A sighting older than 180 s also resets.
- **Strikes may drift.** Only strategy and expiry must match; the order uses the latest plan and still passes the executor's live-price recheck (credit floor / debit cap).
- **Entry window.** Counting continues before `no_entry_before_et`; at 09:45 a ticker with a full count can go immediately.
- **Hourly limit.** Counted from the journal's `submitted` rows in the last hour plus in-cycle submissions; a limited ticker is skipped before planning (`skipped_entry_rate`) and starts its count again afterwards.
- **Restarts.** State is `trade_journal/entry_candidates.json` (atomic temp + rename), because the agent process restarts every cycle.
- **Entry timing — why ratios.** Strikes can move a grid step between cycles, so raw credit / debit is not comparable; credit ÷ width, reward ÷ risk and debit ÷ mid are. The gap term targets the cost that dominated the first live day (four debit fills $86 over mid).
- **A higher credit can mean more risk.** At a fixed delta (the scanner re-picks strikes each cycle) a richer credit mostly reflects IV, but the max-wait chase limit and the normal floors still bound it; a plan that stops passing its floors is never entered.
- **Live waits and skips** journal as `rejected` with `risk: entry_timing_wait` / `risk: entry_timing_skip`; a skip clears the candidate (the next entry starts a new confirmation).
- **Shadow outcomes** are resolved at entry when the rule agrees (`improvement_usd` 0), otherwise when the rule would have entered or skipped in a later cycle (filled legs re-quoted from the next cycle on; trackers older than 30 min are dropped).
- **`entry_confirm_cycles = 1`** restores immediate entry; the Wheel's operator-staged orders (`/propose` → promote) are not affected.

## 5. Cross-References

- `30_profit_target_management.md` — the exit-side counterpart (trailing profit).
- `37_position_caps.md` — per-ticker / sector / total-risk caps applied before these gates.
- `57_agent_supervisor.md` — the ~75 s cycle the counts are measured in.

---

*Last verified against repo HEAD on 2026-10-06.*

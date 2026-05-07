# Order-Submission Idempotency (`client_order_id` + Retry)

> **One-line summary:** Every `POST /v2/orders` carries a unique `client_order_id` UUID generated client-side and reused across up to 2 attempts; Alpaca's server-side dedup collapses any duplicate that survived a network blip back to one order, so a transient timeout can never produce a doubled position.
> **Source of truth:** [`trading_agent/executor.py:60-76`](../../trading_agent/executor.py), [`trading_agent/executor.py:299-348`](../../trading_agent/executor.py), [`trading_agent/executor.py:349-541`](../../trading_agent/executor.py)
> **Phase:** 2  •  **Group:** risk
> **Depends on:** `00_sdlc_and_conventions.md` (broker-port abstraction)
> **Consumed by:** `agent.py:_process_ticker` (Phase VI EXECUTE), the journal `action="warning"` row written when retries exhaust.

---

## 1. Theory & Objective

A naïve `requests.post` on order submission has a silent and expensive failure mode. Alpaca accepts the order, assigns it a server-side ID, begins routing it. Then a network blip — packet loss, TCP reset, the agent-side TLS layer giving up after the read timeout — kills the connection before the 200 response reaches us. From the agent's vantage point the call raised `ReadTimeout`. Without an idempotency key, any retry submits a *second* identical spread. Two contracts at the same strikes, the account is over-committed, the C/W floor invariant is violated by construction.

The fix is to give every order submission a UUID generated client-side **before** the POST goes out. Alpaca treats `client_order_id` as a server-side dedup key: a second request with the same ID returns the original order record instead of creating a new one. We can retry as many times as we like and the broker enforces the at-most-once guarantee for us.

The retry policy itself is conservative — 2 attempts, 1.0s backoff, 4xx short-circuit. We're not trying to recover from broker outages; we're trying to recover from the most common failure (a single-packet network blip) without doubling a position.

## 2. Retry policy

```text
ORDER_RETRY_ATTEMPTS    = 2            # at most one retry
ORDER_RETRY_BACKOFF_S   = 1.0          # constant, no exponential

Decision tree per attempt:
  - 2xx          → return submitted result
  - 4xx (any)    → permanent at the broker (validation, BP, PDT) — do
                   NOT retry; same payload will be rejected the same way
  - 5xx, network → transient — retry with the SAME client_order_id
  - retries exhausted → return status="error" with client_order_id
                        embedded so an operator can search the broker
```

Worst-case latency: `(read_timeout × attempts) + (backoff × (attempts-1))`
= `(15s × 2) + (1.0s × 1) = 31s`. Well inside the cycle's 270s hard guard.

## 3. Reference Python Implementation

### `trading_agent/executor.py:60-76` — module-level retry constants

```python
# How many POST /v2/orders attempts before giving up.  At 2 attempts
# with the default 15s timeout, worst-case latency is ~31s plus
# ORDER_RETRY_BACKOFF_S of sleep — well inside the cycle's 270s hard
# guard.  Each attempt re-uses the SAME client_order_id so a retry
# of an order Alpaca already accepted collapses server-side instead
# of producing a duplicate.
ORDER_RETRY_ATTEMPTS = 2
ORDER_RETRY_BACKOFF_S = 1.0
```

### `trading_agent/executor.py:300-348` — UUID generation + payload

```python
def _submit_order(self, plan: SpreadPlan, plan_path: str,
                  run_id: str, account_balance: float = 0.0) -> Dict:
    ...
    # ── Idempotency key ────────────────────────────────────────────
    # Generate ONCE per plan.  Re-used across every retry of THIS
    # POST so that if Alpaca accepted the first request but the
    # response was lost on the wire, the second POST collapses
    # server-side instead of producing a duplicate spread.
    client_order_id = f"ta-{run_id[:8]}-{uuid.uuid4().hex[:12]}"

    order_payload = {
        "type": "limit",
        "time_in_force": "day",
        "order_class": "mleg",
        "qty": str(qty),
        "limit_price": str(limit_price_value),
        "client_order_id": client_order_id,
        "legs": legs_payload,
    }

    return self._submit_order_with_idempotency(
        plan=plan, plan_path=plan_path, run_id=run_id,
        order_payload=order_payload,
        client_order_id=client_order_id,
    )
```

### `trading_agent/executor.py:349-541` — retry loop

```python
def _submit_order_with_idempotency(self, plan, plan_path, run_id,
                                   order_payload, client_order_id):
    last_error = None
    for attempt in range(1, ORDER_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(
                f"{self.base_url}/orders",
                headers=self._headers(), json=order_payload,
                timeout=ALPACA_TIMEOUT_LONG)
            resp_body = resp.json()
            resp.raise_for_status()
            order_id = resp_body.get("id", "unknown")
            return {
                "status": "submitted",
                "order_id": order_id,
                "client_order_id": client_order_id,
                "retry_attempts": attempt,
                ...
            }

        except requests.RequestException as exc:
            last_error = str(exc)
            # ... log Alpaca response detail ...

            # 4xx is permanent at the broker — don't retry.
            resp_obj = getattr(exc, "response", None)
            if resp_obj is not None and 400 <= resp_obj.status_code < 500:
                logger.error("[%s] Order rejected by Alpaca (HTTP %d) — "
                             "permanent at broker level, NOT retrying. "
                             "client_order_id=%s",
                             plan.ticker, resp_obj.status_code,
                             client_order_id)
                break

            # Transient — retry with the SAME client_order_id.
            if attempt < ORDER_RETRY_ATTEMPTS:
                logger.warning(
                    "[%s] Transient order POST failure (attempt %d/%d): %s. "
                    "Retrying in %.1fs with same client_order_id=%s "
                    "(broker will dedupe if it already accepted).",
                    plan.ticker, attempt, ORDER_RETRY_ATTEMPTS,
                    exc, ORDER_RETRY_BACKOFF_S, client_order_id)
                time.sleep(ORDER_RETRY_BACKOFF_S)
                continue

    # All attempts exhausted (or 4xx short-circuit).
    return {"status": "error", "error": last_error,
            "client_order_id": client_order_id,
            "retry_attempts": ORDER_RETRY_ATTEMPTS,
            ...}
```

### `trading_agent/agent.py:1559-1592` — warning surfacing on exhaustion

The agent calls `journal_kb.log_warning` whenever the executor returns `status="error"`, embedding the `client_order_id` so an operator can search the Alpaca UI to confirm whether *any* attempt landed:

```python
if isinstance(exec_result, dict) and exec_result.get("status") == "error":
    err_msg = exec_result.get("error", "unknown")
    client_order_id = exec_result.get("client_order_id", "n/a")
    attempts = exec_result.get("retry_attempts", 1)
    self.journal_kb.log_warning(
        source="executor",
        ticker=ticker,
        message=f"Order submission failed after {attempts} attempt(s): "
                f"{err_msg}. Search Alpaca for client_order_id="
                f"{client_order_id} to confirm whether the order ever "
                f"landed (the broker dedupes duplicates by this key).",
        context={"client_order_id": client_order_id,
                 "retry_attempts": attempts,
                 "strategy": plan.strategy_name})
```

## 4. Edge Cases / Guardrails

- **`client_order_id` is generated ONCE per plan**, not per retry. The whole point is that retries reuse the same key — generating a new UUID per attempt would defeat the dedup. The retry helper takes `client_order_id` as a parameter to make this invariant explicit at the API surface.

- **UUID format is `ta-<run_id[:8]>-<uuid12>`.** The `ta-` prefix marks it as agent-originated (vs. manual orders the operator places via the broker UI), the run_id slice ties the order to a specific cycle in the journal, the random hex provides the actual collision-avoidance entropy. 22 chars total — well under Alpaca's 128-char limit.

- **4xx is a hard short-circuit, not just "skip remaining retries".** A 4xx means the broker has already evaluated the order and refused (validation error, insufficient buying power, PDT, market closed). Retrying with the same payload is pointless. Distinguished from 5xx + network errors via `getattr(exc, "response", None)` because the `response` attr is only populated on HTTPError, not on ConnectTimeout / ReadTimeout.

- **Alpaca's response body is logged on every attempt.** When the broker DOES respond with an error, the message field carries the actionable reason ("insufficient buying power", "market is closed"). We surface this in both the agent log AND the journal warning so an operator triaging a `status="error"` row doesn't have to scroll through `logs/trading_agent.log` to find the broker's reason.

- **Worst-case dedup behaviour.** If the network is so degraded that BOTH attempts time out before getting a response — but Alpaca actually accepted the first one — the agent returns `status="error"` and the trade looks like it failed. The journal warning row tells the operator to search by `client_order_id`. The order IS at the broker, and a manual cycle on the operator's end can either let it ride or cancel it. The dedup property has held: there's still only ONE order, just one we don't know about until the operator checks.

- **The retry budget is per-call, not per-cycle.** If the next 5-min cycle re-submits the same plan, that's a *new* `client_order_id` and a *new* order. This is correct behaviour — by then the operator should have triaged the previous warning, and the trade plan's pre-conditions (regime, credit, sizing) have been re-evaluated.

- **Time-budget interaction with `_cycle_lock`.** The cycle-singleton lock holds throughout a cycle; retries don't release it. Worst-case 31s of order-submission latency means subsequent triggers (Streamlit watchdog, auto-refresh) skip cleanly via `status="skipped_concurrent"` rather than racing past the lock. See `tests/test_cycle_singleton.py`.

- **`_append_to_plan` is called on both success and error.** The trade plan's `state_history` always records what happened: `order_result` on success, `order_error` on failure. Combined with the `client_order_id` field this gives a deterministic forensic trail per plan.

## 5. Cross-References

- `00_sdlc_and_conventions.md` — broker port abstraction (the executor is the only module that talks to Alpaca's order endpoint).
- `17_close_failure_and_cooldown.md` — companion skill covering the CLOSE side; together they describe the agent's full broker-interaction safety story.
- `19_journal_schema.md` — `action="warning"` row schema and dedup-bypass rules.
- Tests: `tests/test_production_readiness.py` covers first-try success, retry-on-transient (verifies stable `client_order_id`), no-retry-on-4xx, retry-exhausted error path.

---

*Last verified against repo HEAD on 2026-05-06.*

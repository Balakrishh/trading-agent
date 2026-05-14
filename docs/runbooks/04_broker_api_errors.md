# Runbook 04 — Broker API Error Codes

> **Trigger:** You see an Alpaca or Schwab error code in the agent log and want to know what it means and what to do.
> **Status:** Reference table only for now — will expand to a full diagnostic runbook as new codes are encountered.

---

## Alpaca order/position error codes (seen in production)

| Code | Message snippet | Where it appears | Operator action |
|---|---|---|---|
| `40310000` | "account not eligible to trade uncovered option contracts" | Position close, when you're closing a long-protective leg before the short → momentarily naked. | Close ALL short legs FIRST, longs LAST. See runbook 03 §5. |
| `40310000` | "insufficient options buying power for cash-secured put" | Position close on a put leg when remaining account BP < notional. Hits sub-$25K accounts often, especially on high-strike puts (GLD 415, SPY 500+). | Close another smaller position first to free BP, then retry. |
| `40310100` | "trade denied due to pattern day trading protection" | Position close that would constitute a 4th day-trade in a 5-day rolling window on a sub-$25K account. | Wait for the 5-day window to roll. Position may have to ride to expiration. |
| `40310010` | "stock too volatile" / "trading halted" | Rare. Position close during a halt. | Wait for halt to clear, retry manually. |
| HTTP 400 | "invalid order — qty must be > 0" | Order submit when sizing produced qty=0 (max risk too small for the spread's max loss). | Reduce `min_credit_ratio` in preset, OR increase account equity, OR use wider strikes. |
| HTTP 422 | "duplicate client_order_id" | Order submit retry of a SUCCEEDED order. **This is the dedup working correctly — not an error to alarm at.** | None. The original order is the one to track. |
| HTTP 429 | "rate limit exceeded" | Burst of fetches/orders. | Backoff handles it. If it persists, reduce ticker count or extend cycle interval. |
| HTTP 5xx | "service unavailable" / "internal server error" | Transient Alpaca outage. | Retry logic handles it. If it persists across 30+ minutes, check status.alpaca.markets. |

---

## Schwab OAuth error patterns

| Pattern in log | Meaning | Operator action |
|---|---|---|
| "No Schwab tokens on disk yet" | No `~/.schwab_tokens.json` exists. | Run `python -m trading_agent.schwab_oauth login` on the Pi. |
| "Schwab token refresh PERMANENTLY FAILED (HTTP 401)" | Refresh token revoked, or 7-day refresh-token absolute expiry hit. | Run the login flow again. |
| "Schwab token refresh transient error (HTTP 500)" / "(HTTP 503)" | Schwab gateway hiccup; retry logic engaged. | None if it recovers within 3 retries. If it persists, check Schwab developer portal status. |
| "Schwab token refresh failed after 3 attempts" | Permanent retry-budget exhaustion. | Treat as the permanent case → re-login. |
| "Schwab GET /<path> returned 401" | Token expired mid-flight; one auto-refresh retry will follow. | None. Verify it succeeded by absence of subsequent identical row. |
| "Schwab chain DIAGNOSTIC DUMP" | Diagnostic log from the chain adapter (one-time field-name probe). | None — informational only. |

---

## Going deeper

When this stub stops being enough, expand into the full 8-section runbook format covering:

1. The full decision tree for "I see error code X, what's the chain of causation?"
2. How to determine whether to retry, wait, or escalate.
3. Specific cases where the operator must manually intervene at the broker UI.
4. Cooldown / backoff state that the agent tracks per-error-class.

Most of the patterns above were learned in 2026-05-06 (initial XLF zombie), 2026-05-12 (GLD zombie + Schwab token gap), and 2026-05-13 (cooldown counter bug). New patterns should be added inline as they're encountered.

---

## Cross-references

- **Runbook 03** — zombie position recovery, the most common reason to land on this page.
- **Skill 16** — market-data provider routing (Schwab OAuth refresh policy).
- **Skill 17** — close-failure action + cooldown (when broker errors trigger the cooldown lockout).
- **Skill 18** — order-submission idempotency (`client_order_id` dedup that produces the HTTP 422 "duplicate" code).

---

*Stub created 2026-05-13. Expand to full runbook as new error codes are encountered.*

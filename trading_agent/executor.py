"""
Phase IV — ACT
Submits validated credit-spread orders to Alpaca's paper trading API.
Also handles dry-run logging and trade-plan persistence.

Single-file persistence model
------------------------------
Each ticker gets ONE persistent file:  trade_plans/{TICKER}.json

The file contains a "state_history" array so every run is preserved
without cluttering the directory.  Only the last MAX_HISTORY entries
are kept.  Old timestamped files (trade_plan_{TICKER}_{TS}.json) are
left untouched for backward compatibility with the position monitor.

File structure::

    {
      "ticker":        "AAPL",
      "created":       "2026-04-01T15:00:00+00:00",
      "last_updated":  "2026-04-01T15:58:00+00:00",
      "state_history": [
        {
          "run_id":       "20260401_155800",
          "timestamp":    "2026-04-01T15:58:00+00:00",
          "trade_plan":   { ... },
          "risk_verdict": { ... },
          "mode":         "dry_run",
          "order_result": { ... }   // appended after submission
        },
        ...
      ]
    }
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import requests

from trading_agent.market_data import ALPACA_TIMEOUT_LONG
from trading_agent.strategy import SpreadPlan
from trading_agent.risk_manager import RiskVerdict


# ── Order-submission retry policy ───────────────────────────────────────────
# How many POST /v2/orders attempts before giving up.  At 2 attempts with
# the default 15s timeout, worst-case latency is ~31s plus ORDER_RETRY_BACKOFF_S
# of sleep — well inside the cycle's 270s hard guard.  Each attempt re-uses
# the SAME client_order_id (see ``_submit_order_with_idempotency``) so a
# retry of an order Alpaca already accepted collapses server-side instead
# of producing a duplicate.
ORDER_RETRY_ATTEMPTS = 2
# Brief sleep between order POST retries.  Kept short because the next
# 5-min cycle is only 300s away — we'd rather skip and let the operator
# inspect the broker than block the cycle for 30+ seconds.
ORDER_RETRY_BACKOFF_S = 1.0

if TYPE_CHECKING:
    from trading_agent.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

MAX_HISTORY = 200   # max state_history entries kept per ticker


# ----------------------------------------------------------------------
# Position-sizing primitive — single source of truth
# ----------------------------------------------------------------------
#
# Lifted out of OrderExecutor as a module-level free function so the
# backtester (``trading_agent/backtest/account.py``) can import the
# *exact* same sizing math the live executor uses, with no risk of
# drift.  ``OrderExecutor._calculate_qty`` now delegates here.  This
# keeps the live↔backtest sizing parity invariant honest by construction
# — there is one definition, period.
def calculate_position_qty(plan: SpreadPlan, account_balance: float,
                           max_risk_pct: float,
                           live_credit: Optional[float] = None) -> int:
    """
    Size contracts so the position's total max loss stays within the
    same budget the RiskManager validated against (``max_risk_pct ×
    equity`` — guardrail #4).

    ::

        credit                 = live_credit if provided else plan.net_credit
        max_loss_per_contract  = (spread_width − credit) × 100
        qty                    = floor(max_risk_pct × equity
                                       / max_loss_per_contract)

    Parameters
    ----------
    plan
        The validated ``SpreadPlan`` whose contracts are being sized.
    account_balance
        Current equity (live = Alpaca account balance; backtest =
        ``SimAccount.equity``).
    max_risk_pct
        Per-trade ceiling expressed as a fraction of equity.  Must
        match the value the RiskManager used when approving the plan
        (typically ``RiskManager.max_risk_pct`` or
        ``preset.max_risk_pct``) — otherwise sizing and the guardrail
        would diverge.
    live_credit
        When provided, overrides ``plan.net_credit`` — used at
        submission time to size off the haircut credit the order will
        actually carry, not the stale planning credit.

    Returns
    -------
    int
        Number of contracts.  **0** when no integer quantity fits
        inside the budget (e.g. a single contract's max loss alone
        exceeds the ceiling, or inputs are non-positive).  The caller
        MUST treat 0 as "abort submission" — never silently floor to 1,
        which would otherwise bypass the guardrail.
    """
    credit = live_credit if live_credit is not None else plan.net_credit
    max_loss_per_contract = (plan.spread_width - credit) * 100
    if max_loss_per_contract <= 0 or account_balance <= 0:
        return 0
    max_risk_dollars = account_balance * max_risk_pct
    return int(max_risk_dollars // max_loss_per_contract)


class OrderExecutor:
    """
    Fires multi-leg option orders to Alpaca Paper API, or writes
    them to a per-ticker JSON plan file in dry-run mode.
    """

    # Warn if the live credit deviates from the plan by more than this fraction
    PRICE_DRIFT_WARN_PCT = 0.10   # 10 %

    # Standard option NBBO tick size for limit orders >= $0.05 ($0.01 for
    # penny pilot below $3, but our credit spreads always price above that).
    # Submitting at mid - 1 tick lets the order land aggressively enough to
    # actually fill instead of camping at the un-fillable mid.
    OPTION_TICK = 0.05

    # When live quote refresh fails and we fall back to the synthetic
    # plan.net_credit, apply this haircut before submitting. Synthetic
    # credit comes from Black-Scholes pricing, which assumes zero
    # bid-ask spread; real markets always discount that mid by at least
    # 10-15% on the bid side. Without the haircut the limit lands at an
    # un-fillable price and the order sits open until day-end (Alpaca
    # returns DAY-tif orders). The haircut still has to clear the C/W
    # floor — `_recheck_live_economics` runs after this and rejects the
    # submission if the post-haircut credit drops below the floor.
    FALLBACK_HAIRCUT_PCT = 0.15   # 15 %

    def __init__(self, api_key: str, secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets/v2",
                 trade_plan_dir: str = "trade_plans",
                 dry_run: bool = True,
                 data_provider: Optional["MarketDataProvider"] = None,
                 max_risk_pct: float = 0.02,
                 min_credit_ratio: float = 0.33,
                 *,
                 delta_aware_floor: bool = False,
                 edge_buffer: float = 0.10):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.trade_plan_dir = trade_plan_dir
        self.dry_run = dry_run
        self.data_provider = data_provider   # used for live quote refresh
        # Same ceilings the RiskManager enforces as guardrails #2 and #4.
        # Sizing AND the live-credit re-check share these so planning-time
        # validation and execution-time validation never drift apart.
        self.max_risk_pct = max_risk_pct
        self.min_credit_ratio = min_credit_ratio
        # When True, the live-credit recheck and the haircut guard use a
        # delta-aware floor (|Δshort_max|×(1+edge_buffer)) instead of the
        # static ``min_credit_ratio``. Mirrors RiskManager so an adaptive
        # plan never gets rejected at exec time by a stale static floor.
        self.delta_aware_floor = delta_aware_floor
        self.edge_buffer = edge_buffer
        os.makedirs(self.trade_plan_dir, exist_ok=True)

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, verdict: RiskVerdict) -> Dict:
        """
        Execute a trade plan that has passed risk checks.
        Returns a result dict with order details or dry-run log.
        """
        plan = verdict.plan

        # Save the plan (always) — returns (filepath, run_id)
        plan_path, run_id = self._save_plan(plan, verdict)

        if not verdict.approved:
            logger.warning("[%s] Trade REJECTED by risk manager — skipping.",
                           plan.ticker)
            return {
                "status": "rejected",
                "reason": verdict.summary,
                "plan_file": plan_path,
                "run_id": run_id,
            }

        if self.dry_run:
            logger.info("[%s] DRY RUN — trade plan written to %s",
                        plan.ticker, plan_path)
            return {
                "status": "dry_run",
                "plan_file": plan_path,
                "run_id": run_id,
                "plan": plan.to_dict(),
            }

        # Live paper execution
        return self._submit_order(plan, plan_path, run_id, verdict.account_balance)

    # ------------------------------------------------------------------
    # Alpaca order submission
    # ------------------------------------------------------------------

    def _calculate_qty(self, plan: SpreadPlan, account_balance: float,
                       live_credit: Optional[float] = None) -> int:
        """
        Thin instance-bound delegator to the module-level
        :func:`calculate_position_qty` — kept on the class so existing
        callers (and the unit-test suite that mocks ``executor`` instances)
        continue to work unchanged.

        The math is documented on :func:`calculate_position_qty`.  The
        live executor and the backtest's ``SimAccount`` import the same
        free function so sizing is computed by exactly one piece of code.
        """
        return calculate_position_qty(
            plan, account_balance, self.max_risk_pct, live_credit
        )

    def _recheck_live_economics(self, plan: SpreadPlan,
                                live_credit: float,
                                account_balance: float) -> Tuple[bool, str]:
        """
        Re-validate the two economics-bearing guardrails against live bid/ask.

        Between planning (Phase III) and execution (Phase VI) the net credit
        can drift materially with the book.  Only the market-independent
        checks (market-open, account-type, buying-power, underlying
        liquidity) stay valid; ``credit_to_width_ratio`` and ``max_loss``
        must be recomputed from ``live_credit`` and re-validated against
        the same thresholds the RiskManager uses:

          * guardrail #2  — credit / width  ≥  C/W floor
              - static mode:    ``min_credit_ratio``
              - adaptive mode:  ``|Δshort_max| × (1 + edge_buffer)``
                (mirrors RiskManager Check 2 so a scanner-picked plan
                never trips a stale static floor at execution time)
          * guardrail #4  — max_loss (per contract) ≤ ``max_risk_pct × equity``

        Returns
        -------
        (ok, reason)
            ``ok`` is True when both checks pass; ``reason`` is empty on
            success, else a short human-readable "live_credit_risk: ..."
            string describing the first failure.
        """
        width = plan.spread_width
        if width <= 0:
            return (False,
                    f"live_credit_risk: spread_width {width} is non-positive")

        # Resolve C/W floor — must match RiskManager Check 2 exactly so
        # planning-time validation and execution-time recheck never drift.
        if self.delta_aware_floor and plan.legs:
            short_legs = [l for l in plan.legs if l.action == "sell"]
            short_max_delta = (max(abs(l.delta) for l in short_legs)
                               if short_legs else 0.0)
            cw_floor = short_max_delta * (1.0 + self.edge_buffer)
            floor_label = (f"|Δ|×(1+edge)={short_max_delta:.3f}×"
                           f"{1+self.edge_buffer:.2f}={cw_floor:.4f}")
        else:
            cw_floor = self.min_credit_ratio
            floor_label = f"{self.min_credit_ratio}"

        live_ratio = live_credit / width
        if live_ratio < cw_floor:
            return (False,
                    f"live_credit_risk: credit/width {live_ratio:.4f} < "
                    f"{floor_label} "
                    f"(live_credit=${live_credit:.2f}, width=${width:.2f}, "
                    f"planning ratio was {plan.credit_to_width_ratio:.4f})")

        live_max_loss = (width - live_credit) * 100
        max_allowed = account_balance * self.max_risk_pct
        if live_max_loss > max_allowed:
            return (False,
                    f"live_credit_risk: max_loss ${live_max_loss:.2f} > "
                    f"{self.max_risk_pct*100:.0f}% × ${account_balance:,.2f} "
                    f"(=${max_allowed:.2f}) "
                    f"(live_credit=${live_credit:.2f}, "
                    f"planning max_loss was ${plan.max_loss:.2f})")

        return (True, "")

    def _submit_order(self, plan: SpreadPlan, plan_path: str,
                      run_id: str, account_balance: float = 0.0) -> Dict:
        """
        Submit a multi-leg option order to Alpaca.
        Uses POST /v2/orders with order_class='mleg'.

        Alpaca mleg payload format (from API docs):
        - Top-level: type, time_in_force, order_class, qty, limit_price
        - Legs array: symbol, ratio_qty (string), side, position_intent
        - limit_price: string, NEGATIVE for credit, POSITIVE for debit
        - No top-level 'side' field for mleg orders
        """
        legs_payload = []
        for leg in plan.legs:
            if leg.action == "sell":
                position_intent = "sell_to_open"
                side = "sell"
            else:
                position_intent = "buy_to_open"
                side = "buy"

            legs_payload.append({
                "symbol": leg.symbol,
                "ratio_qty": "1",
                "side": side,
                "position_intent": position_intent,
            })

        # Refresh bid/ask from live market right before sending the order.
        # The option chain was fetched during Phase III (planning); by now
        # seconds-to-minutes may have passed.  Use fresh quotes so the
        # limit_price reflects what the market is actually offering.
        live_credit = self._refresh_limit_price(plan)
        if live_credit is None:
            # Quote fetch failed → apply FALLBACK_HAIRCUT_PCT to the
            # synthetic plan credit so the limit price is more aligned
            # with what a real market maker is willing to pay. The
            # original behaviour (use plan.net_credit verbatim) priced
            # too rich and orders sat open at the limit until DAY-tif
            # expiry. The post-haircut credit is still re-validated
            # against the C/W floor below — a haircut that breaches
            # the floor causes the submission to be rejected, not
            # silently downgraded.
            haircut_credit = round(
                plan.net_credit * (1.0 - self.FALLBACK_HAIRCUT_PCT), 2)
            logger.warning(
                "[%s] Quote refresh failed — applying %.0f%% fallback "
                "haircut: planned $%.2f → submit $%.2f (so the limit "
                "lands closer to typical bid).",
                plan.ticker, self.FALLBACK_HAIRCUT_PCT * 100,
                plan.net_credit, haircut_credit)
            live_credit = haircut_credit
        else:
            drift = abs(live_credit - plan.net_credit)
            drift_pct = drift / plan.net_credit if plan.net_credit else 0
            if drift_pct > self.PRICE_DRIFT_WARN_PCT:
                logger.warning(
                    "[%s] Credit drifted %.1f%% since planning "
                    "(plan=$%.2f → live=$%.2f)",
                    plan.ticker, drift_pct * 100,
                    plan.net_credit, live_credit)
            else:
                logger.info("[%s] Live credit $%.2f (plan was $%.2f)",
                            plan.ticker, live_credit, plan.net_credit)

        # Re-validate the economics-bearing guardrails against LIVE credit.
        # The RiskManager approved this plan at planning time using
        # plan.net_credit; if the bid/ask has drifted materially the
        # credit-to-width or max-loss checks may no longer hold.  The
        # other guardrails (market-open, paper, buying-power, underlying
        # liquidity) are environment-dependent and haven't changed.
        ok, recheck_reason = self._recheck_live_economics(
            plan, live_credit, account_balance)
        if not ok:
            logger.error("[%s] Live-credit risk recheck FAILED — %s. "
                         "Aborting order submission.",
                         plan.ticker, recheck_reason)
            result = {
                "status": "rejected",
                "reason": recheck_reason,
                "plan_file": plan_path,
                "run_id": run_id,
            }
            self._append_to_plan(plan_path, run_id, {"order_result": result})
            return result

        # ------------------------------------------------------------------
        # 1-tick "fill haircut": mid-quote credit limits almost never fill
        # in practice — the market makers want a penny, and on a 7-DTE
        # spread theta drains the mid past our static limit within hours.
        # Try shaving one tick ($0.05) off the credit so the limit lands
        # below mid; only do so if the haircut credit still passes the
        # economics floors (C/W ratio + max-loss).  If the haircut breaks
        # the floor, fall back to mid with a warning so we don't violate
        # risk guardrails just to chase a fill.
        # ------------------------------------------------------------------
        haircut_credit = round(live_credit - self.OPTION_TICK, 2)
        submit_credit = live_credit
        if haircut_credit > 0:
            ok_haircut, haircut_reason = self._recheck_live_economics(
                plan, haircut_credit, account_balance)
            if ok_haircut:
                submit_credit = haircut_credit
                logger.info(
                    "[%s] Applying 1-tick fill haircut: limit credit "
                    "$%.2f (live mid $%.2f − $%.2f tick)",
                    plan.ticker, submit_credit, live_credit,
                    self.OPTION_TICK)
            else:
                logger.warning(
                    "[%s] 1-tick haircut would breach guardrails (%s) — "
                    "submitting at live mid $%.2f instead. Order may not "
                    "fill quickly.",
                    plan.ticker, haircut_reason, live_credit)
        else:
            logger.warning(
                "[%s] Live credit $%.2f too small to apply 1-tick "
                "haircut without going non-positive — submitting at mid.",
                plan.ticker, live_credit)

        # Alpaca sign convention: credit → negative limit_price
        limit_price_value = -abs(submit_credit)

        # Size off the credit we're ACTUALLY submitting (post-haircut).
        # qty must reflect the economics on the wire, not the un-haircut
        # mid, so risk sizing stays consistent with what fills.
        qty = self._calculate_qty(plan, account_balance, live_credit=submit_credit)
        if qty < 1:
            max_loss_per_contract = (plan.spread_width - submit_credit) * 100
            max_risk_dollars = account_balance * self.max_risk_pct
            reason = (
                f"qty=0: max_loss_per_contract ${max_loss_per_contract:.2f} "
                f"> sizing budget ${max_risk_dollars:.2f} "
                f"({self.max_risk_pct*100:.0f}% × ${account_balance:,.2f}) "
                f"(submit_credit=${submit_credit:.2f})"
            )
            logger.error(
                "[%s] Position sizing produced qty=0 — %s. Aborting order "
                "submission rather than silently flooring to 1 contract.",
                plan.ticker, reason)
            result = {
                "status": "rejected",
                "reason": reason,
                "plan_file": plan_path,
                "run_id": run_id,
            }
            self._append_to_plan(plan_path, run_id, {"order_result": result})
            return result

        logger.info(
            "[%s] Position size: %d contract(s) "
            "(max_risk_pct=%.0f%%, equity=$%.2f)",
            plan.ticker, qty, self.max_risk_pct * 100, account_balance)

        # ── Idempotency key ───────────────────────────────────────────────
        # Generate ONCE per plan.  Re-used across every retry of THIS POST
        # so that if Alpaca accepted the first request but the response
        # was lost on the wire, the second POST collapses server-side
        # instead of producing a duplicate spread.  The key is also
        # written into the trade_plan_*.json so an operator can search
        # `client_order_id=<uuid>` in the broker UI to confirm whether a
        # specific submission ever landed.
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

        logger.info("[%s] Submitting %s order to Alpaca (client_order_id=%s): %s",
                     plan.ticker, plan.strategy_name, client_order_id,
                     json.dumps(order_payload, indent=2))

        return self._submit_order_with_idempotency(
            plan=plan,
            plan_path=plan_path,
            run_id=run_id,
            order_payload=order_payload,
            client_order_id=client_order_id,
        )

    def _submit_order_with_idempotency(
        self, plan: SpreadPlan, plan_path: str, run_id: str,
        order_payload: Dict, client_order_id: str,
    ) -> Dict:
        """
        POST the order with up to ORDER_RETRY_ATTEMPTS attempts, re-using
        the same client_order_id so the broker collapses duplicates.

        Why this matters
        ----------------
        Without an idempotency key, a network blip after Alpaca accepts
        the order — but before we read the 200 response — produces a
        ConnectTimeout / ReadTimeout on our side.  A naive retry then
        submits a SECOND identical spread, doubling exposure.  Alpaca
        treats ``client_order_id`` as a server-side dedup key: a second
        request with the same ID returns the original order instead of
        creating a new one.

        Retry conditions
        ----------------
        We retry only on transient transport errors that DIDN'T produce
        a successful HTTP response — connection timeouts, read timeouts,
        connection reset.  4xx responses (validation, insufficient
        buying power, PDT) are NOT retried — they're permanent at the
        broker level.

        Failure surface
        ---------------
        After all attempts exhaust we return ``status="error"`` with the
        client_order_id and ``retry_attempts`` count so the operator can
        confirm with the broker whether ANY attempt landed (the broker
        UI accepts ``client_order_id`` as a search filter).
        """
        last_error: Optional[str] = None
        last_resp_body = None

        for attempt in range(1, ORDER_RETRY_ATTEMPTS + 1):
            resp_body = None
            try:
                resp = requests.post(
                    f"{self.base_url}/orders",
                    headers=self._headers(),
                    json=order_payload,
                    timeout=ALPACA_TIMEOUT_LONG,
                )

                try:
                    resp_body = resp.json()
                except Exception:
                    resp_body = resp.text

                # 4xx is permanent at the broker — don't retry.  The
                # broker has already evaluated and rejected the order
                # (validation, buying power, PDT) and a retry will be
                # rejected the same way.  raise_for_status produces a
                # RequestException that we catch in the 4xx branch
                # below (and bail out).
                resp.raise_for_status()

                order_id = (resp_body.get("id", "unknown")
                            if isinstance(resp_body, dict) else "unknown")
                logger.info(
                    "[%s] Order SUBMITTED (attempt %d/%d) — ID: %s "
                    "(client_order_id=%s)",
                    plan.ticker, attempt, ORDER_RETRY_ATTEMPTS,
                    order_id, client_order_id,
                )

                result = {
                    "status": "submitted",
                    "order_id": order_id,
                    "client_order_id": client_order_id,
                    "retry_attempts": attempt,
                    "plan_file": plan_path,
                    "run_id": run_id,
                    "alpaca_response": resp_body,
                }
                self._append_to_plan(plan_path, run_id, {"order_result": result})
                return result

            except requests.RequestException as exc:
                last_error = str(exc)
                last_resp_body = resp_body

                # Pull the broker error detail into the log so the operator
                # can tell apart "Alpaca rejected for buying power" from
                # "the connection timed out".
                if resp_body is not None:
                    logger.error(
                        "[%s] Alpaca response body (attempt %d/%d): %s",
                        plan.ticker, attempt, ORDER_RETRY_ATTEMPTS, resp_body,
                    )
                    last_error += f" | Alpaca detail: {resp_body}"
                elif hasattr(exc, "response") and exc.response is not None:
                    try:
                        detail = exc.response.json()
                        logger.error(
                            "[%s] Alpaca response body (attempt %d/%d): %s",
                            plan.ticker, attempt, ORDER_RETRY_ATTEMPTS, detail,
                        )
                        last_error += f" | Alpaca detail: {detail}"
                    except Exception:
                        raw = exc.response.text
                        logger.error(
                            "[%s] Alpaca raw response (attempt %d/%d): %s",
                            plan.ticker, attempt, ORDER_RETRY_ATTEMPTS, raw,
                        )
                        last_error += f" | Alpaca raw: {raw}"

                # Permanent-error short-circuit: a 4xx HTTP response means
                # Alpaca evaluated the order and refused it.  Retrying with
                # the same payload (same client_order_id) will not change
                # the answer.  Skip directly to the failure return below.
                resp_obj = getattr(exc, "response", None)
                if resp_obj is not None and 400 <= resp_obj.status_code < 500:
                    logger.error(
                        "[%s] Order rejected by Alpaca (HTTP %d, attempt %d/%d) "
                        "— permanent at broker level, NOT retrying. "
                        "client_order_id=%s",
                        plan.ticker, resp_obj.status_code, attempt,
                        ORDER_RETRY_ATTEMPTS, client_order_id,
                    )
                    break

                # Transient transport error — retry with the SAME
                # client_order_id so any duplicate submission collapses
                # server-side at Alpaca.
                if attempt < ORDER_RETRY_ATTEMPTS:
                    logger.warning(
                        "[%s] Transient order POST failure (attempt %d/%d): "
                        "%s. Retrying in %.1fs with same client_order_id=%s "
                        "(broker will dedupe if it already accepted).",
                        plan.ticker, attempt, ORDER_RETRY_ATTEMPTS,
                        exc, ORDER_RETRY_BACKOFF_S, client_order_id,
                    )
                    time.sleep(ORDER_RETRY_BACKOFF_S)
                    continue

        # All attempts exhausted (or 4xx short-circuit).  Surface a
        # detailed error result so the operator can search Alpaca by
        # client_order_id to confirm whether the order ever landed.
        logger.error(
            "[%s] Order FAILED after %d attempt(s): %s "
            "(client_order_id=%s — search Alpaca to confirm whether ANY "
            "attempt landed)",
            plan.ticker, ORDER_RETRY_ATTEMPTS, last_error, client_order_id,
        )
        result = {
            "status": "error",
            "error": last_error or "unknown error",
            "client_order_id": client_order_id,
            "retry_attempts": ORDER_RETRY_ATTEMPTS,
            "plan_file": plan_path,
            "run_id": run_id,
        }
        self._append_to_plan(plan_path, run_id, {"order_error": result})
        return result

    def _refresh_limit_price(self, plan: SpreadPlan) -> Optional[float]:
        """
        Fetch live bid/ask for the plan's leg symbols and recalculate
        net credit using the same bid-for-sold / ask-for-bought convention
        used during planning.

        Returns the fresh net credit, or None if the fetch fails.
        """
        if self.data_provider is None:
            return None

        symbols = [leg.symbol for leg in plan.legs]
        quotes = self.data_provider.fetch_option_quotes(symbols)
        if not quotes:
            return None

        # Identify sold vs bought legs and recalculate credit
        total_credit = 0.0
        for leg in plan.legs:
            q = quotes.get(leg.symbol)
            if q is None:
                logger.warning("[%s] No live quote for %s — aborting refresh",
                               plan.ticker, leg.symbol)
                return None
            if leg.action == "sell":
                total_credit += q["bid"]   # receive the bid
            else:
                total_credit -= q["ask"]   # pay the ask

        return round(total_credit, 2)

    # ------------------------------------------------------------------
    # Close positions
    # ------------------------------------------------------------------

    def close_spread(self, spread) -> Dict:
        """
        Close an open credit spread.
        Uses DELETE /v2/positions/{symbol} for each leg individually.

        Leg ordering — close shorts before longs
        ----------------------------------------
        Alpaca evaluates "is the account uncovered?" between each leg
        DELETE.  If we close the long-call hedge first, the remaining
        short call becomes momentarily *naked*, and Alpaca rejects with
        ``account not eligible to trade uncovered option contracts``
        (or "insufficient buying power for cash-secured put" on a
        put-side leg).

        On 2026-05-06 this produced a flurry of red ERROR lines mid-
        close — the legs eventually closed (the executor retries until
        Alpaca accepts) but the log was noisy and the operator couldn't
        tell at a glance whether the close had succeeded.

        Sorting legs ascending by qty puts shorts (qty<0) first, longs
        (qty>0) last.  Each intermediate state then has *more* longs
        than shorts (or equal), so the account is never naked and
        Alpaca accepts each DELETE on the first try.
        """
        ordered_legs = sorted(
            spread.legs,
            key=lambda leg: getattr(leg, "qty", 0),
        )
        results = []
        for leg in ordered_legs:
            results.append(self._close_single_leg(leg.symbol))

        all_ok = all(r.get("status") == "closed" for r in results)
        summary = {
            "action": "close_spread",
            "underlying": spread.underlying,
            "strategy": spread.strategy_name,
            "signal": spread.exit_signal.value,
            "reason": spread.exit_reason,
            "leg_results": results,
            "all_closed": all_ok,
        }

        if all_ok:
            logger.info("[%s] Spread CLOSED successfully (%s)",
                        spread.underlying, spread.exit_signal.value)
        else:
            logger.warning("[%s] Spread close PARTIAL — some legs failed",
                           spread.underlying)

        return summary

    def _close_single_leg(self, symbol: str) -> Dict:
        """DELETE /v2/positions/{symbol} — close a single option leg."""
        url = f"{self.base_url}/positions/{symbol}"
        resp_body = None
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=ALPACA_TIMEOUT_LONG)
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = resp.text
            resp.raise_for_status()
            logger.info("Closed position: %s", symbol)
            return {"status": "closed", "symbol": symbol, "response": resp_body}

        except requests.RequestException as exc:
            error_msg = str(exc)
            if resp_body is not None:
                logger.error("Close %s response: %s", symbol, resp_body)
                error_msg += f" | Detail: {resp_body}"
            logger.error("Failed to close position %s: %s", symbol, error_msg)
            return {"status": "error", "symbol": symbol, "error": error_msg}

    # ------------------------------------------------------------------
    # Plan file management  — single persistent file per ticker
    # ------------------------------------------------------------------

    def _save_plan(self, plan: SpreadPlan,
                   verdict: RiskVerdict) -> tuple[str, str]:
        """
        Persist the trade plan + risk verdict to a single per-ticker file.

        Returns
        -------
        (filepath, run_id)
        """
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%d_%H%M%S")
        ts = now.isoformat()

        filepath = os.path.join(self.trade_plan_dir,
                                f"trade_plan_{plan.ticker}.json")

        # Load existing file or initialise fresh
        try:
            with open(filepath) as fh:
                persistent = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            persistent = {
                "ticker": plan.ticker,
                "created": ts,
                "state_history": [],
            }

        entry = {
            "run_id": run_id,
            "timestamp": ts,
            "trade_plan": plan.to_dict(),
            "risk_verdict": {
                "approved": verdict.approved,
                "account_balance": verdict.account_balance,
                "max_allowed_loss": verdict.max_allowed_loss,
                "checks_passed": verdict.checks_passed,
                "checks_failed": verdict.checks_failed,
                "summary": verdict.summary,
            },
            "mode": "dry_run" if self.dry_run else "live",
        }

        persistent["last_updated"] = ts
        persistent["state_history"].append(entry)

        # Trim to keep only the most recent MAX_HISTORY runs
        if len(persistent["state_history"]) > MAX_HISTORY:
            persistent["state_history"] = (
                persistent["state_history"][-MAX_HISTORY:]
            )

        with open(filepath, "w") as fh:
            json.dump(persistent, fh, indent=2)

        logger.info("Trade plan saved to %s (run_id=%s, history=%d)",
                    filepath, run_id, len(persistent["state_history"]))

        # Auto-generate companion HTML report
        try:
            from trading_agent.trade_plan_report import generate_report
            html_path = generate_report(filepath)
            logger.debug("HTML report updated: %s", html_path)
        except Exception as exc:
            logger.debug("HTML report generation skipped: %s", exc)

        return filepath, run_id

    def _append_to_plan(self, filepath: str, run_id: str, data: Dict) -> None:
        """
        Merge *data* into the state_history entry matching *run_id*.
        Falls back to updating the last entry if run_id is not found.
        """
        try:
            with open(filepath) as fh:
                persistent = json.load(fh)

            history = persistent.get("state_history", [])
            target = next(
                (e for e in reversed(history) if e.get("run_id") == run_id),
                history[-1] if history else None,
            )
            if target is not None:
                target.update(data)

            persistent["last_updated"] = datetime.now(timezone.utc).isoformat()

            with open(filepath, "w") as fh:
                json.dump(persistent, fh, indent=2)

        except Exception as exc:
            logger.error("Failed to update plan file %s: %s", filepath, exc)

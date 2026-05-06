#!/usr/bin/env python3
"""
cancel_open_options_orders.py — clear out unfilled option orders on Alpaca.

Why this exists
---------------
When the executor's live-quote refresh fails it falls back to the
synthetic ``plan.net_credit``, which is computed from Black-Scholes
mid-pricing and is essentially always richer than what real market
makers will pay. The result is a multi-leg limit order that sits at
an un-fillable price for the rest of the trading day (DAY tif).
``executor.py`` now applies a ``FALLBACK_HAIRCUT_PCT`` to bring the
limit closer to typical bid, but any orders submitted *before* that fix
are still parked at the old rich prices.

This script lists every currently-open option order on the configured
Alpaca account, prints them in a readable table, and asks for explicit
confirmation before cancelling. It's safe to run repeatedly — Alpaca
returns 422 (already-closed) on a second cancel of the same order ID,
which we treat as a no-op.

Usage
-----
    python scripts/cancel_open_options_orders.py             # interactive
    python scripts/cancel_open_options_orders.py --yes       # no prompt
    python scripts/cancel_open_options_orders.py --dry-run   # list only

The script reads ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL
from the project's ``.env`` (same loader the agent uses), so it only
operates on whichever account the agent is wired to. It does NOT touch
filled positions — only resting (open) orders.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

# Read Alpaca creds directly from .env to avoid importing
# trading_agent.config (which transitively pulls in pandas_market_calendars
# via the calendar layer — heavy and unnecessary for an HTTP-only script).
_repo_root = Path(__file__).resolve().parents[1]


def _load_env_file(env_path: Path) -> None:
    """Best-effort .env loader — populates os.environ for KEY=VALUE lines.

    Skips comments and blank lines. Strips surrounding quotes if present.
    Does NOT use python-dotenv so the script can run on stock Python with
    only ``requests`` available.
    """
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split("#", 1)[0].strip()  # strip inline comment
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(_repo_root / ".env")


def _is_option_symbol(sym: str) -> bool:
    """OCC symbols look like ``GLD260515C00415000`` — root + 6-char date +
    C/P + 8-digit strike. Heuristic: ≥15 chars and contains a 'C' or 'P'
    after the date prefix."""
    return len(sym) >= 15 and any(c in sym[-9:] for c in "CP")


def list_open_orders(base_url: str, headers: dict) -> list:
    """GET /v2/orders?status=open — full list of resting orders."""
    resp = requests.get(
        f"{base_url}/orders",
        params={"status": "open", "nested": "true", "limit": 500},
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def cancel_order(base_url: str, headers: dict, order_id: str) -> bool:
    """DELETE /v2/orders/{id} — cancel a single order. Returns True on
    success or already-cancelled, False on error."""
    resp = requests.delete(
        f"{base_url}/orders/{order_id}",
        headers=headers,
        timeout=15,
    )
    if resp.status_code in (200, 204):
        return True
    if resp.status_code == 422:
        # Order already in a terminal state (filled / cancelled / expired).
        # Treat as no-op — nothing left to do.
        return True
    print(f"  [!] DELETE {order_id} → HTTP {resp.status_code}: "
          f"{resp.text[:200]}")
    return False


def _format_legs(order: dict) -> str:
    """Render the leg list for a multi-leg order in one line."""
    legs = order.get("legs") or []
    if not legs:
        return order.get("symbol", "?")
    parts = []
    for leg in legs:
        side = leg.get("side", "?")[0].upper()  # B / S
        sym = leg.get("symbol", "?")
        qty = leg.get("qty", "?")
        parts.append(f"{side}×{qty} {sym}")
    return " · ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List candidate orders but do not cancel.")
    parser.add_argument("--include-stocks", action="store_true",
                        help="Also cancel non-option open orders. Default: "
                             "options only (multi-leg or OCC-shaped symbols).")
    args = parser.parse_args()

    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    base_url   = os.environ.get(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2",
    )
    if not api_key or not secret_key:
        print("[!] ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env",
              file=sys.stderr)
        return 2
    headers = {
        "APCA-API-KEY-ID":     api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type":        "application/json",
    }

    print(f"Endpoint: {base_url}")
    print(f"Mode:     {'DRY-RUN (no cancellations)' if args.dry_run else 'LIVE'}")
    print()

    try:
        orders = list_open_orders(base_url, headers)
    except requests.RequestException as exc:
        print(f"[!] Failed to list orders: {exc}", file=sys.stderr)
        return 2

    # Filter to option orders unless --include-stocks. A multi-leg order
    # has legs[]; otherwise we sniff the symbol.
    candidates = []
    for o in orders:
        legs = o.get("legs") or []
        symbol = o.get("symbol", "")
        is_option = bool(legs) or _is_option_symbol(symbol)
        if args.include_stocks or is_option:
            candidates.append(o)

    if not candidates:
        print("No open option orders. Nothing to cancel.")
        return 0

    print(f"{len(candidates)} open order(s):")
    print(f"  {'ID':<37s} {'Submitted':<26s} {'Type':<6s} "
          f"{'Limit':>8s}  Legs")
    for o in candidates:
        oid = o.get("id", "?")
        submitted = o.get("submitted_at", "")[:25]
        otype = o.get("type", "?")
        limit = o.get("limit_price") or o.get("filled_avg_price") or "—"
        print(f"  {oid:<37s} {submitted:<26s} {otype:<6s} "
              f"{str(limit):>8s}  {_format_legs(o)}")
    print()

    if args.dry_run:
        print("(Dry-run — exiting without cancellation.)")
        return 0

    if not args.yes:
        try:
            answer = input(
                f"Cancel all {len(candidates)} order(s)? [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    cancelled = 0
    failed = 0
    for o in candidates:
        oid = o.get("id", "?")
        if cancel_order(base_url, headers, oid):
            cancelled += 1
            print(f"  [✓] Cancelled {oid}")
        else:
            failed += 1
    print()
    print(f"Done. Cancelled: {cancelled}, Failed: {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

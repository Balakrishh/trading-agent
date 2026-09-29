"""CLI entry: ``python -m trading_agent.daily_reviewer_main`` (skill 56).

Runs the daily reviewer and prints the returned digest to stdout so
launchd's stderr/stdout redirection captures it in the operator log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from trading_agent.daily_reviewer import run


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trading_agent.daily_reviewer_main",
        description="End-of-day trade journal reviewer (skill 56).",
    )
    parser.add_argument("--date", default=None,
                        help="ISO date YYYY-MM-DD to review. Defaults to today.")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Skip the Telegram send (useful for backfills).")
    parser.add_argument("--json", action="store_true",
                        help="Print the returned summary as JSON.")
    args = parser.parse_args(argv)

    if not _env_bool("TRADING_AGENT_DAILY_REVIEWER_ENABLED", default=False):
        print("TRADING_AGENT_DAILY_REVIEWER_ENABLED is not set — exiting cleanly.",
              file=sys.stderr)
        return 0

    result = run(review_date=args.date,
                 send_telegram=not args.no_telegram)
    if args.json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(result["digest"])
        print("")
        print(f"[audit] {result['audit_path']}")
        if result.get("proposal_id"):
            print(f"[proposal] {result['proposal_id']}")
        print(f"[telegram] sent={result['telegram_sent']}")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main())

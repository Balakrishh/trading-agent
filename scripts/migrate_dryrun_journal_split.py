"""Migrate signals_live.jsonl → split dry-run rows into signals_dryrun.jsonl.

Skill 19 §1 decoupling (2026-05-21). Before this change, dry-run
cycles wrote into ``trade_journal/signals_live.jsonl`` with a
``mode="DRY-RUN"`` tag. After this change, dry-run cycles write to
``trade_journal/signals_dryrun.jsonl``. Production consumers
(EOD Telegram recap, dashboard realized-P&L tile,
_render_closed_today) read only ``signals_live.jsonl`` — they MUST
NOT see the historical dry-run rows that pre-date this split.

This script:

  1. Reads every row of ``signals_live.jsonl``.
  2. Classifies each row as ``LIVE`` or ``DRYRUN`` using these
     signals (any one is sufficient to tag a row as DRYRUN):
       * ``raw_signal.mode == "DRY-RUN"``
       * ``action == "dry_run"``
       * ``action == "dry_run_close"``
       * ``raw_signal.fill_status == "dry_run"``
       * ``action`` starts with ``"defensive_roll_dry_run"`` (rare)
  3. Writes:
       * ``signals_live.jsonl``   ← cleaned, LIVE-only rows
       * ``signals_dryrun.jsonl`` ← extracted DRYRUN rows
     Preserving original line order within each file.
  4. Backs up the original to
     ``signals_live.jsonl.pre-split.<UTC-isotimestamp>`` so the
     migration is reversible.

Idempotent: re-running on already-split files is safe (the LIVE
file no longer contains DRYRUN rows, so the dryrun output stays
empty and the LIVE file is rewritten unchanged).

Usage::

    python scripts/migrate_dryrun_journal_split.py
    # → reads ./trade_journal/signals_live.jsonl

    python scripts/migrate_dryrun_journal_split.py --dir /some/other/path
    # → reads /some/other/path/signals_live.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

DEFAULT_JOURNAL_DIR = Path("trade_journal")
LIVE_NAME    = "signals_live.jsonl"
DRYRUN_NAME  = "signals_dryrun.jsonl"


def _is_dryrun_row(rec: dict) -> bool:
    """Classify a journal row as DRYRUN (True) or LIVE (False).

    Conservative: a row is DRYRUN only if at least one of the
    canonical signals fires. Untagged rows (legacy data pre-dating
    the mode-tagging change) are kept on the LIVE side — that's the
    behavior the original mode-filter assumed, and we preserve it.
    """
    action = (rec.get("action") or "")
    rs = rec.get("raw_signal") or {}
    if not isinstance(rs, dict):
        return False
    if (rs.get("mode") or "").upper() == "DRY-RUN":
        return True
    if action in ("dry_run", "dry_run_close"):
        return True
    if action.startswith("defensive_roll_dry_run"):
        return True
    if (rs.get("fill_status") or "").lower() == "dry_run":
        return True
    return False


def _split_lines(
    src_lines: Iterable[str],
) -> Tuple[list, list, int, int]:
    """Walk lines, return (live_rows, dryrun_rows, parse_errors, dryrun_count)."""
    live, dryrun = [], []
    parse_errors = 0
    for raw in src_lines:
        line = raw.rstrip("\n")
        if not line.strip():
            # Preserve blank lines on the LIVE side so the file
            # structure (very rare) stays bit-identical for the
            # caller's other tooling.
            live.append(line)
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # Don't lose lines we can't parse — keep them on the
            # LIVE side and let the operator decide.
            live.append(line)
            parse_errors += 1
            continue
        if _is_dryrun_row(rec):
            dryrun.append(line)
        else:
            live.append(line)
    return live, dryrun, parse_errors, len(dryrun)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split dry-run rows out of signals_live.jsonl"
    )
    parser.add_argument(
        "--dir", default=str(DEFAULT_JOURNAL_DIR),
        help=f"journal directory (default: {DEFAULT_JOURNAL_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report counts but don't write any files (preview mode)",
    )
    args = parser.parse_args()

    journal_dir = Path(args.dir)
    live_path = journal_dir / LIVE_NAME
    dryrun_path = journal_dir / DRYRUN_NAME

    if not live_path.exists():
        print(f"ERR: {live_path} not found — nothing to migrate.",
              file=sys.stderr)
        return 1

    print(f"Reading {live_path} …")
    with open(live_path, "r", encoding="utf-8") as fh:
        live_rows, dryrun_rows, parse_errors, dryrun_count = _split_lines(fh)

    total = len(live_rows) + dryrun_rows.__len__()
    print(f"  total rows scanned:  {total}")
    print(f"  → LIVE   (kept):     {len(live_rows)}")
    print(f"  → DRYRUN (extracted): {dryrun_count}")
    print(f"  unparseable lines (kept on LIVE side): {parse_errors}")

    if args.dry_run:
        print("\n--dry-run mode: no files written. Re-run without "
              "--dry-run to apply.")
        return 0

    if dryrun_count == 0:
        print("\nNothing to migrate — signals_live.jsonl has no "
              "dry-run rows. No-op.")
        return 0

    # ── Backup ──────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = live_path.with_suffix(
        f".jsonl.pre-split.{ts}"
    )
    print(f"\nBacking up original → {backup_path}")
    live_path.rename(backup_path)

    # ── Write LIVE-only file ────────────────────────────────────────
    print(f"Writing cleaned LIVE rows → {live_path}")
    with open(live_path, "w", encoding="utf-8") as fh:
        for line in live_rows:
            fh.write(line + "\n")

    # ── Append/extend the dryrun file ───────────────────────────────
    # Append so re-runs accumulate any new dryrun rows that snuck
    # back into signals_live.jsonl (shouldn't happen post-deploy,
    # but defensive).
    mode = "a" if dryrun_path.exists() else "w"
    print(f"Writing extracted DRYRUN rows → {dryrun_path} (mode={mode})")
    with open(dryrun_path, mode, encoding="utf-8") as fh:
        for line in dryrun_rows:
            fh.write(line + "\n")

    print()
    print(f"OK. Operator action:")
    print(f"  • Confirm dashboard now shows clean LIVE-only data.")
    print(f"  • Keep {backup_path.name} until you've verified the new state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

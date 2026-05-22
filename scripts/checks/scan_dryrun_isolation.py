"""SDD check — production code never reads signals_dryrun.jsonl.

Skill 19 §1 (decoupling, 2026-05-21) — dry-run cycles write to a
SEPARATE journal file so live consumers (Telegram EOD recap,
dashboard realized-P&L tile, _render_closed_today) cannot be
polluted by dry-run synthetic events.

The structural guarantee: production source files read ONLY
``signals_live.jsonl``. The only paths allowed to reference
``signals_dryrun.jsonl`` are:

  * ``trading_agent/journal_kb.py`` (the writer; constructs the
    path via ``f"signals_{run_mode}.jsonl"``)
  * ``trading_agent/agent.py`` (constructs JournalKB with
    ``run_mode="dryrun"`` for dry-run cycles)
  * ``scripts/migrate_dryrun_journal_split.py`` (one-off migration
    helper, not part of the running agent)
  * ``tests/`` (test fixtures may stage either file)
  * ``docs/`` (skill documentation)
  * ``scripts/checks/scan_dryrun_isolation.py`` (this script
    documents the rule)

Any other source file referencing ``signals_dryrun.jsonl`` is a
coupling violation — fix the source file (use ``signals_live.jsonl``
explicitly OR consume the journal through ``JournalKB.jsonl_path``)
or add the file to the ALLOWED set below with a justifying comment.

Exit code 0 when the invariant holds, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files allowed to mention signals_dryrun.jsonl — see module docstring.
ALLOWED_PATHS = {
    "trading_agent/journal_kb.py",
    "trading_agent/agent.py",
    # journal_reader.py is a read-only utility — its docstring mentions
    # both filenames to document the design intent (single class can be
    # pointed at either file), but no concrete production call site
    # passes the dryrun path. Production code that USES this reader
    # is the surface the isolation gate actually wants to constrain.
    "trading_agent/journal_reader.py",
    "scripts/migrate_dryrun_journal_split.py",
    "scripts/checks/scan_dryrun_isolation.py",
}

# Roots to scan. Tests, docs, and one-off scripts are excluded
# because they're allowed to mention the file by design.
SCAN_ROOTS = ("trading_agent",)

# Pattern: any literal mention of signals_dryrun.jsonl (also catches
# variants like signals_dryrun.md just in case someone reads the
# markdown variant by hand).
PATTERN = re.compile(r"signals_dryrun\.(jsonl|md)")


def main() -> int:
    violations: list[tuple[Path, int, str]] = []
    for root in SCAN_ROOTS:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for py in root_path.rglob("*.py"):
            try:
                text = py.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not PATTERN.search(text):
                continue
            rel = py.relative_to(REPO_ROOT).as_posix()
            if rel in ALLOWED_PATHS:
                continue
            # Find each violating line for the operator report
            for n, line in enumerate(text.splitlines(), start=1):
                if PATTERN.search(line):
                    violations.append((py, n, line.strip()))

    if not violations:
        print("SDD dry-run isolation — production code never "
              "references signals_dryrun.jsonl. ✓")
        return 0

    print("SDD dry-run isolation — VIOLATIONS:", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Production source files reference signals_dryrun.jsonl.\n"
        "Dry-run + live journals are physically separate by design "
        "(skill 19 §1). Production code should consume only "
        "signals_live.jsonl OR access the journal through a "
        "JournalKB instance's .jsonl_path attribute.",
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    for path, lineno, line in violations:
        rel = path.relative_to(REPO_ROOT).as_posix()
        print(f"  {rel}:{lineno}", file=sys.stderr)
        print(f"    {line[:120]}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "If the new code path legitimately needs to read the "
        "dryrun journal, add the source file to ALLOWED_PATHS in "
        "this script with a justifying comment.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

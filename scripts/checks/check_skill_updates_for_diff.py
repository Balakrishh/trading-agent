"""
SDD pre-commit helper: surface which skills need attention given a diff.

Reads ``docs/traceability.md`` (the auto-generated coverage matrix) and
``git diff`` to answer the question: **"If I commit these changes,
which skills must I also update or re-stamp?"**

Use cases
---------

* **Before committing** — pipe through to your shell so you see the
  checklist next to the diff::

      python3 scripts/checks/check_skill_updates_for_diff.py
      python3 scripts/checks/check_skill_updates_for_diff.py --ref main
      python3 scripts/checks/check_skill_updates_for_diff.py --staged

* **As a pre-commit hook** — install in ``.git/hooks/pre-commit`` so a
  forgotten skill update fails the commit::

      #!/bin/sh
      python3 scripts/checks/check_skill_updates_for_diff.py --staged
      exit $?

* **As a CI gate** — run against the PR diff. Exit 1 signals "skills
  need attention," typically requiring the contributor to either update
  the affected skill OR explicitly mark it as not-needing-update.

Behavior
--------

* Reads ``docs/traceability.md`` to build the inverse mapping
  ``source_file -> [skill_filename, ...]``. If the matrix is stale,
  re-run ``scripts/checks/build_traceability.py`` first.

* Uses ``git diff --name-only`` to find the set of changed files.

* For each changed source under ``trading_agent/``, looks up the skills
  that cite it. Reports the union as a checklist.

* Marks each skill row with ``[x]`` if that skill's markdown file was
  ALSO in the diff (i.e. the contributor already updated it),
  ``[ ]`` otherwise.

* Exits 1 if any skill needs attention but wasn't updated in the diff.

Limitations
-----------

* The traceability matrix uses file-level granularity. If you touch
  one function in ``chain_scanner.py`` but the skill cites a different
  function in the same file, this still flags the skill. False
  positives are acceptable here — they nudge the contributor to read
  the §3 quote and confirm it's still valid.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACEABILITY_PATH = REPO_ROOT / "docs" / "traceability.md"

# Match a skill row in the traceability matrix: starts with "| NN |"
SKILL_ROW_RE = re.compile(
    r"^\|\s*(\d{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
)


def _build_source_to_skills_map() -> Dict[str, List[str]]:
    """Parse traceability.md to build {source_file: [skill_filename, ...]}.

    The cited-source column is comma-separated with the ``trading_agent/``
    prefix stripped. We re-add the prefix when building the inverse map
    so it matches the paths emitted by ``git diff``.
    """
    if not TRACEABILITY_PATH.exists():
        print(f"ERR  {TRACEABILITY_PATH.relative_to(REPO_ROOT)} missing. "
              f"Run `python3 scripts/checks/build_traceability.py` first.",
              file=sys.stderr)
        sys.exit(2)

    text = TRACEABILITY_PATH.read_text(encoding="utf-8")
    out: Dict[str, List[str]] = defaultdict(list)
    for ln in text.splitlines():
        m = SKILL_ROW_RE.match(ln)
        if not m:
            continue
        number = m.group(1)
        title = m.group(2).strip()
        sources_cell = m.group(3).strip()
        if sources_cell in ("—", ""):
            continue
        # Build skill filename: NN_title_with_underscores.md
        skill_filename = (
            f"docs/skills/{number}_{title.replace(' ', '_')}.md"
        )
        for src in [s.strip() for s in sources_cell.split(",") if s.strip()]:
            # Sources are stored stripped of the trading_agent/ prefix.
            # Re-add it so it matches `git diff` output.
            if not src.startswith("trading_agent/"):
                src = f"trading_agent/{src}"
            out[src].append(skill_filename)
    return out


def _changed_files(ref: str, staged: bool) -> List[str]:
    """Return repo-relative paths of files changed since ``ref``."""
    cmd = ["git", "diff", "--name-only"]
    if staged:
        cmd.append("--cached")
    else:
        cmd.append(ref)
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        print("ERR  git not on PATH", file=sys.stderr)
        sys.exit(2)
    if result.returncode != 0:
        # No diff or ref doesn't exist — empty list is fine.
        return []
    return [f for f in result.stdout.splitlines() if f.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="HEAD",
                        help="Git ref to diff against (default: HEAD — i.e. "
                             "unstaged + staged working-tree changes).")
    parser.add_argument("--staged", action="store_true",
                        help="Only look at staged changes (for pre-commit hook).")
    args = parser.parse_args()

    source_to_skills = _build_source_to_skills_map()
    changed = _changed_files(args.ref, args.staged)
    if not changed:
        print("OK   No changed files in this diff.")
        return 0

    changed_source = [f for f in changed if f.startswith("trading_agent/")
                                            and f.endswith(".py")]
    changed_skill_files: Set[str] = {f for f in changed
                                     if f.startswith("docs/skills/")}

    # Map: skill_filename -> [source_files_that_changed_and_cited]
    affected: Dict[str, List[str]] = defaultdict(list)
    for src in changed_source:
        for sk in source_to_skills.get(src, []):
            affected[sk].append(src)

    if not affected and not changed_skill_files:
        print("OK   No skill-update needed for this diff.")
        return 0

    label = "staged" if args.staged else f"vs {args.ref}"
    print(f"Skill-update checklist ({label}):")
    print()

    has_pending = False
    for sk in sorted(affected):
        already_updated = sk in changed_skill_files
        marker = "[x]" if already_updated else "[ ]"
        if not already_updated:
            has_pending = True
        print(f"  {marker} {sk}")
        for src in sorted(set(affected[sk])):
            print(f"          changed source: {src}")

    # Also surface any skill files updated WITHOUT a related source change —
    # that's a doc-only update, totally fine, but useful to call out.
    doc_only_updates = changed_skill_files - set(affected.keys())
    if doc_only_updates:
        print()
        print(f"Skills updated (no related source change in this diff):")
        for sk in sorted(doc_only_updates):
            print(f"  [x] {sk}  (doc-only)")

    if has_pending:
        print()
        print(f"⚠  {sum(1 for sk in affected if sk not in changed_skill_files)} "
              f"skill(s) need attention. Update each + re-stamp the footer.")
        return 1

    print()
    print(f"OK   All affected skills are included in this diff.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

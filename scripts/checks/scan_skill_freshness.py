"""
SDD freshness check — each skill's footer date vs. its cited source.

Every skill under ``docs/skills/`` carries:

  * A "Source of truth" header citing one or more ``relative/path.py:LINE-LINE``
    locations in the codebase, AND
  * A footer "*Last verified against repo HEAD on YYYY-MM-DD*".

When a cited source file is modified in git AFTER the skill's footer
date, the skill is out-of-date — the §3 Reference Python may no longer
match the live source. This check fails CI in that case so the
contributor either:

  1. Re-stamps the footer (after verifying the §3 quote still matches), or
  2. Updates §3 and re-stamps the footer (when the source genuinely changed).

The pre-2026-05-15 workflow caught some of these via the
"Last verified" convention plus reviewer discipline, but nothing
automated it. This script makes drift a CI failure.

Exits 0 when all skills are fresh, 1 when ≥1 skill is stale.

Limitations
-----------
* Uses ``git log -1 --format=%ad --date=short`` on each cited file to
  read the last commit date. This is more reliable than filesystem
  mtime (which gets reset on fresh checkouts) but does require git
  history to be present (CI environments must clone with ``--depth=0``
  or sufficient depth for this to work).

* Ignores Phase-2 skills (the ones marked "planned, not yet written"
  in ``docs/skills/README.md``) — they don't have content to verify.

* Treats the empty-source case (no cited file exists in the repo)
  as a separate "broken skill" error, not a freshness issue.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "docs" / "skills"

# Regex for the "Source of truth" line in a skill. Captures every
# `file:line[-line]` citation. Robust to single-file or multi-file
# citations separated by commas (see skill 03's two citations).
SOURCE_OF_TRUTH_RE = re.compile(
    r"\*\*Source of truth:\*\*\s*(.+)$",
    re.MULTILINE,
)

# Each citation looks like ``some/relative/path.py:LINE-LINE`` or just
# ``some/path.py`` (no line range). Captures the path; line range is
# ignored for the freshness check (we care about file-level mtime).
#
# We REQUIRE the path to contain a directory separator (e.g.
# ``trading_agent/chain_scanner.py``, not bare ``chain_scanner.py``).
# Skill 15 enumerates filenames inline as prose without their parent
# directory — those aren't real citations and the literal file at
# repo-root doesn't exist. Filtering to "must contain `/`" eliminates
# those false positives.
CITATION_PATH_RE = re.compile(
    # Captures the .py path. The optional ``:...`` after ``.py``
    # tolerates both line-range citations (``:160-162``) AND
    # function-name citations (``:_leg_spread_too_wide``).
    r"`([^`\s]+/[^`\s]+\.py)(?::[^`\s]+)?`",
)

# Footer line: "*Last verified against repo HEAD on YYYY-MM-DD*"
# Tolerates extra text after the date (e.g., "(incl. follow-up)").
FOOTER_DATE_RE = re.compile(
    r"\*Last verified against repo HEAD on (\d{4}-\d{2}-\d{2})",
)


def _git_last_modified(repo_relative_path: str) -> Optional[date]:
    """Return the date of the most recent commit touching ``path``.

    Returns ``None`` if git couldn't find a commit (file may be
    uncommitted or path may not exist).
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ad", "--date=short",
             "--", repo_relative_path],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # git not on PATH — skip the check rather than failing.
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return datetime.strptime(result.stdout.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_skill(skill_path: Path) -> Tuple[List[str], Optional[date]]:
    """Extract the cited source paths and the footer date from a skill.

    Returns ``(paths, footer_date)``. Either may be empty/None if the
    skill is malformed.
    """
    text = skill_path.read_text(encoding="utf-8")

    paths: List[str] = []
    sot_match = SOURCE_OF_TRUTH_RE.search(text)
    if sot_match:
        sot_line = sot_match.group(1)
        paths = list(CITATION_PATH_RE.findall(sot_line))

    footer_date: Optional[date] = None
    footer_match = FOOTER_DATE_RE.search(text)
    if footer_match:
        try:
            footer_date = datetime.strptime(
                footer_match.group(1), "%Y-%m-%d"
            ).date()
        except ValueError:
            footer_date = None

    return paths, footer_date


def main() -> int:
    if not SKILLS_DIR.exists():
        print(f"ERR: {SKILLS_DIR} not found", file=sys.stderr)
        return 1

    skills = sorted(SKILLS_DIR.glob("[0-9][0-9]_*.md"))
    if not skills:
        print(f"ERR: no skill files found under {SKILLS_DIR}", file=sys.stderr)
        return 1

    stale: List[Tuple[str, str, date, date]] = []  # skill, file, footer, src_mod
    broken: List[Tuple[str, str]] = []             # skill, reason
    missing: List[Tuple[str, str]] = []            # skill, missing-path
    checked = 0

    for sf in skills:
        rel_skill = sf.relative_to(REPO_ROOT).as_posix()
        paths, footer_date = _parse_skill(sf)

        if footer_date is None:
            broken.append((rel_skill, "no parseable footer date"))
            continue
        if not paths:
            # This is OK — skill 00 (meta-skill) doesn't have a code citation.
            # We still emit at INFO level so the human knows.
            continue

        for p in paths:
            abs_p = REPO_ROOT / p
            if not abs_p.exists():
                missing.append((rel_skill, p))
                continue

            src_date = _git_last_modified(p)
            if src_date is None:
                # File exists but no commit history (uncommitted change).
                # Skip rather than flag.
                continue

            if src_date > footer_date:
                stale.append((rel_skill, p, footer_date, src_date))

            checked += 1

    print(f"SDD freshness check — scanned {len(skills)} skill files, "
          f"{checked} (skill, source) pairs.")
    print()

    has_errors = False
    if missing:
        has_errors = True
        print("ERR  Cited source files that don't exist in the repo:")
        for sk, p in missing:
            print(f"     - {sk}  →  {p}  (missing)")
        print()
    if broken:
        has_errors = True
        print("ERR  Skills with unparseable footer:")
        for sk, why in broken:
            print(f"     - {sk}  ({why})")
        print()
    if stale:
        has_errors = True
        print(f"FAIL {len(stale)} skill(s) stale — source modified AFTER footer:")
        for sk, p, footer, src in stale:
            days = (src - footer).days
            print(f"     - {sk}")
            print(f"         cited: {p}")
            print(f"         footer:    {footer}")
            print(f"         src last:  {src}  ({days} day(s) newer)")
            print(f"         → re-verify §3 + re-stamp footer to today")
        print()
        return 1
    if has_errors:
        return 1

    print(f"OK   All {checked} (skill, source) pairs are fresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

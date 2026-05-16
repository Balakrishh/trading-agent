"""
SDD quote-match check — §3 Reference Python quotes vs. live source.

Every skill's §3 "Reference Python Implementation" carries one or more
fenced Python blocks. Each block typically has a comment header of the
form:

    # trading_agent/chain_scanner.py:184-207
    def _score_candidate(credit, width, ...):
        ...

This check verifies the FIRST CODE-LINE of each block (e.g., the
``def`` signature, a function call, a constant assignment) appears
verbatim in the cited source file. If the source has drifted such
that the signature changed, the check fails CI — the skill is lying
about the live code.

Why "first non-comment line" only
---------------------------------
A more thorough check would diff the entire block against the source's
function body. But Python source rarely matches markdown formatting
exactly (whitespace, trailing comments, removed blank lines), so a
full-body diff produces false positives. The first non-comment line
is typically the function signature or a key formula — that's what
actually matters for "does the skill describe the same code?".

Heuristic, not formal proof
---------------------------
This catches the failure modes that matter (signature renamed,
formula changed, function moved) without being so strict that every
whitespace change fails CI. For full conformance use
``tests/conformance/`` (skill 00) — those tests assert the documented
behaviour by running the actual code, not by string-matching.

Exits 0 when all quotes match, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "docs" / "skills"

# Fenced Python code blocks. Captures the body (between ```python and ```).
PY_BLOCK_RE = re.compile(
    r"```python\s*\n(.*?)```",
    re.DOTALL,
)

# Comment-header citation: # path/to.py:LINE-LINE   (or no line range)
HEADER_CITATION_RE = re.compile(
    r"^\s*#\s*([\w./_-]+\.py)(?::\d+(?:-\d+)?)?\s*$",
)


def _first_code_line(block: str) -> Optional[str]:
    """Return the first non-blank, non-pure-comment line in ``block``.

    Skips the comment-header citation and any leading docstring blanks.
    Returns ``None`` if the block has no codeable content.
    """
    for raw in block.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return stripped
    return None


def _extract_blocks_with_citation(skill_text: str) -> List[Tuple[str, str]]:
    """For each Python block in the skill, return (cited_path, first_code_line).

    Skips blocks whose comment header doesn't carry a citation (e.g.,
    illustration blocks with no source claim).
    """
    out: List[Tuple[str, str]] = []
    for block in PY_BLOCK_RE.findall(skill_text):
        # First line should be the citation comment header (or sometimes
        # the second line if there's a leading blank).
        cited: Optional[str] = None
        for line in block.splitlines()[:3]:
            m = HEADER_CITATION_RE.match(line)
            if m:
                cited = m.group(1)
                break
        if not cited:
            continue
        first_code = _first_code_line(block)
        if not first_code:
            continue
        out.append((cited, first_code))
    return out


def _line_appears_in(source_text: str, line: str) -> bool:
    """Tolerant substring match. Strips trailing comments + whitespace.

    Returns True if a stripped form of ``line`` appears as a substring of
    the source. We don't require exact match — Python source may carry
    trailing comments, line continuations, or extra blanks that the skill
    omits for readability.
    """
    # Drop the line's trailing comment so we don't require comment-text
    # to be byte-identical with source.
    line_no_comment = line.split("#", 1)[0].rstrip().rstrip(":,)")
    # Re-add the colon/paren since they're meaningful syntactically.
    line_canon = line_no_comment + ":" if line.rstrip().endswith(":") else line_no_comment
    if not line_canon.strip():
        return True
    # Whitespace tolerance: collapse any run of whitespace to a single space.
    source_canon = re.sub(r"\s+", " ", source_text)
    needle = re.sub(r"\s+", " ", line_canon)
    return needle in source_canon


def main() -> int:
    if not SKILLS_DIR.exists():
        print(f"ERR: {SKILLS_DIR} not found", file=sys.stderr)
        return 1

    skills = sorted(SKILLS_DIR.glob("[0-9][0-9]_*.md"))
    if not skills:
        print(f"ERR: no skill files found", file=sys.stderr)
        return 1

    mismatches: List[Tuple[str, str, str]] = []  # skill, cited, line
    missing_files: List[Tuple[str, str]] = []
    checked = 0

    for sf in skills:
        rel_skill = sf.relative_to(REPO_ROOT).as_posix()
        skill_text = sf.read_text(encoding="utf-8")
        for cited, line in _extract_blocks_with_citation(skill_text):
            abs_cited = REPO_ROOT / cited
            if not abs_cited.exists():
                missing_files.append((rel_skill, cited))
                continue
            source_text = abs_cited.read_text(encoding="utf-8")
            if not _line_appears_in(source_text, line):
                mismatches.append((rel_skill, cited, line))
            checked += 1

    print(f"SDD quote-match check — scanned {len(skills)} skills, "
          f"verified {checked} cited code lines.")
    print()

    has_errors = False
    if missing_files:
        has_errors = True
        print(f"ERR  {len(missing_files)} skill(s) cite paths that don't exist:")
        for sk, p in missing_files:
            print(f"     - {sk}  →  {p}")
        print()

    if mismatches:
        has_errors = True
        print(f"FAIL {len(mismatches)} quoted line(s) no longer appear in source:")
        for sk, cited, line in mismatches:
            print(f"     - {sk}")
            print(f"         cited: {cited}")
            print(f"         line:  {line[:100]}")
            print(f"         → update §3 of the skill OR fix source drift")
        print()
        return 1

    if has_errors:
        return 1

    print(f"OK   All {checked} cited code lines still appear in source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

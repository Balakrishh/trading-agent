"""
SDD traceability matrix builder.

Generates ``docs/traceability.md`` — a coverage matrix mapping every
skill to its cited source files, its module tests, its conformance
tests, and the runbooks that cross-link to it. Plus the *inverses*:

  * Orphan source files — code paths with no skill citing them
    (candidates for "is this still in use? does it need a spec?")
  * Orphan skills — skills with no conformance test
    (candidates for the Tier 1 conformance-test back-fill)
  * Orphan tests — test files with no skill reference in their
    docstring (candidates for "what's this test actually verifying?")

This is the first Tier 2 SDD artifact. It does not gate CI — it's a
read-only coverage report. Run on demand::

    python scripts/checks/build_traceability.py

Outputs to ``docs/traceability.md`` (overwrites). Idempotent — running
twice produces the same file.

Limitations
-----------
* Only walks the canonical paths: ``docs/skills/``, ``trading_agent/``,
  ``tests/``, ``docs/runbooks/``. New directories require a code change.
* "Test references skill" is detected via the regex ``[Ss]kill\\s*NN``
  appearing in the test file's content. False positives possible (a
  comment mentioning a skill number for context) but acceptable.
* Source files are considered "cited" if at least one skill's
  ``Source of truth`` header references them — fine-grained line-range
  citations aren't tracked at this layer.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "docs" / "skills"
RUNBOOKS_DIR = REPO_ROOT / "docs" / "runbooks"
TRADING_AGENT_DIR = REPO_ROOT / "trading_agent"
TESTS_DIR = REPO_ROOT / "tests"
OUTPUT_PATH = REPO_ROOT / "docs" / "traceability.md"

SOURCE_OF_TRUTH_RE = re.compile(
    r"\*\*Source of truth:\*\*\s*(.+)$",
    re.MULTILINE,
)
CITATION_PATH_RE = re.compile(
    # Captures the .py path. Everything between ``.py`` and the closing
    # backtick is consumed as the (discarded) citation suffix — this
    # tolerates ALL the suffix shapes used in the repo today:
    #   `path.py`                       (no suffix)
    #   `path.py:160-162`               (single line range)
    #   `path.py:50, 79, 166-190`       (comma-separated line list)
    #   `path.py:_function_name`        (function-name citation)
    # The non-greedy `[^`]*` allows commas and spaces inside the
    # backtick pair without breaking the match — that was the
    # 2026-05-15 bug that hid skills 06 and 11.
    r"`([^`\s]+/[^`\s]+\.py)[^`]*`",
)
FOOTER_DATE_RE = re.compile(
    r"\*Last verified against repo HEAD on (\d{4}-\d{2}-\d{2})",
)
# "Skill 03", "skill 03", "Skill 17", etc. — case-insensitive.
SKILL_REFERENCE_RE = re.compile(r"[Ss]kill\s+(\d{2})\b")
# Skill filename pattern: NN_short_name.md
SKILL_FILENAME_RE = re.compile(r"^(\d{2})_(.+)\.md$")
# Runbook filename pattern: NN_short_name.md (under runbooks/)
RUNBOOK_FILENAME_RE = re.compile(r"^(\d{2})_(.+)\.md$")


def _parse_skill_metadata(skill_path: Path) -> Tuple[List[str], Optional[date]]:
    """Return (cited_source_paths, footer_date) for a skill."""
    text = skill_path.read_text(encoding="utf-8")
    paths: List[str] = []
    m = SOURCE_OF_TRUTH_RE.search(text)
    if m:
        paths = list(CITATION_PATH_RE.findall(m.group(1)))
    # Dedupe while preserving order.
    seen: Set[str] = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    footer_date: Optional[date] = None
    m = FOOTER_DATE_RE.search(text)
    if m:
        from datetime import datetime
        try:
            footer_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    return paths, footer_date


def _tests_referencing_each_skill(
    skill_to_sources: Dict[str, List[str]],
) -> Dict[str, Set[str]]:
    """Walk tests/ and map skill_NN → {test_files referencing it}.

    Three linkage paths, applied in order:

      1. **Conformance tests** (under ``tests/conformance/test_skill_NN_*.py``)
         — detected via filename. Always strongest signal.
      2. **Explicit references in test content** — comments or
         docstrings containing ``Skill NN``. Strongest editorial
         signal but the rarest in practice.
      3. **Import-based linkage** — if a test imports a module that
         a skill cites in its Source-of-truth header, link the test
         to the skill. Catches the bulk of "I tested this function
         but didn't think to mention the skill number" tests.

    Path #3 was added 2026-05-16 after the matrix's first generation
    showed only 2 / 22 skills with module tests, despite the obvious
    coupling between ``test_chain_scanner.py`` and skills 01/03/05/29.
    """
    out: Dict[str, Set[str]] = defaultdict(set)
    if not TESTS_DIR.exists():
        return out

    # Build the inverse map: source_path -> [skill_number, ...] for
    # cheap O(1) lookup during the test scan.
    source_to_skills: Dict[str, List[str]] = defaultdict(list)
    for skill_num, sources in skill_to_sources.items():
        for src in sources:
            source_to_skills[src].append(skill_num)

    # Regex to match imports of trading_agent.X.Y modules. Both
    # ``from trading_agent.X import ...`` and ``import trading_agent.X``
    # supported. Captures the module path so it can be mapped back
    # to ``trading_agent/X/Y.py``.
    import_re = re.compile(
        r"(?:from|import)\s+trading_agent(?:\.[\w.]+)?",
    )
    from_re = re.compile(
        r"from\s+(trading_agent(?:\.[\w]+)*)\s+import",
    )
    direct_re = re.compile(
        r"import\s+(trading_agent(?:\.[\w]+)*)",
    )

    for test_path in TESTS_DIR.rglob("test_*.py"):
        rel = test_path.relative_to(REPO_ROOT).as_posix()
        # Conformance tests: filename carries the skill number.
        m = re.match(r"test_skill_(\d{2})_.*\.py$", test_path.name)
        if m:
            out[m.group(1)].add(rel)
            continue

        try:
            content = test_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Linkage path 2: explicit "Skill NN" references in content.
        for sk in set(SKILL_REFERENCE_RE.findall(content)):
            out[sk].add(rel)

        # Linkage path 3: imports → cited sources → skills.
        modules_imported: Set[str] = set()
        for m in from_re.finditer(content):
            modules_imported.add(m.group(1))
        for m in direct_re.finditer(content):
            modules_imported.add(m.group(1))

        # Convert each imported module to its file path and check
        # whether any skill cites that file.
        for mod in modules_imported:
            # ``trading_agent.chain_scanner`` → ``trading_agent/chain_scanner.py``
            file_path = mod.replace(".", "/") + ".py"
            for sk in source_to_skills.get(file_path, []):
                out[sk].add(rel)
            # Also try package-style: ``trading_agent.backtest`` may
            # import the package's __init__.py.
            pkg_init = mod.replace(".", "/") + "/__init__.py"
            for sk in source_to_skills.get(pkg_init, []):
                out[sk].add(rel)

    return out


def _runbooks_referencing_each_skill() -> Dict[str, Set[str]]:
    """Walk runbooks/ and map skill_NN → {runbook_files referencing it}."""
    out: Dict[str, Set[str]] = defaultdict(set)
    if not RUNBOOKS_DIR.exists():
        return out
    for rb_path in RUNBOOKS_DIR.glob("*.md"):
        try:
            content = rb_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Skip README + meta
        if rb_path.name in ("README.md", "prompts.md"):
            continue
        rel = rb_path.relative_to(REPO_ROOT).as_posix()
        for sk in set(SKILL_REFERENCE_RE.findall(content)):
            out[sk].add(rel)
    return out


def _all_source_files() -> List[str]:
    """Every .py file under trading_agent/, as repo-relative posix paths."""
    if not TRADING_AGENT_DIR.exists():
        return []
    out: List[str] = []
    for p in TRADING_AGENT_DIR.rglob("*.py"):
        # Skip __pycache__ + __init__.py (the latter rarely cited).
        if "__pycache__" in p.parts:
            continue
        out.append(p.relative_to(REPO_ROOT).as_posix())
    return sorted(out)


def _git_last_modified(repo_relative_path: str) -> Optional[date]:
    """Last commit date for the given file. Returns None if unknown."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ad", "--date=short",
             "--", repo_relative_path],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    from datetime import datetime
    try:
        return datetime.strptime(result.stdout.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def build_matrix() -> str:
    """Return the full traceability markdown."""
    skills = sorted(SKILLS_DIR.glob("[0-9][0-9]_*.md"))

    # First pass: parse each skill's SoT citations so the test-import
    # linkage in _tests_referencing_each_skill can resolve source-path
    # references back to skill numbers.
    skill_to_sources: Dict[str, List[str]] = {}
    for sf in skills:
        m = SKILL_FILENAME_RE.match(sf.name)
        if not m:
            continue
        cites, _ = _parse_skill_metadata(sf)
        skill_to_sources[m.group(1)] = cites

    test_map = _tests_referencing_each_skill(skill_to_sources)
    runbook_map = _runbooks_referencing_each_skill()
    all_sources = _all_source_files()

    # Per-skill rows + accumulate the set of cited source files.
    cited_sources: Set[str] = set()
    rows: List[Tuple[str, str, str, str, str, str]] = []
    skill_numbers_with_tests: Set[str] = set()
    for sf in skills:
        m = SKILL_FILENAME_RE.match(sf.name)
        if not m:
            continue
        num = m.group(1)
        title = m.group(2).replace("_", " ")
        cites, footer = _parse_skill_metadata(sf)
        cited_sources.update(cites)

        module_tests = sorted(t for t in test_map.get(num, set())
                              if "conformance/" not in t)
        conf_tests = sorted(t for t in test_map.get(num, set())
                            if "conformance/" in t)
        if conf_tests:
            skill_numbers_with_tests.add(num)
        runbooks = sorted(runbook_map.get(num, set()))

        cites_short = ", ".join(c.replace("trading_agent/", "") for c in cites) or "—"
        mt_short = ", ".join(t.replace("tests/", "") for t in module_tests) or "—"
        ct_short = ", ".join(t.replace("tests/", "") for t in conf_tests) or "❌ MISSING"
        rb_short = ", ".join(r.replace("docs/runbooks/", "") for r in runbooks) or "—"
        footer_str = footer.isoformat() if footer else "?"
        rows.append((num, title, cites_short, mt_short, ct_short, rb_short))

    # Orphan code: source files not in cited_sources.
    orphan_code = [s for s in all_sources if s not in cited_sources
                   and not s.endswith("__init__.py")]

    # Orphan skills (no conformance test).
    orphan_skills = []
    for sf in skills:
        m = SKILL_FILENAME_RE.match(sf.name)
        if m and m.group(1) not in skill_numbers_with_tests:
            orphan_skills.append(sf.name)

    # Compose markdown.
    # Pre-2026-05-17 the header carried ``date.today().isoformat()``.
    # That produced a deterministic-but-daily-ticking diff: every
    # midnight, CI's ``git diff --exit-code`` would flag the file as
    # stale even when zero substantive content had changed. The
    # generator now emits the same content regardless of wall-clock
    # date — git's own log already records WHEN this file was last
    # touched, so the inline date stamp was redundant.
    out = []
    out.append("# Traceability Matrix")
    out.append("")
    out.append("*Auto-generated by `scripts/checks/build_traceability.py`.*")
    out.append("*Do not edit by hand — re-run the script to refresh.*")
    out.append("")
    out.append("This matrix maps every skill to its cited source files, "
               "tests, and runbooks. The inverse views below surface "
               "orphan code (paths no skill cites) and orphan skills "
               "(skills with no conformance test).")
    out.append("")
    out.append("## Skill → source / tests / runbooks")
    out.append("")
    out.append("| # | Skill | Cited source | Module tests | Conformance test | Runbooks |")
    out.append("|---|---|---|---|---|---|")
    for num, title, cites, mt, ct, rb in rows:
        out.append(f"| {num} | {title} | {cites} | {mt} | {ct} | {rb} |")
    out.append("")
    out.append(f"**Skills with conformance tests:** {len(skill_numbers_with_tests)} / {len(rows)} "
               f"({100*len(skill_numbers_with_tests)//max(1,len(rows))}%)")
    out.append("")

    out.append("## Orphan source files")
    out.append("")
    out.append("Code paths under `trading_agent/` that NO skill currently cites. "
               "Each is a candidate for either (a) drafting a skill if the "
               "concept is non-trivial, or (b) confirming the file is "
               "implementation-detail and doesn't warrant a spec.")
    out.append("")
    if orphan_code:
        for src in orphan_code:
            last_mod = _git_last_modified(src)
            mod_str = f" (last modified {last_mod})" if last_mod else ""
            out.append(f"- `{src}`{mod_str}")
    else:
        out.append("*(none — every source file is cited by at least one skill)*")
    out.append("")
    out.append(f"**Orphan source coverage:** {len(orphan_code)} / {len(all_sources)} "
               f"files ({100*len(orphan_code)//max(1,len(all_sources))}% uncovered)")
    out.append("")

    out.append("## Orphan skills (no conformance test)")
    out.append("")
    out.append("Skills with no corresponding `tests/conformance/test_skill_NN_*.py` "
               "file. These are candidates for back-filling conformance tests so the "
               "skill's documented behavior gets pinned by an executable assertion.")
    out.append("")
    if orphan_skills:
        for sk in orphan_skills:
            out.append(f"- `docs/skills/{sk}`")
    else:
        out.append("*(none — every skill has a conformance test)*")
    out.append("")

    out.append("---")
    out.append("")
    out.append("*Run `python scripts/checks/build_traceability.py` to refresh this file.*")
    return "\n".join(out) + "\n"


def main() -> int:
    if not SKILLS_DIR.exists():
        print(f"ERR: {SKILLS_DIR} not found", file=sys.stderr)
        return 1
    text = build_matrix()
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    # Surface a compact summary to stdout so CI / local runs see the
    # coverage shape at a glance.
    skill_count = text.count("\n| ") - 1  # rough header-row discount
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(text):,} bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

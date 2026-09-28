#!/usr/bin/env python3
"""CI gate — every tool listed in `.claude/agents/*.md` frontmatter must
appear in ``trading_agent.mcp.READONLY_TOOLS``.

Prevents a subagent .md from claiming access to a tool that doesn't
exist (typo → silent failure) or to a tool outside the read-only
surface (widening the write path by editing a markdown file).

Skill 48 + skill 55. Exits 0 on OK, 1 on any violation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Set


ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = ROOT / ".claude" / "agents"


def _parse_tools_from_frontmatter(text: str) -> List[str]:
    """Return the ``tools:`` YAML list from the file's frontmatter.

    Kept dependency-free — a tiny hand-parser is sufficient for the
    frontmatter shape this repo uses (skill 47 §3 style).
    """
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return []
    fm = m.group(1)
    tools: List[str] = []
    in_tools = False
    for line in fm.splitlines():
        if line.startswith("tools:"):
            in_tools = True
            # Inline form: ``tools: [a, b]``
            after = line.split(":", 1)[1].strip()
            if after.startswith("["):
                inner = after.strip("[]")
                return [t.strip().strip("'\"")
                        for t in inner.split(",") if t.strip()]
            continue
        if in_tools:
            if line.startswith("  - "):
                tools.append(line[4:].strip().strip("'\""))
            elif line and not line.startswith(" "):
                break
    return tools


def main() -> int:
    if not AGENTS_DIR.exists():
        print("no .claude/agents/ directory — skipping subagent-allowlist check")
        return 0

    # Load the closed READONLY_TOOLS set. Import from source rather than
    # runtime to keep this script dependency-free.
    init_txt = (ROOT / "trading_agent" / "mcp" / "__init__.py").read_text()
    m = re.search(r"READONLY_TOOLS[^(]*\(\s*(.*?)\)", init_txt, re.DOTALL)
    if not m:
        print("ERROR: could not parse READONLY_TOOLS from mcp/__init__.py")
        return 2
    readonly: Set[str] = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    if not readonly:
        print("ERROR: READONLY_TOOLS parsed to empty set")
        return 2

    offenders: List[str] = []
    total = 0
    for md in sorted(AGENTS_DIR.glob("*.md")):
        tools = _parse_tools_from_frontmatter(md.read_text())
        total += 1
        for t in tools:
            if t not in readonly:
                offenders.append(f"{md.name}: {t!r} not in READONLY_TOOLS")

    print(f"SDD subagent-allowlist check — scanned {total} subagent(s), "
          f"{len(readonly)} tools in READONLY_TOOLS.")
    if offenders:
        print("FAIL", *offenders, sep="\n  ")
        return 1
    print("OK   every listed subagent tool is in the read-only surface.")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(main())

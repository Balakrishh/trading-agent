"""Conformance for Phase 4 / Phase 5 — .claude/commands/ + .claude/agents/
files satisfy shape rules so a Claude Code session picks them up
correctly and no subagent widens the read-only surface.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set


_ROOT = Path(__file__).resolve().parents[2]
_CMDS = _ROOT / ".claude" / "commands"
_AGENTS = _ROOT / ".claude" / "agents"


def _frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return {}
    out: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _tools_list(text: str) -> List[str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return []
    fm = m.group(1)
    tools: List[str] = []
    in_tools = False
    for line in fm.splitlines():
        if line.startswith("tools:"):
            in_tools = True
            continue
        if in_tools:
            if line.startswith("  - "):
                tools.append(line[4:].strip())
            elif line and not line.startswith(" "):
                break
    return tools


def _readonly_tools() -> Set[str]:
    src = (_ROOT / "trading_agent" / "mcp" / "__init__.py").read_text()
    return set(re.findall(r'"([a-z_]+)"', src.split("READONLY_TOOLS")[1]))


def test_every_slash_command_has_description():
    """Slash commands must ship a `description:` frontmatter or Claude
    Code won't index them.
    """
    for md in sorted(_CMDS.glob("*.md")):
        fm = _frontmatter(md.read_text())
        assert "description" in fm, f"{md.name} missing `description:`"


def test_slash_commands_never_call_executor():
    """A slash command's prose must not INSTRUCT Claude to import the
    executor or call submit_order/place_order directly.

    We whitelist prohibition sentences ("Do NOT ...") and operator-facing
    shell commands the operator runs themselves ("Run `python -m ...`"),
    since those preserve the invariant rather than break it.
    """
    forbidden = ("trading_agent.executor.", "submit_order(", "place_order(")
    prohibition_markers = (
        "do not", "never", "operator", "themselves", "the operator",
        "`python -m trading_agent.executor_promote",
        "`python -m trading_agent.executor.cancel_stuck`",
    )
    offenders = []
    for md in sorted(_CMDS.glob("*.md")):
        for line in md.read_text().splitlines():
            lower = line.lower()
            if any(m in lower for m in prohibition_markers):
                continue
            for f in forbidden:
                if f in line:
                    offenders.append((md.name, f, line.strip()[:80]))
    assert not offenders, offenders


def test_every_subagent_has_name_and_description():
    for md in sorted(_AGENTS.glob("*.md")):
        fm = _frontmatter(md.read_text())
        assert "name" in fm, f"{md.name} missing `name:`"
        assert "description" in fm, f"{md.name} missing `description:`"


def test_every_subagent_tool_in_readonly_set():
    """The core Phase-4 invariant — a subagent can only access
    tools in READONLY_TOOLS.
    """
    ro = _readonly_tools()
    offenders = []
    for md in sorted(_AGENTS.glob("*.md")):
        for t in _tools_list(md.read_text()):
            if t not in ro:
                offenders.append((md.name, t))
    assert not offenders, (
        f"Subagent tool(s) not in READONLY_TOOLS: {offenders}. "
        "Widening the write path via markdown is forbidden — the "
        "MCP surface (skill 48) is the boundary.")


def test_expected_subagents_present():
    """Phase 4 ships exactly three subagents. If a future change adds
    more, update this test in the same commit that adds them.
    """
    names = {p.stem for p in _AGENTS.glob("*.md")}
    assert "risk-reviewer" in names
    assert "scanner-runner" in names
    assert "journal-analyst" in names


def test_expected_slash_commands_present():
    names = {p.stem for p in _CMDS.glob("*.md")}
    for expected in ("portfolio", "triage", "propose", "incident", "taxreview"):
        assert expected in names, f"missing /{expected}"

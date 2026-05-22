"""SDD gate: bound method length on agent.py + collaborators.

After the close-event collaborator extraction landed (skill 35) and
the JournalReader pattern took over journal queries (skill 19 §1.2),
the goal is to keep agent.py's methods small enough to reason about
without flowcharts. Pre-extraction `_journal_close_event` was 300
lines; the bugs that hid inside it included the -$2,976 phantom-loss
recap, the UTC-vs-ET dedup collision, and the ticker-drop TypeError.

This gate sets a ratcheting cap. Initial threshold (LIMIT) is set
ABOVE the current largest method so existing code passes. As bugs
get fixed by extraction (`_process_ticker`, `_run_cycle_impl`,
`_stage_monitor`, `_maybe_defensive_roll`), this limit drops with
them. A future contributor adding lines to one of those methods is
made to think about whether their change is an extraction
opportunity instead of just landing 80 more lines.

Per-file overrides accommodate intentional outliers (e.g. config
files with one giant validation function). Anything not in the
override list inherits LIMIT.

Exit codes:
  0 — every method ≤ its file's limit
  1 — at least one method exceeds
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Default cap for the trading-agent core modules.
LIMIT = 400

# Per-file overrides where the cap deliberately differs.
PER_FILE_LIMIT: Dict[str, int] = {
    # No overrides yet — the four big methods in agent.py are within
    # LIMIT today and are the next extraction targets.
}

# Files subject to the gate. Tests + scripts + streamlit views are
# excluded — they're either short or read-only declarative.
TARGET_FILES: Tuple[str, ...] = (
    "trading_agent/agent.py",
    "trading_agent/executor.py",
    "trading_agent/strategy.py",
    "trading_agent/chain_scanner.py",
    "trading_agent/decision_engine.py",
    "trading_agent/risk_manager.py",
    "trading_agent/position_monitor.py",
    "trading_agent/journal_kb.py",
    "trading_agent/journal_reader.py",
    "trading_agent/close_event_collaborators.py",
    "trading_agent/exception_monitor.py",
    "trading_agent/regime.py",
    "trading_agent/multi_tf_regime.py",
)


def _method_lengths(path: Path) -> List[Tuple[str, int, int]]:
    """Return list of (qualname, lineno, line_count) for every function
    and method definition in `path`."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    out: List[Tuple[str, int, int]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: List[str] = []

        def _record(self, node) -> None:
            qual = ".".join(self.stack + [node.name])
            # end_lineno is set on Python 3.8+; fall back to the last
            # statement's lineno if missing.
            end = getattr(node, "end_lineno", None)
            if end is None:
                end = max(
                    (getattr(c, "end_lineno", c.lineno) for c in ast.walk(node)),
                    default=node.lineno,
                )
            out.append((qual, node.lineno, end - node.lineno + 1))

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            self._record(node)
            # Don't descend — nested defs are unusual and would inflate
            # the parent count if we recursed.

        def visit_AsyncFunctionDef(self, node):
            self._record(node)

    Visitor().visit(tree)
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    offenders: List[Tuple[str, str, int, int, int]] = []
    total_files = 0
    total_methods = 0
    largest_seen: Tuple[str, str, int] = ("", "", 0)
    for rel in TARGET_FILES:
        path = repo_root / rel
        if not path.is_file():
            continue
        total_files += 1
        limit = PER_FILE_LIMIT.get(rel, LIMIT)
        for qual, lineno, length in _method_lengths(path):
            total_methods += 1
            if length > largest_seen[2]:
                largest_seen = (rel, qual, length)
            if length > limit:
                offenders.append((rel, qual, lineno, length, limit))

    for rel, qual, lineno, length, limit in offenders:
        print(
            f"FAIL {rel}:{lineno}: {qual} is {length} lines "
            f"(limit {limit}) — extract a collaborator (see skill 35)"
        )

    print(
        f"\nSDD method-length gate — scanned {total_files} file(s), "
        f"{total_methods} method(s). Cap: {LIMIT} lines (per-file "
        f"overrides: {len(PER_FILE_LIMIT)})."
    )
    print(
        f"Largest method: {largest_seen[0]}::{largest_seen[1]} "
        f"= {largest_seen[2]} lines."
    )
    if offenders:
        print(f"\nFAIL — {len(offenders)} method(s) exceed limit.")
        return 1
    print(
        "\nOK   every method within length limit. "
        "Drop LIMIT in scan_method_length.py as the next refactor "
        "shrinks the current ceiling."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

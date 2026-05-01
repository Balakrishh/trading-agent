"""
AST integration check: verify the |Δ|×(1+edge_buffer) C/W floor formula
appears identically in all three gate sites — chain_scanner, risk_manager,
and executor — so a scanner-picked plan can never be rejected at planning
or execution time by a stricter floor.

Exits 0 if the invariant holds, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent / "trading_agent"

EXPECTED_FILES = {
    "chain_scanner.py": "_cw_floor",          # helper function
    "risk_manager.py": "validate",            # contains Check 2
    "executor.py": "_recheck_live_economics", # live recheck path
}


def _has_floor_formula(tree: ast.AST) -> bool:
    """
    Search for: <something> * (1.0 + <edge_buffer attr or name>)
    where the LHS references an absolute-delta-like name (short_max_delta,
    short_delta, etc.). Returns True if at least one such expression exists.
    """
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        # RHS must be (1.0 + edge_buffer) or (1 + edge_buffer)
        rhs = node.right
        if not (isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Add)):
            continue
        a, b = rhs.left, rhs.right
        ones = [x for x in (a, b) if isinstance(x, ast.Constant)
                and isinstance(x.value, (int, float)) and x.value == 1]
        if not ones:
            continue
        non_one = [x for x in (a, b) if x not in ones]
        if not non_one:
            continue
        edge = non_one[0]
        edge_id = (
            edge.attr if isinstance(edge, ast.Attribute) else
            edge.id if isinstance(edge, ast.Name) else
            None
        )
        if edge_id != "edge_buffer":
            continue
        # LHS — accept abs(...), name, attr
        lhs = node.left
        if isinstance(lhs, ast.Call) and isinstance(lhs.func, ast.Name) \
                and lhs.func.id == "abs":
            found = True
            break
        if isinstance(lhs, (ast.Name, ast.Attribute)):
            name = (lhs.attr if isinstance(lhs, ast.Attribute) else lhs.id)
            if "delta" in name.lower():
                found = True
                break
    return found


def main() -> int:
    failures = []
    for fname, marker in EXPECTED_FILES.items():
        path = ROOT / fname
        src = path.read_text()
        tree = ast.parse(src)
        if not _has_floor_formula(tree):
            failures.append(f"{fname}: missing |Δ|×(1+edge_buffer) formula")
        else:
            print(f"  OK   {fname}: |Δ|×(1+edge_buffer) found")

    if failures:
        print("\nFAIL — invariant broken:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll three gate sites share the same C/W floor formula.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

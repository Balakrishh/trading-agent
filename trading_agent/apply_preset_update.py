"""Apply a staged preset update — Phase A stub (skill 56).

Reads a proposal from ``pending_preset_updates/<uuid>.json``, prints
the field-by-field diff against the currently-saved preset, and exits
0 without writing.

**Phase A stub.** The actual ``save_active_preset`` call is deliberately
NOT wired here yet — Phase B lands the write plus the 3-predicate
auto-apply gate. Landing them together in the same PR would be a
half-tested change; keeping this file minimal lets the operator
inspect proposed diffs today without the "did it save?" ambiguity.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from trading_agent import pending_preset_updates_writer as _pwriter


log = logging.getLogger("trading_agent.apply_preset_update")


def _render_diff(proposal: dict) -> str:
    snapshot = proposal.get("preset_snapshot") or {}
    preset_diff = proposal.get("preset_diff") or {}
    watchlist_diff = proposal.get("watchlist_diff") or {}
    lines = [
        f"Proposal:      {proposal.get('proposal_id')}",
        f"Review date:   {proposal.get('review_date')}",
        f"Proposed by:   {proposal.get('proposed_by')}",
        f"Confidence:    {proposal.get('confidence', 0.0):.2f}",
        f"LLM reasoning: {proposal.get('llm_reasoning', '')[:200]}",
        "",
        "Preset diff:",
    ]
    if not preset_diff:
        lines.append("  (no preset field changes)")
    else:
        for field, new_val in preset_diff.items():
            old = snapshot.get(field, "<absent>")
            lines.append(f"  {field}: {old!r} → {new_val!r}")
    lines.append("")
    lines.append("Watchlist diff:")
    if not watchlist_diff:
        lines.append("  (no watchlist changes)")
    else:
        for k in ("drops", "adds"):
            v = watchlist_diff.get(k) or []
            if v:
                lines.append(f"  {k}: {', '.join(v)}")
        if watchlist_diff.get("reason"):
            lines.append(f"  reason: {watchlist_diff['reason']}")
    lines.append("")
    obs = proposal.get("observations") or []
    if obs:
        lines.append("Observations:")
        for o in obs:
            lines.append(f"  • {o}")
    return "\n".join(lines)


def apply(proposal_id: str, *, assume_yes: bool = False) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    proposal = _pwriter.read(proposal_id)
    if proposal is None:
        log.error("Proposal not found: %s", proposal_id)
        return 2

    print(_render_diff(proposal))
    print("")
    log.info(
        "Phase A stub: NOT persisting. Phase B lands "
        "strategy_presets.save_active_preset() + watchlist_store writes "
        "behind a 3-predicate auto-apply gate mirroring skill 55. Until "
        "then, edit the preset via the Streamlit UI to apply this diff "
        "manually — the diff above is the exact set of fields to change."
    )
    _ = assume_yes  # reserved for the Phase B gate
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trading_agent.apply_preset_update",
        description="Apply a staged daily-reviewer preset update (skill 56). "
                    "Phase A: prints the diff; does NOT save.",
    )
    parser.add_argument("proposal_id", help="UUID hex from the .json filename.")
    parser.add_argument("--yes", action="store_true",
                        help="Reserved for Phase B — auto-apply gate.")
    args = parser.parse_args(argv)
    return apply(args.proposal_id, assume_yes=args.yes)


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main())

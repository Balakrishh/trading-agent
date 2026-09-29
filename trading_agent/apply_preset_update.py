"""Apply a staged preset update — Phase B write path (skill 56 §3).

Reads a proposal from ``pending_preset_updates/<uuid>.json``, prints
the field-by-field diff, evaluates a 3-predicate auto-apply gate
(env master switch + PresetConfig allowlist + per-field change-size
cap), and either waits for ``--yes`` or auto-applies when every
predicate passes.

Write side:
- preset fields → ``strategy_presets.save_active_preset(...)`` with
  the LLM-proposed field passed as an overlay kwarg.
- watchlist drops/adds → ``watchlist_store.remove_ticker`` /
  ``add_ticker``.

**Invariant.** This module and ``strategy_presets.py`` are the only
two files in the repo permitted to call ``save_active_preset``
(watchlist_store aside). Verified by
``tests/conformance/test_skill_56_daily_reviewer.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from trading_agent import pending_preset_updates_writer as _pwriter


log = logging.getLogger("trading_agent.apply_preset_update")


# ---------------------------------------------------------------------------
# Gate — pure, testable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApplyGateResult:
    """Three predicates for auto-apply. Every field either "pass" or
    a human string describing why it failed.
    """
    master_switch:    str
    field_allowlist:  str
    delta_size_cap:   str

    @property
    def all_pass(self) -> bool:
        return all(
            getattr(self, f) == "pass" for f in (
                "master_switch", "field_allowlist", "delta_size_cap",
            )
        )

    def failures(self) -> List[str]:
        return [f"{f}: {getattr(self, f)}"
                for f in ("master_switch", "field_allowlist", "delta_size_cap")
                if getattr(self, f) != "pass"]


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def evaluate_apply_gate(
    *,
    proposal: Dict[str, Any],
    preset: Any,
) -> ApplyGateResult:
    """Return an ApplyGateResult. Every predicate mirrors skill 56 §3.5.

    Delta-size cap is per-field: (new - old) / max(|old|, 1e-9) is
    compared against ``preset.auto_apply_max_delta_change_pct``. For
    boolean fields (e.g. defensive_roll_enabled) the cap doesn't
    apply — any flip counts as within-cap.
    """
    # 1. Master env switch
    if not _env_bool("TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED"):
        master = "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED is off"
    else:
        master = "pass"

    diff = proposal.get("preset_diff") or {}
    snapshot = proposal.get("preset_snapshot") or {}
    allowed = tuple(getattr(preset, "auto_apply_allowed_fields", ()))
    if not diff:
        # No preset changes proposed — allowlist trivially satisfied.
        allowlist = "pass"
    elif not allowed:
        allowlist = "preset auto_apply_allowed_fields empty (disabled)"
    else:
        outside = [f for f in diff if f not in allowed]
        if outside:
            allowlist = f"fields not in allowlist: {outside}"
        else:
            allowlist = "pass"

    cap_pct = float(getattr(preset, "auto_apply_max_delta_change_pct", 0.0))
    if cap_pct <= 0:
        size_cap = "preset auto_apply_max_delta_change_pct=0 (disabled)"
    else:
        over_cap = []
        for field, new_val in diff.items():
            old_val = snapshot.get(field)
            if isinstance(new_val, bool) or isinstance(old_val, bool):
                # Boolean flip — cap doesn't apply.
                continue
            try:
                old_f = float(old_val)
                new_f = float(new_val)
            except (TypeError, ValueError):
                continue
            denom = max(abs(old_f), 1e-9)
            delta_pct = abs(new_f - old_f) / denom
            if delta_pct > cap_pct:
                over_cap.append(f"{field}: |Δ|={delta_pct:.2%} > cap={cap_pct:.2%}")
        size_cap = "pass" if not over_cap else "; ".join(over_cap)

    return ApplyGateResult(
        master_switch=master,
        field_allowlist=allowlist,
        delta_size_cap=size_cap,
    )


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------

def _apply_preset_diff(preset_diff: Dict[str, Any]) -> bool:
    """Call save_active_preset with the LLM-proposed fields as overlays.

    Returns True on success, False if there's nothing to apply.
    """
    if not preset_diff:
        return False
    # Import at call site keeps the AST walker happy and mirrors the
    # discipline used in executor_promote (see skill 55 §4).
    from trading_agent.strategy_presets import (                      # noqa: PLC0415
        load_active_preset, save_active_preset,
    )
    current = load_active_preset()
    # Read the profile name off the on-disk preset payload so re-save
    # preserves it. Falls back to "custom" if the loader didn't tag it.
    profile_name = getattr(current, "name", "custom")
    profile: str = "custom" if profile_name.lower() == "custom" else profile_name.lower()
    overlays: Dict[str, Any] = {}
    for k in ("edge_buffer", "profit_target_pct", "min_pop",
              "max_leg_spread_cents", "defensive_roll_enabled"):
        if k in preset_diff:
            overlays[k] = preset_diff[k]
    save_active_preset(
        profile=profile,
        directional_bias=getattr(current, "directional_bias", "auto"),
        **overlays,
    )
    return True


def _apply_watchlist_diff(watchlist_diff: Dict[str, Any]) -> Dict[str, int]:
    """Apply drops + adds to the watchlist store."""
    if not watchlist_diff:
        return {"drops": 0, "adds": 0}
    from trading_agent.watchlist_store import (                       # noqa: PLC0415
        add_ticker, remove_ticker,
    )
    drops = watchlist_diff.get("drops") or []
    adds = watchlist_diff.get("adds") or []
    d_count = 0
    for t in drops:
        try:
            remove_ticker(str(t).upper())
            d_count += 1
        except Exception as exc:                                       # noqa: BLE001
            log.warning("Failed to drop %s: %s", t, exc)
    a_count = 0
    for t in adds:
        try:
            add_ticker(str(t).upper())
            a_count += 1
        except Exception as exc:                                       # noqa: BLE001
            log.warning("Failed to add %s: %s", t, exc)
    return {"drops": d_count, "adds": a_count}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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


def _prompt_yes() -> bool:
    try:
        ans = input("Apply this update? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def apply(proposal_id: str,
          *,
          assume_yes: bool = False,
          dry_run: bool = False) -> int:
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

    if dry_run:
        log.info("--dry-run: printed diff, not applying.")
        return 0

    # Gate evaluation (Phase B addition).
    try:
        from trading_agent.strategy_presets import load_active_preset  # noqa: PLC0415
        preset = load_active_preset()
    except Exception as exc:                                          # noqa: BLE001
        log.warning("Could not load preset (%s); manual approval only.", exc)
        preset = None

    gate = evaluate_apply_gate(proposal=proposal, preset=preset)
    auto_ok = gate.all_pass

    if not (assume_yes or auto_ok):
        if not gate.all_pass:
            log.info("Auto-apply NOT eligible. Gate failures:\n  - %s",
                     "\n  - ".join(gate.failures()))
        if not _prompt_yes():
            log.info("Operator declined.")
            return 3

    # Apply.
    changed_preset = _apply_preset_diff(proposal.get("preset_diff") or {})
    wl_counts = _apply_watchlist_diff(proposal.get("watchlist_diff") or {})

    log.info("Applied — preset_changed=%s watchlist=%s",
             changed_preset, wl_counts)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trading_agent.apply_preset_update",
        description="Apply a staged daily-reviewer preset update (skill 56).",
    )
    parser.add_argument("proposal_id", help="UUID hex from the .json filename.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive prompt (still runs the gate).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the diff and exit without applying.")
    args = parser.parse_args(argv)
    return apply(args.proposal_id, assume_yes=args.yes, dry_run=args.dry_run)


if __name__ == "__main__":                                            # pragma: no cover
    sys.exit(main())

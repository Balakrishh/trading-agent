"""Write proposals to ``pending_orders/*.json`` (skill 51 §3).

CRITICAL: this module MUST NOT import ``trading_agent.executor`` or
any order-submission primitive. Its only job is to serialize a
proposal to disk so the promote CLI (skill 55) can review and
optionally submit it. The read/write boundary is enforced by
``tests/conformance/test_skill_55_promote_is_sole_writer.py``.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PENDING_DIR = Path(os.environ.get(
    "TRADING_AGENT_PENDING_ORDERS_DIR",
    "pending_orders",
)).resolve()


def write(
    *,
    underlying: str,
    strategy: str,
    params: Dict[str, Any],
    verdict: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
    preset_name: str,
    auto_promote_requested: bool = False,
    proposed_by: str = "claude-code",
) -> str:
    """Persist a proposal atomically. Returns the proposal UUID.

    Atomicity uses the temp-plus-rename pattern from the SDLC
    conventions (skill 00 §4 — Frozen dataclasses & atomic writes).
    """
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    proposal_id = uuid.uuid4().hex
    proposed_at = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "proposal_id":           proposal_id,
        "proposed_at_utc":       proposed_at,
        "proposed_by":           proposed_by,
        "underlying":            underlying.upper(),
        "strategy":              strategy,
        "params":                dict(params or {}),
        "verdict":               dict(verdict or {}),
        "risk_snapshot":         dict(risk_snapshot or {}),
        "preset_name":           preset_name,
        "auto_promote_requested": bool(auto_promote_requested),
        "schema_version":        1,
    }
    fp = PENDING_DIR / f"{proposal_id}.json"
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(fp)
    return proposal_id


def read(proposal_id: str) -> Optional[Dict[str, Any]]:
    """Return the parsed proposal, or None if the file is missing."""
    fp = PENDING_DIR / f"{proposal_id}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def list_proposals() -> list[str]:
    """Return every proposal UUID currently in the pending dir."""
    if not PENDING_DIR.exists():
        return []
    return sorted(p.stem for p in PENDING_DIR.glob("*.json"))

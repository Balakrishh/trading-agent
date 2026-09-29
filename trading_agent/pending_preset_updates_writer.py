"""Write LLM-proposed preset updates to ``pending_preset_updates/*.json``
(skill 56 §3).

CRITICAL: this module MUST NOT import ``trading_agent.executor``,
``trading_agent.strategy_presets.save_active_preset``, or any writer
that mutates live config. Its only job is to serialize a proposal so
the apply CLI can review and optionally persist it. The invariant is
verified by
``tests/conformance/test_skill_56_daily_reviewer.py``.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PENDING_DIR = Path(os.environ.get(
    "TRADING_AGENT_PENDING_PRESET_UPDATES_DIR",
    "pending_preset_updates",
)).resolve()


def write(
    *,
    proposed_by: str = "daily_reviewer",
    review_date: str,
    preset_snapshot: Dict[str, Any],
    preset_diff: Dict[str, Any],
    watchlist_diff: Optional[Dict[str, Any]] = None,
    observations: Optional[list[str]] = None,
    llm_reasoning: str = "",
    confidence: float = 0.0,
) -> str:
    """Persist a review proposal atomically. Returns the proposal UUID.

    Same temp-plus-rename discipline as ``pending_orders_writer`` — a
    concurrently-launched apply CLI can never observe a half-written
    file (skill 00 §4).
    """
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    proposal_id = uuid.uuid4().hex
    payload: Dict[str, Any] = {
        "proposal_id":     proposal_id,
        "proposed_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposed_by":     proposed_by,
        "review_date":     review_date,
        "preset_snapshot": dict(preset_snapshot or {}),
        "preset_diff":     dict(preset_diff or {}),
        "watchlist_diff":  dict(watchlist_diff or {}),
        "observations":    list(observations or []),
        "llm_reasoning":   llm_reasoning,
        "confidence":      float(confidence),
        "schema_version":  1,
    }
    fp = PENDING_DIR / f"{proposal_id}.json"
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(fp)
    return proposal_id


def read(proposal_id: str) -> Optional[Dict[str, Any]]:
    fp = PENDING_DIR / f"{proposal_id}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def list_proposals() -> list[str]:
    if not PENDING_DIR.exists():
        return []
    return sorted(p.stem for p in PENDING_DIR.glob("*.json"))

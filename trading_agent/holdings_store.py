"""
holdings_store.py — JSON-backed persistence for the operator's pasted holdings.

Skill: ``docs/skills/41_positions_provider.md`` §3.4.

Mirrors ``watchlist_store.py`` exactly so the two share atomic-write
semantics, schema versioning, RLock, and the same gitignored
``knowledge_base/`` location.

Why a snapshot file (not JSONL event log)
-----------------------------------------
Holdings are a *current-state* artefact, not a sequence of events. The
operator pastes their book once, the Long-Term Evaluator re-parses on
every refresh. Re-parsing the raw blob (rather than serialising
``Position`` dataclasses) means parser improvements automatically apply
to historical pastes — a JSONL of pre-parsed positions would freeze
whatever shape the parser produced on the day of the paste.

The file lives at ``knowledge_base/holdings.json`` (gitignored
alongside ``watchlist.json``) so the operator's real-money positions
never get committed.

Schema (v1)::

    {
      "schema_version": 1,
      "saved_at": "2026-06-16T20:14:23Z",
      "raw_paste": "<verbatim JSON blob the operator pasted>",
      "parsed_count": 13,
      "notes": ""
    }

``raw_paste`` is the bytes-equal text the operator put in the textarea
— Schwab export, canonical shape, mixed format — whatever they typed.
``parsed_count`` is decorative; it lets the UI show "13 positions
saved" without re-parsing on the load path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_HOLDINGS_PATH = Path("knowledge_base/holdings.json")
SCHEMA_VERSION = 1

# Single process-wide reentrant lock — same pattern as watchlist_store. A
# plain Lock would self-deadlock when save_holdings() is called from
# inside a load → mutate → save sequence (none today, future-proofing).
_WRITE_LOCK = threading.RLock()


@dataclass
class HoldingsSnapshot:
    """In-memory representation of the saved holdings paste."""

    raw_paste: str = ""
    saved_at: str = ""
    parsed_count: int = 0
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def is_empty(self) -> bool:
        return not (self.raw_paste or "").strip()


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_holdings(path: Path = DEFAULT_HOLDINGS_PATH) -> HoldingsSnapshot:
    """Read the persisted holdings paste; return an empty snapshot if missing."""
    p = Path(path)
    if not p.exists():
        return HoldingsSnapshot()

    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Holdings file unreadable (%s) — returning empty.", exc)
        return HoldingsSnapshot()

    version = int(raw.get("schema_version", 1))
    if version != SCHEMA_VERSION:
        logger.warning(
            "Holdings schema_version=%d differs from current=%d; "
            "loading best-effort.", version, SCHEMA_VERSION,
        )

    return HoldingsSnapshot(
        raw_paste=str(raw.get("raw_paste", "")),
        saved_at=str(raw.get("saved_at", "")),
        parsed_count=int(raw.get("parsed_count", 0) or 0),
        notes=str(raw.get("notes", "")),
        schema_version=SCHEMA_VERSION,
    )


def save_holdings(
    snapshot: HoldingsSnapshot,
    path: Path = DEFAULT_HOLDINGS_PATH,
) -> HoldingsSnapshot:
    """Atomic write to *path*. Creates parent dir if needed.

    The on-disk ``saved_at`` is always stamped to now (UTC) regardless
    of what the snapshot carried in — the operator wants "when did I
    last save?" not "when did I last edit?".
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    snapshot.saved_at = now
    payload = asdict(snapshot)
    payload["schema_version"] = SCHEMA_VERSION
    tmp = p.with_suffix(p.suffix + ".tmp")
    with _WRITE_LOCK:
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, p)
    logger.info(
        "Holdings saved (%d positions, %d chars) → %s",
        snapshot.parsed_count, len(snapshot.raw_paste), p,
    )
    return snapshot


def clear_holdings(path: Path = DEFAULT_HOLDINGS_PATH) -> None:
    """Delete the persisted holdings file. Idempotent (no-op if missing).

    Used by the Streamlit "Reset" button when the operator wants to
    purge their saved paste — e.g. before sharing screenshots, or
    when they've made a structural mistake they want to redo from
    scratch rather than edit-in-place.
    """
    p = Path(path)
    with _WRITE_LOCK:
        try:
            p.unlink()
            logger.info("Holdings file cleared → %s", p)
        except FileNotFoundError:
            logger.debug("Holdings file already absent at %s — no-op.", p)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", p, exc)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def update_paste(
    raw_paste: str,
    parsed_count: int,
    notes: Optional[str] = None,
    path: Path = DEFAULT_HOLDINGS_PATH,
) -> HoldingsSnapshot:
    """Atomic save with the just-pasted blob + parse count.

    The Streamlit UI calls this after a successful parse so the file
    state always reflects "the last thing the operator confirmed
    works", not whatever's mid-edit in the textarea.
    """
    snapshot = HoldingsSnapshot(
        raw_paste=raw_paste,
        parsed_count=parsed_count,
        notes=notes if notes is not None else "",
    )
    return save_holdings(snapshot, path)

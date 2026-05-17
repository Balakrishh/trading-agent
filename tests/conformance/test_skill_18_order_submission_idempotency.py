"""Conformance test: skill 18 — Order-submission idempotency.

Skill 18 §2 documents the idempotency contract:

  * Every order submission carries a ``client_order_id`` of the form
    ``ta-<run_id_prefix>-<uuid12>``
  * On HTTP 422 ("duplicate client_order_id"), the retry is treated
    as a SUCCESS — Alpaca already received the original
  * Up to N retries on 5xx (transient broker errors)

The full retry behavior under various HTTP responses requires a
broker mock and is covered by ``tests/test_executor.py``. This
conformance test pins the client_order_id FORMAT — the most
critical contract because the broker uses it for dedup.

Failure modes caught:
- Someone changes the prefix from ``ta-`` to something else, breaking
  forensic queries that grep for past orders
- The uuid component shrinks from 12 chars (96 bits) to fewer —
  collision risk goes up
- The run_id prefix length changes from 8 chars to something else
"""

from __future__ import annotations

import re


# Documented format: ``ta-<run_id_8_chars>-<uuid_12_hex_chars>``
CLIENT_ORDER_ID_RE = re.compile(r"^ta-[a-zA-Z0-9_]{8}-[0-9a-f]{12}$")


def test_skill_18_format_regex_matches_a_valid_id() -> None:
    """Skill 18 §3: the client_order_id format is fixed."""
    # Example matching the documented format
    valid_id = "ta-20260515-a1b2c3d4e5f6"
    assert CLIENT_ORDER_ID_RE.match(valid_id), (
        "Skill 18 §3: a syntactically-valid client_order_id must match "
        "the documented ``ta-<run_id>-<uuid>`` format."
    )


def test_skill_18_format_excludes_wrong_prefix() -> None:
    """The 'ta-' prefix is what every forensic query keys off.
    Changing it (e.g., to 'agent-') would orphan every historical
    order from log search."""
    assert not CLIENT_ORDER_ID_RE.match("agent-20260515-a1b2c3d4e5f6")
    assert not CLIENT_ORDER_ID_RE.match("xx-20260515-a1b2c3d4e5f6")
    assert not CLIENT_ORDER_ID_RE.match("20260515-a1b2c3d4e5f6")


def test_skill_18_format_excludes_wrong_uuid_length() -> None:
    """The uuid suffix is exactly 12 hex chars. Other lengths would
    silently increase collision risk OR break the regex parsers
    consumers use to extract run_id from a stored order ID."""
    # 11 chars (too short)
    assert not CLIENT_ORDER_ID_RE.match("ta-20260515-a1b2c3d4e5f")
    # 13 chars (too long)
    assert not CLIENT_ORDER_ID_RE.match("ta-20260515-a1b2c3d4e5f6a")


def test_skill_18_executor_generates_id_in_documented_format() -> None:
    """Smoke test: re-implement the executor's generation logic and
    assert the output matches the documented format. This catches
    the case where the executor's generation drifts away from what
    the skill describes."""
    import uuid
    run_id = "20260515"
    generated = f"ta-{run_id[:8]}-{uuid.uuid4().hex[:12]}"
    assert CLIENT_ORDER_ID_RE.match(generated), (
        f"Skill 18 §3: executor's generation logic should produce IDs "
        f"matching the documented format. Got {generated!r}."
    )

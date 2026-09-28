"""Bearer-token auth for the Schwab data server. Skill 47 §3.2."""

from __future__ import annotations

import hmac
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised by ``check_bearer`` on missing / mismatched credentials.

    Callers should map this to HTTP 401 with a fixed body
    ``{"error": "unauthorized"}`` — same for missing header and wrong
    key, no oracle for an attacker.
    """


def check_bearer(
    authorization_header: Optional[str],
    *,
    expected_key: Optional[str],
) -> None:
    """Enforce ``Authorization: Bearer <expected_key>`` when set.

    * ``expected_key is None`` → auth disabled; no-op.
    * ``authorization_header`` missing / malformed / wrong → ``AuthError``.

    Comparison uses ``hmac.compare_digest`` for constant-time equality
    so an attacker can't time-side-channel the correct key.
    """
    if not expected_key:
        return   # Tailscale-only gating

    if not authorization_header:
        raise AuthError("missing")

    parts = authorization_header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("malformed")

    provided = parts[1].strip()
    if not hmac.compare_digest(provided, expected_key):
        raise AuthError("mismatch")


def build_fastapi_dependency(
    expected_key: Optional[str],
) -> Callable:
    """Return a FastAPI ``Depends()`` callable that enforces the header.

    Import lives inside the function so this module doesn't require
    FastAPI at import time — the auth logic itself is FastAPI-free and
    directly unit-testable via ``check_bearer``.
    """
    from fastapi import Header, HTTPException

    async def _dep(authorization: Optional[str] = Header(default=None)) -> None:
        try:
            check_bearer(authorization, expected_key=expected_key)
        except AuthError as exc:
            # Log at INFO with a redacted marker — never the attempted value.
            logger.info("auth denied (reason=%s)", str(exc))
            raise HTTPException(
                status_code=401,
                detail={"error": "unauthorized"},
            )

    return _dep

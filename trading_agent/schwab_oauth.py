"""
schwab_oauth.py — Schwab Trader API OAuth 2.0 helper
=====================================================

Schwab uses authorization-code OAuth 2.0 with these characteristics:

  * Access tokens last **30 minutes**.
  * Refresh tokens last **7 days** (absolute, NOT rolling — re-auth weekly).
  * The refresh token rotates every time it's used; the latest one MUST
    be persisted, otherwise the next refresh fails with 401 and the
    agent halts.

This helper hides all of that behind one method: ``get_access_token()``.
It returns a valid token, refreshing automatically when fewer than
``REFRESH_LEEWAY_SEC`` seconds remain.  Tokens persist atomically to
``~/.schwab_tokens.json`` (override via ``SCHWAB_TOKEN_PATH``) so a
restart of the agent or Streamlit doesn't trigger an unnecessary
refresh.

Headless flow
-------------
The first time, a human has to visit the authorization URL in a
browser, approve, and paste the redirected URL back to the helper.
After that the agent is autonomous for 7 days.  The CLI entry point
``python -m trading_agent.schwab_oauth login`` walks an operator
through this once.

Why we don't use a third-party Schwab SDK
------------------------------------------
The CLAUDE.md soft rules favour single-source-of-truth for primitives,
and the OAuth flow is short enough that hand-rolling it is cheaper
than vendoring a library that has its own update cadence.  The adapter
in ``market_data_schwab.py`` is the only consumer.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests

logger = logging.getLogger(__name__)

# ---- Endpoints (constants, not env-vars: only Schwab can change these) ----
AUTHORIZE_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

# Refresh the access token when this many seconds remain on its lifetime.
# 30-min token → refresh at ~2 min remaining, generous enough that a
# concurrent cycle won't see a stale token mid-call.
REFRESH_LEEWAY_SEC = 120

# (connect, read) timeout for token exchange.  Tight connect so DNS /
# VPN issues fail fast; generous read so a slow upstream still completes.
TOKEN_TIMEOUT = (3, 15)

# Default token cache path.  Override via SCHWAB_TOKEN_PATH env var.
DEFAULT_TOKEN_PATH = Path.home() / ".schwab_tokens.json"


# ---------------------------------------------------------------------------
# Token state
# ---------------------------------------------------------------------------
@dataclass
class TokenSet:
    """
    Container for the access + refresh token pair plus their expiry
    epochs.  Persisted as JSON; never logged in full (the helper logs
    only the first 6 characters of each token, padded with `…`).
    """
    access_token: str = ""
    refresh_token: str = ""
    # Epoch seconds when the access_token expires (issued_at + expires_in).
    expires_at: float = 0.0
    # Epoch seconds when the refresh_token expires (issued_at + 7 days).
    # Schwab returns 'expires_in' for access only, so we compute this
    # ourselves at issuance time.
    refresh_expires_at: float = 0.0
    # Token type — Schwab always returns "Bearer", but persist for safety.
    token_type: str = "Bearer"
    # Scope, ID token, and any extra fields we don't structurally consume
    # but want to keep around for debugging.
    extras: dict = field(default_factory=dict)

    def is_access_expiring(self, leeway: float = REFRESH_LEEWAY_SEC) -> bool:
        return time.time() + leeway >= self.expires_at

    def is_refresh_expired(self) -> bool:
        return time.time() >= self.refresh_expires_at

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "refresh_expires_at": self.refresh_expires_at,
            "token_type": self.token_type,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenSet":
        return cls(
            access_token=d.get("access_token", ""),
            refresh_token=d.get("refresh_token", ""),
            expires_at=float(d.get("expires_at", 0.0)),
            refresh_expires_at=float(d.get("refresh_expires_at", 0.0)),
            token_type=d.get("token_type", "Bearer"),
            extras=d.get("extras", {}),
        )


def _redact(token: str, keep: int = 6) -> str:
    """Return a log-safe preview: 'abc123…(45ch)' for an opaque token."""
    if not token:
        return "<empty>"
    return f"{token[:keep]}…(+{max(0, len(token) - keep)}ch)"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
class SchwabOAuth:
    """
    Manages a Schwab OAuth token pair: load → refresh → expose.

    Typical usage from the adapter::

        oauth = SchwabOAuth.from_env()        # reads .env
        token = oauth.get_access_token()      # auto-refreshes
        r = requests.get(URL, headers={"Authorization": f"Bearer {token}"})

    Construction
    ------------
    Use :meth:`from_env` for production wiring (reads SCHWAB_CLIENT_ID,
    SCHWAB_CLIENT_SECRET, SCHWAB_REDIRECT_URI, SCHWAB_TOKEN_PATH).
    Use the regular constructor in tests so the secret material can be
    injected without polluting ``os.environ``.
    """

    def __init__(self, client_id: str, client_secret: str,
                 redirect_uri: str,
                 token_path: Path = DEFAULT_TOKEN_PATH):
        if not client_id or not client_secret:
            raise ValueError(
                "SchwabOAuth requires both client_id and client_secret. "
                "Did you forget to add SCHWAB_CLIENT_ID / SCHWAB_CLIENT_SECRET "
                "to .env?"
            )
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_path = Path(token_path)
        self._tokens: Optional[TokenSet] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "SchwabOAuth":
        """
        Build from os.environ. Required env vars:
          - SCHWAB_CLIENT_ID
          - SCHWAB_CLIENT_SECRET
          - SCHWAB_REDIRECT_URI  (e.g. https://127.0.0.1:8182)
        Optional:
          - SCHWAB_TOKEN_PATH    (defaults to ~/.schwab_tokens.json)
        """
        token_path_str = os.environ.get("SCHWAB_TOKEN_PATH", "").strip()
        token_path = Path(token_path_str) if token_path_str else DEFAULT_TOKEN_PATH
        return cls(
            client_id=os.environ.get("SCHWAB_CLIENT_ID", "").strip(),
            client_secret=os.environ.get("SCHWAB_CLIENT_SECRET", "").strip(),
            redirect_uri=os.environ.get(
                "SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182"
            ).strip(),
            token_path=token_path,
        )

    # ------------------------------------------------------------------
    # Persistence — atomic temp+rename per CLAUDE.md soft rule
    # ------------------------------------------------------------------
    def _save(self, ts: TokenSet) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.token_path.with_suffix(self.token_path.suffix + ".tmp")
        tmp.write_text(json.dumps(ts.to_dict(), indent=2))
        tmp.replace(self.token_path)
        # Make sure no one else can read your refresh token on a shared host.
        try:
            os.chmod(self.token_path, 0o600)
        except OSError:  # noqa: PERF203 — Windows / FS without permission bits
            pass

    def _load(self) -> Optional[TokenSet]:
        if not self.token_path.exists():
            return None
        try:
            data = json.loads(self.token_path.read_text())
            return TokenSet.from_dict(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Schwab token cache at %s is unreadable (%s) — "
                "treating as missing; re-auth required.",
                self.token_path, exc,
            )
            return None

    # ------------------------------------------------------------------
    # OAuth primitives
    # ------------------------------------------------------------------
    def _basic_auth_header(self) -> dict:
        """Schwab's token endpoint takes Basic auth: client_id:client_secret."""
        creds = f"{self.client_id}:{self.client_secret}".encode()
        b64 = base64.b64encode(creds).decode()
        return {
            "Authorization": f"Basic {b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def _ingest_token_response(self, body: dict) -> TokenSet:
        """
        Convert a Schwab token-response payload into a :class:`TokenSet`,
        compute absolute expiry epochs, and persist atomically.
        """
        now = time.time()
        access_lifetime = float(body.get("expires_in", 1800))   # 30 min default
        # Schwab doesn't expose refresh_expires_in; their docs say 7 days.
        # Compute it from issuance time so the helper can pre-emptively
        # warn when re-auth is needed.
        refresh_lifetime_sec = 7 * 24 * 3600   # 7 days

        ts = TokenSet(
            access_token=str(body.get("access_token", "")),
            refresh_token=str(body.get("refresh_token", "")),
            expires_at=now + access_lifetime,
            refresh_expires_at=now + refresh_lifetime_sec,
            token_type=str(body.get("token_type", "Bearer")),
            extras={
                k: v for k, v in body.items()
                if k not in {"access_token", "refresh_token", "token_type",
                             "expires_in"}
            },
        )
        self._save(ts)
        self._tokens = ts
        logger.info(
            "Schwab tokens refreshed — access expires in %.0f min, "
            "refresh expires in %.1f days. (access=%s, refresh=%s)",
            access_lifetime / 60, refresh_lifetime_sec / 86400,
            _redact(ts.access_token), _redact(ts.refresh_token),
        )
        return ts

    # ------------------------------------------------------------------
    # Authorization-code exchange (one-time, manual)
    # ------------------------------------------------------------------
    def build_authorization_url(self) -> str:
        """
        Return the URL the operator must visit in a browser to grant
        access.  Schwab will redirect back to ``redirect_uri?code=...``
        upon approval; the operator copies that full URL into
        :meth:`exchange_authorization_code`.
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            # 'readonly' is the documented scope on the swagger; Schwab
            # also accepts no scope for full read+trade if the app is
            # provisioned that way.
            "scope": "readonly",
        }
        return f"{AUTHORIZE_URL}?{urlencode(params)}"

    def exchange_authorization_code(self, redirect_url_or_code: str) -> TokenSet:
        """
        Trade the authorization code for the first token pair.

        ``redirect_url_or_code`` accepts either the bare ``code`` query
        param value, or the full URL Schwab redirected the operator to
        (the helper extracts ``?code=...`` from the URL).
        """
        code = self._extract_code(redirect_url_or_code)
        if not code:
            raise ValueError(
                "Could not parse authorization code from input. "
                "Paste either the bare `code=...` value or the full "
                "redirect URL Schwab sent you."
            )
        body = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }
        resp = requests.post(
            TOKEN_URL,
            headers=self._basic_auth_header(),
            data=body,
            timeout=TOKEN_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Schwab token exchange failed: HTTP {resp.status_code} — "
                f"{resp.text[:300]}"
            )
        return self._ingest_token_response(resp.json())

    @staticmethod
    def _extract_code(s: str) -> str:
        """Pull the `code` value out of either a URL or a bare string."""
        s = (s or "").strip()
        if not s:
            return ""
        if s.startswith(("http://", "https://")):
            qs = parse_qs(urlparse(s).query)
            vals = qs.get("code") or []
            return vals[0] if vals else ""
        if s.startswith("code="):
            return s.partition("=")[2]
        return s

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def _refresh(self, ts: TokenSet) -> TokenSet:
        """
        Call POST /token with grant_type=refresh_token.  Schwab returns
        a brand-new refresh_token alongside the new access_token; we
        persist the new one immediately so a crash mid-cycle doesn't
        invalidate the cache.
        """
        if not ts.refresh_token:
            raise RuntimeError(
                "No refresh_token available. Run "
                "`python -m trading_agent.schwab_oauth login` "
                "to perform the one-time authorization-code exchange."
            )
        if ts.is_refresh_expired():
            raise RuntimeError(
                "Schwab refresh token expired (>7 days old). Run "
                "`python -m trading_agent.schwab_oauth login` to re-auth."
            )
        body = {
            "grant_type": "refresh_token",
            "refresh_token": ts.refresh_token,
        }
        resp = requests.post(
            TOKEN_URL,
            headers=self._basic_auth_header(),
            data=body,
            timeout=TOKEN_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Schwab token refresh failed: HTTP {resp.status_code} — "
                f"{resp.text[:300]}"
            )
        return self._ingest_token_response(resp.json())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_access_token(self) -> str:
        """
        Return a valid access token, refreshing it transparently when
        fewer than :data:`REFRESH_LEEWAY_SEC` seconds remain on its
        lifetime.

        Raises
        ------
        RuntimeError
            * No token cache and no auth code yet provided.
            * The refresh token has expired (must re-auth manually).
            * Schwab returned a non-200 on the refresh exchange.
        """
        if self._tokens is None:
            self._tokens = self._load()
        if self._tokens is None:
            raise RuntimeError(
                "No Schwab tokens on disk yet. Run "
                "`python -m trading_agent.schwab_oauth login` to perform "
                "the one-time authorization-code exchange."
            )
        if self._tokens.is_access_expiring():
            logger.info(
                "Schwab access token expiring in %.0fs — refreshing.",
                max(0.0, self._tokens.expires_at - time.time()),
            )
            self._tokens = self._refresh(self._tokens)
        return self._tokens.access_token

    def authorization_status(self) -> dict:
        """Operator-facing diagnostic — what's our current state?"""
        if self._tokens is None:
            self._tokens = self._load()
        if self._tokens is None:
            return {"state": "unauthorized",
                    "message": "No tokens — run the login flow."}
        now = time.time()
        return {
            "state": "ok" if not self._tokens.is_refresh_expired() else "expired",
            "access_expires_in_sec": int(self._tokens.expires_at - now),
            "refresh_expires_in_days": round(
                (self._tokens.refresh_expires_at - now) / 86400, 2),
            "access_token_preview": _redact(self._tokens.access_token),
            "refresh_token_preview": _redact(self._tokens.refresh_token),
            "token_path": str(self.token_path),
        }


# ---------------------------------------------------------------------------
# CLI entry point: `python -m trading_agent.schwab_oauth {login,status}`
# ---------------------------------------------------------------------------
def _cli_login(oauth: SchwabOAuth) -> int:
    print("Schwab Trader API — one-time authorization-code exchange")
    print("=" * 60)
    print()
    print("1. Open this URL in a browser logged into your Schwab account:")
    print()
    print(f"    {oauth.build_authorization_url()}")
    print()
    print("2. Approve the app. Schwab will redirect to a URL like:")
    print(f"    {oauth.redirect_uri}/?code=XYZ&session=...")
    print()
    print("3. Copy the FULL redirect URL (or just the `code=...` value) "
          "and paste it below.")
    print()
    try:
        pasted = input("Redirect URL or code: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 1
    if not pasted:
        print("Empty input — aborting.")
        return 1
    try:
        ts = oauth.exchange_authorization_code(pasted)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 2
    print()
    print(f"OK. Tokens saved to {oauth.token_path}.")
    print(f"Access expires at  {time.ctime(ts.expires_at)}")
    print(f"Refresh expires at {time.ctime(ts.refresh_expires_at)}  "
          f"(re-run this command before then to avoid downtime)")
    return 0


def _cli_status(oauth: SchwabOAuth) -> int:
    status = oauth.authorization_status()
    print(json.dumps(status, indent=2))
    return 0 if status.get("state") == "ok" else 2


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "status"

    try:
        oauth = SchwabOAuth.from_env()
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    if cmd == "login":
        return _cli_login(oauth)
    if cmd == "status":
        return _cli_status(oauth)
    print(f"Unknown subcommand: {cmd!r}. Use `login` or `status`.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

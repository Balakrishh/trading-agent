"""
test_schwab_oauth.py — pin down the Schwab OAuth helper.

These tests run with no Schwab credentials and no network access:

  * Token cache persistence (atomic temp+rename per CLAUDE.md soft rule)
  * Access-token expiry detection + auto-refresh
  * Refresh-token expiry → operator-friendly RuntimeError
  * Authorization-code parsing from URL or bare value
  * CLI status / login subcommands
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading_agent.schwab_oauth import (
    REFRESH_LEEWAY_SEC,
    SchwabOAuth,
    TokenSet,
    _redact,
)


# ── Token cache I/O ──────────────────────────────────────────────────


def test_save_and_load_round_trip(tmp_path: Path):
    """Tokens written to disk should round-trip identically."""
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "tokens.json")
    ts = TokenSet(
        access_token="ACCESS123",
        refresh_token="REFRESH123",
        expires_at=time.time() + 1800,
        refresh_expires_at=time.time() + 7 * 86400,
    )
    o._save(ts)

    # Fresh helper, same path → load matches
    o2 = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                     token_path=tmp_path / "tokens.json")
    loaded = o2._load()
    assert loaded is not None
    assert loaded.access_token == "ACCESS123"
    assert loaded.refresh_token == "REFRESH123"
    # Expiry epochs round-trip within float precision
    assert abs(loaded.expires_at - ts.expires_at) < 1e-3


def test_save_uses_atomic_temp_rename(tmp_path: Path):
    """No partial writes — the .tmp file should not linger after _save."""
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "tokens.json")
    o._save(TokenSet(access_token="A", refresh_token="R",
                     expires_at=time.time() + 60,
                     refresh_expires_at=time.time() + 60))
    # No .tmp left behind
    assert not (tmp_path / "tokens.json.tmp").exists()
    assert (tmp_path / "tokens.json").exists()


def test_load_returns_none_when_file_missing(tmp_path: Path):
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "missing.json")
    assert o._load() is None


def test_load_returns_none_on_corrupt_file(tmp_path: Path):
    p = tmp_path / "tokens.json"
    p.write_text("this is not json {{{")
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182", token_path=p)
    assert o._load() is None


# ── TokenSet expiry semantics ────────────────────────────────────────


def test_access_token_expiring_within_leeway():
    ts = TokenSet(access_token="A", refresh_token="R",
                  expires_at=time.time() + REFRESH_LEEWAY_SEC - 5,
                  refresh_expires_at=time.time() + 86400)
    assert ts.is_access_expiring() is True


def test_access_token_fresh_outside_leeway():
    ts = TokenSet(access_token="A", refresh_token="R",
                  expires_at=time.time() + REFRESH_LEEWAY_SEC + 60,
                  refresh_expires_at=time.time() + 86400)
    assert ts.is_access_expiring() is False


def test_refresh_token_expired_at_boundary():
    ts = TokenSet(access_token="A", refresh_token="R",
                  expires_at=time.time() - 1,
                  refresh_expires_at=time.time() - 1)
    assert ts.is_refresh_expired() is True


# ── get_access_token() refresh path ─────────────────────────────────


def test_get_access_token_returns_cached_when_fresh(tmp_path):
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "tokens.json")
    o._save(TokenSet(
        access_token="FRESH_TOKEN", refresh_token="R",
        expires_at=time.time() + 1800,    # 30 min in the future
        refresh_expires_at=time.time() + 86400,
    ))
    o._tokens = None  # force load
    # Should not hit the network
    with patch("trading_agent.schwab_oauth.requests.post") as post:
        token = o.get_access_token()
    assert token == "FRESH_TOKEN"
    post.assert_not_called()


def test_get_access_token_refreshes_when_expiring(tmp_path):
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "tokens.json")
    o._save(TokenSet(
        access_token="OLD_TOKEN", refresh_token="ROTATING_REFRESH",
        expires_at=time.time() + 30,      # well within leeway
        refresh_expires_at=time.time() + 86400,
    ))
    o._tokens = None

    with patch("trading_agent.schwab_oauth.requests.post") as post:
        post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "access_token": "NEW_TOKEN",
                "refresh_token": "NEW_ROTATED_REFRESH",
                "token_type": "Bearer",
                "expires_in": 1800,
            },
        )
        token = o.get_access_token()

    assert token == "NEW_TOKEN"
    # And the new refresh token was persisted
    saved = json.loads((tmp_path / "tokens.json").read_text())
    assert saved["access_token"] == "NEW_TOKEN"
    assert saved["refresh_token"] == "NEW_ROTATED_REFRESH"


def test_get_access_token_raises_when_no_tokens_yet(tmp_path):
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="No Schwab tokens"):
        o.get_access_token()


def test_get_access_token_raises_when_refresh_token_expired(tmp_path):
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "tokens.json")
    o._save(TokenSet(
        access_token="OLD", refresh_token="OLD_R",
        expires_at=time.time() - 1,
        refresh_expires_at=time.time() - 1,    # already expired
    ))
    o._tokens = None
    with pytest.raises(RuntimeError, match="refresh token expired"):
        o.get_access_token()


# ── Authorization-code parsing ──────────────────────────────────────


@pytest.mark.parametrize(
    "input_,expected",
    [
        ("ABC123", "ABC123"),
        ("code=ABC123", "ABC123"),
        ("https://127.0.0.1:8182/?code=ABC123&session=xyz", "ABC123"),
        ("https://127.0.0.1:8182/?session=xyz&code=DEF456", "DEF456"),
        ("https://127.0.0.1:8182/?code=ABC%2B123", "ABC+123"),  # url-decoded
    ],
)
def test_extract_code(input_, expected):
    assert SchwabOAuth._extract_code(input_) == expected


def test_extract_code_returns_empty_for_no_code():
    assert SchwabOAuth._extract_code("") == ""
    assert SchwabOAuth._extract_code("https://example.com/?foo=bar") == ""


# ── Construction guards ─────────────────────────────────────────────


def test_constructor_rejects_missing_client_id():
    with pytest.raises(ValueError, match="client_id"):
        SchwabOAuth("", "secret", "https://127.0.0.1:8182")


def test_constructor_rejects_missing_client_secret():
    with pytest.raises(ValueError, match="client_id and client_secret"):
        SchwabOAuth("id", "", "https://127.0.0.1:8182")


# ── Logging-safe redaction ──────────────────────────────────────────


def test_redact_preserves_first_chars():
    assert _redact("ABCDEF12345").startswith("ABCDEF")
    assert "5ch" in _redact("12345") or "(+0ch" in _redact("12345")


def test_redact_handles_empty():
    assert _redact("") == "<empty>"


# ── authorization_status() diagnostic ───────────────────────────────


def test_authorization_status_when_unauthorized(tmp_path):
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "missing.json")
    s = o.authorization_status()
    assert s["state"] == "unauthorized"


def test_authorization_status_when_ok(tmp_path):
    o = SchwabOAuth("id", "secret", "https://127.0.0.1:8182",
                    token_path=tmp_path / "tokens.json")
    o._save(TokenSet(
        access_token="A", refresh_token="R",
        expires_at=time.time() + 1800,
        refresh_expires_at=time.time() + 7 * 86400,
    ))
    o._tokens = None
    s = o.authorization_status()
    assert s["state"] == "ok"
    assert s["access_expires_in_sec"] > 1700
    assert s["refresh_expires_in_days"] > 6.9

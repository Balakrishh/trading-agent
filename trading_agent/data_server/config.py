"""Environment-driven configuration for the Schwab data server. Skill 47 §3.4."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_PORT = 8765
DEFAULT_BIND = "127.0.0.1"       # localhost by default; operator overrides for tailnet
DEFAULT_PRICE_TTL_SEC = 60       # only applied when SCHWAB_API_CACHE_ENABLED=true
DEFAULT_SNAPSHOT_TTL_SEC = 90    # only applied when SCHWAB_API_CACHE_ENABLED=true


@dataclass(frozen=True)
class CacheTTLs:
    """TTL (in seconds) per cached endpoint. Chain fetches are cached
    inside the provider (skill 16) — this dataclass only carries the
    TTLs the server layer needs on top.

    When ``cache_enabled=False`` on the parent ServerConfig, ALL TTLs
    materialize as 0 regardless of the per-endpoint env vars. Callers
    ready their fresh-vs-cached posture with a single master switch."""
    price_sec: int = DEFAULT_PRICE_TTL_SEC
    snapshot_sec: int = DEFAULT_SNAPSHOT_TTL_SEC


@dataclass(frozen=True)
class ServerConfig:
    """Complete server configuration, materialized from env vars.

    Constructed via ``ServerConfig.from_env()``. Tests pass explicit
    values instead of relying on ambient environment.
    """
    api_key:       Optional[str] = None
    port:          int = DEFAULT_PORT
    bind:          str = DEFAULT_BIND
    ttls:          CacheTTLs = field(default_factory=lambda: CacheTTLs(0, 0))
    log_file:      Optional[str] = None
    cache_enabled: bool = False    # master switch — see skill 47 §3.4

    @classmethod
    def from_env(cls) -> "ServerConfig":
        cache_on = _env_bool("SCHWAB_API_CACHE_ENABLED", default=False)
        if cache_on:
            price_ttl = int(_env_str("SCHWAB_API_PRICE_TTL_SEC")
                            or DEFAULT_PRICE_TTL_SEC)
            snap_ttl = int(_env_str("SCHWAB_API_SNAPSHOT_TTL_SEC")
                           or DEFAULT_SNAPSHOT_TTL_SEC)
        else:
            # Master switch off → every request goes live. Per-endpoint
            # TTL env vars are IGNORED in this mode — operator flips
            # SCHWAB_API_CACHE_ENABLED=true first, then tunes TTLs.
            price_ttl = 0
            snap_ttl = 0
        return cls(
            api_key=_env_str("SCHWAB_API_SERVER_KEY"),
            port=int(_env_str("SCHWAB_API_PORT") or DEFAULT_PORT),
            bind=_env_str("SCHWAB_API_BIND") or DEFAULT_BIND,
            ttls=CacheTTLs(price_sec=price_ttl, snapshot_sec=snap_ttl),
            log_file=_env_str("SCHWAB_API_LOG_FILE"),
            cache_enabled=cache_on,
        )

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


def _env_str(name: str) -> Optional[str]:
    v = os.environ.get(name, "").strip()
    return v or None


def _env_bool(name: str, *, default: bool = False) -> bool:
    """Parse an env var as a boolean. Recognized truthy values (case-insensitive):
    ``1``, ``true``, ``yes``, ``on``. Anything else — including unset — is falsy.
    """
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")

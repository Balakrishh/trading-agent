"""Environment-driven configuration for the Schwab data server. Skill 47 §3.4."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_PORT = 8765
DEFAULT_BIND = "127.0.0.1"       # localhost by default; operator overrides for tailnet
DEFAULT_PRICE_TTL_SEC = 60
DEFAULT_SNAPSHOT_TTL_SEC = 90


@dataclass(frozen=True)
class CacheTTLs:
    """TTL (in seconds) per cached endpoint. Chain fetches are cached
    inside the provider itself (skill 16) — this dataclass only carries
    the TTLs the server layer needs to add on top."""
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
    ttls:          CacheTTLs = field(default_factory=CacheTTLs)
    log_file:      Optional[str] = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            api_key=_env_str("SCHWAB_API_SERVER_KEY"),
            port=int(_env_str("SCHWAB_API_PORT") or DEFAULT_PORT),
            bind=_env_str("SCHWAB_API_BIND") or DEFAULT_BIND,
            ttls=CacheTTLs(
                price_sec=int(_env_str("SCHWAB_API_PRICE_TTL_SEC")
                              or DEFAULT_PRICE_TTL_SEC),
                snapshot_sec=int(_env_str("SCHWAB_API_SNAPSHOT_TTL_SEC")
                                 or DEFAULT_SNAPSHOT_TTL_SEC),
            ),
            log_file=_env_str("SCHWAB_API_LOG_FILE"),
        )

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


def _env_str(name: str) -> Optional[str]:
    v = os.environ.get(name, "").strip()
    return v or None

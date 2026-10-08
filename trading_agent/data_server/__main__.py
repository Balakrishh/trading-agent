"""CLI entry: ``python -m trading_agent.data_server``.

Reads env vars via ``ServerConfig.from_env()``, constructs the real
``SchwabMarketDataProvider``, wires the FastAPI app, and hands to
Uvicorn. Skill 47 §3.4.

Command line:
    python -m trading_agent.data_server              # 127.0.0.1:8765
    python -m trading_agent.data_server --bind 100.115.216.79
    python -m trading_agent.data_server --port 9000

Environment overrides (all optional):
    SCHWAB_API_SERVER_KEY    — shared bearer token (required, ≥ 24 chars;
                               the server refuses to start without one)
    SCHWAB_API_ALLOW_NO_KEY  — true → start with no key (network gating
                               only; never behind Tailscale Funnel)
    SCHWAB_API_PORT          — port to bind (default 8765)
    SCHWAB_API_BIND          — address to bind (default 127.0.0.1)
    SCHWAB_API_PRICE_TTL_SEC — price cache TTL (default 60)
    SCHWAB_API_SNAPSHOT_TTL_SEC — snapshot cache TTL (default 90)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

from dotenv import dotenv_values, find_dotenv, load_dotenv

from trading_agent.data_server.app import build_app
from trading_agent.data_server.auth import key_fingerprint
from trading_agent.data_server.config import ServerConfig


def _build_default_provider():
    """Construct the real SchwabMarketDataProvider via the same factory
    the trading agent uses. Kept in a function so tests can monkeypatch
    or bypass this path entirely.
    """
    from trading_agent.config import load_config
    from trading_agent.market_data_factory import build_market_data_provider

    cfg = load_config()
    return build_market_data_provider(
        alpaca_api_key=cfg.alpaca.api_key,
        alpaca_secret_key=cfg.alpaca.secret_key,
        alpaca_data_url=cfg.alpaca.data_url,
        alpaca_base_url=cfg.alpaca.base_url,
        # Force Schwab: this server exists specifically to expose it.
        surface="data_api",
        default_provider="schwab",
    )


_KEY_VAR = "SCHWAB_API_SERVER_KEY"
_ALLOW_NO_KEY_VAR = "SCHWAB_API_ALLOW_NO_KEY"
MIN_KEY_LENGTH = 24
EXIT_KEY_POLICY = 4


def key_policy_error(api_key: Optional[str], allow_no_key: bool) -> Optional[str]:
    """Startup refusal reason, or None when the key is acceptable.

    The server may be published to the internet (Tailscale Funnel), where
    the bearer key is the only protection, so a missing or short key is a
    hard stop (2026-10-08). ``allow_no_key`` (SCHWAB_API_ALLOW_NO_KEY)
    opts back in to keyless, network-gated mode; it never waives the
    length check for a key that *is* set."""
    key = (api_key or "").strip()
    if not key:
        if allow_no_key:
            return None
        return (f"{_KEY_VAR} is not set — refusing to start: without it the "
                f"server answers anyone who can reach it (including through "
                f"Tailscale Funnel). Set it in .env (e.g. python -c \"import "
                f"secrets; print(secrets.token_urlsafe(32))\"), or set "
                f"{_ALLOW_NO_KEY_VAR}=true for a tailnet-only server.")
    if len(key) < MIN_KEY_LENGTH:
        return (f"{_KEY_VAR} is only {len(key)} characters — refusing to start; "
                f"use at least {MIN_KEY_LENGTH} (e.g. secrets.token_urlsafe(32)).")
    return None


def describe_key_source(shell_key: str, dotenv_key: str,
                        effective_key: str) -> Tuple[str, Optional[str]]:
    """Return (source label, optional mismatch warning) for the startup log.

    ``shell_key`` — value exported in the launching shell (before .env
    load); ``dotenv_key`` — value in .env; ``effective_key`` — what the
    server enforces. An exported var wins over .env (load_dotenv default),
    which silently diverges from a client that only reads .env.
    """
    if shell_key:
        source = "shell env (overrides .env)" if dotenv_key else "shell env"
    elif dotenv_key:
        source = ".env"
    else:
        source = "unset"
    warning = None
    if shell_key and dotenv_key and shell_key != dotenv_key:
        warning = (
            f"{_KEY_VAR} in the launching shell (fp={key_fingerprint(shell_key)}) "
            f"DIFFERS from .env (fp={key_fingerprint(dotenv_key)}) — the shell "
            f"value is enforced, so clients using the .env key will get 401. "
            f"Restart with: env -u {_KEY_VAR} python -m trading_agent.data_server"
        )
    return source, warning


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trading_agent.data_server",
        description="Schwab data API — local HTTP server (skill 47).",
    )
    parser.add_argument(
        "--bind", default=None,
        help="Address to bind. Default: env SCHWAB_API_BIND or 127.0.0.1.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to bind. Default: env SCHWAB_API_PORT or 8765.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("trading_agent.data_server")

    # Load .env BEFORE reading SCHWAB_API_SERVER_KEY. Pre-2026-09-29 the
    # key was read here from the launching shell only; .env was loaded
    # later (inside load_config() when the provider was built), so a key
    # set in .env never took effect and the MCP — which reads .env —
    # got 401 on every call. An already-exported var still wins.
    shell_key = os.environ.get(_KEY_VAR, "").strip()
    dotenv_key = (dotenv_values(find_dotenv()).get(_KEY_VAR) or "").strip()
    load_dotenv()
    cfg = ServerConfig.from_env()
    key_source, key_warning = describe_key_source(
        shell_key, dotenv_key, cfg.api_key or "")
    bind = args.bind or cfg.bind
    port = args.port or cfg.port

    # Startup-info line the operator scans to confirm the right thing
    # is running. NEVER log the actual API key (skill 47 §4).
    cache_mode = (
        f"ENABLED (price_ttl={cfg.ttls.price_sec}s "
        f"snapshot_ttl={cfg.ttls.snapshot_sec}s)"
        if cfg.cache_enabled
        else "DISABLED (every request live — set "
             "SCHWAB_API_CACHE_ENABLED=true to enable)"
    )
    log.info(
        "Schwab Data API starting — bind=%s port=%d auth=%s cache=%s",
        bind, port,
        "bearer-token" if cfg.auth_enabled else "TAILSCALE-ONLY (no api key)",
        cache_mode,
    )
    # Fingerprint only (sha256[:8]) — compare with the client's
    # `printf %s "$SCHWAB_API_SERVER_KEY" | shasum -a 256 | cut -c1-8`.
    log.info("API key source=%s fingerprint=%s",
             key_source, key_fingerprint(cfg.api_key))
    if key_warning:
        log.warning(key_warning)
    from trading_agent.data_server.config import _env_bool
    refusal = key_policy_error(cfg.api_key, _env_bool(_ALLOW_NO_KEY_VAR, default=False))
    if refusal:
        log.error(refusal)
        return EXIT_KEY_POLICY
    if not cfg.auth_enabled:
        log.warning(
            "SCHWAB_API_SERVER_KEY is not set and %s=true — server relies on "
            "network gating only. Never publish it with Tailscale Funnel.",
            _ALLOW_NO_KEY_VAR,
        )

    try:
        provider = _build_default_provider()
    except Exception as exc:                                    # noqa: BLE001
        log.error("Failed to construct market-data provider: %s", exc)
        return 2

    app = build_app(provider=provider, config=cfg)

    try:
        import uvicorn
    except ImportError:
        log.error(
            "uvicorn is not installed. Run: pip install 'uvicorn[standard]>=0.27'"
        )
        return 3

    uvicorn.run(app, host=bind, port=port, log_level="info")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main())

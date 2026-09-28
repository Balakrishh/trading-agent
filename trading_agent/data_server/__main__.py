"""CLI entry: ``python -m trading_agent.data_server``.

Reads env vars via ``ServerConfig.from_env()``, constructs the real
``SchwabMarketDataProvider``, wires the FastAPI app, and hands to
Uvicorn. Skill 47 §3.4.

Command line:
    python -m trading_agent.data_server              # 127.0.0.1:8765
    python -m trading_agent.data_server --bind 100.115.216.79
    python -m trading_agent.data_server --port 9000

Environment overrides (all optional):
    SCHWAB_API_SERVER_KEY    — shared bearer token; None → Tailscale-only
    SCHWAB_API_PORT          — port to bind (default 8765)
    SCHWAB_API_BIND          — address to bind (default 127.0.0.1)
    SCHWAB_API_PRICE_TTL_SEC — price cache TTL (default 60)
    SCHWAB_API_SNAPSHOT_TTL_SEC — snapshot cache TTL (default 90)
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from trading_agent.data_server.app import build_app
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

    cfg = ServerConfig.from_env()
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
    if not cfg.auth_enabled:
        log.warning(
            "SCHWAB_API_SERVER_KEY is not set — server relies on network "
            "gating only. Set the env var to require Authorization: Bearer <key>."
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

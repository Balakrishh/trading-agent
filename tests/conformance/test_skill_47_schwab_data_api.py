"""Conformance tests for skill 47 — Schwab data API server.

Pinned behaviors:

- READ-ONLY invariant: no forbidden imports in data_server/*.py — §4
- Bearer-token auth: missing / wrong / correct key mapping — §3.2
- Ticker validation: happy path + rejection cases — §4
- Cache-hit avoids re-calling the provider — §3.3
- POST /quotes and /snapshots reject empty payloads — §4
- Upstream Schwab exceptions map to 502 with a stable error body — §4
- CLI entry compiles and reads env config — §3.4

TestClient shortcut used everywhere is FastAPI's built-in ASGI test
client — no live network, no live Schwab, no sleep-and-see. Every
test is deterministic in milliseconds.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# §4 — READ-ONLY invariant: no forbidden imports in the data_server package.
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORT_MODULES = (
    "trading_agent.executor",     # order submission module
    "trading_agent.journal_kb",   # journal writes
)
FORBIDDEN_NAMES_IMPORTED = (
    "submit_order",
    "place_order",
    "TradingClient",              # alpaca-py trading client
)


def _walk_imports(source: str):
    """Yield (module_name, imported_name) tuples for every import in ``source``.

    Uses AST so prose in docstrings + comments doesn't match. For
    ``import foo``, yields ``("foo", "foo")``. For ``from foo import bar``,
    yields ``("foo", "bar")``. Wildcard imports yield ``(module, "*")``.
    """
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, alias.name.split(".")[-1]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                yield module, alias.name


def test_skill_47_readonly_no_forbidden_imports():
    """Skill 47 §4 — data_server/*.py must NOT import order/journal code.

    Uses AST inspection so the invariant applies to REAL imports only;
    the module docstring can still mention forbidden token names as prose.
    """
    server_dir = Path(__file__).resolve().parents[2] / "trading_agent" / "data_server"
    assert server_dir.is_dir(), f"data_server/ not found at {server_dir}"

    py_files = list(server_dir.glob("*.py"))
    assert py_files, "no .py files under data_server/"

    for fp in py_files:
        source = fp.read_text(encoding="utf-8")
        for module, name in _walk_imports(source):
            for bad_mod in FORBIDDEN_IMPORT_MODULES:
                assert not module.startswith(bad_mod), (
                    f"data_server/{fp.name} imports from {module!r}; "
                    f"forbidden per skill 47 §4."
                )
            assert name not in FORBIDDEN_NAMES_IMPORTED, (
                f"data_server/{fp.name} imports name {name!r}; "
                f"forbidden per skill 47 §4."
            )


def test_skill_47_server_read_only_sentinel_exists():
    """The module-level SERVER_READ_ONLY constant must be True."""
    from trading_agent.data_server import SERVER_READ_ONLY
    assert SERVER_READ_ONLY is True


# ---------------------------------------------------------------------------
# §3.2 — Bearer-token auth
# ---------------------------------------------------------------------------

def test_skill_47_check_bearer_disabled_when_no_key():
    """expected_key=None → no-op regardless of header value."""
    from trading_agent.data_server.auth import check_bearer
    check_bearer(None, expected_key=None)          # no header, no key
    check_bearer("Bearer whatever", expected_key=None)  # bogus header, no key


def test_skill_47_check_bearer_missing_header_raises():
    from trading_agent.data_server.auth import AuthError, check_bearer
    with pytest.raises(AuthError):
        check_bearer(None, expected_key="secret")


def test_skill_47_check_bearer_wrong_key_raises():
    from trading_agent.data_server.auth import AuthError, check_bearer
    with pytest.raises(AuthError):
        check_bearer("Bearer nope", expected_key="secret")


def test_skill_47_check_bearer_malformed_raises():
    from trading_agent.data_server.auth import AuthError, check_bearer
    with pytest.raises(AuthError):
        check_bearer("Basic user:pass", expected_key="secret")
    with pytest.raises(AuthError):
        check_bearer("something-without-scheme", expected_key="secret")


def test_skill_47_check_bearer_correct_key_passes():
    from trading_agent.data_server.auth import check_bearer
    check_bearer("Bearer secret", expected_key="secret")
    check_bearer("bearer secret", expected_key="secret")  # case-insensitive scheme


# ---------------------------------------------------------------------------
# §3.3 — TTLCache behavior
# ---------------------------------------------------------------------------

def test_skill_47_cache_hits_avoid_refetch():
    from trading_agent.data_server.cache import TTLCache
    cache = TTLCache(ttl_seconds=60)
    calls = {"n": 0}
    def _fetch():
        calls["n"] += 1
        return 42
    v1 = cache.get_or_set("k", _fetch)
    v2 = cache.get_or_set("k", _fetch)
    assert v1 == v2 == 42
    assert calls["n"] == 1  # second call served from cache


def test_skill_47_cache_zero_ttl_always_refetches():
    from trading_agent.data_server.cache import TTLCache
    cache = TTLCache(ttl_seconds=0)
    calls = {"n": 0}
    def _fetch():
        calls["n"] += 1
        return calls["n"]
    cache.get_or_set("k", _fetch)
    cache.get_or_set("k", _fetch)
    assert calls["n"] == 2  # both calls hit fetcher


def test_skill_47_cache_negative_ttl_rejected():
    from trading_agent.data_server.cache import TTLCache
    with pytest.raises(ValueError):
        TTLCache(ttl_seconds=-1)


# ---------------------------------------------------------------------------
# Fixtures for HTTP tests
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Deterministic MarketDataPort stub for HTTP tests."""
    def __init__(self):
        self.calls = {"price": 0, "chain": 0, "quotes": 0, "snapshots": 0}
        self.raise_on = None    # set to Exception instance to force upstream fail

    def _maybe_raise(self):
        if self.raise_on is not None:
            raise self.raise_on

    def get_current_price(self, ticker):
        self.calls["price"] += 1
        self._maybe_raise()
        return 542.30 if ticker == "SPY" else 100.0

    def fetch_option_chain(self, underlying, expiration_date, option_type):
        self.calls["chain"] += 1
        self._maybe_raise()
        return [{"symbol": f"{underlying}260918C00100000",
                 "strike": 100.0, "type": option_type, "bid": 1.0, "ask": 1.1,
                 "delta": 0.25, "dte": 21}]

    def fetch_option_quotes(self, symbols):
        self.calls["quotes"] += 1
        self._maybe_raise()
        return [{"symbol": s, "bid": 1.0, "ask": 1.1} for s in symbols]

    def fetch_batch_snapshots(self, tickers):
        self.calls["snapshots"] += 1
        self._maybe_raise()
        return {t: {"price": 100.0, "day_change_pct": 0.5} for t in tickers}

    def is_market_open(self):
        return True


def _client(*, api_key=None):
    """Return a FastAPI TestClient with a FakeProvider and optional auth."""
    from fastapi.testclient import TestClient
    from trading_agent.data_server.app import build_app
    from trading_agent.data_server.config import ServerConfig
    provider = _FakeProvider()
    cfg = ServerConfig(api_key=api_key)
    app = build_app(provider=provider, config=cfg)
    return TestClient(app), provider


def _auth_headers(key):
    return {"Authorization": f"Bearer {key}"} if key else {}


# ---------------------------------------------------------------------------
# §2 — Endpoint happy paths
# ---------------------------------------------------------------------------

def test_health_returns_ok_without_auth():
    client, _ = _client(api_key="secret")
    r = client.get("/health")  # no auth header
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_returns_ok_when_provider_healthy():
    client, _ = _client()
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_price_endpoint_returns_provider_price():
    client, provider = _client()
    r = client.get("/price/SPY")
    assert r.status_code == 200
    body = r.json()
    assert body == {"ticker": "SPY", "price": 542.30}
    # Second call should hit the cache — provider called ONCE.
    r2 = client.get("/price/SPY")
    assert r2.status_code == 200
    assert provider.calls["price"] == 1


def test_chain_endpoint_returns_contracts():
    client, _ = _client()
    r = client.get("/chain/SPY?expiration=2026-10-17&option_type=call")
    assert r.status_code == 200
    body = r.json()
    assert body["underlying"] == "SPY"
    assert body["expiration"] == "2026-10-17"
    assert body["option_type"] == "call"
    assert body["count"] == 1
    assert body["contracts"][0]["symbol"].startswith("SPY")


def test_quotes_endpoint_returns_batch():
    client, _ = _client()
    r = client.post("/quotes", json={"symbols": ["SPY", "QQQ"]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert [q["symbol"] for q in body["quotes"]] == ["SPY", "QQQ"]


def test_snapshots_endpoint_returns_batch():
    client, provider = _client()
    r = client.post("/snapshots", json={"tickers": ["SPY", "QQQ"]})
    assert r.status_code == 200
    body = r.json()
    assert set(body["snapshots"].keys()) == {"SPY", "QQQ"}
    # Same-order re-request should cache-hit (key is sorted, so any order works).
    r2 = client.post("/snapshots", json={"tickers": ["QQQ", "SPY"]})
    assert r2.status_code == 200
    assert provider.calls["snapshots"] == 1


def test_market_status_returns_bool():
    client, _ = _client()
    r = client.get("/market-status")
    assert r.status_code == 200
    assert "open" in r.json()


# ---------------------------------------------------------------------------
# §3.2 — HTTP-level auth enforcement
# ---------------------------------------------------------------------------

def test_price_requires_bearer_when_configured():
    client, _ = _client(api_key="s3cret")
    r = client.get("/price/SPY")  # no header
    assert r.status_code == 401
    assert r.json()["detail"] == {"error": "unauthorized"}


def test_price_rejects_wrong_bearer():
    client, _ = _client(api_key="s3cret")
    r = client.get("/price/SPY", headers=_auth_headers("wrong"))
    assert r.status_code == 401


def test_price_accepts_correct_bearer():
    client, _ = _client(api_key="s3cret")
    r = client.get("/price/SPY", headers=_auth_headers("s3cret"))
    assert r.status_code == 200


def test_health_stays_open_when_auth_configured():
    """Liveness must not require auth so launchd/uptime probes work."""
    client, _ = _client(api_key="s3cret")
    r = client.get("/health")  # no auth header
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# §4 — Validation + error mapping
# ---------------------------------------------------------------------------

def test_price_rejects_invalid_ticker():
    client, _ = _client()
    for bad in ("aapl", "SPY;DROP", "TOOLONGTICKER", "../etc", "", "spy"):
        r = client.get(f"/price/{bad}")
        # 404 acceptable when route parser rejects; 400 when handler rejects.
        assert r.status_code in (400, 404), f"{bad!r} → {r.status_code}"


def test_chain_rejects_invalid_option_type():
    client, _ = _client()
    r = client.get("/chain/SPY?expiration=2026-10-17&option_type=straddle")
    assert r.status_code == 400


def test_chain_rejects_bad_expiration():
    client, _ = _client()
    r = client.get("/chain/SPY?expiration=next-friday&option_type=call")
    assert r.status_code == 400


def test_quotes_rejects_empty_symbol_list():
    client, _ = _client()
    r = client.post("/quotes", json={"symbols": []})
    assert r.status_code == 422  # Pydantic min_length=1


def test_snapshots_rejects_empty_tickers_list():
    client, _ = _client()
    r = client.post("/snapshots", json={"tickers": []})
    assert r.status_code == 422


def test_upstream_schwab_error_maps_to_502():
    client, provider = _client()
    provider.raise_on = RuntimeError("Schwab HTTP 503 from /chains")
    r = client.get("/price/SPY")
    assert r.status_code == 502
    assert r.json()["detail"]["error"] in ("schwab_upstream", "schwab_auth")


def test_upstream_schwab_auth_error_maps_to_schwab_auth():
    client, provider = _client()
    provider.raise_on = RuntimeError("Schwab auth failed: refresh token expired")
    r = client.get("/price/SPY")
    assert r.status_code == 502
    assert r.json()["detail"]["error"] == "schwab_auth"


# ---------------------------------------------------------------------------
# §3.4 — CLI entry point
# ---------------------------------------------------------------------------

def test_cli_module_importable():
    """The CLI entry module must import without side effects."""
    import trading_agent.data_server.__main__ as cli
    assert hasattr(cli, "main")


def test_serverconfig_reads_env(monkeypatch):
    monkeypatch.setenv("SCHWAB_API_SERVER_KEY", "abc123")
    monkeypatch.setenv("SCHWAB_API_PORT", "9999")
    monkeypatch.setenv("SCHWAB_API_BIND", "100.115.216.79")
    monkeypatch.setenv("SCHWAB_API_PRICE_TTL_SEC", "30")
    monkeypatch.setenv("SCHWAB_API_SNAPSHOT_TTL_SEC", "45")

    from trading_agent.data_server.config import ServerConfig
    cfg = ServerConfig.from_env()
    assert cfg.api_key == "abc123"
    assert cfg.port == 9999
    assert cfg.bind == "100.115.216.79"
    assert cfg.ttls.price_sec == 30
    assert cfg.ttls.snapshot_sec == 45
    assert cfg.auth_enabled is True


def test_serverconfig_defaults_when_env_unset(monkeypatch):
    for k in ("SCHWAB_API_SERVER_KEY", "SCHWAB_API_PORT", "SCHWAB_API_BIND",
              "SCHWAB_API_PRICE_TTL_SEC", "SCHWAB_API_SNAPSHOT_TTL_SEC",
              "SCHWAB_API_LOG_FILE"):
        monkeypatch.delenv(k, raising=False)
    from trading_agent.data_server.config import ServerConfig
    cfg = ServerConfig.from_env()
    assert cfg.api_key is None
    assert cfg.port == 8765
    assert cfg.bind == "127.0.0.1"
    assert cfg.auth_enabled is False

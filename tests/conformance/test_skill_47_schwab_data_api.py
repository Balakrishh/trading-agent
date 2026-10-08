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


def _client(*, api_key=None, cache_enabled=False):
    """Return a FastAPI TestClient with a FakeProvider and optional auth.
    Caching is off by default since 29bb020 (SCHWAB_API_CACHE_ENABLED)."""
    from fastapi.testclient import TestClient
    from trading_agent.data_server.app import build_app
    from trading_agent.data_server.config import CacheTTLs, ServerConfig
    provider = _FakeProvider()
    # Direct construction defaults ttls to 0; from_env() only fills the
    # real TTLs when the master switch is on — mirror that here.
    cfg = ServerConfig(api_key=api_key, cache_enabled=cache_enabled,
                       ttls=CacheTTLs() if cache_enabled else CacheTTLs(0, 0))
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
    client, provider = _client(cache_enabled=True)
    r = client.get("/price/SPY")
    assert r.status_code == 200
    body = r.json()
    assert body == {"ticker": "SPY", "price": 542.30}
    # Second call should hit the cache — provider called ONCE.
    r2 = client.get("/price/SPY")
    assert r2.status_code == 200
    assert provider.calls["price"] == 1


def test_cache_disabled_by_default_every_request_is_live():
    """29bb020: SCHWAB_API_CACHE_ENABLED defaults off — no caching."""
    client, provider = _client()
    client.get("/price/SPY")
    client.get("/price/SPY")
    assert provider.calls["price"] == 2


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
    client, provider = _client(cache_enabled=True)
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
# Regression tests for the 2026-09-27 "POST endpoints treated as query params" bug.
# Root cause: Pydantic BaseModel classes defined inside build_app() weren't
# recognized by FastAPI's parameter introspection on some version combos —
# the parameter fell back to query-param treatment, POSTs returned 422
# "field required as query", and /openapi.json 500'd on schema generation.
# Fix: models declared at module scope + Body(...) annotation on the params.
# ---------------------------------------------------------------------------

def test_openapi_json_generates_successfully():
    """/openapi.json must return 200 with a valid schema. Regression for the
    schema-generation crash when models lived inside build_app()."""
    client, _ = _client()
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "openapi" in schema
    assert "paths" in schema
    # The POST endpoints must show a requestBody in the schema — proof
    # FastAPI recognized the Pydantic model as a body type, not a query param.
    for post_route in ("/quotes", "/snapshots"):
        route_spec = schema["paths"].get(post_route, {}).get("post", {})
        assert "requestBody" in route_spec, (
            f"{post_route} has no requestBody in OpenAPI schema — the "
            f"Pydantic model isn't being recognized as a body type."
        )


def test_quotes_endpoint_accepts_json_body_not_query():
    """POST /quotes must consume {'symbols': [...]} as JSON body."""
    client, _ = _client()
    r = client.post("/quotes", json={"symbols": ["SPY", "QQQ"]})
    assert r.status_code == 200, (
        f"expected 200, got {r.status_code}: {r.text[:200]}"
    )


def test_snapshots_endpoint_accepts_json_body_not_query():
    """POST /snapshots must consume {'tickers': [...]} as JSON body."""
    client, _ = _client()
    r = client.post("/snapshots", json={"tickers": ["SPY", "QQQ"]})
    assert r.status_code == 200, (
        f"expected 200, got {r.status_code}: {r.text[:200]}"
    )


def test_request_models_defined_at_module_scope():
    """Skill 47 §3.5 — QuotesRequest / SnapshotsRequest must be defined at
    module scope, not inside build_app(). Prevents the closure-scope regression
    from ever coming back.
    """
    from trading_agent.data_server import app as app_module
    assert hasattr(app_module, "QuotesRequest"), (
        "QuotesRequest must be importable from trading_agent.data_server.app"
    )
    assert hasattr(app_module, "SnapshotsRequest"), (
        "SnapshotsRequest must be importable from trading_agent.data_server.app"
    )


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
    for bad in ("SPY;DROP", "TOOLONGTICKER", "../etc", "", "1SPY", "spy aapl"):
        r = client.get(f"/price/{bad}")
        # 404 acceptable when route parser rejects; 400 when handler rejects.
        assert r.status_code in (400, 404), f"{bad!r} → {r.status_code}"


def test_price_normalises_lowercase_ticker():
    """_validate_ticker has uppercased before matching since 4557be8:
    'spy' is served as SPY (callers pass user input through MCP)."""
    client, _ = _client()
    r = client.get("/price/spy")
    assert r.status_code == 200 and r.json()["ticker"] == "SPY"


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

def test_cli_module_defines_main():
    """The CLI entry module must define a ``main`` function.

    Uses AST inspection (not runtime import) so this test doesn't need
    fastapi/pydantic installed. The runtime-import path is exercised
    by test_openapi_json_generates_successfully and its siblings,
    which do need the deps and only run when FastAPI is present.
    """
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] /
           "trading_agent" / "data_server" / "__main__.py").read_text()
    tree = ast.parse(src)
    func_names = {n.name for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)}
    assert "main" in func_names


def test_serverconfig_reads_env(monkeypatch):
    monkeypatch.setenv("SCHWAB_API_SERVER_KEY", "abc123")
    monkeypatch.setenv("SCHWAB_API_PORT", "9999")
    monkeypatch.setenv("SCHWAB_API_BIND", "100.115.216.79")
    # Per-endpoint TTL vars are only honoured when the master switch is on.
    monkeypatch.setenv("SCHWAB_API_CACHE_ENABLED", "true")
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
              "SCHWAB_API_LOG_FILE", "SCHWAB_API_CACHE_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    from trading_agent.data_server.config import ServerConfig
    cfg = ServerConfig.from_env()
    assert cfg.api_key is None
    assert cfg.port == 8765
    assert cfg.bind == "127.0.0.1"
    assert cfg.auth_enabled is False


# ---------------------------------------------------------------------------
# §3.4 — Cache master switch
# ---------------------------------------------------------------------------

def test_cache_disabled_by_default(monkeypatch):
    """Skill 47 §3.4 — SCHWAB_API_CACHE_ENABLED defaults to false. When
    unset, both price and snapshot TTLs are 0 (every request live)."""
    for k in ("SCHWAB_API_CACHE_ENABLED", "SCHWAB_API_PRICE_TTL_SEC",
              "SCHWAB_API_SNAPSHOT_TTL_SEC"):
        monkeypatch.delenv(k, raising=False)
    from trading_agent.data_server.config import ServerConfig
    cfg = ServerConfig.from_env()
    assert cfg.cache_enabled is False
    assert cfg.ttls.price_sec == 0
    assert cfg.ttls.snapshot_sec == 0


def test_cache_master_switch_true_uses_default_ttls(monkeypatch):
    for k in ("SCHWAB_API_PRICE_TTL_SEC", "SCHWAB_API_SNAPSHOT_TTL_SEC"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SCHWAB_API_CACHE_ENABLED", "true")
    from trading_agent.data_server.config import ServerConfig
    cfg = ServerConfig.from_env()
    assert cfg.cache_enabled is True
    assert cfg.ttls.price_sec == 60
    assert cfg.ttls.snapshot_sec == 90


def test_cache_master_switch_ignores_per_endpoint_ttls_when_disabled(monkeypatch):
    """Even if per-endpoint TTLs are set, the master switch overrides them.

    Documents the "flip the master first, THEN tune per-endpoint" workflow
    the skill 47 §3.4 note describes.
    """
    monkeypatch.setenv("SCHWAB_API_CACHE_ENABLED", "false")
    monkeypatch.setenv("SCHWAB_API_PRICE_TTL_SEC", "300")     # tries to set 300
    monkeypatch.setenv("SCHWAB_API_SNAPSHOT_TTL_SEC", "300")  # tries to set 300
    from trading_agent.data_server.config import ServerConfig
    cfg = ServerConfig.from_env()
    assert cfg.cache_enabled is False
    assert cfg.ttls.price_sec == 0        # ignored, forced to 0
    assert cfg.ttls.snapshot_sec == 0


def test_cache_master_switch_true_respects_per_endpoint_overrides(monkeypatch):
    monkeypatch.setenv("SCHWAB_API_CACHE_ENABLED", "true")
    monkeypatch.setenv("SCHWAB_API_PRICE_TTL_SEC", "5")
    monkeypatch.setenv("SCHWAB_API_SNAPSHOT_TTL_SEC", "10")
    from trading_agent.data_server.config import ServerConfig
    cfg = ServerConfig.from_env()
    assert cfg.cache_enabled is True
    assert cfg.ttls.price_sec == 5
    assert cfg.ttls.snapshot_sec == 10


def test_env_bool_parses_common_truthy_values(monkeypatch):
    """The parser recognizes 1, true, yes, on (case-insensitive)."""
    from trading_agent.data_server.config import _env_bool
    for truthy in ("1", "true", "True", "TRUE", "yes", "YES", "on", "ON"):
        monkeypatch.setenv("SCHWAB_API_TEST_BOOL", truthy)
        assert _env_bool("SCHWAB_API_TEST_BOOL") is True, f"failed for {truthy!r}"
    for falsy in ("0", "false", "FALSE", "no", "off", "", "banana"):
        monkeypatch.setenv("SCHWAB_API_TEST_BOOL", falsy)
        assert _env_bool("SCHWAB_API_TEST_BOOL") is False, f"failed for {falsy!r}"
    monkeypatch.delenv("SCHWAB_API_TEST_BOOL", raising=False)
    assert _env_bool("SCHWAB_API_TEST_BOOL", default=True) is True
    assert _env_bool("SCHWAB_API_TEST_BOOL", default=False) is False


def test_skill_47_main_loads_dotenv_before_reading_server_key(monkeypatch):
    """Regression 2026-09-29: ServerConfig.from_env() ran before .env was
    loaded, so a SCHWAB_API_SERVER_KEY set only in .env never applied and
    the MCP (which reads .env) got 401 on every call."""
    import trading_agent.data_server.__main__ as entry

    order = []

    class _Stop(Exception):
        pass

    def fake_from_env():
        order.append("from_env")
        raise _Stop

    monkeypatch.setattr(entry, "load_dotenv", lambda *a, **k: order.append("dotenv"))
    monkeypatch.setattr(entry.ServerConfig, "from_env", staticmethod(fake_from_env))
    try:
        entry.main([])
    except _Stop:
        pass
    assert order == ["dotenv", "from_env"]


def test_skill_47_key_fingerprint_matches_shasum_and_never_leaks():
    """sha256[:8] — same as `printf %s KEY | shasum -a 256 | cut -c1-8`."""
    from trading_agent.data_server.auth import key_fingerprint
    assert key_fingerprint("test") == "9f86d081"
    assert key_fingerprint("") == key_fingerprint(None) == "e3b0c442"
    assert "secret-value" not in key_fingerprint("secret-value")


@pytest.mark.parametrize("shell,dotenv,source,warns", [
    ("", "", "unset", False),
    ("", "k1", ".env", False),
    ("k1", "", "shell env", False),
    ("k1", "k1", "shell env (overrides .env)", False),
    ("k1", "k2", "shell env (overrides .env)", True),
])
def test_skill_47_describe_key_source(shell, dotenv, source, warns):
    """Startup log must name where the enforced key came from and warn
    when a shell export shadows a different .env key (2026-09-29: stale
    export → MCP using .env key got 401 on every call)."""
    from trading_agent.data_server.__main__ import describe_key_source
    got_source, warning = describe_key_source(shell, dotenv, shell or dotenv)
    assert got_source == source
    assert (warning is not None) == warns
    if warning:
        assert "k1" not in warning and "k2" not in warning   # fingerprints only
        assert "env -u SCHWAB_API_SERVER_KEY" in warning


# ── 2026-10-08: refuse to start without a strong key (Funnel exposure) ──

@pytest.mark.parametrize("key,allow,refused", [
    (None, False, True),
    ("", False, True),
    ("   ", False, True),
    (None, True, False),                      # explicit tailnet-only opt-in
    ("short-key", False, True),
    ("short-key", True, True),                # opt-in never waives length
    ("x" * 23, False, True),
    ("x" * 24, False, False),
])
def test_skill_47_key_policy(key, allow, refused):
    from trading_agent.data_server.__main__ import key_policy_error
    err = key_policy_error(key, allow)
    assert (err is not None) is refused
    if err and (key or "").strip():
        assert key.strip() not in err          # never echo the key


def _run_main_with(monkeypatch, env):
    import trading_agent.data_server.__main__ as entry
    for k in ("SCHWAB_API_SERVER_KEY", "SCHWAB_API_ALLOW_NO_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(entry, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(entry, "dotenv_values", lambda *a, **k: {})
    monkeypatch.setattr(entry, "find_dotenv", lambda *a, **k: "")
    built = []
    monkeypatch.setattr(entry, "_build_default_provider",
                        lambda: built.append(1) or (_ for _ in ()).throw(SystemExit(99)))
    try:
        return entry.main([]), built
    except SystemExit as e:
        return e.code, built


def test_skill_47_main_refuses_without_key(monkeypatch):
    code, built = _run_main_with(monkeypatch, {})
    assert code == 4 and built == []           # stops before touching Schwab


def test_skill_47_main_proceeds_with_strong_key_or_opt_in(monkeypatch):
    assert _run_main_with(monkeypatch, {"SCHWAB_API_SERVER_KEY": "k" * 43}) == (99, [1])
    assert _run_main_with(monkeypatch, {"SCHWAB_API_ALLOW_NO_KEY": "true"}) == (99, [1])


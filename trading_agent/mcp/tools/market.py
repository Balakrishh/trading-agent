"""Market-data + alerts read wrappers (skill 48 §2).

Delegates to the skill-47 data server surface when it's reachable
(preferred — one place owns Schwab OAuth), else to the in-process
market data provider as fallback. Both paths are read-only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import os
import urllib.request
import urllib.error
import json


def _data_server_url() -> Optional[str]:
    """Return the data-server base URL if configured, else None."""
    base = os.environ.get("SCHWAB_API_BASE_URL", "").strip()
    return base or None


def _data_server_get(path: str) -> Optional[Dict[str, Any]]:
    base = _data_server_url()
    if not base:
        return None
    key = os.environ.get("SCHWAB_API_SERVER_KEY", "")
    req = urllib.request.Request(f"{base.rstrip('/')}/{path.lstrip('/')}")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def get_quote(symbol: str) -> Dict[str, Any]:
    """Current mark for an equity or option symbol."""
    if not symbol:
        raise ValueError("symbol is required")
    remote = _data_server_get(f"price/{symbol.upper()}")
    if remote is not None:
        return remote
    # Fallback: in-process. Best-effort — market_data_factory build
    # requires credentials the MCP process may not have.
    return {"symbol": symbol.upper(), "price": None,
            "source": "unavailable", "note": "data server unreachable"}


def get_chain(
    underlying: str,
    expiration: Optional[str] = None,
    option_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Option chain — thin passthrough to the data server."""
    if not underlying:
        raise ValueError("underlying is required")
    q = []
    if expiration:
        q.append(f"expiration={expiration}")
    if option_type:
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        q.append(f"option_type={option_type}")
    suffix = ("?" + "&".join(q)) if q else ""
    remote = _data_server_get(f"chain/{underlying.upper()}{suffix}")
    if remote is not None:
        return remote
    return {"underlying": underlying.upper(),
            "source": "unavailable",
            "note": "data server unreachable"}


def get_market_status() -> Dict[str, Any]:
    """Whether the market is currently open (US equity session)."""
    remote = _data_server_get("market-status")
    if remote is not None:
        return remote
    try:
        from trading_agent.market_hours import is_within_market_hours
        return {"open": bool(is_within_market_hours()),
                "source": "in-process"}
    except Exception as exc:  # noqa: BLE001
        return {"open": None, "source": "unavailable", "error": str(exc)}


def get_recent_alerts(hours: Any = 24) -> Dict[str, Any]:
    """Read the ExceptionMonitor's recent-events log (skill 34).

    ``hours`` is coerced from string/float since some MCP clients JSON-
    serialize numeric arguments as strings.
    """
    try:
        hours = int(hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hours must be an integer, got {hours!r}") from exc
    if hours <= 0:
        raise ValueError("hours must be positive")
    try:
        from trading_agent.journal_reader import JournalReader
        reader = JournalReader()
        return {
            "window_hours": hours,
            "silenced_today": [
                str(e) for e in reader.silenced_exceptions_today()
            ],
            "error_count_today": reader.error_count_today(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"window_hours": hours, "error": str(exc), "alerts": []}

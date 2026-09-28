"""Thread-safe TTL cache used by /price and /snapshots endpoints. Skill 47 §3.3."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Tuple


class TTLCache:
    """A minimal thread-safe TTL cache.

    Two operations:
      * ``get_or_set(key, fetch)`` — return cached value if fresh,
        otherwise call ``fetch()``, cache the result, and return it.
      * ``clear()``               — drop everything (used by tests).

    Not intended as a general-purpose LRU. Keys are held forever until
    ``clear()`` or process death. The endpoints that use this call it
    with a small bounded set of keys (tickers) so unbounded growth in
    practice is fine.
    """

    def __init__(self, ttl_seconds: int) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        self._ttl = int(ttl_seconds)
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get_or_set(self, key: str, fetch: Callable[[], Any]) -> Any:
        """Return cached value if within TTL, else fetch + store."""
        now = time.monotonic()
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                cached_at, value = cached
                if (now - cached_at) < self._ttl:
                    return value
        # Cache miss — fetch OUTSIDE the lock so a slow upstream call
        # doesn't block other keys. Then write inside the lock again.
        value = fetch()
        with self._lock:
            self._store[key] = (time.monotonic(), value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

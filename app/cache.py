from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional, Tuple


@dataclass
class _Entry:
    expires_at: float
    value: Any


class TTLCache:
    """Very small in-memory TTL cache (process-local)."""

    def __init__(self, ttl_seconds: int = 30, max_items: int = 256):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._store: Dict[Hashable, _Entry] = {}

    def get(self, key: Hashable) -> Optional[Any]:
        now = time.monotonic()
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expires_at <= now:
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: Hashable, value: Any) -> None:
        now = time.monotonic()
        # simple eviction: drop one expired or oldest-ish when over max
        if len(self._store) >= self.max_items:
            self._evict_one(now)
        self._store[key] = _Entry(expires_at=now + self.ttl_seconds, value=value)

    def _evict_one(self, now: float) -> None:
        # remove any expired
        for k in list(self._store.keys()):
            if self._store[k].expires_at <= now:
                self._store.pop(k, None)
                return
        # otherwise remove an arbitrary item (insertion order not guaranteed)
        if self._store:
            self._store.pop(next(iter(self._store.keys())), None)


def make_key(*parts: Any) -> Tuple[Any, ...]:
    return tuple(parts)

"""Per-key asyncio lock manager used to serialize idempotent sampling."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict


class KeyedLocks:
    """
    A manager of per-key ``asyncio.Lock``s with reference-counted cleanup.

    Used to serialize sampling per ``(tuner_id, run_id, request_hash)`` so
    concurrent retries of the same turn don't both generate + record a
    duplicate sibling completion. Locks are created lazily and dropped once
    no coroutine holds or waits on them, so the table doesn't grow unbounded
    across the many distinct turns of a training run.
    """

    def __init__(self) -> None:
        self._locks: Dict[Any, asyncio.Lock] = {}
        self._refcounts: Dict[Any, int] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, key: Any) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            self._refcounts[key] = self._refcounts.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                self._refcounts[key] -= 1
                if self._refcounts[key] <= 0:
                    self._refcounts.pop(key, None)
                    self._locks.pop(key, None)

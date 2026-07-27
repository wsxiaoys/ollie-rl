"""Small in-process cache primitives."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar, cast

T = TypeVar("T")
_MISSING = object()


class AsyncTTLValue(Generic[T]):
    """Cache one asynchronously loaded value for a bounded period.

    Concurrent cache misses are collapsed behind a lock, so only one caller
    invokes ``load``. Exceptions are not cached; callers may use
    :meth:`stale_or` inside ``load`` when stale-on-error behavior is desired.
    """

    def __init__(self, ttl_seconds: float):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._value: object = _MISSING
        self._expires_at = 0.0

    async def get(self, load: Callable[[], Awaitable[T]]) -> T:
        """Return the fresh value, loading it once when absent or expired."""
        if self._is_fresh():
            return cast(T, self._value)

        async with self._lock:
            if self._is_fresh():
                return cast(T, self._value)

            value = await load()
            self.set(value)
            return value

    def set(self, value: T) -> None:
        """Store ``value`` and restart its TTL."""
        self._value = value
        self._expires_at = time.monotonic() + self._ttl_seconds

    def stale_or(self, default: T) -> T:
        """Return the cached value even if expired, or ``default`` if absent."""
        if self._value is _MISSING:
            return default
        return cast(T, self._value)

    def _is_fresh(self) -> bool:
        return self._value is not _MISSING and time.monotonic() < self._expires_at

import asyncio
import logging
from typing import Any, Coroutine, Set

logger = logging.getLogger(__name__)


class BackgroundJob:
    """Owns fire-and-forget asyncio tasks so they aren't garbage-collected.

    asyncio only keeps a *weak* reference to a running task, so a bare
    ``asyncio.create_task(...)`` whose handle is discarded can be collected
    mid-flight -- silently dropping the work. ``BackgroundJob`` holds a strong
    reference to every spawned task and drops it once the task finishes, and
    logs any exception that escapes the coroutine as a safety net.

    NOTE: these tasks live only in this process and do NOT survive a reboot.
    A dropped task (crash, GC, or restart) is currently reclaimed only by
    ad-hoc reconciliation (e.g. a train-op poll is re-spawned lazily on trainer
    restore). For real durability this should move to a persistent task queue
    (e.g. taskiq) so pending work is picked up by any worker after a restart.
    """

    def __init__(self) -> None:
        self._tasks: Set[asyncio.Task] = set()

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Schedule ``coro`` as a tracked background task and return it."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # Safety net: callers are expected to handle their own errors, so
            # anything reaching here is an unhandled escape worth surfacing.
            logger.error("Background job raised an unhandled exception", exc_info=exc)

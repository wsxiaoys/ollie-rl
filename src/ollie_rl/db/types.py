"""
Shared datetime utilities and SQLAlchemy types for working with UTC.

The goals are:

1. All datetimes persisted to the database are stored as timezone-aware UTC.
2. All datetimes loaded from the database come back as timezone-aware UTC,
   even if the underlying backend (e.g. SQLite) doesn't actually persist
   timezone information.
3. Application code never has to think about ``timezone.utc`` again — just
   call :func:`utcnow` and use the :class:`UtcDateTime` column type.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Prefer this over ``datetime.now()`` / ``datetime.utcnow()`` everywhere in
    the codebase so we have a single canonical source of "now".
    """
    return datetime.now(timezone.utc)


class UtcDateTime(TypeDecorator):
    """A ``DateTime`` column that always stores and returns UTC.

    - On write: naive datetimes are assumed to already be UTC and are tagged
      with ``tzinfo=UTC``; aware datetimes are converted to UTC.
    - On read: values are returned with ``tzinfo=UTC`` regardless of whether
      the underlying driver preserved timezone information (this matters for
      SQLite, which stores naive strings).
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: Optional[datetime], dialect: object
    ) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            # Assume legacy/naive input is already UTC.
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(
        self, value: Optional[datetime], dialect: object
    ) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

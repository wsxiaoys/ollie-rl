"""Cursor helpers for the in-flight chat-completion workload listing."""

import base64
import binascii
from datetime import datetime
from typing import Tuple

from ollie_rl.service.tuner.errors import InvalidWorkloadCursorError


def encode_workload_cursor(created_at: datetime, run_id: str, request_hash: str) -> str:
    """Encode the workload listing's stable three-column sort position."""
    raw = f"{created_at.isoformat()}|{run_id}|{request_hash}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_workload_cursor(cursor: str) -> Tuple[datetime, str, str]:
    """Decode a workload cursor, raising a client-facing error if malformed."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        created_at_str, run_id, request_hash = raw.rsplit("|", 2)
        return datetime.fromisoformat(created_at_str), run_id, request_hash
    except (ValueError, UnicodeDecodeError, binascii.Error) as e:
        raise InvalidWorkloadCursorError(f"Invalid workload cursor: {cursor!r}") from e

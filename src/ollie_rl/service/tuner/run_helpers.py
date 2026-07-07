"""Pure helpers for deriving run status, cursors, and run list items."""

import base64
import binascii
from datetime import datetime
from typing import Optional, Tuple

from ollie_rl.db.models import RunModel
from ollie_rl.service.tuner.constants import RUN_EXPIRE_GENERATION_BUDGET_MS  # noqa: F401
from ollie_rl.service.tuner.errors import InvalidRunCursorError
from ollie_rl.types import RunItem, RunStatus


def last_train_op_duration_seconds(state_data: object) -> Optional[float]:
    """Derive the most recent completed train op's execution time (seconds).

    Reads the LRO timing captured under
    ``last_train_op.metadata.{create_time, update_time}`` and returns
    ``update_time - create_time``. Robust to camelCase serialization and
    tolerant of missing/partial timing (returns ``None``). Trainers that don't
    persist a ``last_train_op`` (e.g. inline backends) yield ``None``.
    """
    if not isinstance(state_data, dict):
        return None
    op = state_data.get("last_train_op")
    if not isinstance(op, dict):
        return None
    meta = op.get("metadata")
    if not isinstance(meta, dict):
        return None
    create = meta.get("create_time") or meta.get("createTime")
    update = meta.get("update_time") or meta.get("updateTime")
    if not isinstance(create, str) or not isinstance(update, str):
        return None
    try:
        start = datetime.fromisoformat(create.replace("Z", "+00:00"))
        end = datetime.fromisoformat(update.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (end - start).total_seconds()


def _run_status(
    run: RunModel,
    now: datetime,
    is_expired: bool,
    is_length: bool,
    is_content_filter: bool,
) -> RunStatus:
    """Derive a single mutually-exclusive lifecycle label for a run.

    Priority mirrors how the bookkeeping columns accumulate: a run that
    has been trained or requeued (rejected) takes precedence over its
    reward/lease state.

    ``is_length`` marks runs where at least one completion exceeded the recipe's
    ``max_context_window`` and was rewritten to a length sample.
    ``is_content_filter`` marks runs whose completion was content-filtered
    (malformed) and carried the ``content_filter_penalty`` reward. Both are
    surfaced before ordinary rewarded state so dashboard run rows can
    distinguish these automatic penalty paths from user-provided rewards. They
    are mutually exclusive in practice (a run terminates at its first
    behavior-penalty completion); ``length`` takes precedence if both were ever
    recorded.

    Once a run is past its lease with no reward, ``is_expired`` splits it into
    ``expired`` vs ``lost``. ``is_expired`` is true when the run either still
    has a lingering ``InFlightChatCompletionModel`` row (the generation itself
    stalled past the lease) or has burned at least ``RUN_EXPIRE_GENERATION_BUDGET_MS`` of
    total generation time without a reward -- both signal wasted compute on a run
    that never finished. Otherwise the run is ``lost`` (crashed/abandoned
    worker, or ops finished but no reward was ever posted).
    """
    if run.trained_count > 0:
        return "trained"
    if run.rejected_count > 0:
        return "rejected"
    if is_length:
        return "length"
    if is_content_filter:
        return "content_filter"
    if run.reward is not None:
        return "rewarded"
    if run.expires_at > now:
        return "in_flight"
    return "expired" if is_expired else "lost"


def encode_run_cursor(created_at: datetime, run_id: str) -> str:
    """Encode a ``(created_at, id)`` run position into an opaque cursor.

    The two fields form the stable sort key used by ``list_runs``; base64 keeps
    the token opaque so clients treat it as a handle rather than parsing it.
    """
    raw = f"{created_at.isoformat()}|{run_id}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_run_cursor(cursor: str) -> Tuple[datetime, str]:
    """Decode a cursor produced by :func:`encode_run_cursor`.

    Raises ``InvalidRunCursorError`` for malformed tokens so the API layer can
    surface a 400 rather than a 500.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        created_at_str, run_id = raw.rsplit("|", 1)
        return datetime.fromisoformat(created_at_str), run_id
    except (ValueError, UnicodeDecodeError, binascii.Error) as e:
        raise InvalidRunCursorError(f"Invalid runs cursor: {cursor!r}") from e


def build_run_item(
    run: RunModel,
    completion_count: int,
    now: datetime,
    policy_generation: Optional[int] = None,
    duration_ms_total: Optional[int] = None,
    context_window_tokens_max: Optional[int] = None,
    is_expired: bool = False,
    is_length: bool = False,
    is_content_filter: bool = False,
) -> RunItem:
    return RunItem(
        run_id=run.id,
        datum_id=run.datum_id,
        status=_run_status(run, now, is_expired, is_length, is_content_filter),
        reward=run.reward,
        policy_generation=policy_generation,
        trained_count=run.trained_count,
        rejected_count=run.rejected_count,
        completion_count=completion_count,
        duration_ms_total=duration_ms_total,
        context_window_tokens_max=context_window_tokens_max,
        created_at=run.created_at,
        expires_at=run.expires_at,
    )

import array
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid
from sqlalchemy import (
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    ForeignKey,
    Float,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from ollie_rl.db.types import UtcDateTime, utcnow


class _PackedIntList(TypeDecorator[List[int]]):
    """
    Stores a `List[int]` as a compact int64 little-endian BLOB.

    Encoding/decoding stays inside the model layer so the service /
    trainer layers can read and write the column as a plain
    `List[int]`. Uses the stdlib `array` module ("q" = signed int64),
    ample for any practical vocab size, no extra dependency.
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(
        self, value: Optional[List[int]], dialect
    ) -> Optional[bytes]:
        if value is None:
            return None
        return array.array("q", value).tobytes()

    def process_result_value(
        self, value: Optional[bytes], dialect
    ) -> Optional[List[int]]:
        if value is None:
            return None
        buf = array.array("q")
        buf.frombytes(value)
        return list(buf)


class _PackedFloatList(TypeDecorator[List[float]]):
    """
    Stores a `List[float]` as a compact float32 little-endian BLOB.

    Float32 matches the precision tinker stores on `SampledSequence`,
    so the round-trip is lossless for the values we actually see.
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(
        self, value: Optional[List[float]], dialect
    ) -> Optional[bytes]:
        if value is None:
            return None
        return array.array("f", value).tobytes()

    def process_result_value(
        self, value: Optional[bytes], dialect
    ) -> Optional[List[float]]:
        if value is None:
            return None
        buf = array.array("f")
        buf.frombytes(value)
        return list(buf)


def generate_tuner_id() -> str:
    return f"tuner_{uuid.uuid4().hex}"


def generate_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


class BaseModel(DeclarativeBase):
    """SQLAlchemy Declarative Base"""

    pass


class TunerModel(BaseModel):
    """
    SQLAlchemy model representing a single persisted tuner's metadata and state.
    """

    __tablename__ = "tuners"

    id: Mapped[str] = mapped_column(
        String(255), primary_key=True, default=generate_tuner_id
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    recipe: Mapped[str] = mapped_column(String(255), nullable=False)
    trainer: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # `trainer_state` is populated by the Tuner itself via its StateStore. It is
    # NULL between row creation and the Tuner's first save.
    trainer_state: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ChatCompletionModel(BaseModel):
    """
    SQLAlchemy model representing a recorded chat completion.

    Within the same tuner:
      1. id: Represents a single LLM request and response interaction.
      2. trajectory_id: Represents a continuous series of chat completions (currently not
         recorded because we do not differentiate them at this stage).
      3. run_id: Represents a specific run for a data row; it can contain multiple trajectories
         (e.g., in multi-agent or sub-agent architectures).
      4. policy_generation: Represents the version of the tuner when serving this chat completion.
      5. datum_id: Represents the reference ID in the dataset; a dataset item can have multiple task runs.

    For a typical GRPO-style training step:
      1. Each group is defined by the same `data_id` and multiple `run_id`s within a tuner version
         range (e.g., steps 3-5), and usually requires a minimum number of runs.
      2. A training step is triggered once a sufficient number of groups satisfy condition #1,
         at which point these completions are consumed.
    """

    __tablename__ = "chat_completions"

    # Almost every read is tuner-scoped and then narrowed / grouped by run_id
    # (progress aggregation, list_runs, run + completion detail lookups, and
    # the training-batch staleness scan). `tuner_id` alone had no index, so
    # those queries fell back to a full table scan. A composite
    # (tuner_id, run_id) index serves the equality lookups and the group-by
    # without scanning the (large, blob-heavy) table.
    __table_args__ = (
        Index("ix_chat_completions_tuner_id_run_id", "tuner_id", "run_id"),
        # Idempotent-sample lookup: find a prior completion for the same turn
        # (same request prompt) within a run so a retry replays it instead of
        # recording a duplicate sibling. See `request_hash` below.
        Index(
            "ix_chat_completions_tuner_id_run_id_request_hash",
            "tuner_id",
            "run_id",
            "request_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    tuner_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tuners.id"), nullable=False
    )
    policy_generation: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    # trajectory_id: Mapped[Optional[str]] = mapped_column(
    #     String(255), nullable=False, index=True
    # )
    run_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=False, index=True
    )
    datum_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # SHA-256 hex digest of the request prompt (messages), used to make
    # sampling idempotent per run. A slow/cancelled request is retried by the
    # client with the *identical* prompt; since an agent conversation is
    # linear, a repeat `(tuner_id, run_id, request_hash)` is always such a
    # retry and must replay the stored completion rather than generate a new
    # sibling (which would fork the trajectory and pollute training). NULL for
    # rows written before this column existed.
    request_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Optional cached sample-time tensors written by trainers that need
    # to replay rollouts at train time (Tinker). Encoding/decoding is
    # handled transparently by the `_PackedIntList` / `_PackedFloatList`
    # type decorators so the rest of the codebase reads and writes
    # plain `List[int]` / `List[float]`. NULL for trainers that retain
    # candidates server-side (e.g. gemini_msrl) or that do not train
    # at all (fake).
    tokens: Mapped[Optional[List[int]]] = mapped_column(_PackedIntList(), nullable=True)
    logprobs: Mapped[Optional[List[float]]] = mapped_column(
        _PackedFloatList(), nullable=True
    )
    request: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    response: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    # Wall-clock time (in milliseconds) spent generating this completion,
    # measured around the trainer's sample/wait span. Every newly recorded
    # completion carries it (`record_chat_completion` requires it); NULL only
    # for legacy rows written before this column existed. (The idempotent
    # replay path records no new row; it returns the original completion,
    # which retains its own duration.)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class InFlightChatCompletionModel(BaseModel):
    """
    Durable resume state for a chat completion whose backend op is still
    in flight for a given turn.

    Keyed by the exact turn identity we already dedup on
    (``(tuner_id, run_id, request_hash)``), it stores the op's serializable
    resume state (``op.save_state()`` -- e.g. a Gemini LRO ``op_name``) the
    moment the op is submitted. A later retry of the *same* turn re-attaches to
    that already-running op instead of spawning a fresh one, so a generation
    longer than the poll budget can complete across retries on a single op
    (no orphaned ops, no lease burn).

    Rows are short-lived: created at first submit, and deleted on recorded
    success or on terminal op failure. They are intentionally NOT cleared on
    cancel/timeout, since those mean the backend op is still progressing and the
    next retry must re-attach.
    """

    __tablename__ = "in_flight_chat_completions"

    # Natural key = the turn identity. At most one in-flight op per turn.
    tuner_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tuners.id"), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), primary_key=True)

    # `op.save_state()` (the op resource name for gemini_msrl).
    state: Mapped[str] = mapped_column(String(512), nullable=False)
    # First-submit time, used as the start for end-to-end duration across any
    # re-attach cycles.
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class RunModel(BaseModel):
    __tablename__ = "runs"

    # Two hot, tuner-scoped access patterns that the lone `tuner_id` index left
    # as filter-then-sort / full scans over a tuner's runs:
    #   1. `list_runs` keyset pagination orders by (created_at DESC, id DESC)
    #      within a tuner. A composite (tuner_id, created_at, id) lets the DB
    #      walk the index in order instead of sorting every matching run.
    #   2. `_collect_consumable_batch` selects unconsumed runs by
    #      trained_count/rejected_count within a tuner on every training
    #      attempt. A composite (tuner_id, trained_count, rejected_count)
    #      narrows straight to the candidates.
    __table_args__ = (
        Index("ix_runs_tuner_id_created_at_id", "tuner_id", "created_at", "id"),
        Index(
            "ix_runs_tuner_id_trained_count_rejected_count",
            "tuner_id",
            "trained_count",
            "rejected_count",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(255), primary_key=True, default=generate_run_id
    )
    tuner_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tuners.id"), index=True
    )
    datum_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trained_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expires_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class DatumRowModel(BaseModel):
    __tablename__ = "datum_rows"

    tuner_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tuners.id"), primary_key=True
    )
    datum_id: Mapped[str] = mapped_column(String(255), primary_key=True)

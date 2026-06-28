import array
from datetime import datetime
from typing import List, Optional
import uuid
from sqlalchemy import (
    Integer,
    LargeBinary,
    String,
    Text,
    ForeignKey,
    Float,
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
    return f"tuner_{uuid.uuid4()}"


def generate_run_id() -> str:
    return f"run_{uuid.uuid4()}"


class BaseModel(DeclarativeBase):
    """SQLAlchemy Declarative Base"""

    pass


class TunerModel(BaseModel):
    """
    SQLAlchemy model representing a single persisted tuner's metadata and state.
    """

    __tablename__ = "tuners"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_tuner_id
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    recipe: Mapped[str] = mapped_column(String(255), nullable=False)
    trainer: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # JSON-serialized dictionary of trainer-specific bootstrap parameters.
    trainer_params: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # `state` is populated by the Tuner itself via its StateStore. It is
    # NULL between row creation and the Tuner's first save.
    state: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


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

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    tuner_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tuners.id"), nullable=False
    )
    policy_generation: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    # trajectory_id: Mapped[Optional[str]] = mapped_column(
    #     String(255), nullable=False, index=True
    # )
    run_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=False, index=True
    )
    datum_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
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
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class RunModel(BaseModel):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(
        String(255), primary_key=True, default=generate_run_id
    )
    tuner_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tuners.id"), index=True
    )
    datum_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trained_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
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

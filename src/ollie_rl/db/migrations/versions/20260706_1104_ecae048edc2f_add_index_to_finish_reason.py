"""add finish_reason functional index to chat_completions

Speeds up the length-limited-run scan (`_length_datums`) that backs the
frequently polled tuner progress snapshot. The predicate is
``tuner_id = ? AND json(response).choices[0].finish_reason = 'length'``; a
functional index over the *extracted* finish reason lets the planner satisfy it
from the index instead of reading and JSON-parsing every (potentially large)
``response`` blob. ``run_id`` is carried as a trailing key so the run-id listing
is served from the index alone.

The index expression is built from SQLAlchemy's generic JSON accessor so it
renders per dialect and matches the expression the ORM query emits (letting the
planner actually use the index):

* Postgres:
  ``CAST(((response -> 'choices') -> 0) ->> 'finish_reason' AS VARCHAR)``
* SQLite:
  ``JSON_EXTRACT(response, '$."choices"[0]."finish_reason"')``

Revision ID: ecae048edc2f
Revises: c995075b7dc6
Create Date: 2026-07-06 11:04:14.861958

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ecae048edc2f"
down_revision: Union[str, None] = "c995075b7dc6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


INDEX_NAME = "ix_chat_completions_tuner_id_finish_reason"


def _finish_reason_expr():
    """`response.choices[0].finish_reason` as text, rendered per dialect.

    Reuses the generic JSON accessor so the DDL matches exactly what the ORM
    query compiles to; the planner only uses an expression index when the
    query's expression matches the index's.
    """
    return sa.column("response", sa.JSON())["choices"][0]["finish_reason"].as_string()


def upgrade() -> None:
    op.create_index(
        INDEX_NAME,
        "chat_completions",
        ["tuner_id", _finish_reason_expr(), "run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="chat_completions")

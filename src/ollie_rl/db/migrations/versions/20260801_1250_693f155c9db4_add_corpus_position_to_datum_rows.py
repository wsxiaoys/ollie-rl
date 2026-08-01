"""add corpus position to datum rows

Revision ID: 693f155c9db4
Revises: c8b6cf17314d
Create Date: 2026-08-01 12:50:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "693f155c9db4"
down_revision: Union[str, None] = "c8b6cf17314d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


INDEX_NAME = "ix_datum_rows_tuner_id_kind_position"


def upgrade() -> None:
    op.add_column(
        "datum_rows",
        sa.Column("position", sa.Integer(), nullable=True),
    )
    # Existing tuners did not persist corpus order. Give them a stable fallback
    # order per split; newly created tuners store the caller-provided position.
    op.execute(
        sa.text(
            """
            UPDATE datum_rows
            SET position = (
                SELECT COUNT(*)
                FROM datum_rows AS preceding
                WHERE preceding.tuner_id = datum_rows.tuner_id
                  AND preceding.kind = datum_rows.kind
                  AND preceding.datum_id < datum_rows.datum_id
            )
            """
        )
    )
    with op.batch_alter_table("datum_rows") as batch_op:
        batch_op.alter_column("position", existing_type=sa.Integer(), nullable=False)
    op.create_index(
        INDEX_NAME,
        "datum_rows",
        ["tuner_id", "kind", "position"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="datum_rows")
    with op.batch_alter_table("datum_rows") as batch_op:
        batch_op.drop_column("position")

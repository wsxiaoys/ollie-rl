"""remove quarantine persistence artifacts

Revision ID: c8b6cf17314d
Revises: 8d71a5c32e4b
Create Date: 2026-08-01 11:49:36.542448

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8b6cf17314d"
down_revision: Union[str, None] = "8d71a5c32e4b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


INDEX_NAME = "ix_chat_completions_tuner_id_policy_generation"


def upgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="chat_completions")
    with op.batch_alter_table("in_flight_chat_completions") as batch_op:
        batch_op.drop_column("policy_generation")


def downgrade() -> None:
    with op.batch_alter_table("in_flight_chat_completions") as batch_op:
        batch_op.add_column(sa.Column("policy_generation", sa.Integer(), nullable=True))
    op.create_index(
        INDEX_NAME,
        "chat_completions",
        ["tuner_id", "policy_generation"],
        unique=False,
    )

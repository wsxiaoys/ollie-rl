"""add sample op state to chat completion

Revision ID: 8d71a5c32e4b
Revises: abb57b16bc98
Create Date: 2026-07-30 17:53:40.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8d71a5c32e4b"
down_revision: Union[str, None] = "abb57b16bc98"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_completions",
        sa.Column("sample_op_state", sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_completions", "sample_op_state")

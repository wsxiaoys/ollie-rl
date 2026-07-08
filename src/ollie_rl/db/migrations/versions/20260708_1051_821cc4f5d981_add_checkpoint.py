"""add checkpoint

Held-out evaluation split (per-checkpoint eval), Pass 1 scaffolding:

* ``datum_rows.kind`` (``server_default="train"``): marks a datum as ``train``
  (dispensable pool) or ``eval`` (held out, scored per checkpoint only). The
  server default backfills existing rows to training, so today's tuners are
  unchanged.
* ``checkpoints`` table: one row per checkpoint a completed train step yields
  (a ``policy_generation`` stamp plus the backend's opaque ``ref`` handle, or
  the ``LIVE_POLICY_CHECKPOINT`` sentinel).
* ``runs.checkpoint_id`` (nullable FK to ``checkpoints.id``): the checkpoint an
  *evaluation* run scores. NULL for ordinary training runs.

Revision ID: 821cc4f5d981
Revises: ecae048edc2f
Create Date: 2026-07-08 10:51:17.760940

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import ollie_rl.db.types


# revision identifiers, used by Alembic.
revision: str = '821cc4f5d981'
down_revision: Union[str, None] = 'ecae048edc2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # New checkpoint table (created first so the runs FK target exists).
    op.create_table(
        'checkpoints',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('tuner_id', sa.String(length=255), nullable=False),
        sa.Column('ref', sa.String(length=512), nullable=False),
        sa.Column('policy_generation', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            ollie_rl.db.types.UtcDateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['tuner_id'], ['tuners.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('checkpoints', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_checkpoints_tuner_id'), ['tuner_id'], unique=False
        )

    # datum_rows.kind: train | eval, backfilled to "train" on existing rows.
    with op.batch_alter_table('datum_rows', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'kind',
                sa.String(length=16),
                server_default='train',
                nullable=False,
            )
        )

    # runs.checkpoint_id: nullable FK; only eval runs set it. batch mode so the
    # foreign key is created via table-rebuild on SQLite (no native ALTER ADD
    # CONSTRAINT).
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('checkpoint_id', sa.String(length=255), nullable=True)
        )
        batch_op.create_foreign_key(
            'fk_runs_checkpoint_id_checkpoints',
            'checkpoints',
            ['checkpoint_id'],
            ['id'],
        )


def downgrade() -> None:
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.drop_constraint(
            'fk_runs_checkpoint_id_checkpoints', type_='foreignkey'
        )
        batch_op.drop_column('checkpoint_id')

    with op.batch_alter_table('datum_rows', schema=None) as batch_op:
        batch_op.drop_column('kind')

    with op.batch_alter_table('checkpoints', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_checkpoints_tuner_id'))
    op.drop_table('checkpoints')

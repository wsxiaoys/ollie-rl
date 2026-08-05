"""persist resolved tuner recipes

Revision ID: 2f78c5e4a1b9
Revises: 693f155c9db4
Create Date: 2026-08-05 10:35:00.000000

"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2f78c5e4a1b9"
down_revision: Union[str, None] = "693f155c9db4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Historical values are embedded so a future cookbook edit cannot change the
# result of replaying this migration.
_PRESET_SNAPSHOTS: dict[str, dict[str, Any]] = {
    "grpo_16x32": {
        "group_size": 16,
        "num_groups_per_batch": 32,
        "max_off_policy_generation": 4,
        "sampler_promotion_every": 4,
        "eval_group_size": 4,
        "content_filter_penalty": -1.0,
        "length_penalty": -10.0,
        "max_context_window": 60_000,
    },
    "grpo_4x8": {
        "group_size": 4,
        "num_groups_per_batch": 8,
        "max_off_policy_generation": 4,
        "sampler_promotion_every": 4,
        "eval_group_size": 4,
        "content_filter_penalty": -1.0,
        "length_penalty": -10.0,
        "max_context_window": 60_000,
    },
}


def upgrade() -> None:
    op.add_column("tuners", sa.Column("recipe_config", sa.JSON(), nullable=True))

    tuners = sa.table(
        "tuners",
        sa.column("id", sa.String()),
        sa.column("recipe", sa.String()),
        sa.column("recipe_config", sa.JSON()),
    )
    connection = op.get_bind()
    rows = connection.execute(sa.select(tuners.c.id, tuners.c.recipe)).all()
    for tuner_id, recipe_name in rows:
        snapshot = _PRESET_SNAPSHOTS.get(recipe_name)
        if snapshot is None:
            raise RuntimeError(
                f"Cannot migrate tuner {tuner_id!r}: unknown recipe {recipe_name!r}"
            )
        connection.execute(
            sa.update(tuners)
            .where(tuners.c.id == tuner_id)
            .values(recipe_config=snapshot)
        )

    with op.batch_alter_table("tuners") as batch_op:
        batch_op.drop_column("recipe")
        batch_op.alter_column(
            "recipe_config",
            new_column_name="recipe",
            existing_type=sa.JSON(),
            nullable=False,
        )


def downgrade() -> None:
    op.add_column(
        "tuners", sa.Column("recipe_name", sa.String(length=255), nullable=True)
    )
    tuners = sa.table(
        "tuners",
        sa.column("id", sa.String()),
        sa.column("recipe", sa.JSON()),
        sa.column("recipe_name", sa.String()),
    )
    connection = op.get_bind()
    rows = connection.execute(sa.select(tuners.c.id, tuners.c.recipe)).all()
    for tuner_id, recipe in rows:
        recipe_name = next(
            (
                name
                for name, snapshot in _PRESET_SNAPSHOTS.items()
                if snapshot == recipe
            ),
            None,
        )
        if recipe_name is None:
            raise RuntimeError(
                f"Cannot downgrade tuner {tuner_id!r}: its resolved recipe "
                "cannot be represented by the legacy named-recipe schema"
            )
        connection.execute(
            sa.update(tuners)
            .where(tuners.c.id == tuner_id)
            .values(recipe_name=recipe_name)
        )

    with op.batch_alter_table("tuners") as batch_op:
        batch_op.drop_column("recipe")
        batch_op.alter_column(
            "recipe_name",
            new_column_name="recipe",
            existing_type=sa.String(length=255),
            nullable=False,
        )

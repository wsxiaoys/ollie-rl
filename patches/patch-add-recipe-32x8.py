#!/usr/bin/env python3
"""Add grpo_32x8: the faithful ollie mapping of the 4999 production group axis.

4999 config: samples_per_task=32 (= GRPO group size) x tasks_per_train_step=8.
ollie recipe naming is grpo_{group_size}x{num_groups_per_batch}, so the correct
recipe is group_size=32, num_groups_per_batch=8 = grpo_32x8 (256 rewards/step).
(grpo_8x32 is the TRANSPOSE -- group_size 8 -- so it is NOT the right group axis.)

Quarantine OFF to match grpo_8x32 / grpo_16x8 (train every datum). Leaves
sampler_promotion_every at the recipe default (4) -- the 4999 value of 2 was
intentionally skipped per user.

Mirrors patch-add-recipe-16x8.py: define the Recipe instance in recipes.py and
register it in cookbook/__init__.py (import + RECIPES map).

Idempotent. Run on VM: python3 ~/patch-add-recipe-32x8.py [--revert]
"""

import pathlib
import sys

RECIPES = pathlib.Path.home() / "ollie-rl/src/ollie_rl/cookbook/recipes.py"
INIT = pathlib.Path.home() / "ollie-rl/src/ollie_rl/cookbook/__init__.py"

# Append the new instance after GRPO_16x8 (anchor on its full block).
RECIPES_ANCHOR = """GRPO_16x8 = Recipe(
    group_size=16,
    num_groups_per_batch=8,  # 16 x 8 = 128 rewarded rollouts per train step
    # Quarantine OFF (matches grpo_8x32): never exclude a datum from dispensing.
    max_unhealthy_finish_ratio=None,
    max_succeed_ratio=None,
)"""
RECIPES_NEW = (
    RECIPES_ANCHOR
    + """

GRPO_32x8 = Recipe(
    group_size=32,
    num_groups_per_batch=8,  # 32 x 8 = 256 rewards/step; the 4999 group axis
    # (4999 samples_per_task=32 IS the GRPO group size; tasks_per_train_step=8.)
    # Quarantine OFF (matches grpo_8x32): never exclude a datum from dispensing.
    max_unhealthy_finish_ratio=None,
    max_succeed_ratio=None,
)"""
)

# import + register in __init__.py (anchor on the current 16x8 lines)
INIT_IMPORT_ANCHOR = (
    "from .recipes import Recipe, GRPO_16x32, GRPO_4x8, GRPO_8x32, GRPO_16x8\n"
)
INIT_IMPORT_NEW = "from .recipes import Recipe, GRPO_16x32, GRPO_4x8, GRPO_8x32, GRPO_16x8, GRPO_32x8\n"

INIT_MAP_ANCHOR = '    "grpo_16x8": GRPO_16x8,\n}'
INIT_MAP_NEW = '    "grpo_16x8": GRPO_16x8,\n    "grpo_32x8": GRPO_32x8,\n}'


def apply(path, old, new, revert):
    src = path.read_text()
    a, b = (new, old) if revert else (old, new)
    if b in src:
        return "already " + ("reverted" if revert else "patched")
    if a not in src:
        return "ANCHOR NOT FOUND"
    path.write_text(src.replace(a, b, 1))
    return "REVERTED" if revert else "PATCHED"


revert = "--revert" in sys.argv
print("recipes.py instance:", apply(RECIPES, RECIPES_ANCHOR, RECIPES_NEW, revert))
print("init import:        ", apply(INIT, INIT_IMPORT_ANCHOR, INIT_IMPORT_NEW, revert))
print("init RECIPES map:   ", apply(INIT, INIT_MAP_ANCHOR, INIT_MAP_NEW, revert))

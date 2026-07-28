#!/usr/bin/env python3
"""Add a grpo_16x8 recipe: group_size 16 x num_groups_per_batch 8 = 128
rewarded rollouts per train step.

Mirrors how grpo_8x32 was added: define the Recipe instance in recipes.py and
register it in the cookbook __init__ RECIPES map + import. Quarantine is
DISABLED (both ratios None) to match grpo_8x32 / the current "train every datum"
preference -- flip these back to 1.0 if you want default quarantine for this
recipe.

Idempotent. Run on VM: python3 ~/patch-add-recipe-16x8.py [--revert]
"""

import pathlib
import sys

RECIPES = pathlib.Path.home() / "ollie-rl/src/ollie_rl/cookbook/recipes.py"
INIT = pathlib.Path.home() / "ollie-rl/src/ollie_rl/cookbook/__init__.py"

# 1. Append the new instance after GRPO_8x32 (anchor on its closing paren block).
RECIPES_ANCHOR = """GRPO_8x32 = Recipe(
    group_size=8,
    num_groups_per_batch=32,
    # Quarantine OFF: never exclude a datum from dispensing. Every datum keeps
    # training regardless of how often it hits an unhealthy finish or how
    # reliably it succeeds; degenerate/too-easy datums still train at their
    # assigned (lowest_)reward instead of being dropped from the pool.
    max_unhealthy_finish_ratio=None,
    max_succeed_ratio=None,
)"""
RECIPES_NEW = (
    RECIPES_ANCHOR
    + """

GRPO_16x8 = Recipe(
    group_size=16,
    num_groups_per_batch=8,  # 16 x 8 = 128 rewarded rollouts per train step
    # Quarantine OFF (matches grpo_8x32): never exclude a datum from dispensing.
    max_unhealthy_finish_ratio=None,
    max_succeed_ratio=None,
)"""
)

# 2. import + register in __init__.py
INIT_IMPORT_ANCHOR = "from .recipes import Recipe, GRPO_16x32, GRPO_4x8, GRPO_8x32\n"
INIT_IMPORT_NEW = (
    "from .recipes import Recipe, GRPO_16x32, GRPO_4x8, GRPO_8x32, GRPO_16x8\n"
)

INIT_MAP_ANCHOR = '    "grpo_8x32": GRPO_8x32,\n}'
INIT_MAP_NEW = '    "grpo_8x32": GRPO_8x32,\n    "grpo_16x8": GRPO_16x8,\n}'


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

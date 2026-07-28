#!/usr/bin/env python3
"""Add a grpo_8x32 recipe (group_size 8, num_groups_per_batch 32 = 256
rewards/train-step) to the VM's ollie-rl cookbook. Idempotent.
Run on VM: python3 ~/patch-add-recipe.py [--revert]
"""

import sys, pathlib

RECIPES = pathlib.Path.home() / "ollie-rl/src/ollie_rl/cookbook/recipes.py"
INIT = pathlib.Path.home() / "ollie-rl/src/ollie_rl/cookbook/__init__.py"

REC_ANCHOR = """GRPO_4x8 = Recipe(
    group_size=4,
    num_groups_per_batch=8,
)
"""
REC_ADD = """
GRPO_8x32 = Recipe(
    group_size=8,
    num_groups_per_batch=32,
)
"""

INIT_IMPORT_ANCHOR = "from .recipes import Recipe, GRPO_16x32, GRPO_4x8\n"
INIT_IMPORT_NEW = "from .recipes import Recipe, GRPO_16x32, GRPO_4x8, GRPO_8x32\n"

INIT_REG_ANCHOR = '    "grpo_4x8": GRPO_4x8,\n'
INIT_REG_NEW = '    "grpo_4x8": GRPO_4x8,\n    "grpo_8x32": GRPO_8x32,\n'


def apply(path, pairs, revert):
    src = path.read_text()
    for anchor, patched in pairs:
        if revert:
            if patched in src:
                src = src.replace(patched, anchor)
        else:
            if patched in src:
                continue
            if anchor not in src:
                return f"ANCHOR NOT FOUND in {path.name}"
            src = src.replace(anchor, patched, 1)
    path.write_text(src)
    return "OK"


revert = "--revert" in sys.argv
print("recipes.py:", apply(RECIPES, [(REC_ANCHOR, REC_ANCHOR + REC_ADD)], revert))
print(
    "__init__.py:",
    apply(
        INIT,
        [
            (INIT_IMPORT_ANCHOR, INIT_IMPORT_NEW),
            (INIT_REG_ANCHOR, INIT_REG_NEW),
        ],
        revert,
    ),
)

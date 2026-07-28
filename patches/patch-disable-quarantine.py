#!/usr/bin/env python3
"""Disable datum quarantine for the grpo_8x32 recipe so NO datum is ever
excluded from dispensing (every datum keeps training, scored at whatever reward
the verifier / lowest_reward assigns).

ollie-rl's quarantine (`quarantined_datums`) excludes a whole datum from the
dispense pool once, over its rewarded attempts, EITHER:
  * (length + content_filter) / rewarded >= max_unhealthy_finish_ratio, or
  * succeeded / rewarded >= max_succeed_ratio.
The predicate already treats `None` as "filter disabled" (it `continue`s past a
threshold that is None). But the `Recipe` fields are typed `float`, so setting
them to None fails pydantic validation. So this patch does two things:

  1. widen both fields to `Optional[float]` on the Recipe model (matches the
     `quarantined_datums(..., max_unhealthy_finish_ratio: Optional[float]=None,
     max_succeed_ratio: Optional[float]=None)` signature), and
  2. set both to None on the GRPO_8x32 instance.

Result: quarantine is fully off for grpo_8x32 — degenerate-looking and
too-easy datums keep getting dispensed and trained. Other recipes are
unchanged (still default 1.0/1.0). The two-phase probe gate in `pick_datum`
still works (it keys off quarantine_min_samples, not the ratios), so batch
formation is unaffected.

Idempotent. Run on VM: python3 ~/patch-disable-quarantine.py [--revert]
"""

import pathlib
import sys

RECIPES = pathlib.Path.home() / "ollie-rl/src/ollie_rl/cookbook/recipes.py"

# 1. widen field types to Optional[float]
TYPES_OLD = """    max_unhealthy_finish_ratio: float = 1.0
    max_succeed_ratio: float = 1.0"""
TYPES_NEW = """    max_unhealthy_finish_ratio: Optional[float] = 1.0
    max_succeed_ratio: Optional[float] = 1.0"""

# ensure Optional is imported (typing import line at top)
IMPORT_OLD = "from typing import Literal\n"
IMPORT_NEW = "from typing import Literal, Optional\n"

# 2. disable quarantine on GRPO_8x32
RECIPE_OLD = """GRPO_8x32 = Recipe(
    group_size=8,
    num_groups_per_batch=32,
)"""
RECIPE_NEW = """GRPO_8x32 = Recipe(
    group_size=8,
    num_groups_per_batch=32,
    # Quarantine OFF: never exclude a datum from dispensing. Every datum keeps
    # training regardless of how often it hits an unhealthy finish or how
    # reliably it succeeds; degenerate/too-easy datums still train at their
    # assigned (lowest_)reward instead of being dropped from the pool.
    max_unhealthy_finish_ratio=None,
    max_succeed_ratio=None,
)"""


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
print("Optional import:", apply(RECIPES, IMPORT_OLD, IMPORT_NEW, revert))
print("field types:    ", apply(RECIPES, TYPES_OLD, TYPES_NEW, revert))
print("GRPO_8x32:      ", apply(RECIPES, RECIPE_OLD, RECIPE_NEW, revert))

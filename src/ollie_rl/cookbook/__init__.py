from typing import Dict

from .recipes import Recipe, GRPO_16x32, GRPO_4x8

RECIPES: Dict[str, Recipe] = {
    "grpo_16x32": GRPO_16x32,
    "grpo_4x8": GRPO_4x8,
}


class Cookbook:
    """Lookup of named recipes."""

    @classmethod
    def get(cls, recipe_kind: str) -> Recipe:
        recipe = RECIPES.get(recipe_kind)
        if recipe is None:
            raise ValueError(
                f"Recipe '{recipe_kind}' not found. Available: {list(RECIPES.keys())}"
            )
        return recipe

    @classmethod
    def has(cls, recipe_kind: str) -> bool:
        return RECIPES.get(recipe_kind) is not None


__all__ = ["Cookbook", "Recipe"]

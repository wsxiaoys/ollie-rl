from typing import Dict

from .recipes import GRPO_16x32, GRPO_4x8, Recipe, RecipeInput

RECIPES: Dict[str, Recipe] = {
    "grpo_16x32": GRPO_16x32,
    "grpo_4x8": GRPO_4x8,
}

RecipeSpec = str | RecipeInput


class Cookbook:
    """Lookup and one-time resolution of named or inline recipes."""

    @classmethod
    def get(cls, recipe_kind: str) -> Recipe:
        recipe = RECIPES.get(recipe_kind)
        if recipe is None:
            raise ValueError(
                f"Recipe '{recipe_kind}' not found. Available: {list(RECIPES.keys())}"
            )
        return recipe

    @classmethod
    def resolve(cls, spec: RecipeSpec) -> Recipe:
        """Resolve a named recipe or fields layered over Recipe defaults."""
        if isinstance(spec, str):
            return cls.get(spec)

        overrides = spec.model_dump(exclude_unset=True)
        return Recipe.model_validate({**Recipe().model_dump(), **overrides})

    @classmethod
    def has(cls, recipe_kind: str) -> bool:
        return RECIPES.get(recipe_kind) is not None


__all__ = ["Cookbook", "Recipe", "RecipeInput", "RecipeSpec"]

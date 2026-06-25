from typing import Dict
from .types import Recipe, Tuner
from .gemini_msrl import GeminiMsrlRecipe

RECIPES: Dict[str, Recipe] = {
    "gemini_msrl": GeminiMsrlRecipe(),
}


class Cookbook:
    """
    Factory class to dynamically create or restore tuner instances.
    """

    @classmethod
    async def create(cls, kind: str, name: str) -> Tuner:
        recipe = RECIPES.get(kind)
        if not recipe:
            raise ValueError(
                f"Recipe template '{kind}' not found. Available templates: {list(RECIPES.keys())}"
            )
        return await recipe.create(name)

    @classmethod
    async def restore(cls, kind: str, state: str) -> Tuner:
        recipe = RECIPES.get(kind)
        if not recipe:
            raise ValueError(
                f"Recipe template '{kind}' not found. Available templates: {list(RECIPES.keys())}"
            )
        return await recipe.restore(state)


__all__ = ["Tuner", "Cookbook"]

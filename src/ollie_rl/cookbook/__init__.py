from typing import Dict
from .types import Recipe, Tuner
from .gemini_msrl import GeminiMsrlRecipe
from .tinker_rl import TinkerRlRecipe

RECIPES: Dict[str, Recipe] = {
    "gemini_msrl": GeminiMsrlRecipe(),
    "tinker_rl": TinkerRlRecipe(),
}


class Cookbook:
    """
    Factory class to dynamically create or restore tuner instances.
    """

    @classmethod
    async def create(cls, kind: str, model_id: str) -> Tuner:
        recipe = RECIPES.get(kind)
        if not recipe:
            raise ValueError(
                f"Recipe template '{kind}' not found. Available templates: {list(RECIPES.keys())}"
            )
        return await recipe.create(model_id)

    @classmethod
    async def restore(cls, kind: str, state: str) -> Tuner:
        recipe = RECIPES.get(kind)
        if not recipe:
            raise ValueError(
                f"Recipe template '{kind}' not found. Available templates: {list(RECIPES.keys())}"
            )
        return await recipe.restore(state)


__all__ = ["Tuner", "Cookbook"]

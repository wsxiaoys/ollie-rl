from typing import Dict
from .types import Recipe, StateStore, Tuner
from .gemini_msrl import GeminiMsrlRecipe

RECIPES: Dict[str, Recipe] = {
    "gemini_msrl": GeminiMsrlRecipe(),
}


class Cookbook:
    """
    Factory class to dynamically open tuner instances.
    """

    @classmethod
    async def open(cls, kind: str, name: str, state_store: StateStore) -> Tuner:
        recipe = RECIPES.get(kind)
        if not recipe:
            raise ValueError(
                f"Recipe template '{kind}' not found. Available templates: {list(RECIPES.keys())}"
            )
        return await recipe.create(name, state_store)


__all__ = ["Tuner", "Cookbook", "StateStore"]

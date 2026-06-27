from typing import Dict, List
from .types import TrainerFactory

_REGISTRY: Dict[str, TrainerFactory] = {}


def register(kind: str, factory: TrainerFactory) -> None:
    _REGISTRY[kind] = factory


def get(kind: str) -> TrainerFactory:
    factory = _REGISTRY.get(kind)
    if factory is None:
        raise ValueError(
            f"Trainer factory for kind '{kind}' not found. Available: {list(_REGISTRY.keys())}"
        )
    return factory


def available() -> List[str]:
    return list(_REGISTRY.keys())

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable

TRANSFORMER_REGISTRY: dict[str, Callable[..., QueryTransformer]] = {}

def register(name: str) -> Callable[[Callable[..., QueryTransformer]], Callable[..., QueryTransformer]]:
    def decorator(factory: Callable[..., QueryTransformer]) -> Callable[..., QueryTransformer]:
        TRANSFORMER_REGISTRY[name] = factory
        return factory
    return decorator

def get_transformer(name: str, **kwargs) -> QueryTransformer:
    if name not in TRANSFORMER_REGISTRY:
        raise ValueError(f"Transformer '{name}' not found. Available: {list(TRANSFORMER_REGISTRY.keys())}")
    return TRANSFORMER_REGISTRY[name](**kwargs)

class QueryTransformer(ABC):
    @abstractmethod
    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        pass

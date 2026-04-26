from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from tqdm import tqdm

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


class BatchProgressCallback(BaseCallbackHandler):
    """tqdm progress bar driven by LLM completions inside ``chain.batch()``."""

    def __init__(self, total: int, desc: str = "Processing"):
        super().__init__()
        self.pbar = tqdm(total=total, desc=desc)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.pbar.update(1)

    def close(self) -> None:
        self.pbar.close()


class QueryTransformer(ABC):
    @abstractmethod
    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        pass


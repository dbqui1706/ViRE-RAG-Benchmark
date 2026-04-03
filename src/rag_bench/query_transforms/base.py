"""Query transformation strategies — base class + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Registry
_REGISTRY: dict[str, Callable[..., QueryTransformer]] = {}

def register(name: str) -> Callable[[type[QueryTransformer]], type[QueryTransformer]]:
    """Decorator to register a query transformer class."""
    def decorator(cls: type[QueryTransformer]) -> type[QueryTransformer]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_transformer(key: str, **kwargs) -> QueryTransformer:
    """Instantiate a transformer by registry key.

    Args:
        key: Strategy name (e.g. "passthrough", "multi_query").
        **kwargs: Passed to the transformer constructor (llm, n_queries, etc.).

    Returns:
        QueryTransformer instance.

    Raises:
        KeyError: If key is not registered.
    """
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown query transformer '{key}'. Available: {available}")
    return _REGISTRY[key](**kwargs)


def list_strategies() -> list[str]:
    """List all registered strategy keys."""
    return sorted(_REGISTRY.keys())


class QueryTransformer(ABC):
    """Abstract base class for query transformation strategies."""

    def __init__(self, llm: ChatOpenAI | None = None, **kwargs):
        self.llm = llm

    @abstractmethod
    def transform(self, query: str) -> list[str]:
        """Transform a query into multiple queries."""
        ...

    @property
    def strategy_name(self) -> str:
        """Short key for this strategy (used in output paths)."""
        for key, cls in _REGISTRY.items():
            if isinstance(self, cls):
                return key
        return self.__class__.__name__.lower()


# Helpers
def dedup_documents(doc_lists: list[list[Document]]) -> list[Document]:
    """Remove duplicate documents based on page_content."""
    seen: set[str] = set()
    merged: list[Document] = []
    for docs in doc_lists:
        for doc in docs:
            content_hash = doc.page_content.strip()
            if content_hash not in seen:
                seen.add(content_hash)
                merged.append(doc)
    return merged

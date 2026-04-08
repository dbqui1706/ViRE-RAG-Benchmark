"""Modular retrieval strategies for ViRAG-Bench.

Each strategy is registered via the ``@register`` decorator and
instantiated via ``get_retriever()``. Follows the same Registry Pattern
used in ``rag_bench.embeddings`` and ``rag_bench.query_transforms``.

Usage::

    from rag_bench.retrievers import get_retriever, register
    from rag_bench.retrievers.base import BaseRetriever

    @register("dense")
    class DenseRetriever(BaseRetriever): ...

    retriever = get_retriever("dense", vectorstore=vs, top_k=5)
    docs = retriever.retrieve("Câu hỏi tiếng Việt")
"""
from __future__ import annotations

from .base import BaseRetriever, RetrievalResult

_REGISTRY: dict[str, type[BaseRetriever]] = {}


def register(name: str):
    """Decorator to register a retriever class under a strategy key.

    Args:
        name: The strategy identifier (e.g. ``'dense'``, ``'bm25_syl'``).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        TypeError: If the decorated class is not a BaseRetriever subclass.
    """
    def decorator(cls: type[BaseRetriever]) -> type[BaseRetriever]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_retriever(name: str, **kwargs) -> BaseRetriever:
    """Instantiate a registered retriever by strategy name.

    Args:
        name: Registered strategy key.
        **kwargs: Forwarded to the retriever constructor.

    Returns:
        An instance of the corresponding ``BaseRetriever`` subclass.

    Raises:
        KeyError: If the strategy name is not registered.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Unknown retrieval strategy: '{name}'. Available: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_strategies() -> list[str]:
    """Return all registered retrieval strategy names.

    Returns:
        A list of strategy key strings currently in the registry.
    """
    return list(_REGISTRY.keys())


from . import dense as _dense
from . import bm25 as _bm25
from . import hybrid as _hybrid
from . import expanded as _expanded

__all__ = ["BaseRetriever", "RetrievalResult", "get_retriever", "list_strategies", "register"]

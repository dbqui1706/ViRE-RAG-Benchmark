"""Abstract base class for all retrieval strategies in ViRAG-Bench."""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseRetriever(ABC):
    """Abstract base class every retrieval strategy must implement.

    Subclasses are registered via the ``@register`` decorator in
    ``rag_bench.retrievers`` and instantiated through ``get_retriever()``.

    Example:
        >>> from rag_bench.retrievers import register, get_retriever
        >>> from rag_bench.retrievers.base import BaseRetriever
        >>> @register("my_strategy")
        ... class MyRetriever(BaseRetriever):
        ...     def retrieve(self, query, **kwargs):
        ...         return []
        >>> r = get_retriever("my_strategy")
    """

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Retrieve relevant documents for a given query.

        Args:
            query: The search query string.
            **kwargs: Additional retriever-specific parameters.

        Returns:
            A list of retrieved Document objects, ordered by relevance.
        """

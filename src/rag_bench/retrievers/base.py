"""Abstract base class and shared data structures for retrieval strategies."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.documents import Document
from tqdm import tqdm


@dataclass
class RetrievalResult:
    """Result of retrieving context for a single question.

    Attributes:
        question: The original query string.
        documents: Retrieved documents ordered by relevance.
        retrieval_ms: Wall-clock retrieval latency in milliseconds.
    """

    question: str
    documents: list[Document]
    retrieval_ms: float


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

    def batch_retrieve(self, queries: list[str], **kwargs) -> list[RetrievalResult]:
        """Retrieve documents for multiple queries.

        Default implementation calls ``retrieve()`` sequentially.
        Subclasses (e.g. ``DenseRetriever``) can override this for
        true batch operations using native database APIs.

        Args:
            queries: List of search query strings.
            **kwargs: Forwarded to ``retrieve()``.

        Returns:
            A list of ``RetrievalResult``, one per query.
        """
        results: list[RetrievalResult] = []
        for q in tqdm(queries, desc="Retrieving"):
            t0 = time.perf_counter()
            docs = self.retrieve(q, **kwargs)
            ms = (time.perf_counter() - t0) * 1000
            results.append(RetrievalResult(question=q, documents=docs, retrieval_ms=ms))
        return results

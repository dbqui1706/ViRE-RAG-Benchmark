"""Dense (vector similarity) retrieval strategy.

Wraps ChromaDB ``similarity_search`` and ``max_marginal_relevance_search``
behind the ``BaseRetriever`` interface and registers them under the key
``'dense'``.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from . import register
from .base import BaseRetriever


@register("dense")
class DenseRetriever(BaseRetriever):
    """Retrieves documents using dense vector similarity search.

    Supports two search modes:

    - ``'similarity'``: cosine / dot-product similarity (default).
    - ``'mmr'``: Maximal Marginal Relevance — trades off relevance for
      diversity by over-fetching ``top_k * 3`` candidates first.

    Args:
        vectorstore: A LangChain-wrapped ChromaDB collection.
        top_k: Number of documents to return.
        search_type: ``'similarity'`` or ``'mmr'``.

    Example:
        >>> retriever = DenseRetriever(vectorstore=vs, top_k=5)
        >>> docs = retriever.retrieve("Câu hỏi tiếng Việt")
    """

    def __init__(
        self,
        vectorstore: Chroma,
        top_k: int = 5,
        search_type: str = "similarity",
    ) -> None:
        self._vs = vectorstore
        self._top_k = top_k
        self._search_type = search_type

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Retrieve top-k documents using the configured search type.

        Args:
            query: The search query string.
            **kwargs: Ignored (reserved for future use).

        Returns:
            A list of up to ``top_k`` Document objects.

        Raises:
            ValueError: If ``search_type`` is not ``'similarity'`` or ``'mmr'``.
        """
        if self._search_type == "similarity":
            return self._vs.similarity_search(query, k=self._top_k)
        if self._search_type == "mmr":
            return self._vs.max_marginal_relevance_search(
                query, k=self._top_k, fetch_k=self._top_k * 3
            )
        raise ValueError(
            f"Unknown search_type: '{self._search_type}'. "
            "Expected 'similarity' or 'mmr'."
        )

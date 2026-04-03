"""Dense (vector similarity) retrieval strategy.

Wraps ChromaDB ``similarity_search`` and ``max_marginal_relevance_search``
behind the ``BaseRetriever`` interface and registers them under the key
``'dense'``.
"""
from __future__ import annotations

import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from . import register
from .base import BaseRetriever, RetrievalResult


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
        self._vectorstore = vectorstore
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
            return self._vectorstore.similarity_search(query, k=self._top_k)
        if self._search_type == "mmr":
            return self._vectorstore.max_marginal_relevance_search(
                query, k=self._top_k, fetch_k=self._top_k * 3
            )
        raise ValueError(
            f"Unknown search_type: '{self._search_type}'. "
            "Expected 'similarity' or 'mmr'."
        )

    def batch_retrieve(self, queries: list[str], **kwargs) -> list[RetrievalResult]:
        """Batch retrieve using native ChromaDB ``collection.query()``.

        For ``search_type='similarity'``, bypasses the LangChain wrapper and
        calls ChromaDB's native batch API which embeds all queries in **one
        call** then searches in parallel — significantly faster than calling
        ``similarity_search()`` N times.

        For ``search_type='mmr'``, falls back to the sequential default
        because LangChain's MMR implementation requires special handling
        (``fetch_k``, ``lambda_mult``).

        Args:
            queries: List of search query strings.
            **kwargs: Ignored.

        Returns:
            A list of ``RetrievalResult``, one per query.
        """
        if self._search_type != "similarity":
            # MMR requires LangChain's special logic — use sequential fallback
            return super().batch_retrieve(queries, **kwargs)

        collection = self._vectorstore._collection  # native chromadb.Collection
        embed_fn = self._vectorstore._embedding_function  # same model used to build index

        t0 = time.perf_counter()
        query_embeddings = embed_fn.embed_documents(queries)
        raw = collection.query(query_embeddings=query_embeddings, n_results=self._top_k)
        total_ms = (time.perf_counter() - t0) * 1000
        per_query_ms = total_ms / max(len(queries), 1)

        results: list[RetrievalResult] = []
        for i, query in enumerate(queries):
            docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(
                    raw["documents"][i], raw["metadatas"][i]
                )
            ]
            results.append(
                RetrievalResult(
                    question=query, documents=docs, retrieval_ms=per_query_ms
                )
            )
        return results

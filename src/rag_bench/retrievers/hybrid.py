"""Hybrid retrieval strategy — Dense + BM25 with Reciprocal Rank Fusion.

Combines dense vector search (via ChromaDB) with sparse BM25 retrieval
(syllable-level) and merges results using RRF scoring. Registered under
the key ``'hybrid'``.

This is the modular replacement for the legacy ``retriever.py`` hybrid
implementation.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from . import register
from .base import BaseRetriever
from .bm25 import BM25SylRetriever
from .dense import DenseRetriever

# ---------------------------------------------------------------------------
# RRF helper (public — also tested directly)
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    doc_lists: list[list[Document]], k_rrf: int = 60,
) -> list[Document]:
    """Merge multiple ranked document lists using Reciprocal Rank Fusion.

    RRF score for each document = sum(1 / (k + rank + 1)) across all lists
    where the document appears (rank is 0-indexed).

    Documents are deduplicated by ``page_content`` and returned in
    descending RRF score order.

    Args:
        doc_lists: Ordered lists of documents from different retrievers.
        k_rrf: RRF smoothing constant (default 60, per the original paper).

    Returns:
        Merged, deduplicated list of documents sorted by RRF score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


@register("hybrid")
class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining Dense and BM25 (syllable) with RRF.

    Internally constructs a ``DenseRetriever`` and a ``BM25SylRetriever``,
    retrieves from both, and merges results using Reciprocal Rank Fusion.

    Args:
        vectorstore: A LangChain-wrapped ChromaDB collection (for dense).
        documents: Chunked document list (for building BM25 index).
        top_k: Number of documents to return per query.

    Example:
        >>> retriever = HybridRetriever(vectorstore=vs, documents=docs, top_k=5)
        >>> results = retriever.retrieve("Luật đất đai")
    """

    def __init__(
        self,
        vectorstore: Chroma,
        documents: list[Document],
        top_k: int = 5,
    ) -> None:
        self._dense = DenseRetriever(
            vectorstore=vectorstore, top_k=top_k, search_type="similarity"
        )
        self._sparse = BM25SylRetriever(documents=documents, top_k=top_k)
        self._top_k = top_k

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Retrieve documents using hybrid Dense + BM25 with RRF.

        Args:
            query: The search query string.
            **kwargs: Ignored.

        Returns:
            A deduplicated list of up to ``top_k`` documents, ranked by
            RRF score.
        """
        dense_docs = self._dense.retrieve(query)
        sparse_docs = self._sparse.retrieve(query)
        merged = reciprocal_rank_fusion([dense_docs, sparse_docs])
        return merged[: self._top_k]

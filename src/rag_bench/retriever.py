"""Legacy retrieval helpers — kept for backward-compatible hybrid search.

.. deprecated::
    New retrieval strategies use the registry in ``rag_bench.retrievers``.
    This module will be removed once the hybrid strategy is migrated to
    ``rag_bench.retrievers.hybrid`` in Phase 3.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """Result of retrieving context for a single question."""

    question: str
    documents: list[Document]
    retrieval_ms: float


# ---------------------------------------------------------------------------
# Hybrid Search helpers (RRF — used by legacy 'hybrid' search_type)
# ---------------------------------------------------------------------------


def _reciprocal_rank_fusion(
    doc_lists: list[list[Document]], k_rrf: int = 60,
) -> list[Document]:
    """Merge multiple ranked doc lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank + 1)) for each list where doc appears.

    Args:
        doc_lists: Ordered lists of documents from different retrievers.
        k_rrf: RRF constant (default 60 per the original paper).

    Returns:
        Merged, re-ranked list of documents.
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


def _search_hybrid(
    vectorstore: Chroma,
    bm25_retriever,
    query: str,
    k: int,
) -> list[Document]:
    """Hybrid search: BM25 (sparse) + Vector (dense) merged with RRF.

    Args:
        vectorstore: ChromaDB vector store.
        bm25_retriever: LangChain BM25Retriever instance.
        query: Search query.
        k: Number of documents to return.

    Returns:
        Top-k documents merged via RRF.
    """
    dense_docs = vectorstore.similarity_search(query, k=k)
    sparse_docs = bm25_retriever.invoke(query)[:k]
    return _reciprocal_rank_fusion([dense_docs, sparse_docs])[:k]


def batch_advanced_retrieve(
    vectorstore: Chroma,
    questions: list[str],
    k: int = 5,
    reranker=None,
    rerank_factor: int = 3,
    search_type: str = "hybrid",
    bm25_retriever=None,
    # Kept for backward compatibility — ignored
    transformer=None,
) -> list[RetrievalResult]:
    """Batch retrieval for the legacy 'hybrid' search_type.

    Retrieves using BM25+Dense RRF fusion, then optionally reranks.

    Args:
        vectorstore: ChromaDB vector store.
        questions: List of query strings.
        k: Number of documents to return per query.
        reranker: Optional reranker instance.
        rerank_factor: Over-retrieve k*factor before reranking.
        search_type: Must be ``'hybrid'`` for this path.
        bm25_retriever: LangChain BM25Retriever instance.
        transformer: Ignored (kept for API compatibility).

    Returns:
        List of RetrievalResult, one per question.
    """
    results = []
    for question in questions:
        t0 = time.perf_counter()
        effective_k = k * rerank_factor if reranker else k

        docs = _search_hybrid(vectorstore, bm25_retriever, question, effective_k)

        # Dedup by page_content
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)

        # Optional reranking
        if reranker and len(unique) > k:
            doc_texts = [d.page_content for d in unique]
            rerank_results = reranker.rerank(question, doc_texts, top_n=k)
            reranked_indices = [r.index for r in rerank_results]
            unique = [unique[i] for i in reranked_indices]

        retrieval_ms = (time.perf_counter() - t0) * 1000
        results.append(
            RetrievalResult(
                question=question,
                documents=unique[:k],
                retrieval_ms=retrieval_ms,
            )
        )
    return results

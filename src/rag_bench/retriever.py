"""Retrieval — separated from generation for batch processing."""

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


def retrieve_context(
    vectorstore: Chroma,
    question: str,
    k: int = 5,
) -> RetrievalResult:
    """Retrieve top-K relevant documents for a question."""
    t0 = time.perf_counter()
    docs = vectorstore.similarity_search(question, k=k)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    return RetrievalResult(question=question, documents=docs, retrieval_ms=retrieval_ms)


def batch_retrieve(
    vectorstore: Chroma,
    questions: list[str],
    k: int = 5,
) -> list[RetrievalResult]:
    """Retrieve contexts for all questions (sequential)."""
    return [retrieve_context(vectorstore, q, k) for q in questions]


# ---------------------------------------------------------------------------
# Hybrid Search helpers
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    doc_lists: list[list[Document]], k_rrf: int = 60,
) -> list[Document]:
    """Merge multiple ranked doc lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank)) for each list where doc appears.
    Higher score = more relevant.
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


def _search_dense(vectorstore: Chroma, query: str, k: int, search_type: str) -> list[Document]:
    """Dense retrieval: similarity or MMR."""
    if search_type == "mmr":
        return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
    return vectorstore.similarity_search(query, k=k)


def _search_hybrid(
    vectorstore: Chroma,
    bm25_retriever,
    query: str,
    k: int,
) -> list[Document]:
    """Hybrid search: BM25 (sparse) + Vector (dense) merged with RRF."""
    dense_docs = vectorstore.similarity_search(query, k=k)
    sparse_docs = bm25_retriever.invoke(query)[:k]
    return _reciprocal_rank_fusion([dense_docs, sparse_docs])[:k]


# ---------------------------------------------------------------------------
# Advanced retrieve (with transform, search_type, rerank)
# ---------------------------------------------------------------------------

def advanced_retrieve(
    vectorstore: Chroma,
    question: str,
    k: int = 5,
    transformer=None,
    reranker=None,
    rerank_factor: int = 3,
    search_type: str = "similarity",
    bm25_retriever=None,
) -> RetrievalResult:
    """Retrieve with optional query transformation, search type, and reranking.

    Flow:
        1. Transform: question -> [q1, q2, ...] via transformer
        2. Retrieve: using search_type (similarity | mmr | hybrid)
        3. Merge + dedup by page_content
        4. (Optional) Rerank: re-score, keep top-k
        5. Return top-k documents
    """
    t0 = time.perf_counter()

    # 1. Transform query
    queries = transformer.transform(question) if transformer else [question]
    if transformer:
        print(f"[Advanced Retrieve] Transformed query: {queries}")

    # 2. Determine effective k (over-retrieve if reranking)
    effective_k = k * rerank_factor if reranker else k

    # 3. Retrieve for all queries
    all_docs: list[Document] = []
    for q in queries:
        if search_type == "hybrid" and bm25_retriever is not None:
            all_docs.extend(_search_hybrid(vectorstore, bm25_retriever, q, effective_k))
        else:
            all_docs.extend(_search_dense(vectorstore, q, effective_k, search_type))

    # 4. Dedup by page_content
    seen: set[str] = set()
    unique_docs: list[Document] = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    # 5. Rerank if enabled
    if reranker and len(unique_docs) > k:
        doc_texts = [d.page_content for d in unique_docs]
        rerank_results = reranker.rerank(question, doc_texts, top_n=k)
        reranked_indices = [r.index for r in rerank_results]
        unique_docs = [unique_docs[i] for i in reranked_indices]

    retrieval_ms = (time.perf_counter() - t0) * 1000
    return RetrievalResult(
        question=question,
        documents=unique_docs[:k],
        retrieval_ms=retrieval_ms,
    )


def batch_advanced_retrieve(
    vectorstore: Chroma,
    questions: list[str],
    k: int = 5,
    transformer=None,
    reranker=None,
    rerank_factor: int = 3,
    search_type: str = "similarity",
    bm25_retriever=None,
) -> list[RetrievalResult]:
    """Batch version of advanced_retrieve."""
    return [
        advanced_retrieve(
            vectorstore, q, k, transformer, reranker, rerank_factor,
            search_type, bm25_retriever,
        )
        for q in questions
    ]

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
    """Retrieve top-K relevant documents for a question.

    Args:
        vectorstore: Chroma vector store.
        question: The user question.
        k: Number of documents to retrieve.

    Returns:
        RetrievalResult with documents and timing.
    """
    t0 = time.perf_counter()
    docs = vectorstore.similarity_search(question, k=k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    return RetrievalResult(
        question=question,
        documents=docs,
        retrieval_ms=retrieval_ms,
    )


def batch_retrieve(
    vectorstore: Chroma,
    questions: list[str],
    k: int = 5,
) -> list[RetrievalResult]:
    """Retrieve contexts for all questions (sequential — fast enough for vector search).

    Args:
        vectorstore: Chroma vector store.
        questions: List of questions.
        k: Number of documents per question.

    Returns:
        List of RetrievalResult in same order.
    """
    return [retrieve_context(vectorstore, q, k) for q in questions]


def advanced_retrieve(
    vectorstore: Chroma,
    question: str,
    k: int = 5,
    transformer=None,
    reranker=None,
    rerank_factor: int = 3,
) -> RetrievalResult:
    """Retrieve with optional query transformation and reranking.

    Flow:
        1. Transform: question -> [q1, q2, ...] via transformer
        2. Retrieve: similarity_search for each qi
        3. Merge + dedup by page_content
        4. (Optional) Rerank: re-score, keep top-k
        5. Return top-k documents
    """
    t0 = time.perf_counter()

    # 1. Transform query
    queries = transformer.transform(question) if transformer else [question]

    # 2. Determine effective k (over-retrieve if reranking)
    effective_k = k * rerank_factor if reranker else k

    # 3. Retrieve for all queries
    all_docs = []
    for q in queries:
        all_docs.extend(vectorstore.similarity_search(q, k=effective_k))

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
) -> list[RetrievalResult]:
    """Batch version of advanced_retrieve."""
    return [
        advanced_retrieve(vectorstore, q, k, transformer, reranker, rerank_factor)
        for q in questions
    ]


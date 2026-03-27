"""Retrieval — separated from generation for batch processing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

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

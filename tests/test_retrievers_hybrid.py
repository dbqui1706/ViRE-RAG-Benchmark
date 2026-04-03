"""Tests for HybridRetriever (Dense + BM25 with RRF)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_bench.retrievers.base import RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_docs():
    """Three sample documents for testing."""
    return [
        Document(page_content="Thủ tục hành chính về đất đai", metadata={"id": "1"}),
        Document(page_content="Bảo hiểm y tế cho người lao động", metadata={"id": "2"}),
        Document(page_content="Luật đất đai năm 2024", metadata={"id": "3"}),
    ]


@pytest.fixture
def mock_vectorstore():
    """Mock ChromaDB vectorstore."""
    vs = MagicMock()
    return vs


# ---------------------------------------------------------------------------
# RRF logic tests
# ---------------------------------------------------------------------------


def test_rrf_merge_order():
    """RRF should rank a doc appearing in both lists higher than one in only one."""
    from rag_bench.retrievers.hybrid import reciprocal_rank_fusion

    doc_a = Document(page_content="A", metadata={})
    doc_b = Document(page_content="B", metadata={})
    doc_c = Document(page_content="C", metadata={})

    # List 1: A, B
    # List 2: A, C
    # A appears in both → highest RRF score
    merged = reciprocal_rank_fusion([[doc_a, doc_b], [doc_a, doc_c]])
    assert merged[0].page_content == "A"
    assert len(merged) == 3  # A, B, C (deduped)


def test_rrf_empty_lists():
    """RRF with empty lists should return empty."""
    from rag_bench.retrievers.hybrid import reciprocal_rank_fusion

    assert reciprocal_rank_fusion([[], []]) == []


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


def test_hybrid_registered():
    """'hybrid' key should be in the registry."""
    import rag_bench.retrievers.hybrid  # noqa: F401 — trigger @register
    from rag_bench.retrievers import list_strategies

    assert "hybrid" in list_strategies()


def test_hybrid_retrieve_calls_both(mock_vectorstore, sample_docs):
    """retrieve() should call both the dense and sparse sub-retrievers."""
    import rag_bench.retrievers.hybrid  # noqa: F401
    from rag_bench.retrievers import get_retriever

    # Dense retriever returns first 2 docs
    mock_vectorstore.similarity_search.return_value = sample_docs[:2]

    retriever = get_retriever(
        "hybrid",
        vectorstore=mock_vectorstore,
        documents=sample_docs,
        top_k=2,
    )
    results = retriever.retrieve("đất đai")

    # Should have called dense (via similarity_search)
    mock_vectorstore.similarity_search.assert_called_once()
    # Should return documents (from both dense + sparse merged)
    assert len(results) > 0
    assert all(isinstance(d, Document) for d in results)


def test_hybrid_returns_top_k(mock_vectorstore, sample_docs):
    """HybridRetriever should return at most top_k documents."""
    import rag_bench.retrievers.hybrid  # noqa: F401
    from rag_bench.retrievers import get_retriever

    mock_vectorstore.similarity_search.return_value = sample_docs

    retriever = get_retriever(
        "hybrid",
        vectorstore=mock_vectorstore,
        documents=sample_docs,
        top_k=2,
    )
    results = retriever.retrieve("thủ tục")

    assert len(results) <= 2


def test_hybrid_dedup(mock_vectorstore):
    """Duplicate documents across dense and sparse should be deduped."""
    import rag_bench.retrievers.hybrid  # noqa: F401
    from rag_bench.retrievers import get_retriever

    shared_doc = Document(page_content="same content", metadata={"id": "dup"})
    unique_doc = Document(page_content="unique doc", metadata={"id": "u1"})

    # Dense returns the same doc that BM25 will also find
    mock_vectorstore.similarity_search.return_value = [shared_doc, unique_doc]

    docs = [shared_doc, unique_doc]
    retriever = get_retriever(
        "hybrid",
        vectorstore=mock_vectorstore,
        documents=docs,
        top_k=5,
    )
    results = retriever.retrieve("same content")

    # "same content" should appear only once
    contents = [d.page_content for d in results]
    assert contents.count("same content") == 1


def test_hybrid_batch_retrieve(mock_vectorstore, sample_docs):
    """batch_retrieve() should return one RetrievalResult per query."""
    import rag_bench.retrievers.hybrid  # noqa: F401
    from rag_bench.retrievers import get_retriever

    mock_vectorstore.similarity_search.return_value = sample_docs[:1]

    retriever = get_retriever(
        "hybrid",
        vectorstore=mock_vectorstore,
        documents=sample_docs,
        top_k=2,
    )
    queries = ["query 1", "query 2", "query 3"]
    results = retriever.batch_retrieve(queries)

    assert len(results) == 3
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert results[0].question == "query 1"
    assert results[2].question == "query 3"

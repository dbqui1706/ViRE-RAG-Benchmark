"""Tests for the Dense retriever strategy."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture()
def mock_vectorstore():
    """A mock ChromaDB vectorstore."""
    vs = MagicMock()
    vs.similarity_search.return_value = [
        Document(page_content="doc 1", metadata={"source": "a"}),
        Document(page_content="doc 2", metadata={"source": "b"}),
    ]
    vs.max_marginal_relevance_search.return_value = [
        Document(page_content="diverse doc", metadata={}),
    ]
    return vs


def test_dense_retriever_registered():
    """'dense' key must be present in the registry after import."""
    import rag_bench.retrievers.dense  # noqa: F401 — trigger @register
    from rag_bench.retrievers import list_strategies

    assert "dense" in list_strategies()


def test_dense_retriever_similarity(mock_vectorstore):
    """Default search_type='similarity' calls similarity_search."""
    import rag_bench.retrievers.dense  # noqa: F401
    from rag_bench.retrievers import get_retriever

    retriever = get_retriever("dense", vectorstore=mock_vectorstore, top_k=2)
    docs = retriever.retrieve("test query")

    mock_vectorstore.similarity_search.assert_called_once_with("test query", k=2)
    assert len(docs) == 2
    assert docs[0].page_content == "doc 1"


def test_dense_retriever_mmr(mock_vectorstore):
    """search_type='mmr' calls max_marginal_relevance_search."""
    import rag_bench.retrievers.dense  # noqa: F401
    from rag_bench.retrievers import get_retriever

    retriever = get_retriever(
        "dense", vectorstore=mock_vectorstore, top_k=1, search_type="mmr"
    )
    docs = retriever.retrieve("test query")

    mock_vectorstore.max_marginal_relevance_search.assert_called_once_with(
        "test query", k=1, fetch_k=3
    )
    assert len(docs) == 1
    assert docs[0].page_content == "diverse doc"


def test_dense_retriever_invalid_search_type(mock_vectorstore):
    """Unknown search_type raises ValueError."""
    import rag_bench.retrievers.dense  # noqa: F401
    from rag_bench.retrievers import get_retriever

    retriever = get_retriever(
        "dense", vectorstore=mock_vectorstore, top_k=5, search_type="invalid"
    )
    with pytest.raises(ValueError, match="Unknown search_type"):
        retriever.retrieve("test query")

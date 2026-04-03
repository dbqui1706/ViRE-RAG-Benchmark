"""Tests for BM25 retrieval strategies (syllable-level and word-level)."""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    Document(page_content="thủ tục hành chính cấp giấy phép", metadata={"source": "legal"}),
    Document(page_content="quy định về bảo hiểm y tế", metadata={"source": "medical"}),
    Document(page_content="hướng dẫn đăng ký doanh nghiệp", metadata={"source": "biz"}),
]


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_bm25_syl_registered():
    """'bm25_syl' must be in the registry after importing the module."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.retrievers import list_strategies

    assert "bm25_syl" in list_strategies()


def test_bm25_word_registered():
    """'bm25_word' must be in the registry after importing the module."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.retrievers import list_strategies

    assert "bm25_word" in list_strategies()


# ---------------------------------------------------------------------------
# Syllable-level BM25
# ---------------------------------------------------------------------------


def test_bm25_syl_tokenizer_splits_spaces():
    """BM25Syl tokenizer should split on whitespace (no word segmentation)."""
    import rag_bench.retrievers.bm25 as bm25_mod

    tokens = bm25_mod._tokenize_syllable("thủ tục hành chính")
    assert tokens == ["thủ", "tục", "hành", "chính"]


def test_bm25_syl_returns_documents():
    """BM25SylRetriever.retrieve() should return a list of Documents."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.retrievers import get_retriever

    retriever = get_retriever("bm25_syl", documents=SAMPLE_DOCS, top_k=2)
    results = retriever.retrieve("thủ tục hành chính")

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(d, Document) for d in results)


def test_bm25_syl_top_k_respected():
    """top_k must cap the number of returned documents."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.retrievers import get_retriever

    retriever = get_retriever("bm25_syl", documents=SAMPLE_DOCS, top_k=1)
    results = retriever.retrieve("giấy phép")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Word-level BM25 (underthesea)
# ---------------------------------------------------------------------------


def test_bm25_word_tokenizer_calls_underthesea():
    """BM25Word tokenizer must call underthesea.word_tokenize."""
    import rag_bench.retrievers.bm25 as bm25_mod

    with patch("rag_bench.retrievers.bm25.word_tokenize") as mock_wt:
        mock_wt.return_value = ["thủ_tục", "hành_chính"]
        tokens = bm25_mod._tokenize_word("thủ tục hành chính")

    mock_wt.assert_called_once_with("thủ tục hành chính")
    assert "thủ_tục" in tokens


def test_bm25_word_returns_documents():
    """BM25WordRetriever.retrieve() should return a list of Documents."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.retrievers import get_retriever

    retriever = get_retriever("bm25_word", documents=SAMPLE_DOCS, top_k=2)
    results = retriever.retrieve("thủ tục hành chính")

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(d, Document) for d in results)


def test_bm25_word_top_k_respected():
    """top_k must cap the number of returned documents."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.retrievers import get_retriever

    retriever = get_retriever("bm25_word", documents=SAMPLE_DOCS, top_k=1)
    results = retriever.retrieve("bảo hiểm y tế")
    assert len(results) == 1


def test_bm25_word_empty_corpus_returns_empty():
    """Retrieving from an empty corpus should return an empty list."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.retrievers import get_retriever

    retriever = get_retriever("bm25_word", documents=[], top_k=5)
    results = retriever.retrieve("bất kỳ câu hỏi nào")
    assert results == []

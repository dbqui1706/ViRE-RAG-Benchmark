"""Tests verifying the registry-based retriever is wired into the pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_bench.config import RagConfig


# ---------------------------------------------------------------------------
# Helper: minimal fake config
# ---------------------------------------------------------------------------

def _base_config(tmp_path, sample_csv, search_type: str = "similarity") -> RagConfig:
    return RagConfig(
        csv_path=str(sample_csv),
        embed_model="bge-small-en-v1.5",
        llm_api_key="fake-key",
        max_samples=3,
        top_k=2,
        chroma_dir=str(tmp_path / "chroma"),
        output_dir=str(tmp_path / "output"),
        search_type=search_type,
    )


# ---------------------------------------------------------------------------
# Unit-level tests: _build_retriever factory
# ---------------------------------------------------------------------------


def test_build_retriever_similarity_returns_dense(tmp_path, sample_csv):
    """search_type='similarity' should return a DenseRetriever."""
    import rag_bench.retrievers.dense  # noqa: F401 — register
    from rag_bench.pipeline import _build_retriever
    from rag_bench.retrievers.dense import DenseRetriever

    vs = MagicMock()
    cfg = _base_config(tmp_path, sample_csv, search_type="similarity")
    retriever = _build_retriever(cfg, vectorstore=vs, docs=[])

    assert isinstance(retriever, DenseRetriever)


def test_build_retriever_mmr_returns_dense_mmr(tmp_path, sample_csv):
    """search_type='mmr' should return a DenseRetriever with search_type='mmr'."""
    import rag_bench.retrievers.dense  # noqa: F401
    from rag_bench.pipeline import _build_retriever
    from rag_bench.retrievers.dense import DenseRetriever

    vs = MagicMock()
    cfg = _base_config(tmp_path, sample_csv, search_type="mmr")
    retriever = _build_retriever(cfg, vectorstore=vs, docs=[])

    assert isinstance(retriever, DenseRetriever)
    assert retriever._search_type == "mmr"


def test_build_retriever_bm25_syl(tmp_path, sample_csv):
    """search_type='bm25_syl' should return a BM25SylRetriever."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.pipeline import _build_retriever
    from rag_bench.retrievers.bm25 import BM25SylRetriever

    docs = [Document(page_content="thủ tục hành chính", metadata={})]
    cfg = _base_config(tmp_path, sample_csv, search_type="bm25_syl")
    retriever = _build_retriever(cfg, vectorstore=MagicMock(), docs=docs)

    assert isinstance(retriever, BM25SylRetriever)


def test_build_retriever_bm25_word(tmp_path, sample_csv):
    """search_type='bm25_word' should return a BM25WordRetriever."""
    import rag_bench.retrievers.bm25  # noqa: F401
    from rag_bench.pipeline import _build_retriever
    from rag_bench.retrievers.bm25 import BM25WordRetriever

    docs = [Document(page_content="bảo hiểm y tế", metadata={})]
    cfg = _base_config(tmp_path, sample_csv, search_type="bm25_word")
    retriever = _build_retriever(cfg, vectorstore=MagicMock(), docs=docs)

    assert isinstance(retriever, BM25WordRetriever)


def test_build_retriever_unknown_raises(tmp_path, sample_csv):
    """Unknown search_type should raise ValueError."""
    from rag_bench.pipeline import _build_retriever

    cfg = _base_config(tmp_path, sample_csv, search_type="does_not_exist")
    with pytest.raises(ValueError, match="Unknown search_type"):
        _build_retriever(cfg, vectorstore=MagicMock(), docs=[])

"""Tests for TF-IDF and Weighted Hybrid retrievers.

All tests use mocks — no external APIs or GPU required.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    Document(page_content="thủ tục hành chính cấp giấy phép xây dựng"),
    Document(page_content="quy trình khám bệnh tại bệnh viện đa khoa"),
    Document(page_content="hướng dẫn nộp thuế thu nhập cá nhân trực tuyến"),
    Document(page_content="đăng ký kinh doanh hộ cá thể theo quy định mới"),
    Document(page_content="giấy phép lái xe hạng B2 thi sát hạch lý thuyết"),
]


# ---------------------------------------------------------------------------
# TF-IDF Retriever Tests
# ---------------------------------------------------------------------------


class TestTfidfRetriever:
    """Tests for TfidfSylRetriever."""

    def test_retrieve_returns_relevant_docs(self):
        """TF-IDF should return docs containing query terms, ranked by similarity."""
        from rag_bench.retrievers.tfidf import TfidfSylRetriever

        retriever = TfidfSylRetriever(documents=SAMPLE_DOCS, top_k=3)
        results = retriever.retrieve("giấy phép xây dựng")

        assert len(results) > 0
        # The doc about "giấy phép xây dựng" should be first (highest overlap)
        assert "xây dựng" in results[0].page_content

    def test_retrieve_respects_top_k(self):
        """Should return at most top_k documents."""
        from rag_bench.retrievers.tfidf import TfidfSylRetriever

        retriever = TfidfSylRetriever(documents=SAMPLE_DOCS, top_k=2)
        results = retriever.retrieve("thủ tục hành chính")

        assert len(results) <= 2

    def test_retrieve_empty_corpus(self):
        """Empty corpus should return empty list."""
        from rag_bench.retrievers.tfidf import TfidfSylRetriever

        retriever = TfidfSylRetriever(documents=[], top_k=3)
        results = retriever.retrieve("bất kỳ câu hỏi nào")

        assert results == []

    def test_retrieve_no_match(self):
        """Query with no overlapping terms should return empty (all scores = 0)."""
        from rag_bench.retrievers.tfidf import TfidfSylRetriever

        retriever = TfidfSylRetriever(documents=SAMPLE_DOCS, top_k=3)
        results = retriever.retrieve("zzzzxxxx yyyyyy")

        assert results == []

    def test_batch_retrieve(self):
        """batch_retrieve should return one RetrievalResult per query."""
        from rag_bench.retrievers.tfidf import TfidfSylRetriever

        retriever = TfidfSylRetriever(documents=SAMPLE_DOCS, top_k=2)
        queries = ["giấy phép", "khám bệnh"]
        results = retriever.batch_retrieve(queries)

        assert len(results) == 2
        # Each result should have a question and documents list
        assert results[0].question == "giấy phép"
        assert results[1].question == "khám bệnh"
        assert len(results[0].documents) <= 2
        assert len(results[1].documents) <= 2

    def test_tfidf_l2_normalized_dot_product_equals_cosine(self):
        """Verify that TfidfVectorizer produces L2-normalized vectors,
        so dot product == cosine similarity."""
        from rag_bench.retrievers.tfidf import TfidfSylRetriever
        import numpy as np

        retriever = TfidfSylRetriever(documents=SAMPLE_DOCS, top_k=5)
        # Check L2 norm of each row is ~1.0
        for i in range(retriever._doc_matrix.shape[0]):
            row = retriever._doc_matrix[i].toarray().flatten()
            norm = np.linalg.norm(row)
            assert abs(norm - 1.0) < 1e-6, f"Row {i} norm = {norm}, expected 1.0"

    def test_top_k_larger_than_corpus(self):
        """top_k > len(docs) should return all matching docs without error."""
        from rag_bench.retrievers.tfidf import TfidfSylRetriever

        retriever = TfidfSylRetriever(documents=SAMPLE_DOCS, top_k=100)
        results = retriever.retrieve("giấy phép")

        assert len(results) <= len(SAMPLE_DOCS)


# ---------------------------------------------------------------------------
# _normalize_scores Tests
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    """Tests for the Min-Max normalization helper."""

    def test_normal_case(self):
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        scores = {"a": 10.0, "b": 20.0, "c": 5.0}
        result = _normalize_scores(scores)

        assert result["c"] == pytest.approx(0.0)     # min → 0
        assert result["b"] == pytest.approx(1.0)     # max → 1
        assert result["a"] == pytest.approx(1 / 3)   # (10-5)/(20-5)

    def test_empty_map(self):
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        assert _normalize_scores({}) == {}

    def test_all_zeros_returns_zeros(self):
        """When all scores are 0, should return 0.0 (no signal)."""
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        scores = {"a": 0.0, "b": 0.0, "c": 0.0}
        result = _normalize_scores(scores)

        assert all(v == 0.0 for v in result.values())

    def test_all_equal_nonzero_returns_ones(self):
        """When all scores are equal but > 0, should return 1.0 (equal signal)."""
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        scores = {"a": 5.0, "b": 5.0}
        result = _normalize_scores(scores)

        assert all(v == 1.0 for v in result.values())

    def test_single_entry(self):
        """Single entry with positive value → 1.0."""
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        result = _normalize_scores({"only": 42.0})
        assert result["only"] == 1.0

    def test_single_entry_zero(self):
        """Single entry with zero → 0.0."""
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        result = _normalize_scores({"only": 0.0})
        assert result["only"] == 0.0

    def test_negative_values(self):
        """Negative values (from negated distances) should normalize correctly."""
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        scores = {"close": -0.1, "mid": -0.5, "far": -0.9}
        result = _normalize_scores(scores)

        # -0.1 is highest (closest doc) → 1.0
        assert result["close"] == pytest.approx(1.0)
        # -0.9 is lowest (farthest doc) → 0.0
        assert result["far"] == pytest.approx(0.0)
        # -0.5 is middle → 0.5
        assert result["mid"] == pytest.approx(0.5)

    def test_output_range(self):
        """All normalized values should be in [0.0, 1.0]."""
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        import random
        random.seed(42)
        scores = {f"d{i}": random.uniform(-100, 100) for i in range(50)}
        result = _normalize_scores(scores)

        for v in result.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Weighted Hybrid Retriever — Unit Tests (mocked)
# ---------------------------------------------------------------------------


class TestWeightedHybridRetriever:
    """Tests for WeightedHybridRetriever using mocks."""

    def _make_retriever(self, dense_results, bm25_docs, bm25_scores,
                        top_k=3, alpha=0.3):
        """Create a WeightedHybridRetriever with mocked Dense + BM25."""
        from rag_bench.retrievers.hybrid_weighted import WeightedHybridRetriever

        # Mock vectorstore
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = dense_results

        # Create retriever with mocked BM25
        with patch.object(
            WeightedHybridRetriever, '__init__', lambda self, **kw: None
        ):
            retriever = WeightedHybridRetriever.__new__(WeightedHybridRetriever)
            retriever._vectorstore = mock_vs
            retriever._top_k = top_k
            retriever._alpha = alpha

            # Mock BM25 sparse retriever internals
            mock_sparse = MagicMock()
            mock_sparse._index = MagicMock()
            mock_sparse._documents = [Document(page_content="dummy")]
            mock_sparse._tokenizer_func = lambda text: text.lower().split()

            import numpy as np
            mock_sparse._index.retrieve.return_value = (
                [bm25_docs],     # results[0]
                [bm25_scores],   # scores[0]
            )
            retriever._sparse = mock_sparse

        return retriever

    def test_fusion_formula(self):
        """Verify: final = alpha * dense_norm + (1-alpha) * sparse_norm."""
        doc_a = Document(page_content="doc_a")
        doc_b = Document(page_content="doc_b")

        # Dense: doc_a=distance 0.1 (close), doc_b=distance 0.9 (far)
        dense_results = [(doc_a, 0.1), (doc_b, 0.9)]
        # BM25: doc_a=score 2.0 (low), doc_b=score 8.0 (high)
        bm25_docs = [doc_a, doc_b]
        bm25_scores = [2.0, 8.0]

        retriever = self._make_retriever(
            dense_results, bm25_docs, bm25_scores, top_k=2, alpha=0.3
        )
        results = retriever.retrieve("test query")

        # Dense negated: doc_a=-0.1, doc_b=-0.9
        # Dense normalized: doc_a=1.0, doc_b=0.0
        # BM25 normalized: doc_a=0.0, doc_b=1.0
        # Final: doc_a = 0.3*1.0 + 0.7*0.0 = 0.3
        # Final: doc_b = 0.3*0.0 + 0.7*1.0 = 0.7
        # doc_b should rank first!
        assert len(results) == 2
        assert results[0].page_content == "doc_b"
        assert results[1].page_content == "doc_a"

    def test_alpha_0_means_only_sparse(self):
        """alpha=0 → only sparse scores matter."""
        doc_a = Document(page_content="doc_a")
        doc_b = Document(page_content="doc_b")

        # Dense: doc_a is closer
        dense_results = [(doc_a, 0.1), (doc_b, 0.9)]
        # BM25: doc_b has higher score
        bm25_docs = [doc_a, doc_b]
        bm25_scores = [1.0, 10.0]

        retriever = self._make_retriever(
            dense_results, bm25_docs, bm25_scores, top_k=2, alpha=0.0
        )
        results = retriever.retrieve("test")

        # With alpha=0: final = 0*dense + 1*sparse → doc_b first
        assert results[0].page_content == "doc_b"

    def test_alpha_1_means_only_dense(self):
        """alpha=1 → only dense scores matter."""
        doc_a = Document(page_content="doc_a")
        doc_b = Document(page_content="doc_b")

        # Dense: doc_a is closer (lower distance)
        dense_results = [(doc_a, 0.1), (doc_b, 0.9)]
        # BM25: doc_b has higher score
        bm25_docs = [doc_a, doc_b]
        bm25_scores = [1.0, 10.0]

        retriever = self._make_retriever(
            dense_results, bm25_docs, bm25_scores, top_k=2, alpha=1.0
        )
        results = retriever.retrieve("test")

        # With alpha=1: final = 1*dense + 0*sparse → doc_a first (closer)
        assert results[0].page_content == "doc_a"

    def test_respects_top_k(self):
        """Should return at most top_k documents."""
        docs = [Document(page_content=f"doc_{i}") for i in range(5)]

        dense_results = [(d, 0.1 * (i + 1)) for i, d in enumerate(docs)]
        bm25_docs = docs
        bm25_scores = [float(5 - i) for i in range(5)]

        retriever = self._make_retriever(
            dense_results, bm25_docs, bm25_scores, top_k=2
        )
        results = retriever.retrieve("test")

        assert len(results) <= 2

    def test_disjoint_dense_and_sparse_results(self):
        """Documents appearing in only one source get imputed scores."""
        doc_dense = Document(page_content="dense_only")
        doc_sparse = Document(page_content="sparse_only")
        doc_both = Document(page_content="in_both")

        dense_results = [(doc_dense, 0.2), (doc_both, 0.5)]
        bm25_docs = [doc_sparse, doc_both]
        bm25_scores = [5.0, 3.0]

        retriever = self._make_retriever(
            dense_results, bm25_docs, bm25_scores, top_k=3, alpha=0.3
        )
        results = retriever.retrieve("test")

        # All 3 docs should appear
        contents = {r.page_content for r in results}
        assert contents == {"dense_only", "sparse_only", "in_both"}

    def test_all_zero_bm25_scores_no_inflate(self):
        """When all BM25 scores are 0, sparse component should contribute 0."""
        from rag_bench.retrievers.hybrid_weighted import _normalize_scores

        scores = {"a": 0.0, "b": 0.0, "c": 0.0}
        result = _normalize_scores(scores)

        # Should be 0.0, not 1.0 — no BM25 signal means no contribution
        assert all(v == 0.0 for v in result.values())

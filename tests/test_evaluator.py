import math

import pytest
from langchain_core.documents import Document

from rag_bench.evaluator import (
    evaluate_answer,
    evaluate_retrieval,
    exact_match,
    map_at_k,
    ndcg_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)

# === Section 1: Generation Quality ===

def test_exact_match_identical():
    assert exact_match("Hello world", "hello world") == 1.0


def test_exact_match_different():
    assert exact_match("Hello", "World") == 0.0


def test_token_f1_perfect():
    assert token_f1("the cat sat", "the cat sat") == 1.0


def test_token_f1_partial():
    f1 = token_f1("the cat", "the cat sat on mat")
    assert 0.0 < f1 < 1.0


def test_rouge_l():
    score = rouge_l("the cat sat on the mat", "the cat on the mat")
    assert 0.0 < score <= 1.0


def test_evaluate_answer():
    result = evaluate_answer("Paris", "paris is the capital")
    assert "f1" in result
    assert "rouge_l" in result
    assert result["f1"] > 0.0  # "paris" overlaps


def test_evaluate_answer_no_semantic():
    """Default call should NOT include semantic metrics."""
    result = evaluate_answer("test answer", "test answer")
    assert "bert_score" not in result
    assert "semantic_sim" not in result


# === Section 2: Retrieval Quality (index-based matching via qid) ===


def test_evaluate_retrieval_no_nodes():
    result = evaluate_retrieval([], gold_context_id="q1")
    assert result["precision"] == 0.0
    assert result["mrr"] == 0.0
    assert result["hit_rate"] == 0.0
    assert result["map_at_5"] == 0.0
    assert result["first_relevant_rank"] == float("inf")


def test_evaluate_retrieval_with_match():
    """Chunk at position 2 has matching qid → relevant."""
    docs = [
        Document(page_content="irrelevant text", metadata={"context_id": "q99"}),
        Document(page_content="the cat sat on the mat", metadata={"context_id": "q1"}),
        Document(page_content="another irrelevant", metadata={"context_id": "q50"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=3)
    assert result["mrr"] == 0.5  # 1/2 (second position)
    assert result["hit_rate"] == 1.0
    assert result["first_relevant_rank"] == 2


def test_evaluate_retrieval_no_match():
    docs = [
        Document(page_content="completely unrelated", metadata={"context_id": "q99"}),
        Document(page_content="another unrelated", metadata={"context_id": "q50"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=2)
    assert result["mrr"] == 0.0
    assert result["hit_rate"] == 0.0
    assert result["first_relevant_rank"] == float("inf")


def test_evaluate_retrieval_multiple_relevant_chunks():
    """Multiple chunks from the same gold context (after chunking)."""
    docs = [
        Document(page_content="chunk A of gold", metadata={"context_id": "q1"}),
        Document(page_content="irrelevant", metadata={"context_id": "q99"}),
        Document(page_content="chunk B of gold", metadata={"context_id": "q1"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)

    # n_relevant = 2 (two chunks with qid=q1)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall_at_3"] == pytest.approx(1.0)  # 2/2 found
    assert result["first_relevant_rank"] == 1


# === Section 2b: Unit Tests for recall_at_k, ndcg_at_k, map_at_k ===


def test_recall_at_k_continuous():
    """Recall@K is continuous: hits_in_topk / total_relevant, not binary.

    With 3 relevant docs total (R=3) and 1 found in top-3,
    recall@3 should be 1/3, NOT 1.0.
    """
    matches = [False, True, False, False, False]  # 1 relevant at rank 2
    n_relevant = 3  # but there are 3 relevant docs total

    r = recall_at_k(matches, k=3, n_relevant=n_relevant)
    assert r == pytest.approx(1.0 / 3.0)

    # With all 3 found in top-5
    matches_all = [True, True, False, True, False]
    r_all = recall_at_k(matches_all, k=5, n_relevant=3)
    assert r_all == pytest.approx(1.0)


def test_recall_at_k_single_relevant():
    """Single relevant doc (R=1) — recall@K is 1.0 if found, 0.0 if not."""
    matches = [False, True, False]
    assert recall_at_k(matches, k=1, n_relevant=1) == 0.0
    assert recall_at_k(matches, k=2, n_relevant=1) == 1.0
    assert recall_at_k(matches, k=3, n_relevant=1) == 1.0


def test_ndcg_at_k_with_total_relevant():
    """NDCG@K uses total R for IDCG, not just relevant-in-top-K.

    With R=2 but only 1 found at rank 2 in top-3:
    DCG@3 = 1/log2(3) ≈ 0.6309
    IDCG@3 = 1/log2(2) + 1/log2(3) ≈ 1.0 + 0.6309 = 1.6309
    NDCG@3 = 0.6309 / 1.6309 ≈ 0.387
    """
    matches = [False, True, False]
    n_relevant = 2

    ndcg = ndcg_at_k(matches, k=3, n_relevant=n_relevant)
    expected_dcg = 1.0 / math.log2(3)  # rank 2 → log2(2+1)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg == pytest.approx(expected_dcg / expected_idcg, abs=1e-4)


def test_ndcg_at_k_single_relevant_perfect():
    """Single relevant (R=1) at rank 1 → NDCG = 1.0."""
    matches = [True, False, False]
    assert ndcg_at_k(matches, k=3, n_relevant=1) == pytest.approx(1.0)


def test_ndcg_at_k_single_relevant_at_rank2():
    """Single relevant (R=1) at rank 2 → NDCG = 1/log2(3) ≈ 0.631."""
    matches = [False, True, False]
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(matches, k=3, n_relevant=1) == pytest.approx(expected, abs=1e-4)


def test_ndcg_at_k_no_relevant():
    """No relevant docs → NDCG = 0.0."""
    matches = [False, False, False]
    assert ndcg_at_k(matches, k=3, n_relevant=0) == 0.0


def test_map_at_k_basic():
    """MAP@K with one relevant doc at rank 2.

    AP@3 = (1/2) / min(R, 3) = 0.5 / 1 = 0.5 (R=1)
    """
    matches = [False, True, False]
    ap = map_at_k(matches, k=3, n_relevant=1)
    assert ap == pytest.approx(0.5)


def test_map_at_k_two_relevant():
    """MAP@K with two relevant docs at ranks 1 and 3.

    AP@5: hits at rank 1 (prec=1/1) and rank 3 (prec=2/3)
    AP = (1.0 + 2/3) / min(R=2, 5) = 1.6667 / 2 = 0.8333
    """
    matches = [True, False, True, False, False]
    ap = map_at_k(matches, k=5, n_relevant=2)
    expected = (1.0 + 2.0 / 3.0) / 2.0
    assert ap == pytest.approx(expected, abs=1e-4)


def test_map_at_k_no_relevant():
    """No relevant docs → AP = 0.0."""
    matches = [False, False, False]
    assert map_at_k(matches, k=3, n_relevant=0) == 0.0


# === Section 2c: evaluate_retrieval output shape and new metrics ===


def test_evaluate_retrieval_recall_at_k_values():
    """Recall@K at multiple K values — match at position 2."""
    docs = [
        Document(page_content="irrelevant", metadata={"context_id": "q99"}),
        Document(page_content="gold chunk", metadata={"context_id": "q1"}),
        Document(page_content="another irrelevant", metadata={"context_id": "q50"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)

    # Single relevant doc (R=1), so recall is binary-equivalent
    assert result["recall_at_1"] == 0.0   # not in top-1
    assert result["recall_at_3"] == 1.0   # found in top-3
    assert result["recall_at_5"] == 1.0   # found in top-5
    assert result["recall_at_10"] == 1.0  # found in top-10


def test_evaluate_retrieval_ndcg_at_k_values():
    """NDCG@K — match at position 2 gives partial score."""
    docs = [
        Document(page_content="irrelevant", metadata={"context_id": "q99"}),
        Document(page_content="gold chunk", metadata={"context_id": "q1"}),
        Document(page_content="another irrelevant", metadata={"context_id": "q50"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)

    # NDCG@1: no match at pos 1 → 0.0
    assert result["ndcg_at_1"] == 0.0

    # NDCG@3: DCG = 1/log2(3) ≈ 0.6309, IDCG = 1.0 → ~0.631
    assert 0.62 < result["ndcg_at_3"] < 0.64

    # NDCG@5 and NDCG@10 same as NDCG@3 (only one relevant doc)
    assert result["ndcg_at_5"] == result["ndcg_at_3"]


def test_evaluate_retrieval_has_map():
    """evaluate_retrieval should include MAP@K metrics."""
    docs = [
        Document(page_content="gold chunk", metadata={"context_id": "q1"}),
        Document(page_content="irrelevant", metadata={"context_id": "q99"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)

    # Match at rank 1: AP@1 = 1.0, AP@3 = 1.0, etc.
    assert result["map_at_1"] == 1.0
    assert result["map_at_3"] == 1.0
    assert result["map_at_5"] == 1.0


def test_evaluate_retrieval_has_mrr_at_k():
    """MRR should be K-aware: mrr_at_1, mrr_at_3, etc."""
    docs = [
        Document(page_content="irrelevant", metadata={"context_id": "q99"}),
        Document(page_content="gold chunk", metadata={"context_id": "q1"}),
        Document(page_content="another", metadata={"context_id": "q50"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)

    # First relevant at rank 2
    assert result["mrr_at_1"] == 0.0    # rank 2 > k=1
    assert result["mrr_at_3"] == 0.5    # 1/2, within k=3
    assert result["mrr_at_5"] == 0.5    # 1/2, within k=5
    assert result["mrr_at_10"] == 0.5   # 1/2, within k=10


def test_evaluate_retrieval_has_first_relevant_rank():
    """Should report first_relevant_rank."""
    docs = [
        Document(page_content="irrelevant", metadata={"context_id": "q99"}),
        Document(page_content="gold chunk", metadata={"context_id": "q1"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)
    assert result["first_relevant_rank"] == 2


def test_recall_ndcg_no_match():
    """All zeros when no relevant doc is found."""
    docs = [
        Document(page_content="completely unrelated", metadata={"context_id": "q99"}),
        Document(page_content="another unrelated", metadata={"context_id": "q50"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)
    assert result["recall_at_1"] == 0.0
    assert result["recall_at_5"] == 0.0
    assert result["ndcg_at_1"] == 0.0
    assert result["ndcg_at_5"] == 0.0
    assert result["map_at_5"] == 0.0


def test_recall_ndcg_empty():
    """Empty retrieval → all zeros."""
    result = evaluate_retrieval([], gold_context_id="q1")
    assert result["recall_at_1"] == 0.0
    assert result["ndcg_at_1"] == 0.0
    assert result["map_at_1"] == 0.0


def test_ndcg_at_k_perfect():
    """Match at position 1 → NDCG = 1.0."""
    docs = [Document(page_content="gold", metadata={"context_id": "q1"})]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)
    assert result["ndcg_at_1"] == 1.0
    assert result["ndcg_at_5"] == 1.0
    assert result["recall_at_1"] == 1.0


def test_evaluate_retrieval_doc_without_metadata():
    """Documents without qid metadata should be treated as non-relevant."""
    docs = [
        Document(page_content="no metadata doc"),
        Document(page_content="gold", metadata={"context_id": "q1"}),
    ]
    result = evaluate_retrieval(docs, gold_context_id="q1", k=5)
    assert result["first_relevant_rank"] == 2
    assert result["hit_rate"] == 1.0


# === Section 3: RAGAS (import test only — actual eval needs LLM) ===

# def test_ragas_importable():
#     """Verify RAGAS function is importable when ragas is installed."""
#     pytest.importorskip("ragas")
#     from rag_bench.evaluator import run_ragas_evaluation
#     assert callable(run_ragas_evaluation)

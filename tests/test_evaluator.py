import pytest
from langchain_core.documents import Document

from rag_bench.evaluator import (
    context_match,
    context_overlap,
    evaluate_answer,
    evaluate_retrieval,
    exact_match,
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


# === Section 2: Retrieval Quality ===


def test_context_match_exact():
    assert context_match("the cat sat on the mat", "the cat sat on the mat") is True


def test_context_overlap_partial():
    # 3/5 gold tokens = 0.6 >= 0.5 threshold → True
    overlap = context_overlap("the cat dog", "the cat sat on mat")
    assert 0.3 < overlap < 0.7


def test_context_match_low_overlap():
    # 1/5 gold tokens = 0.2 < 0.5 threshold → False
    assert context_match("dog", "the cat sat on mat") is False


def test_context_match_empty_gold():
    assert context_match("some text", "") is False


def test_evaluate_retrieval_no_nodes():
    result = evaluate_retrieval([], "gold context")
    assert result["context_precision"] == 0.0
    assert result["context_recall"] == 0.0
    assert result["mrr"] == 0.0
    assert result["hit_rate"] == 0.0
    assert result["best_overlap"] == 0.0


def test_evaluate_retrieval_with_match():
    """Use LangChain Documents with page_content."""
    docs = [
        Document(page_content="irrelevant text about dogs"),
        Document(page_content="the cat sat on the mat nicely"),
        Document(page_content="another irrelevant document"),
    ]
    gold = "the cat sat on the mat"

    result = evaluate_retrieval(docs, gold, k=3)
    assert result["context_recall"] == 1.0  # Found at position 2
    assert result["mrr"] == 0.5  # 1/2 (second position)
    assert result["hit_rate"] == 1.0
    assert result["best_overlap"] > 0.5


def test_evaluate_retrieval_no_match():
    docs = [
        Document(page_content="completely unrelated"),
        Document(page_content="another unrelated"),
    ]
    result = evaluate_retrieval(docs, "the cat sat on the mat", k=2)
    assert result["context_recall"] == 0.0
    assert result["mrr"] == 0.0
    assert result["hit_rate"] == 0.0


# === Section 2b: Multi-K Retrieval Metrics (Recall@K, NDCG@K) ===


def test_recall_at_k_values():
    """Recall@K at multiple K values — match at position 2."""
    docs = [
        Document(page_content="irrelevant text about dogs"),
        Document(page_content="the cat sat on the mat nicely"),
        Document(page_content="another irrelevant document"),
    ]
    gold = "the cat sat on the mat"
    result = evaluate_retrieval(docs, gold, k=5)

    assert result["recall_at_1"] == 0.0   # not in top-1
    assert result["recall_at_3"] == 1.0   # found in top-3
    assert result["recall_at_5"] == 1.0   # found in top-5
    assert result["recall_at_10"] == 1.0  # found in top-10


def test_ndcg_at_k_values():
    """NDCG@K — match at position 2 gives partial score."""
    docs = [
        Document(page_content="irrelevant text about dogs"),
        Document(page_content="the cat sat on the mat nicely"),
        Document(page_content="another irrelevant document"),
    ]
    gold = "the cat sat on the mat"
    result = evaluate_retrieval(docs, gold, k=5)

    # NDCG@1: no match at pos 1 → 0.0
    assert result["ndcg_at_1"] == 0.0

    # NDCG@3: DCG = 1/log2(3) ≈ 0.6309, IDCG = 1.0 → ~0.631
    assert 0.62 < result["ndcg_at_3"] < 0.64

    # NDCG@5 and NDCG@10 same as NDCG@3 (only one relevant doc)
    assert result["ndcg_at_5"] == result["ndcg_at_3"]


def test_recall_ndcg_no_match():
    """All zeros when no relevant doc is found."""
    docs = [
        Document(page_content="completely unrelated"),
        Document(page_content="another unrelated"),
    ]
    result = evaluate_retrieval(docs, "the cat sat on the mat", k=5)
    assert result["recall_at_1"] == 0.0
    assert result["recall_at_5"] == 0.0
    assert result["ndcg_at_1"] == 0.0
    assert result["ndcg_at_5"] == 0.0


def test_recall_ndcg_empty():
    """Empty retrieval → all zeros."""
    result = evaluate_retrieval([], "gold context")
    assert result["recall_at_1"] == 0.0
    assert result["ndcg_at_1"] == 0.0


def test_ndcg_at_k_perfect():
    """Match at position 1 → NDCG = 1.0."""
    docs = [Document(page_content="the cat sat on the mat")]
    result = evaluate_retrieval(docs, "the cat sat on the mat", k=5)
    assert result["ndcg_at_1"] == 1.0
    assert result["ndcg_at_5"] == 1.0
    assert result["recall_at_1"] == 1.0


# === Section 3: RAGAS (import test only — actual eval needs LLM) ===

def test_ragas_importable():
    """Verify RAGAS function is importable when ragas is installed."""
    pytest.importorskip("ragas")
    from rag_bench.evaluator import run_ragas_evaluation
    assert callable(run_ragas_evaluation)

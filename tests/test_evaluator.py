from langchain_core.documents import Document

from rag_bench.evaluator import (
    exact_match, token_f1, rouge_l, evaluate_answer,
    context_match, context_overlap, evaluate_retrieval,
    run_ragas_evaluation,
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


def test_evaluate_answer_vietnamese():
    result = evaluate_answer("thủ đô việt nam", "Thủ đô của Việt Nam là Hà Nội")
    assert "f1" in result
    assert "rouge_l" in result
    assert result["f1"] > 0.0
    assert result["exact_match"] == 0.0


def test_evaluate_answer_empty_strings():
    result = evaluate_answer("", "")
    assert result["exact_match"] == 1.0
    assert result["f1"] == 0.0

    result2 = evaluate_answer("text", "")
    assert result2["exact_match"] == 0.0
    assert result2["f1"] == 0.0

    result3 = evaluate_answer("", "text")
    assert result3["exact_match"] == 0.0
    assert result3["f1"] == 0.0


def test_evaluate_answer_with_semantic(mocker):
    """Test include_semantic=True with mocked semantic functions."""
    mocker.patch("rag_bench.evaluator.compute_bert_score", return_value=0.85)
    mocker.patch("rag_bench.evaluator.compute_semantic_similarity", return_value=0.90)

    result = evaluate_answer("test answer", "test answer", include_semantic=True)

    assert result["bert_score"] == 0.85
    assert result["semantic_sim"] == 0.90


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


def test_context_overlap_edge_cases():
    # Empty texts
    assert context_overlap("", "") == 0.0
    assert context_overlap("text", "") == 0.0
    assert context_overlap("", "text") == 0.0

    # Single words
    assert context_overlap("word", "word") == 1.0
    assert context_overlap("word", "different") == 0.0


def test_context_match_thresholds():
    gold = "one two three four"
    # overlap between "one two" and "one two three four":
    # forward (gold coverage): 2 / 4 = 0.5
    # backward (chunk precision): 2 / 2 = 1.0
    # max = 1.0
    assert context_match("one two", gold, threshold=0.5) is True

    # "one extra" and "one two three four":
    # forward: 1 / 4 = 0.25
    # backward: 1 / 2 = 0.5
    # max = 0.5
    assert context_match("one extra", gold, threshold=0.5) is True
    assert context_match("one extra", gold, threshold=0.6) is False


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


def test_evaluate_retrieval_overlap_scenarios():
    gold = "one two three four five six"

    # 0% overlap
    docs_0 = [Document(page_content="seven eight nine ten")]
    res_0 = evaluate_retrieval(docs_0, gold)
    assert res_0["context_recall"] == 0.0
    assert res_0["hit_rate"] == 0.0

    # 50% overlap (3/6 tokens)
    docs_50 = [Document(page_content="one two three extra words here")]
    res_50 = evaluate_retrieval(docs_50, gold)
    assert res_50["context_recall"] == 0.5
    assert res_50["hit_rate"] == 1.0  # 3/5 overlapping ret tokens (0.6) > 0.5 threshold

    # 100% overlap
    docs_100 = [Document(page_content="one two three four five six")]
    res_100 = evaluate_retrieval(docs_100, gold)
    assert res_100["context_recall"] == 1.0
    assert res_100["hit_rate"] == 1.0


# === Section 3: RAGAS (import test only — actual eval needs LLM) ===

def test_ragas_importable():
    """Verify RAGAS function is importable."""
    assert callable(run_ragas_evaluation)

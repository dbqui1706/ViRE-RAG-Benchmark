"""Test for Section 3: RAGAS Evaluation (LLM-based metrics).

Tests run_ragas_evaluation() using the RAGAS v0.4+ collections API with
experiment() decorator and asyncio.gather for concurrent metric scoring.

Usage:
    python -m tests.test_ragas_evaluation           # from project root
    python tests/test_ragas_evaluation.py            # direct run
"""

import asyncio
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Vietnamese text
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from rag_bench.evaluator import run_ragas_evaluation, build_eval_dataset, ExperimentResult


# ---------------------------------------------------------------------------
# Sample data: 3 Vietnamese QA scenarios
# ---------------------------------------------------------------------------

SAMPLE_DATA = [
    {
        # Perfect case: context fully supports reference, response is correct
        "user_input": "Thủ đô của Việt Nam là gì?",
        "retrieved_contexts": [
            "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam, "
            "đồng thời là thành phố lớn thứ hai cả nước về dân số."
        ],
        "response": "Thủ đô của Việt Nam là Hà Nội.",
        "reference": "Hà Nội.",
    },
    {
        # Partial match: response is more verbose than reference
        "user_input": "Sông nào dài nhất Việt Nam?",
        "retrieved_contexts": [
            "Sông Mê Kông (hay còn gọi là sông Cửu Long) chảy qua Việt Nam "
            "với chiều dài khoảng 4.350 km, là sông dài nhất Đông Nam Á."
        ],
        "response": "Sông dài nhất Việt Nam là sông Mê Kông, còn gọi là sông Cửu Long, dài khoảng 4.350 km.",
        "reference": "Sông Mê Kông (Cửu Long).",
    },
    {
        # Incorrect response: context provides different info than reference
        "user_input": "Ai là người sáng lập ra nhà Trần?",
        "retrieved_contexts": [
            "Nhà Trần là một triều đại phong kiến Việt Nam, tồn tại từ năm 1226 đến năm 1400. "
            "Trần Thái Tông là vị vua đầu tiên của nhà Trần."
        ],
        "response": "Trần Thái Tông là người sáng lập ra nhà Trần.",
        "reference": "Trần Thủ Độ là người có công sáng lập nhà Trần, còn Trần Thái Tông là vị vua đầu tiên.",
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_eval_dataset():
    """build_eval_dataset converts list[dict] → EvaluationDataset correctly."""
    dataset = build_eval_dataset(SAMPLE_DATA)
    assert len(dataset) == 3
    # Check first sample has correct fields
    first = list(dataset)[0]
    assert first.user_input == SAMPLE_DATA[0]["user_input"]
    assert first.response == SAMPLE_DATA[0]["response"]
    assert first.reference == SAMPLE_DATA[0]["reference"]
    assert first.retrieved_contexts == SAMPLE_DATA[0]["retrieved_contexts"]


def test_experiment_result_schema():
    """ExperimentResult Pydantic model accepts and validates fields."""
    result = ExperimentResult(
        faithfulness=0.9,
        factual_correctness=0.8,
        context_precision=0.7,
        context_recall=0.6,
        answer_relevancy=None,
    )
    assert result.faithfulness == 0.9
    assert result.answer_relevancy is None

    # With answer_relevancy
    result2 = ExperimentResult(
        faithfulness=0.9,
        factual_correctness=0.8,
        context_precision=0.7,
        context_recall=0.6,
        answer_relevancy=0.85,
    )
    assert result2.answer_relevancy == 0.85


def test_run_ragas_evaluation_requires_client():
    """run_ragas_evaluation raises ValueError when client is None."""
    import pytest
    with pytest.raises(ValueError, match="client"):
        asyncio.run(run_ragas_evaluation(SAMPLE_DATA, client=None))


# ---------------------------------------------------------------------------
# Live integration test (calls OpenAI API)
# ---------------------------------------------------------------------------

async def _run_live_test():
    """Run the full RAGAS evaluation with real API calls."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("SKIP: OPENAI_API_KEY not set")
        return None

    client = AsyncOpenAI(api_key=api_key)

    print(f"\nRunning RAGAS evaluation on {len(SAMPLE_DATA)} samples...\n")
    for i, d in enumerate(SAMPLE_DATA, 1):
        print(f"  Sample {i}:")
        print(f"    Q: {d['user_input']}")
        print(f"    A: {d['response'][:60]}...")
        print(f"    Ref: {d['reference'][:60]}...")
        print()

    # Test WITHOUT answer_relevancy (faster — no embeddings needed)
    scores = await run_ragas_evaluation(
        per_query_data=SAMPLE_DATA,
        model="gpt-4o-mini",
        client=client,
        include_answer_relevancy=False,
    )

    print("\n" + "=" * 50)
    print("RAGAS Evaluation Results")
    print("=" * 50)
    for metric, score in scores.items():
        print(f"  {metric:25s}: {score:.4f}")
    print("=" * 50)

    # Validate output structure
    assert isinstance(scores, dict), "scores should be a dict"
    for key in ["faithfulness", "factual_correctness", "context_precision", "context_recall"]:
        assert key in scores, f"Missing metric: {key}"
        assert 0.0 <= scores[key] <= 1.0 or scores[key] != scores[key], \
            f"{key}={scores[key]} out of [0,1] range"
    # answer_relevancy should NOT be present when disabled
    assert "answer_relevancy" not in scores, \
        "answer_relevancy should not be in scores when include_answer_relevancy=False"

    print("\n✅ All assertions passed!")
    return scores


def main():
    """Entry point for direct execution."""
    scores = asyncio.run(_run_live_test())
    return scores

# if __name__ == "__main__":
#     main()

"""Integration test for the full pipeline with a mock LLM."""

from unittest.mock import MagicMock, patch

from rag_bench.config import RagConfig
from rag_bench.pipeline import run_pipeline


def test_pipeline_with_mock(sample_csv, tmp_path):
    """Verify the pipeline runs end-to-end with a mock LLM."""
    config = RagConfig(
        csv_path=str(sample_csv),
        embed_model="bge-small-en-v1.5",
        llm_api_key="fake-key",
        llm_base_url="https://fake.api",
        max_samples=5,
        top_k=2,
        chroma_dir=str(tmp_path / "chroma"),
        output_dir=str(tmp_path / "output"),
    )

    # Mock the LCEL chain's invoke and batch methods
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Mock answer"
    mock_chain.batch.return_value = ["Mock answer"] * 5

    with patch("rag_bench.generator.build_prompt") as mock_build:
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_build.return_value = mock_prompt
        with patch.object(mock_chain, "__or__", return_value=mock_chain):
            with patch("rag_bench.generator.ChatOpenAI"):
                from rag_bench.generator import FPTGenerator
                gen = FPTGenerator(
                    model="test", api_key="fake", base_url="https://fake.api"
                )
                gen.chain = mock_chain
                
                with patch("rag_bench.pipeline.FPTGenerator", return_value=gen):
                    results = run_pipeline(config)

    assert results["config"]["dataset"] == "test_dataset"
    # Generation metrics
    assert "f1" in results["generation_metrics"]
    assert "rouge_l" in results["generation_metrics"]
    # Retrieval metrics
    assert "context_precision" in results["retrieval_metrics"]
    assert "context_recall" in results["retrieval_metrics"]
    assert "mrr" in results["retrieval_metrics"]
    assert "hit_rate" in results["retrieval_metrics"]
    # RAGAS metrics empty when eval_faithfulness is False
    assert results["ragas_metrics"] == {}

    # Per-query
    assert len(results["per_query"]) == 5
    # Verify output files
    out_dir = tmp_path / "output" / "test_dataset" / "bge-small-en-v1.5"
    assert (out_dir / "report.md").exists()
    assert (out_dir / "metrics_summary.json").exists()

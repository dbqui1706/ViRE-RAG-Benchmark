"""Smoke test for unified index pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rag_bench.config import RagConfig
from rag_bench.pipeline import run_unified_pipeline


@pytest.fixture()
def tiny_csvs(tmp_path: Path):
    rows_a = [
        {"question": f"Q_A{i}?", "answer": f"Answer_A{i}", "context": f"Context A{i} text " * 10}
        for i in range(5)
    ]
    rows_b = [
        {"question": f"Q_B{i}?", "answer": "", "context": f"Context B{i} text " * 10}
        for i in range(5)
    ]
    csv_a = tmp_path / "DatasetA.csv"
    csv_b = tmp_path / "DatasetB.csv"
    pd.DataFrame(rows_a).to_csv(csv_a, index=False)
    pd.DataFrame(rows_b).to_csv(csv_b, index=False)
    all_rows = (
        [dict(r, dataset="DatasetA") for r in rows_a]
        + [dict(r, dataset="DatasetB") for r in rows_b]
    )
    unified_csv = tmp_path / "unified.csv"
    pd.DataFrame(all_rows).to_csv(unified_csv, index=False)
    return csv_a, csv_b, unified_csv


def _mock_gen():
    m = MagicMock()
    m.text = "mock answer"
    m.generation_ms = 1.0
    m.input_tokens = 10
    m.output_tokens = 5
    return m


def test_unified_pipeline_runs(tiny_csvs, tmp_path):
    """Unified pipeline builds one index and evaluates each dataset."""
    csv_a, csv_b, unified_csv = tiny_csvs
    from langchain_core.embeddings import FakeEmbeddings
    with patch("rag_bench.pipeline.OpenAIGenerator") as MockLLM, \
         patch("rag_bench.pipeline.get_embed_model", return_value=FakeEmbeddings(size=384)):
        MockLLM.return_value.batch_generate.return_value = [_mock_gen()] * 5
        config = RagConfig.from_env(
            csv_path=str(csv_a),
            embed_model="bge-small-en-v1.5",
            unified_index_csv=str(unified_csv),
            output_dir=str(tmp_path / "outputs"),
            chroma_dir=str(tmp_path / "chroma"),
            max_samples=5,
            max_workers=1,
            chunk_strategy="passthrough",
            llm_api_key="test-key",
        )
        results = run_unified_pipeline(config, [str(csv_a), str(csv_b)])

    assert len(results) == 2
    for res in results:
        assert "retrieval_metrics" in res
        assert "generation_metrics" in res
        assert res["config"]["index_source"] == "unified"

    # DatasetA has gold answers -> float metrics
    gen_a = results[0]["generation_metrics"]
    assert gen_a["f1"] is not None and isinstance(gen_a["f1"], float)

    # DatasetB has no gold answers -> None metrics
    gen_b = results[1]["generation_metrics"]
    assert gen_b["f1"] is None
    assert gen_b["exact_match"] is None

    # Retrieval metrics present for both
    assert "context_precision" in results[0]["retrieval_metrics"]
    assert "hit_rate" in results[1]["retrieval_metrics"]

    # Output files exist
    out = tmp_path / "outputs"
    assert (out / "DatasetA_unified" / "baseline" / "bge-small-en-v1.5" / "evaluations.json").exists()
    assert (out / "DatasetA_unified" / "baseline" / "bge-small-en-v1.5" / "generations.json").exists()
    assert (out / "DatasetB_unified" / "baseline" / "bge-small-en-v1.5" / "evaluations.json").exists()


def test_unified_pipeline_requires_unified_index_csv():
    """Should raise AssertionError if unified_index_csv is not set."""
    config = RagConfig.from_env(
        csv_path="data/ALQAC.csv",
        embed_model="bge-small-en-v1.5",
        llm_api_key="test-key",
    )
    with pytest.raises(AssertionError, match="unified_index_csv must be set"):
        run_unified_pipeline(config, ["data/ALQAC.csv"])

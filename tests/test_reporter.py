import json

from rag_bench.reporter import save_results


def test_save_results(tmp_path):
    results = {
        "config": {"dataset": "CSConDa", "embed_model": "default", "llm_model": "llama-3.1-8b", "max_samples": 200},
        "metrics": {"em": 0.5, "f1": 0.7, "rouge_l": 0.65},
        "latency": {"mean_total_ms": 120.0, "total_cost_usd": 0.05},
    }
    save_results(results, tmp_path)

    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "report.md").exists()

    data = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert data["metrics"]["em"] == 0.5

    report = (tmp_path / "report.md").read_text()
    assert "CSConDa" in report
    assert "EM" in report


def test_save_with_per_query(tmp_path):
    results = {
        "config": {"dataset": "Test"},
        "metrics": {"em": 1.0},
        "latency": {"mean_total_ms": 50.0},
        "per_query": [{"qid": "1", "answer": "test"}],
    }
    save_results(results, tmp_path)
    assert (tmp_path / "results.json").exists()

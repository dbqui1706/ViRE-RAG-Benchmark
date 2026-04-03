from rag_bench.timer import QueryMetrics, aggregate_metrics


def test_query_metrics_total():
    m = QueryMetrics(retrieval_ms=10.0, generation_ms=50.0, input_tokens=100, output_tokens=30)
    assert m.total_ms == 60.0


def test_query_metrics_cost():
    m = QueryMetrics(retrieval_ms=10.0, generation_ms=50.0, input_tokens=1000, output_tokens=1000)
    assert m.estimated_cost_usd > 0
    # 1000/1000 * 0.0003 + 1000/1000 * 0.0006 = 0.0009
    assert abs(m.estimated_cost_usd - 0.0009) < 1e-6


def test_aggregate_metrics():
    metrics = [
        QueryMetrics(retrieval_ms=10, generation_ms=50, input_tokens=100, output_tokens=30),
        QueryMetrics(retrieval_ms=20, generation_ms=40, input_tokens=150, output_tokens=40),
    ]
    agg = aggregate_metrics(metrics)
    assert agg["mean_total_ms"] == 60.0
    assert agg["total_queries"] == 2
    assert agg["mean_retrieval_ms"] == 15.0
    assert agg["mean_generation_ms"] == 45.0
    assert agg["total_cost_usd"] > 0

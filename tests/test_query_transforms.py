import pytest
from rag_bench.query_transforms import get_transformer

def test_passthrough_transformer():
    transformer = get_transformer("passthrough")
    queries = ["q1", "q2"]
    result = transformer.batch_transform(queries)
    assert result == [["q1"], ["q2"]]

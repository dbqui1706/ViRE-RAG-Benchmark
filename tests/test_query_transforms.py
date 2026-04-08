import pytest
from unittest.mock import MagicMock, patch
from rag_bench.query_transforms import get_transformer

def test_passthrough_transformer():
    transformer = get_transformer("passthrough")
    queries = ["q1", "q2"]
    result = transformer.batch_transform(queries)
    assert result == [["q1"], ["q2"]]

from langchain_core.messages import AIMessage

def test_multi_query_transformer():
    with patch("rag_bench.query_transforms.multi_query.ChatOpenAI"):
        transformer = get_transformer("multi_query", llm_model="gpt-4o-mini", n_variations=3)
        mock_response = AIMessage(content="variant 1\nvariant 2\nvariant 3")
        transformer.chain = MagicMock()
        transformer.chain.invoke.return_value = mock_response
        
        result = transformer.batch_transform(["test question"])
        
        assert len(result) == 1
        assert result[0] == ["test question", "variant 1", "variant 2", "variant 3"]

import pytest
from unittest.mock import MagicMock, patch
from rag_bench.query_transforms import get_transformer

def test_passthrough_transformer():
    transformer = get_transformer("passthrough")
    queries = ["q1", "q2"]
    result = transformer.batch_transform(queries)
    assert result == [["q1"], ["q2"]]

from rag_bench.query_transforms.hyde import HydeResponse

def test_hyde_transformer():
    with patch("rag_bench.query_transforms.hyde.ChatOpenAI"):
        transformer = get_transformer("hyde", llm_model="gpt-4o-mini")
        mock_response = HydeResponse(document="This is a hypothetical document.")
        transformer.chain = MagicMock()
        transformer.chain.invoke.return_value = mock_response
        
        result = transformer.batch_transform(["test question"])
        
        assert len(result) == 1
        assert result[0] == ["This is a hypothetical document."]

from rag_bench.query_transforms.query_expansion import ExpansionResponse

def test_query_expansion_transformer():
    with patch("rag_bench.query_transforms.query_expansion.ChatOpenAI"):
        transformer = get_transformer("query_expansion", llm_model="gpt-4o-mini")
        mock_response = ExpansionResponse(keywords="keyword1, keyword2, keyword3")
        transformer.chain = MagicMock()
        transformer.chain.invoke.return_value = mock_response
        
        result = transformer.batch_transform(["test question"])
        
        assert len(result) == 1
        assert result[0] == ["test question keyword1, keyword2, keyword3"]

from rag_bench.query_transforms.step_back import StepBackResponse

def test_step_back_transformer():
    with patch("rag_bench.query_transforms.step_back.ChatOpenAI"):
        transformer = get_transformer("step_back", llm_model="gpt-4o-mini")
        mock_response = StepBackResponse(question="This is a generalized step-back question?")
        transformer.chain = MagicMock()
        transformer.chain.invoke.return_value = mock_response
        
        result = transformer.batch_transform(["test detailed question?"])
        
        assert len(result) == 1
        assert result[0] == ["test detailed question?", "This is a generalized step-back question?"]

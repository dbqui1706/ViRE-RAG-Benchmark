import pytest
from _pytest.monkeypatch import monkeypatch
from rag_bench.retrievers import get_retriever
from rag_bench.retrievers.base import RetrievalResult
from unittest.mock import MagicMock

def test_expanded_retriever_batch():
    mock_base = MagicMock()
    # base returns RetrievalResult for each query variant
    doc1 = MagicMock(page_content="doc1")
    doc2 = MagicMock(page_content="doc2")
    doc3 = MagicMock(page_content="doc3")
    
    res1 = RetrievalResult(question="q1_v1", documents=[doc1, doc2], retrieval_ms=10.0)
    res2 = RetrievalResult(question="q1_v2", documents=[doc2, doc3], retrieval_ms=10.0)
    
    mock_base.batch_retrieve.return_value = [res1, res2]
    
    mock_transformer = MagicMock()
    mock_transformer.batch_transform.return_value = [["q1_v1", "q1_v2"]]
    
    retriever = get_retriever("expanded", base_retriever=mock_base, transformer=mock_transformer, top_k=2)
    results = retriever.batch_retrieve(["q1"])
    
    assert len(results) == 1
    # We expect RRF fusion to merge docs, returning max 2 distinct docs
    assert len(results[0].documents) <= 2
    assert isinstance(results[0], RetrievalResult)
    mock_transformer.batch_transform.assert_called_once_with(["q1"])
    # assert the base was called with the flattened query list
    mock_base.batch_retrieve.assert_called_once_with(["q1_v1", "q1_v2"])

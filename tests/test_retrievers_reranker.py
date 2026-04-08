import pytest
from langchain_core.documents import Document

from rag_bench.retrievers.reranker import RerankRetriever
from rag_bench.retrievers.base import BaseRetriever

class MockBaseRetriever(BaseRetriever):
    def __init__(self):
        self._top_k = 5
        
    def retrieve(self, query: str, **kwargs):
        return [
            Document(page_content="Bad doc"),
            Document(page_content="Good doc")
        ]

def test_reranker_retriever(mocker):
    mock_base = MockBaseRetriever()
    
    rerank_retriever = RerankRetriever(
        base_retriever=mock_base,
        api_key="test_key",
        top_k=1,
        top_m=2
    )
    
    # Check that it properly overrided the inner top_k to match top_m safely
    assert mock_base._top_k == 2
    
    mock_rerank_method = mocker.patch.object(rerank_retriever.rerank_client, "rerank")
    
    class RR:
        def __init__(self, i, s): 
            self.index=i
            self.relevance_score=s
            
    # API says index 1 is best
    mock_rerank_method.return_value = [RR(1, 0.99)]
    
    docs = rerank_retriever.retrieve("query")
    assert len(docs) == 1
    assert docs[0].page_content == "Good doc"
    assert docs[0].metadata["rerank_score"] == 0.99
    
def test_reranker_retriever_fallback(mocker):
    mock_base = MockBaseRetriever()
    
    rerank_retriever = RerankRetriever(
        base_retriever=mock_base,
        api_key="test_key",
        top_k=1
    )
    
    mock_rerank_method = mocker.patch.object(rerank_retriever.rerank_client, "rerank")
    mock_rerank_method.side_effect = Exception("API down")
    
    docs = rerank_retriever.retrieve("query")
    # Fallback to base retriever order truncated to top_k
    assert len(docs) == 1
    assert docs[0].page_content == "Bad doc"

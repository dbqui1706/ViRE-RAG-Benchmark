import pytest
import numpy as np
from langchain_core.documents import Document

from rag_bench.retrievers.dartboard import DartboardRetriever, cosine_similarity
from rag_bench.retrievers.base import BaseRetriever

class MockBaseRetriever(BaseRetriever):
    def retrieve(self, query: str, **kwargs):
        # Return 3 documents
        return [
            Document(page_content="A"),
            Document(page_content="B"),
            Document(page_content="C"),
        ]

class MockEmbedModel:
    def embed_documents(self, texts):
        # A, B are identical. C is opposite.
        # Query is close to A and B.
        # We want diversity, so it should pick A then C (bypassing B due to high similarity to A).
        embeds = []
        for t in texts:
            if t == "A": embeds.append([1.0, 0.0])
            elif t == "B": embeds.append([0.9, 0.1])
            elif t == "C": embeds.append([0.0, 1.0])
        return embeds
        
    def embed_query(self, text):
        return [1.0, 0.0]

def test_cosine_similarity():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    v3 = np.array([1.0, 0.0])
    assert cosine_similarity(v1, v2) == 0.0
    assert cosine_similarity(v1, v3) == 1.0

def test_dartboard_retriever():
    mock_base = MockBaseRetriever()
    embed_model = MockEmbedModel()
    
    # We want top 2
    retriever = DartboardRetriever(
        base_retriever=mock_base,
        embed_model=embed_model,
        oversample_factor=3,
        diversity_lambda=0.2, # Strong penalty for redundancy
        top_k=2
    )
    
    docs = retriever.retrieve("query")
    assert len(docs) == 2
    assert docs[0].page_content == "A"  # Best relevance
    assert docs[1].page_content == "C"  # Highest diversity against A

def test_dartboard_retriever_fallback():
    mock_base = MockBaseRetriever()
    class BadEmbedModel:
        def embed_query(self, t): raise Exception("offline")
        def embed_documents(self, t): raise Exception("offline")
        
    retriever = DartboardRetriever(
        base_retriever=mock_base,
        embed_model=BadEmbedModel(),
        top_k=2
    )
    
    docs = retriever.retrieve("query")
    assert len(docs) == 2
    assert docs[0].page_content == "A"
    assert docs[1].page_content == "B"

import pytest
from langchain_core.documents import Document

from rag_bench.retrievers.rrf_ensemble import (
    WeightedEnsembleRetriever,
    weighted_reciprocal_rank_fusion,
)
from rag_bench.retrievers.base import BaseRetriever


def test_weighted_rrf_fusion():
    docs_a = [Document(page_content="doc1"), Document(page_content="doc2")]
    docs_b = [Document(page_content="doc2"), Document(page_content="doc3")]

    # Weight A = 1.0, Weight B = 0.5
    # doc1 score: 1.0 * (1/(60+0+1)) = 1/61 ~ 0.01639
    # doc2 score: 1.0 * (1/(60+1+1)) + 0.5 * (1/(60+0+1)) = 1/62 + 0.5/61 ~ 0.01612 + 0.00819 = 0.0243
    # doc3 score: 0.5 * (1/(60+1+1)) = 0.5/62 ~ 0.00806
    # Order should be: doc2, doc1, doc3

    merged = weighted_reciprocal_rank_fusion([(docs_a, 1.0), (docs_b, 0.5)])

    assert len(merged) == 3
    assert merged[0].page_content == "doc2"
    assert merged[1].page_content == "doc1"
    assert merged[2].page_content == "doc3"


class MockRetrieverA(BaseRetriever):
    def retrieve(self, query: str, **kwargs):
        return [Document(page_content="A1"), Document(page_content="A2")]

class MockRetrieverB(BaseRetriever):
    def retrieve(self, query: str, **kwargs):
        return [Document(page_content="A2"), Document(page_content="B1")]


def test_weighted_ensemble_retriever():
    retrieverA = MockRetrieverA()
    retrieverB = MockRetrieverB()

    # If weights are missing, error is raised if lengths mismatch. Handled at init.
    ensemble = WeightedEnsembleRetriever(
        retrievers=[retrieverA, retrieverB],
        weights=[0.8, 0.2],
        top_k=2
    )

    results = ensemble.retrieve("query")
    
    # A2 gets score from both, so it will be top 1.
    # A1 gets 0.8 * 1/61 ~ 0.0131
    # A2 gets 0.8 * 1/62 + 0.2 * 1/61 ~ 0.0129 + 0.0032 = 0.0161
    assert len(results) == 2
    assert results[0].page_content == "A2"
    assert results[1].page_content == "A1"


def test_weighted_ensemble_invalid_weights():
    with pytest.raises(ValueError):
        WeightedEnsembleRetriever(
            retrievers=[MockRetrieverA(), MockRetrieverB()],
            weights=[1.0] # missing one weight
        )

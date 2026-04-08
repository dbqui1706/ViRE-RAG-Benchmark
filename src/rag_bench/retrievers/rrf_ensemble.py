"""Weighted Ensemble retrieval strategy — N methods with weighted RRF.

Combines multiple retrievers using a weighted variant of Reciprocal Rank Fusion.
Registered under the key ``'rrf_ensemble'``.
"""
from __future__ import annotations

from langchain_core.documents import Document

from . import register
from .base import BaseRetriever


def weighted_reciprocal_rank_fusion(
    retriever_results: list[tuple[list[Document], float]],
    k_rrf: int = 60,
) -> list[Document]:
    """Merge ranked document lists using weighted Reciprocal Rank Fusion.

    RRF score for each document = sum(weight * (1 / (k + rank + 1)))

    Args:
        retriever_results: List of tuples. Each tuple is (List_Of_Docs, weight).
        k_rrf: Smoothing constant.

    Returns:
        Merged, deduplicated list of documents sorted by the weighted RRF score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list, weight in retriever_results:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
            
            # Weighted RRF penalty formula
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k_rrf + rank + 1))

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [doc_map[k] for k in sorted_keys]


@register("rrf_ensemble")
class WeightedEnsembleRetriever(BaseRetriever):
    """Ensemble retriever combining multiple retrievers via weighted RRF.

    Args:
        retrievers: List of BaseRetriever instances.
        weights: List of float weights corresponding to each retriever.
                 If None, weights default to 1.0.
        c: Int for the RRF smoothing constant `k`. Default 60.
        top_k: Number of documents to return per query.
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        weights: list[float] | None = None,
        c: int = 60,
        top_k: int = 5,
    ) -> None:
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("Number of weights must match number of retrievers.")
        
        self.retrievers = retrievers
        self.weights = weights if weights is not None else [1.0] * len(retrievers)
        self.c = c
        self._top_k = top_k

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Retrieve documents using all retrievers and merge.

        Args:
            query: The search query string.
            **kwargs: Ignored.

        Returns:
            A deduplicated list of up to ``top_k`` documents, ranked by
            weighted RRF score.
        """
        results_with_weights = []
        for i, retriever in enumerate(self.retrievers):
            docs = retriever.retrieve(query)
            results_with_weights.append((docs, self.weights[i]))

        merged = weighted_reciprocal_rank_fusion(results_with_weights, k_rrf=self.c)
        return merged[: self._top_k]

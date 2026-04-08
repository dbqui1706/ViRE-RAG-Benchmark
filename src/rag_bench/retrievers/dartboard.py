"""Dartboard retrieval strategy — Diversity undersampling.

Retrieves an oversampled set of documents from a base retriever,
then applies a diversity-penalty (distance maximization) algorithm
to return a subset ensuring information gain and minimizing redundancy.
Registered under the key ``'dartboard'``.
"""
from __future__ import annotations

import numpy as np
from langchain_core.documents import Document

from . import register
from .base import BaseRetriever


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


@register("dartboard")
class DartboardRetriever(BaseRetriever):
    """Retriever focusing on diversity via oversampling and penalty.

    Args:
        base_retriever: BaseRetriever instance to pull initial candidates.
        embed_model: Configured embedding model to calculate distance.
        oversample_factor: Multiplier for initial retrieval. Default 5.
        diversity_lambda: Weight for exploration vs exploitation (lower is more diverse). Default 0.5.
        top_k: Final number of documents to return.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        embed_model,
        oversample_factor: int = 5,
        diversity_lambda: float = 0.5,
        top_k: int = 5,
    ) -> None:
        self.base_retriever = base_retriever
        self.embed_model = embed_model
        
        self.oversample_factor = oversample_factor
        self.diversity_lambda = diversity_lambda
        self._top_k = top_k
        
        if hasattr(self.base_retriever, '_top_k'):
            self.base_retriever._top_k = self._top_k * self.oversample_factor

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Perform diverse selection over an oversampled pool.

        Args:
            query: The search query string.
            **kwargs: Ignored.

        Returns:
            A diverse subset of up to ``top_k`` documents.
        """
        # 1. Oversampled retrieval
        candidates = self.base_retriever.retrieve(query)
        if len(candidates) <= self._top_k:
            return candidates

        # 2. Embed the candidates
        texts = [doc.page_content for doc in candidates]
        try:
            embeddings = self.embed_model.embed_documents(texts)
            query_embedding = self.embed_model.embed_query(query)
        except Exception:
            # Fallback if embed_model disconnected
            return candidates[:self._top_k]

        candidates_emb = [np.array(e) for e in embeddings]
        query_emb = np.array(query_embedding)

        # 3. Dartboard / MMR-like selection
        selected_indices = []
        unselected_indices = list(range(len(candidates)))
        
        # Select first item globally best to query
        initial_scores = [cosine_similarity(query_emb, e) for e in candidates_emb]
        best_first = int(np.argmax(initial_scores))
        
        selected_indices.append(best_first)
        unselected_indices.remove(best_first)

        # Select the remaining top_k - 1 documents
        while len(selected_indices) < self._top_k and unselected_indices:
            best_score = -float("inf")
            best_idx = -1

            for idx in unselected_indices:
                # Relevance to query
                rel = cosine_similarity(query_emb, candidates_emb[idx])
                
                # Max similarity to any already selected document (Redundancy)
                sim_to_selected = max(
                    cosine_similarity(candidates_emb[idx], candidates_emb[sel]) 
                    for sel in selected_indices
                )
                
                # MMR Penalty Formula
                mmr_score = self.diversity_lambda * rel - (1 - self.diversity_lambda) * sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]

"""Weighted Hybrid retrieval strategy — Dense + BM25 with Linear Score Fusion."""
from __future__ import annotations

import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from . import register
from .base import BaseRetriever, RetrievalResult
from .bm25 import BM25WordRetriever

# ---------------------------------------------------------------------------
# Score normalization helper
# ---------------------------------------------------------------------------


def _normalize_scores(doc_score_map: dict[str, float]) -> dict[str, float]:
    """Min-Max scale a dictionary of raw scores into [0.0, 1.0].

    Edge cases:
        - Empty map → empty map.
        - All values equal and zero → all map to 0.0 (no signal).
        - All values equal and non-zero → all map to 1.0 (equal signal).
    """
    if not doc_score_map:
        return {}
    
    min_val = min(doc_score_map.values())
    max_val = max(doc_score_map.values())
    
    if max_val == min_val:
        # No discriminating power — return 0.0 if all zeros, 1.0 otherwise
        fill = 0.0 if max_val == 0.0 else 1.0
        return {k: fill for k in doc_score_map}
        
    return {k: (v - min_val) / (max_val - min_val) for k, v in doc_score_map.items()}


# ---------------------------------------------------------------------------
# WeightedHybridRetriever
# ---------------------------------------------------------------------------


@register("hybrid_weighted")
class WeightedHybridRetriever(BaseRetriever):
    """Hybrid retriever combining Dense and BM25 (word) via Weighted Sum.

    1. Get top (K*3) candidates from Dense + Sparse.
    2. Score all unique candidates using both methods.
    3. Min-Max scale Dense scores and Sparse scores to [0.0, 1.0].
    4. Compute final score: alpha * Dense_Score + (1 - alpha) * Sparse_Score.
    
    Args:
        vectorstore: A LangChain-wrapped ChromaDB collection (for dense).
        documents: Chunked document list (for building BM25 index).
        top_k: Number of documents to return per query.
        alpha: Weight for the Dense score. Sparse gets (1.0 - alpha).
               Default is 0.3.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        documents: list[Document],
        top_k: int = 5,
        alpha: float = 0.3,
    ) -> None:
        self._vectorstore = vectorstore
        # BM25WordRetriever handles the caching for us out of the box
        self._sparse = BM25WordRetriever(documents=documents, top_k=top_k)
        self._top_k = top_k
        self._alpha = alpha
        
    def _get_page_content(self, doc: Document) -> str:
        return doc.page_content

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Retrieve documents using weighted linear combination of scores."""
        fetch_k = self._top_k * 3
        
        # 1. Fetch from Dense
        # similarity_search_with_score returns (Document, distance)
        # Distance: lower is better (L2 or cosine distance).
        # Negate so higher = more relevant, then Min-Max normalize later.
        dense_results = self._vectorstore.similarity_search_with_score(query, k=fetch_k)
        dense_scores_map = {}
        doc_obj_map = {}
        for doc, distance in dense_results:
            key = self._get_page_content(doc)
            dense_scores_map[key] = -float(distance)
            doc_obj_map[key] = doc
            
        # 2. Fetch from BM25 Sparse
        # We access the internal bm25s index to get raw scores
        sparse_scores_map = {}
        if self._sparse._index is not None and self._sparse._documents:
            tokenized_query = self._sparse._tokenizer_func(query)
            bm25_k = min(fetch_k, len(self._sparse._documents))
            
            # bm25s returns numpy arrays nested in lists
            sparse_res, sparse_scs = self._sparse._index.retrieve(
                [tokenized_query], corpus=self._sparse._documents, k=bm25_k
            )
            
            for doc, score in zip(sparse_res[0], sparse_scs[0]):
                key = self._get_page_content(doc)
                sparse_scores_map[key] = float(score)
                if key not in doc_obj_map:
                    doc_obj_map[key] = doc
                    
        # 3. Handle missing scores and Normalize
        # First, unique documents have a score in both maps
        unique_keys = set(dense_scores_map.keys()).union(set(sparse_scores_map.keys()))
        
        # Impute missing dense scores with the minimum dense score observed, or 0
        min_dense = min(dense_scores_map.values()) if dense_scores_map else 0.0
        for k in unique_keys:
            if k not in dense_scores_map:
                dense_scores_map[k] = min_dense
                
        # Impute missing sparse scores with 0.0 (BM25 minimum)
        for k in unique_keys:
            if k not in sparse_scores_map:
                sparse_scores_map[k] = 0.0
                
        # Normalize to [0.0, 1.0]
        norm_dense = _normalize_scores(dense_scores_map)
        norm_sparse = _normalize_scores(sparse_scores_map)
        
        # 4. Final fusion
        final_scores = {}
        for k in unique_keys:
            final_scores[k] = self._alpha * norm_dense[k] + (1.0 - self._alpha) * norm_sparse[k]
            
        # Sort descending
        sorted_keys = sorted(final_scores, key=final_scores.__getitem__, reverse=True)
        
        # Select top k
        top_k_keys = sorted_keys[:self._top_k]
        return [doc_obj_map[k] for k in top_k_keys]

    def batch_retrieve(self, queries: list[str], **kwargs) -> list[RetrievalResult]:
        """Fall back to sequential retrieval.
        
        Batch retrieving with scores across different bounds requires per-query normalization, 
        making vectorization complex.
        """
        return super().batch_retrieve(queries, **kwargs)

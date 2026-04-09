from __future__ import annotations

import time
from itertools import islice

from langchain_core.documents import Document
from tqdm import tqdm

from .base import BaseRetriever, RetrievalResult
from . import register
from rag_bench.query_transforms.base import QueryTransformer


@register("expanded")
class ExpandedRetriever(BaseRetriever):
    """Retriever that expands queries via a transformer, then fuses results with RRF."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        transformer: QueryTransformer,
        top_k: int = 5,
        rrf_k: int = 60,
        **kwargs,
    ):
        self.base_retriever = base_retriever
        self.transformer = transformer
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        return self.batch_retrieve([query], **kwargs)[0].documents

    def batch_retrieve(self, queries: list[str], **kwargs) -> list[RetrievalResult]:
        t0 = time.perf_counter()
        k_val = kwargs.get("top_k", self.top_k)

        # 1. Expand queries
        expanded_queries_list = self.transformer.batch_transform(queries)

        # 2. Flatten & batch-retrieve in one call
        flat_queries = [q for group in expanded_queries_list for q in group]
        base_results = self.base_retriever.batch_retrieve(flat_queries, **kwargs)

        # 3. Re-group and fuse via RRF
        results_iter = iter(base_results)
        total_ms = (time.perf_counter() - t0) * 1000
        per_query_ms = total_ms / len(queries)

        final_results = []
        for i, q_group in enumerate(tqdm(expanded_queries_list, desc="Merging query results")):
            group_docs = [r.documents for r in islice(results_iter, len(q_group))]
            merged = self._rrf_fuse(group_docs, k_val)
            final_results.append(
                RetrievalResult(question=queries[i], documents=merged, retrieval_ms=per_query_ms)
            )

        return final_results

    def _rrf_fuse(self, doc_lists: list[list[Document]], top_k: int) -> list[Document]:
        """Merge multiple ranked lists via Reciprocal Rank Fusion.

        Formula: RRF(d) = Σ 1 / (rank(d, r) + k)  for each ranking r
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for rank_list in doc_lists:
            for rank, doc in enumerate(rank_list):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank + 1 + self.rrf_k)
                if key not in doc_map:
                    doc_map[key] = doc

        ranked_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [doc_map[k] for k in ranked_keys]

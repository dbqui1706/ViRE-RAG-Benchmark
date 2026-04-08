from __future__ import annotations
import time
from langchain_core.documents import Document
from .base import BaseRetriever, RetrievalResult
from . import register
from rag_bench.query_transforms.base import QueryTransformer

@register("expanded")
class ExpandedRetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, transformer: QueryTransformer, top_k: int = 5, **kwargs):
        self.base_retriever = base_retriever
        self.transformer = transformer
        self.top_k = top_k
        self.k = 60 # RRF constant

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        return self.batch_retrieve([query], **kwargs)[0].documents

    def batch_retrieve(self, queries: list[str], **kwargs) -> list[RetrievalResult]:
        t0 = time.perf_counter()
        k_val = kwargs.get("top_k", self.top_k)
        expanded_queries_list = self.transformer.batch_transform(queries)
        
        # Flatten all queries
        flat_queries = []
        for q_list in expanded_queries_list:
            flat_queries.extend(q_list)
            
        # Retrieve all at once using base retriever
        base_results = self.base_retriever.batch_retrieve(flat_queries, **kwargs)
        
        # Re-group and merge via RRF
        final_results = []
        idx = 0
        for i, q_list in enumerate(expanded_queries_list):
            group_results = [res.documents for res in base_results[idx : idx + len(q_list)]]
            idx += len(q_list)
            
            # RRF fusion
            doc_scores = {}
            doc_map = {}
            for rank_list in group_results:
                for rank, doc in enumerate(rank_list):
                    content = doc.page_content
                    if content not in doc_scores:
                        doc_scores[content] = 0.0
                        doc_map[content] = doc
                    doc_scores[content] += 1.0 / (rank + 1 + self.k)
            
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            merged = [doc_map[content] for content, _ in sorted_docs[:k_val]]
            
            ms = (time.perf_counter() - t0) * 1000 / len(queries)
            final_results.append(RetrievalResult(question=queries[i], documents=merged, retrieval_ms=ms))
            
        return final_results

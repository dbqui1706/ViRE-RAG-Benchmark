"""Two-stage retrieval wrapping a base retriever and an API Reranker.

First retrieves Top-M documents using the base strategy, then
re-ranks those documents using an external Reranker Client, and returns Top-K.
Registered under the key ``'reranker'``.
"""
from __future__ import annotations

from langchain_core.documents import Document

from . import register
from .base import BaseRetriever
from rag_bench.reranker import FPTReranker


@register("reranker")
class RerankRetriever(BaseRetriever):
    """Reranker retriever conducting two-stage retrieval.

    Args:
        base_retriever: A BaseRetriever instance for the first stage.
            Over-retrieval (top_m) is controlled by the pipeline at
            construction time — the base retriever is already built
            with the inflated top_k.
        api_key: The API Key for the Rerank API.
        base_url: The Base URL for the Rerank service.
        model: Reranker model endpoint identifier.
        top_k: Number of documents to return after reranking. Default 5.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        api_key: str,
        base_url: str = "https://mkp-api.fptcloud.com",
        model: str = "bge-reranker-v2-m3",
        top_k: int = 5,
        **kwargs,
    ) -> None:
        self.base_retriever = base_retriever
        self.rerank_client = FPTReranker(api_key=api_key, base_url=base_url, model=model)
        self._top_k = top_k

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Perform two-stage retrieval: base_retriever -> rerank.

        Args:
            query: The search query string.
            **kwargs: Ignored.

        Returns:
            A list of up to ``top_k`` documents, sorted by new rerank scores.
        """
        # Stage 1: Initial Retrieval
        initial_docs = self.base_retriever.retrieve(query)
        if not initial_docs:
            return []
            
        doc_texts = [d.page_content for d in initial_docs]
        
        # Stage 2: Reranking
        try:
            # Re-rank API call
            rerank_results = self.rerank_client.rerank(
                query=query, 
                documents=doc_texts, 
                top_n=self._top_k
            )
            
            # Map back to original Documents based on index
            final_docs = []
            for rr in rerank_results:
                original_doc = initial_docs[rr.index]
                # Attach the relevance score to metadata
                original_doc.metadata["rerank_score"] = rr.relevance_score
                final_docs.append(original_doc)
            
            return final_docs
            
        except Exception:
            # API failure fallback - return standard first stage docs limited to top_k
            return initial_docs[:self._top_k]

"""Reranker — FPT Marketplace bge-reranker-v2-m3 API client."""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class RerankResult:
    """Single rerank result with original index and relevance score."""

    index: int
    relevance_score: float


class FPTReranker:
    """Rerank documents via FPT Marketplace bge-reranker-v2-m3 API.

    Endpoint: POST {base_url}/v1/rerank
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://mkp-api.fptcloud.com",
        model: str = "bge-reranker-v2-m3",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def rerank(
        self, query: str, documents: list[str], top_n: int,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_n: Number of top results to return.

        Returns:
            List of RerankResult sorted by relevance_score descending.
        """
        response = requests.post(
            f"{self.base_url}/v1/rerank",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            RerankResult(index=r["index"], relevance_score=r["relevance_score"])
            for r in results
        ]

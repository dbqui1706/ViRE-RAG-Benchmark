"""Query engine wrapper with latency tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .timer import QueryMetrics


@dataclass
class QueryResult:
    """Result of a single RAG query with timing."""

    question: str
    answer: str
    source_nodes: list[Any]
    metrics: QueryMetrics


def query_with_timing(
    query_engine: Any,
    question: str,
    llm: Any,
) -> QueryResult:
    """Run a query and track retrieval + generation latency.

    Args:
        query_engine: LlamaIndex QueryEngine.
        question: The question to ask.
        llm: The FPTGenerator (to extract last_metrics).

    Returns:
        QueryResult with answer, sources, and timing.
    """
    t0 = time.perf_counter()
    response = query_engine.query(question)
    total_ms = (time.perf_counter() - t0) * 1000

    llm_metrics = llm.get_last_metrics() if hasattr(llm, "get_last_metrics") else {}
    gen_ms = llm_metrics.get("generation_ms", 0.0)
    retrieval_ms = total_ms - gen_ms

    metrics = QueryMetrics(
        retrieval_ms=max(retrieval_ms, 0.0),
        generation_ms=gen_ms,
        input_tokens=llm_metrics.get("input_tokens", 0),
        output_tokens=llm_metrics.get("output_tokens", 0),
    )

    return QueryResult(
        question=question,
        answer=str(response),
        source_nodes=response.source_nodes if hasattr(response, "source_nodes") else [],
        metrics=metrics,
    )

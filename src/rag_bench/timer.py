"""Latency and cost tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FPT pricing estimate (USD per 1K tokens) — update with actual pricing
INPUT_COST_PER_1K = 0.0003
OUTPUT_COST_PER_1K = 0.0006


@dataclass
class QueryMetrics:
    """Timing and token metrics for a single query."""

    retrieval_ms: float
    generation_ms: float
    input_tokens: int
    output_tokens: int

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens / 1000 * INPUT_COST_PER_1K
            + self.output_tokens / 1000 * OUTPUT_COST_PER_1K
        )


def aggregate_metrics(metrics: list[QueryMetrics]) -> dict:
    """Aggregate per-query metrics into summary statistics."""
    totals = [m.total_ms for m in metrics]
    return {
        "total_queries": len(metrics),
        "mean_total_ms": float(np.mean(totals)),
        "median_total_ms": float(np.median(totals)),
        "p95_total_ms": float(np.percentile(totals, 95)),
        "p99_total_ms": float(np.percentile(totals, 99)),
        "mean_retrieval_ms": float(np.mean([m.retrieval_ms for m in metrics])),
        "mean_generation_ms": float(np.mean([m.generation_ms for m in metrics])),
        "total_cost_usd": sum(m.estimated_cost_usd for m in metrics),
    }

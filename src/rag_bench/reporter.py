"""Result reporting — JSON + Markdown."""

from __future__ import annotations

import json
from pathlib import Path


def save_results(results: dict, output_dir: str | Path) -> None:
    """Save experiment results as JSON and Markdown.

    Args:
        results: Dict with 'config', 'metrics', 'latency', and optionally 'per_query'.
        output_dir: Directory to save files to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Full per-query results (if present)
    if "per_query" in results:
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results["per_query"], f, indent=2, ensure_ascii=False)

    # Markdown report
    cfg = results.get("config", {})
    metrics = results.get("metrics", {})
    latency = results.get("latency", {})

    md = [
        "# RAG Benchmark Report",
        "",
        f"**Dataset:** {cfg.get('dataset', 'N/A')}",
        f"**Embedding:** {cfg.get('embed_model', 'N/A')}",
        f"**LLM:** {cfg.get('llm_model', 'N/A')}",
        f"**Samples:** {cfg.get('max_samples', 'N/A')}",
        "",
        "## Answer Quality",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        md.append(f"| {k.upper()} | {v:.4f} |")

    md.extend(
        [
            "",
            "## Latency",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
    )
    for k, v in latency.items():
        unit = "ms" if "ms" in k else ("USD" if "cost" in k else "")
        md.append(f"| {k} | {v:.2f} {unit} |")

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

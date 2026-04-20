"""Generate comprehensive retrieval benchmark report with ALL metrics."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path("outputs/retrieval_benchmark")
RESULTS_DIR = OUTPUT_DIR / "results"

DATASETS = [
    "CSConDa", "UIT-ViQuAD2", "ViMedAQA_v2",
    "ViNewsQA", "ViRHE4QA_v2", "ViRe4MRC_v2", "VlogQA_2",
]
DS_SHORT = {
    "CSConDa": "CSConDa", "UIT-ViQuAD2": "ViQuAD2",
    "ViMedAQA_v2": "ViMedAQA", "ViNewsQA": "ViNewsQA",
    "ViRHE4QA_v2": "ViRHE4QA", "ViRe4MRC_v2": "ViRe4MRC",
    "VlogQA_2": "VlogQA",
}

# Metric groups (aligned with index-based evaluation)
METRIC_GROUPS = {
    "Hit & Recall": [
        ("hit_rate",     "Hit@10"),
        ("recall_at_1",  "R@1"),
        ("recall_at_3",  "R@3"),
        ("recall_at_5",  "R@5"),
        ("recall_at_10", "R@10"),
    ],
    "Ranking Quality": [
        ("mrr",        "MRR"),
        ("ndcg_at_1",  "NDCG@1"),
        ("ndcg_at_3",  "NDCG@3"),
        ("ndcg_at_5",  "NDCG@5"),
        ("ndcg_at_10", "NDCG@10"),
    ],
    "Precision & MAP": [
        ("precision",  "P@10"),
        ("map_at_1",   "MAP@1"),
        ("map_at_3",   "MAP@3"),
        ("map_at_5",   "MAP@5"),
        ("map_at_10",  "MAP@10"),
    ],
}

ALL_METRICS = []
for metrics in METRIC_GROUPS.values():
    ALL_METRICS.extend(metrics)

# Strategy taxonomy
STRATEGY_TYPE = {
    "bm25_word": "Sparse",
    "tfidf_word": "Sparse",
    "dense": "Dense",
    "hybrid_rrf": "Hybrid",
    "hybrid_weighted": "Hybrid",
}


EMBED_SHORT = {
    "multilingual-e5-large": "e5-large",
    "vi-bi-encoder": "bi-encoder",
}


def _embed_display(r: dict) -> str:
    """Short embed model name from result dict."""
    raw = r.get("embed_model", "?")
    return EMBED_SHORT.get(raw, raw)


def load_results() -> dict[str, dict]:
    """Load all result JSONs, keyed by filename stem."""
    results = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        key = f.stem
        results[key] = data
    return results


def fmt(val: float, d: int = 2) -> str:
    return f"{val * 100:.{d}f}"


def generate_report(results: dict[str, dict]) -> str:
    lines: list[str] = []
    w = lines.append

    # -- Detect fixed params from first result --
    first = next(iter(results.values()))
    chunking = first.get("chunking", {})
    n_chunks = first.get("n_chunks", "?")

    # -- HEADER --
    w("# Retrieval Benchmark — Full Metrics Report")
    w("")
    w(f"> **Chunking:** {chunking.get('strategy', '?')} "
      f"(size={chunking.get('size', '?')}, overlap={chunking.get('overlap', '?')})")
    w(f"> **Chunks:** {n_chunks:,} | **Top-K:** 10 | **Eval:** 500/dataset | **Seed:** 42")
    w(f"> **Strategies:** {len(results)} | **Datasets:** {len(DATASETS)} | **Metrics:** {len(ALL_METRICS)}")
    w("> **Matching:** index-based (`context_id` hash) | **R:** pre-computed from corpus")
    w("")
    w("---")
    w("")

    # -- Metric Reference --
    w("## Metric Reference")
    w("")
    w("| Group | Metrics | Description |")
    w("|---|---|---|")
    w("| **Hit & Recall** | Hit@10, R@1..10 | Hit: binary. Recall: fraction of ALL relevant chunks found |")
    w("| **Ranking** | MRR, NDCG@1..10 | Ranking quality — penalizes relevant docs at lower positions |")
    w("| **Precision & MAP** | P@10, MAP@1..10 | Precision: fraction of top-K that is relevant. MAP: avg precision |")
    w("")
    w("> **Note:** R = total relevant chunks in *entire corpus* (not just top-K). "
      "Aligned with ViRE (EACL 2026).")
    w("")
    w("---")
    w("")

    # ================================================================
    # GLOBAL LEADERBOARD (grouped by type)
    # ================================================================
    w("## 1. Global Leaderboard — All Metrics")
    w("")
    w("Grouped by retrieval type. Sorted by NDCG@5 within each group. All values x100.")
    w("")

    leaderboard_cols = [
        ("mrr", "MRR"),
        ("precision", "P@10"),
        ("hit_rate", "Hit@10"),
    ]
    recall_cols = [
        ("recall_at_1", "R@1"),
        ("recall_at_3", "R@3"),
        ("recall_at_5", "R@5"),
        ("recall_at_10", "R@10"),
    ]
    ndcg_cols = [
        ("ndcg_at_1", "N@1"),
        ("ndcg_at_3", "N@3"),
        ("ndcg_at_5", "N@5"),
        ("ndcg_at_10", "N@10"),
    ]
    map_cols = [
        ("map_at_5", "MAP@5"),
    ]
    all_lb_cols = leaderboard_cols + recall_cols + ndcg_cols + map_cols

    # Headers
    col_headers = [short for _, short in all_lb_cols]
    w("| Type | Strategy | Embed | " + " | ".join(col_headers) + " |")
    w("|---|---|---|" + "|".join([":---:"] * len(all_lb_cols)) + "|")

    # Global best per metric
    all_results_list = list(results.values())
    metric_maxs = {}
    for key, _ in all_lb_cols:
        metric_maxs[key] = max(r["overall"].get(key, 0) for r in all_results_list)

    # Group by type
    type_order = ["Sparse", "Dense", "Hybrid"]
    for typ in type_order:
        type_results = [
            (k, r) for k, r in results.items()
            if STRATEGY_TYPE.get(r["retrieval"]["strategy"], "?") == typ
        ]
        type_results.sort(key=lambda x: x[1]["overall"].get("ndcg_at_5", 0), reverse=True)

        for i, (file_key, r) in enumerate(type_results):
            ret = r["retrieval"]
            m = r["overall"]

            type_cell = typ if i == 0 else ""

            cells = []
            for key, _ in all_lb_cols:
                val = m.get(key, 0)
                is_best = (val == metric_maxs[key])
                cells.append(f"**{fmt(val)}**" if is_best else fmt(val))

            embed = _embed_display(r)
            w(f"| {type_cell} | {ret['label']} | {embed} | " + " | ".join(cells) + " |")
    w("")

    # ================================================================
    # PER-DATASET BREAKDOWN
    # ================================================================
    w("---")
    w("")
    w("## 2. Per-Dataset Breakdown")
    w("")

    section_idx = 1
    for group_name, group_metrics in METRIC_GROUPS.items():
        w(f"### 2.{section_idx} {group_name}")
        w("")
        section_idx += 1

        for metric_key, metric_short in group_metrics:
            w(f"#### {metric_short}")
            w("")
            ds_headers = [DS_SHORT[d] for d in DATASETS]
            w("| Strategy | " + " | ".join(ds_headers) + " | **Avg** |")
            w("|---|" + "|".join([":---:"] * len(DATASETS)) + "|:---:|")

            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1]["overall"].get("ndcg_at_5", 0),
                reverse=True,
            )

            for file_key, r in sorted_results:
                label = f"{r['retrieval']['label']} ({_embed_display(r)})"
                ds_vals = [r["per_dataset"].get(d, {}).get(metric_key, 0) for d in DATASETS]

                cells = []
                for i_ds, d in enumerate(DATASETS):
                    val = ds_vals[i_ds]
                    col_vals = [rr["per_dataset"].get(d, {}).get(metric_key, 0)
                                for rr in all_results_list]
                    is_best = (val == max(col_vals))
                    cells.append(f"**{fmt(val)}**" if is_best else fmt(val))

                avg = sum(ds_vals) / len(ds_vals)
                avg_vals = [
                    sum(rr["per_dataset"].get(d, {}).get(metric_key, 0) for d in DATASETS) / len(DATASETS)
                    for rr in all_results_list
                ]
                is_best_avg = (avg == max(avg_vals))
                avg_cell = f"**{fmt(avg)}**" if is_best_avg else fmt(avg)

                w(f"| {label} | " + " | ".join(cells) + f" | {avg_cell} |")
            w("")

    # ================================================================
    # BEST STRATEGY PER DATASET
    # ================================================================
    w("---")
    w("")
    w("## 3. Best Strategy per Dataset")
    w("")
    w("| Dataset | Best (Hit@10) | Score | Best (NDCG@5) | Score | Best (MAP@5) | Score |")
    w("|---|---|:---:|---|:---:|---|:---:|")

    for d in DATASETS:
        ds_short = DS_SHORT[d]
        best_hit = max(all_results_list,
                       key=lambda r: r["per_dataset"].get(d, {}).get("hit_rate", 0))
        best_ndcg = max(all_results_list,
                        key=lambda r: r["per_dataset"].get(d, {}).get("ndcg_at_5", 0))
        best_map = max(all_results_list,
                       key=lambda r: r["per_dataset"].get(d, {}).get("map_at_5", 0))

        w(f"| {ds_short} "
          f"| {best_hit['retrieval']['label']} "
          f"| {fmt(best_hit['per_dataset'][d].get('hit_rate', 0))} "
          f"| {best_ndcg['retrieval']['label']} "
          f"| {fmt(best_ndcg['per_dataset'][d].get('ndcg_at_5', 0))} "
          f"| {best_map['retrieval']['label']} "
          f"| {fmt(best_map['per_dataset'][d].get('map_at_5', 0))} |")
    w("")

    # ================================================================
    # APPENDIX
    # ================================================================
    w("---")
    w("")
    w("## Appendix: Complete Raw Data")
    w("")

    for file_key in sorted(results.keys()):
        r = results[file_key]
        ret = r["retrieval"]
        m = r["overall"]
        embed = r.get("embed_model", "?")
        w("<details>")
        w(f"<summary><b>{ret['label']}</b> — {ret['strategy']} | "
          f"embed={embed} | {r.get('elapsed_s', '?')}s</summary>")
        w("")

        ds_headers = [DS_SHORT[d] for d in DATASETS]
        w("| Metric | " + " | ".join(ds_headers) + " | **Overall** |")
        w("|---|" + "|".join([":---:"] * len(DATASETS)) + "|:---:|")

        for metric_key, metric_short in ALL_METRICS:
            overall_val = fmt(m.get(metric_key, 0))
            ds_cells = [fmt(r["per_dataset"].get(d, {}).get(metric_key, 0))
                        for d in DATASETS]
            w(f"| {metric_short} | " + " | ".join(ds_cells) + f" | **{overall_val}** |")
        w("")
        w("</details>")
        w("")

    return "\n".join(lines)


def main():
    print("Loading results...")
    results = load_results()
    print(f"  Found {len(results)} strategies: {', '.join(results.keys())}")

    # Rebuild summary.json
    summary_list = list(results.values())
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2, ensure_ascii=False)
    print(f"  Updated: {summary_path}")

    report = generate_report(results)
    report_path = OUTPUT_DIR / "retrieval_full_metrics_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report: {report_path} ({len(report)} chars)")
    print("Done!")


if __name__ == "__main__":
    main()

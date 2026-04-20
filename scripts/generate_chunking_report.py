"""Generate comprehensive chunking benchmark report with ALL metrics."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path("outputs/chunking_benchmark")
RESULTS_DIR = OUTPUT_DIR / "results"

EXPERIMENTS = {
    1: {
        "name": "Chunk Size Curve (Recursive, overlap=50)",
        "configs": ["C4-256-50", "C4-512-50", "C4-1024-50"],
    },
    2: {
        "name": "Overlap Curve (Recursive, size=512)",
        "configs": ["C4-512-0", "C4-512-25", "C4-512-50", "C4-512-100", "C4-512-200"],
    },
    3: {
        "name": "Method × Domain (5 strategies)",
        "configs": ["C1-512", "C2-sentence", "C3-paragraph", "C4-512", "C5-semantic"],
    },
}

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

# ALL metrics in logical groups (aligned with index-based evaluation)
METRIC_GROUPS = {
    "Hit & Recall (tim duoc bao nhieu?)": [
        ("hit_rate",     "Hit@10"),
        ("recall_at_1",  "R@1"),
        ("recall_at_3",  "R@3"),
        ("recall_at_5",  "R@5"),
        ("recall_at_10", "R@10"),
    ],
    "Ranking Quality (dung o vi tri nao?)": [
        ("mrr",        "MRR"),
        ("ndcg_at_1",  "NDCG@1"),
        ("ndcg_at_3",  "NDCG@3"),
        ("ndcg_at_5",  "NDCG@5"),
        ("ndcg_at_10", "NDCG@10"),
    ],
    "Precision & MAP (chat luong top-K?)": [
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


def load_results() -> dict[str, dict]:
    results = {}
    for f in sorted(RESULTS_DIR.glob("unified_*.json")):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        label = data["chunking"]["label"]
        results[label] = data
    return results


def fmt(val: float, d: int = 2) -> str:
    return f"{val * 100:.{d}f}"  # val * 100


def generate_report(results: dict[str, dict]) -> str:
    lines: list[str] = []
    w = lines.append

    # -- HEADER --
    w("# Chunking Benchmark — Full Metrics Report")
    w("")
    w("> **Embedding:** `multilingual-e5-large` (1024d) | **Top-K:** 10 | **Eval:** 500/dataset | **Seed:** 42  ")
    w(f"> **Configs:** {len(results)} | **Datasets:** {len(DATASETS)} | **Metrics:** {len(ALL_METRICS)}")
    w("> **Matching:** index-based (`context_id` hash) | **R:** pre-computed from corpus")
    w("")
    w("---")
    w("")

    # -- NOTE about metric relationships --
    w("## Metric Reference")
    w("")
    w("| Group | Metrics | Description |")
    w("|---|---|---|")
    w("| **Hit & Recall** | Hit@10, R@1..10 | Hit: binary (any relevant in top-K?). Recall: fraction of ALL relevant chunks found |")
    w("| **Ranking** | MRR, NDCG@1..10 | Ranking quality — penalizes relevant docs at lower positions |")
    w("| **Precision & MAP** | P@10, MAP@1..10 | Precision: fraction of top-K that is relevant. MAP: avg precision at each relevant rank |")
    w("")
    w("> **Note:** R = total relevant chunks in *entire corpus* (not just top-K). "
      "Recall@K = hits_in_topK / R. NDCG uses R for IDCG. Aligned with ViRE (EACL 2026).")
    w("")
    w("---")
    w("")

    # ================================================================
    # GLOBAL LEADERBOARD
    # ================================================================
    w("## 1. Global Leaderboard — All Metrics")
    w("")
    w("Grouped by strategy, sorted by NDCG@5 within each group. All values x100 (macro-avg across 7 datasets).")
    w("")

    # Column order: Strategy | Config | Chunks | MRR | P@10 | Hit@10 | R@1..R@10 | NDCG@1..NDCG@10 | MAP@5
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

    # Header
    col_headers = [short for _, short in all_lb_cols]
    w("| Strategy | Config | Chunks | " + " | ".join(col_headers) + " |")
    w("|---|---|---:|" + "|".join([":---:"] * len(all_lb_cols)) + "|")

    # Group configs by strategy
    strategy_order = ["recursive", "fixed", "paragraph", "sentence", "semantic"]
    strategy_display = {
        "recursive": "Recursive",
        "fixed": "Fixed",
        "paragraph": "Paragraph",
        "sentence": "Sentence",
        "semantic": "Semantic",
    }

    # Get all values per metric to find global best
    all_results_list = list(results.values())
    metric_vals = {}
    for key, _ in all_lb_cols:
        metric_vals[key] = [r["overall"].get(key, 0) for r in all_results_list]

    for strat in strategy_order:
        strat_results = [r for r in all_results_list if r["chunking"]["strategy"] == strat]
        strat_results.sort(key=lambda r: r["overall"].get("ndcg_at_5", 0), reverse=True)

        for i, r in enumerate(strat_results):
            c = r["chunking"]
            m = r["overall"]

            # Strategy column: show name only on first row of group
            strat_cell = strategy_display.get(strat, strat) if i == 0 else ""

            cells = []
            for key, _ in all_lb_cols:
                val = m.get(key, 0)
                is_best = (val == max(metric_vals[key]))
                cells.append(f"**{fmt(val)}**" if is_best else fmt(val))

            w(f"| {strat_cell} | {c['label']} | {r['n_chunks']:,} | " + " | ".join(cells) + " |")
    w("")

    # ================================================================
    # PER EXPERIMENT — FULL TABLES
    # ================================================================
    for exp_id, exp in EXPERIMENTS.items():
        exp_results = [results[c] for c in exp["configs"] if c in results]
        if not exp_results:
            continue
        w("---")
        w("")
        w(f"## {exp_id + 1}. Experiment {exp_id} — {exp['name']}")
        w("")

        # -- Overall table with ALL metrics --
        w(f"### {exp_id + 1}.1 Overall (macro-avg across 7 datasets)")
        w("")

        w("| Config | Chunks | " + " | ".join(short for _, short in ALL_METRICS) + " |")
        w("|---|---:|" + "|".join([":---:"] * len(ALL_METRICS)) + "|")

        exp_vals = {}
        for key, _ in ALL_METRICS:
            exp_vals[key] = [r["overall"].get(key, 0) for r in exp_results]

        for r in exp_results:
            m = r["overall"]
            cells = []
            for key, _ in ALL_METRICS:
                val = m.get(key, 0)
                is_best = (val == max(exp_vals[key]))
                cells.append(f"**{fmt(val)}**" if is_best else fmt(val))
            w(f"| `{r['chunking']['label']}` | {r['n_chunks']:,} | " + " | ".join(cells) + " |")
        w("")

        # -- Per-dataset breakdown for EACH metric group --
        section_idx = 2
        for group_name, group_metrics in METRIC_GROUPS.items():
            w(f"### {exp_id + 1}.{section_idx} Per-Dataset: {group_name}")
            w("")
            section_idx += 1

            for metric_key, metric_short in group_metrics:
                w(f"#### {metric_short}")
                w("")
                ds_headers = [DS_SHORT[d] for d in DATASETS]
                w("| Config | " + " | ".join(ds_headers) + " | **Avg** |")
                w("|---|" + "|".join([":---:"] * len(DATASETS)) + "|:---:|")

                for r in exp_results:
                    ds_vals = []
                    for d in DATASETS:
                        val = r["per_dataset"].get(d, {}).get(metric_key, 0)
                        ds_vals.append(val)

                    cells = []
                    for i_ds, d in enumerate(DATASETS):
                        val = ds_vals[i_ds]
                        col_vals = [er["per_dataset"].get(d, {}).get(metric_key, 0) for er in exp_results]
                        is_best = (val == max(col_vals))
                        cells.append(f"**{fmt(val)}**" if is_best else fmt(val))

                    avg = sum(ds_vals) / len(ds_vals)
                    avg_vals = [
                        sum(er["per_dataset"].get(d, {}).get(metric_key, 0) for d in DATASETS) / len(DATASETS)
                        for er in exp_results
                    ]
                    is_best_avg = (avg == max(avg_vals))
                    avg_cell = f"**{fmt(avg)}**" if is_best_avg else fmt(avg)

                    w(f"| `{r['chunking']['label']}` | " + " | ".join(cells) + f" | {avg_cell} |")
                w("")

    # ================================================================
    # APPENDIX — FULL RAW DATA PER CONFIG
    # ================================================================
    w("---")
    w("")
    w("## Appendix: Complete Raw Data")
    w("")

    for label in sorted(results.keys()):
        r = results[label]
        c = r["chunking"]
        m = r["overall"]
        w("<details>")
        w(f"<summary><b>{c['label']}</b> — {c['strategy']} | size={c['chunk_size']} "
          f"| overlap={c['chunk_overlap']} | {r['n_chunks']:,} chunks | {r['elapsed_s']}s</summary>")
        w("")

        ds_headers = [DS_SHORT[d] for d in DATASETS]
        w("| Metric | " + " | ".join(ds_headers) + " | **Overall** |")
        w("|---|" + "|".join([":---:"] * len(DATASETS)) + "|:---:|")

        for metric_key, metric_short in ALL_METRICS:
            overall_val = fmt(m.get(metric_key, 0), 2)
            ds_cells = [fmt(r["per_dataset"].get(d, {}).get(metric_key, 0), 2) for d in DATASETS]
            w(f"| {metric_short} | " + " | ".join(ds_cells) + f" | **{overall_val}** |")
        w("")
        w("</details>")
        w("")

    return "\n".join(lines)


def main():
    print("Loading results...")
    results = load_results()
    print(f"  Found {len(results)} configs")

    report = generate_report(results)
    report_path = OUTPUT_DIR / "chunking_full_metrics_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report: {report_path} ({len(report)} chars)")
    print("Done!")


if __name__ == "__main__":
    main()

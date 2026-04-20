"""Generate comprehensive generation benchmark report."""

from __future__ import annotations

import json
from pathlib import Path

MODEL = "llama3.3-70b-instruct"
OUTPUT_DIR = Path(f"outputs/generation_benchmark_llama70b")

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

# Metric groups
GEN_METRICS = [
    ("exact_match", "EM"),
    ("f1", "F1"),
    ("rouge_l", "ROUGE-L"),
]

RET_METRICS_BINARY = [
    ("recall_at_1",  "R@1"),
    ("recall_at_3",  "R@3"),
    ("recall_at_5",  "R@5"),
    ("recall_at_10", "R@10"),
]

RET_METRICS_RANKING = [
    ("mrr",        "MRR"),
    ("ndcg_at_1",  "NDCG@1"),
    ("ndcg_at_3",  "NDCG@3"),
    ("ndcg_at_5",  "NDCG@5"),
    ("ndcg_at_10", "NDCG@10"),
]

RET_METRICS_COVERAGE = [
    ("context_precision", "C.Prec"),
    ("context_recall",    "C.Recall"),
]


def fmt(val: float | None, d: int = 2) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.{d}f}"


def load_evaluations() -> dict[str, dict]:
    """Load evaluations.json for each dataset."""
    results = {}
    for ds in DATASETS:
        eval_path = list(OUTPUT_DIR.glob(f"{ds}/**/evaluations.json"))
        if eval_path:
            with open(eval_path[0], encoding="utf-8") as f:
                results[ds] = json.load(f)
    return results


def generate_report(results: dict[str, dict]) -> str:
    lines: list[str] = []
    w = lines.append

    # Get config from first result
    first = next(iter(results.values()))
    cfg = first.get("config", {})

    # ── HEADER ──
    w("# 📊 Generation Benchmark — Full Metrics Report")
    w("")
    w(f"> **Pipeline:** {cfg.get('chunk_strategy', '?')} chunking → "
      f"{cfg.get('search_type', '?')} retrieval → {cfg.get('llm_model', '?')} generation")
    w(f"> **Embedding:** `{cfg.get('embed_model', '?')}` | "
      f"**Top-K:** {cfg.get('top_k', '?')} | "
      f"**Samples:** {cfg.get('max_samples', '?')}/dataset | "
      f"**Datasets:** {len(results)}")
    w(f"> **Index:** unified (shared) | **Generation:** {cfg.get('generation_strategy', 'standard')}")
    w("")
    w("---")
    w("")

    # ══════════════════════════════════════════════════════════════
    # 1. GENERATION QUALITY — Main story
    # ══════════════════════════════════════════════════════════════
    w("## 1. Generation Quality (LLM Output)")
    w("")
    w("How well does the LLM answer match the gold answer?")
    w("")

    ds_headers = [DS_SHORT[d] for d in DATASETS if d in results]
    active_ds = [d for d in DATASETS if d in results]

    w("| Metric | " + " | ".join(ds_headers) + " | **Avg** |")
    w("|---|" + "|".join([":---:"] * len(active_ds)) + "|:---:|")

    for key, short in GEN_METRICS:
        vals = [results[d]["generation_metrics"].get(key) for d in active_ds]
        cells = []
        valid_vals = [v for v in vals if v is not None]
        max_val = max(valid_vals) if valid_vals else None
        for v in vals:
            if v is not None and max_val is not None and v == max_val:
                cells.append(f"**{fmt(v)}**")
            else:
                cells.append(fmt(v))
        avg = sum(v for v in vals if v is not None) / len(valid_vals) if valid_vals else None
        avg_cell = fmt(avg) if avg is not None else "—"
        w(f"| {short} | " + " | ".join(cells) + f" | {avg_cell} |")
    w("")

    # ══════════════════════════════════════════════════════════════
    # 2. RETRIEVAL QUALITY — Supporting metrics
    # ══════════════════════════════════════════════════════════════
    w("---")
    w("")
    w("## 2. Retrieval Quality (Context fed to LLM)")
    w("")

    # Combined table per group
    for group_name, group_metrics in [
        ("Binary Retrieval (có tìm thấy không?)", RET_METRICS_BINARY),
        ("Ranking Quality (đúng ở vị trí nào?)", RET_METRICS_RANKING),
        ("Coverage (nội dung cover bao nhiêu?)", RET_METRICS_COVERAGE),
    ]:
        w(f"### 2.x {group_name}")
        w("")
        w("| Metric | " + " | ".join(ds_headers) + " | **Avg** |")
        w("|---|" + "|".join([":---:"] * len(active_ds)) + "|:---:|")

        for key, short in group_metrics:
            vals = [results[d]["retrieval_metrics"].get(key, 0) for d in active_ds]
            max_val = max(vals)
            cells = []
            for v in vals:
                cells.append(f"**{fmt(v)}**" if v == max_val else fmt(v))
            avg = sum(vals) / len(vals)
            w(f"| {short} | " + " | ".join(cells) + f" | {fmt(avg)} |")
        w("")

    # ══════════════════════════════════════════════════════════════
    # 3. COST & LATENCY
    # ══════════════════════════════════════════════════════════════
    w("---")
    w("")
    w("## 3. Token Usage & Latency")
    w("")
    w("| Dataset | Input Tokens | Output Tokens | Total Tokens | Avg Latency (ms) |")
    w("|---|---:|---:|---:|---:|")

    total_in = 0
    total_out = 0
    for d in active_ds:
        lat = results[d].get("latency", {})
        in_tok = lat.get("total_input_tokens", 0)
        out_tok = lat.get("total_output_tokens", 0)
        avg_ms = lat.get("mean_total_ms", 0)
        total_in += in_tok
        total_out += out_tok
        w(f"| {DS_SHORT[d]} | {in_tok:,} | {out_tok:,} | {in_tok + out_tok:,} | {avg_ms:.0f} |")

    w(f"| **Total** | **{total_in:,}** | **{total_out:,}** | **{total_in + total_out:,}** | — |")
    w("")

    # Cost estimate (gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output)
    cost_in = total_in * 0.15 / 1_000_000
    cost_out = total_out * 0.60 / 1_000_000
    w(f"> **Estimated cost** (gpt-4o-mini): "
      f"${cost_in:.4f} (input) + ${cost_out:.4f} (output) = **${cost_in + cost_out:.4f}** total")
    w("")

    # ══════════════════════════════════════════════════════════════
    # 4. CROSS-VIEW: Gen vs Ret correlation
    # ══════════════════════════════════════════════════════════════
    w("---")
    w("")
    w("## 4. Generation vs Retrieval — Per Dataset")
    w("")
    w("| Dataset | F1 | ROUGE-L | NDCG@5 | R@10 | C.Recall | C.Prec |")
    w("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for d in active_ds:
        gm = results[d]["generation_metrics"]
        rm = results[d]["retrieval_metrics"]
        w(f"| {DS_SHORT[d]} "
          f"| {fmt(gm.get('f1'))} "
          f"| {fmt(gm.get('rouge_l'))} "
          f"| {fmt(rm.get('ndcg_at_5', 0))} "
          f"| {fmt(rm.get('recall_at_10', 0))} "
          f"| {fmt(rm.get('context_recall', 0))} "
          f"| {fmt(rm.get('context_precision', 0))} |")
    w("")

    # Averages row
    def _avg(section: str, key: str) -> float | None:
        vals = [results[d][section].get(key) for d in active_ds]
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else None

    w(f"| **Average** "
      f"| **{fmt(_avg('generation_metrics', 'f1'))}** "
      f"| **{fmt(_avg('generation_metrics', 'rouge_l'))}** "
      f"| **{fmt(_avg('retrieval_metrics', 'ndcg_at_5'))}** "
      f"| **{fmt(_avg('retrieval_metrics', 'recall_at_10'))}** "
      f"| **{fmt(_avg('retrieval_metrics', 'context_recall'))}** "
      f"| **{fmt(_avg('retrieval_metrics', 'context_precision'))}** |")
    w("")

    # ══════════════════════════════════════════════════════════════
    # APPENDIX
    # ══════════════════════════════════════════════════════════════
    w("---")
    w("")
    w("## Appendix: Raw Data per Dataset")
    w("")

    for d in active_ds:
        gm = results[d]["generation_metrics"]
        rm = results[d]["retrieval_metrics"]
        lat = results[d].get("latency", {})
        w(f"<details>")
        w(f"<summary><b>{DS_SHORT[d]}</b> — "
          f"F1={fmt(gm.get('f1'))} | NDCG@5={fmt(rm.get('ndcg_at_5', 0))} | "
          f"{lat.get('total_input_tokens', 0) + lat.get('total_output_tokens', 0):,} tokens"
          f"</summary>")
        w("")

        w("| Category | Metric | Value |")
        w("|---|---|:---:|")
        for key, short in GEN_METRICS:
            w(f"| Generation | {short} | {fmt(gm.get(key))} |")
        for key, short in RET_METRICS_BINARY + RET_METRICS_RANKING + RET_METRICS_COVERAGE:
            w(f"| Retrieval | {short} | {fmt(rm.get(key, 0))} |")
        w(f"| Latency | Avg total (ms) | {lat.get('mean_total_ms', 0):.0f} |")
        w(f"| Tokens | Input | {lat.get('total_input_tokens', 0):,} |")
        w(f"| Tokens | Output | {lat.get('total_output_tokens', 0):,} |")
        w("")
        w("</details>")
        w("")

    return "\n".join(lines)


def main():
    print("Loading evaluations...")
    results = load_evaluations()
    print(f"  Found {len(results)} datasets: {', '.join(DS_SHORT[d] for d in results)}")

    report = generate_report(results)
    report_path = OUTPUT_DIR / "generation_full_metrics_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report: {report_path} ({len(report)} chars)")
    print("Done!")


if __name__ == "__main__":
    main()

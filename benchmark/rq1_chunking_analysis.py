"""RQ1 Chunking Analysis — Chunk statistics + retrieval evaluation.

Produces all tables for RQ1a-d:
  RQ1a: Chunk statistics (no retrieval needed)
  RQ1b: Retrieval performance (macro-averaged)
  RQ1c: Per-domain NDCG@5 breakdown
  RQ1d: Overlap sensitivity ablation

Usage:
    python benchmark/rq1_chunking_analysis.py --stats-only          # RQ1a only
    python benchmark/rq1_chunking_analysis.py --experiment 1        # RQ1a+b+c
    python benchmark/rq1_chunking_analysis.py --experiment 2        # RQ1d
    python benchmark/rq1_chunking_analysis.py --experiment all      # Everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.documents import Document
from rag_bench.chunker import get_chunker
from rag_bench.config import RagConfig
from rag_bench.embeddings.registry import get_embed_model
from rag_bench.evaluator import evaluate_retrieval
from rag_bench.indexer import build_vectorstore
from rag_bench.retrievers.dense import DenseRetriever

# ── Defaults ──────────────────────────────────────────────────────────────
SEED = 42
BENCHMARK_CSV = "data/processed/benchmark.csv"
EMBED_MODEL_KEY = "multilingual-e5-large"
TOP_K = 10
MAX_EVAL_SAMPLES = 500

METRIC_KEYS = [
    "hit_rate", "mrr", "precision",
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
    "map_at_1", "map_at_3", "map_at_5", "map_at_10",
]

# ── Experiment Configs ────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    label: str
    strategy: str
    chunk_size: int = 512
    chunk_overlap: int = 128
    embed_model: str = "intfloat/multilingual-e5-large"


EXPERIMENTS = {
    1: ("RQ1a+b+c: Strategy Comparison", [
        ChunkingConfig("Token-512", "token", 512, 128),
        ChunkingConfig("Sentence", "sentence", 512, 128),
        ChunkingConfig("Paragraph", "paragraph", 512, 128),
        ChunkingConfig("Recursive-512", "recursive", 512, 128),
        ChunkingConfig("Semantic", "semantic", 512, 128),
        ChunkingConfig("Neural", "neural", 512, 128), 
    ]),
    2: ("RQ1d: Overlap Sensitivity (Token-512)", [
        ChunkingConfig("Token-512-ov0", "token", 512, 0),
        ChunkingConfig("Token-512-ov128", "token", 512, 128),
        ChunkingConfig("Token-512-ov256", "token", 512, 256),
    ]),
}


# ── Data Loading ──────────────────────────────────────────────────────────

def load_benchmark_data(
    csv_path: str, max_eval: int,
) -> tuple[list[Document], dict[str, list[dict]], int]:
    """Load CSV -> docs for indexing + QA pairs for eval (sampled, seed=42).

    Returns:
        (all_docs, qa_by_dataset, n_unique_contexts)
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["answer"] = df["answer"].fillna("")
    df["context_id"] = df["context"].apply(
        lambda c: hashlib.md5(str(c).encode("utf-8")).hexdigest()[:12]
    )

    unique_ctx = df.drop_duplicates(subset=["context"])
    n_contexts = len(unique_ctx)
    all_docs = [
        Document(page_content=str(r["context"]),
                 metadata={"dataset": str(r["dataset"]),
                           "context_id": str(r["context_id"])})
        for _, r in unique_ctx.iterrows()
    ]

    rng = random.Random(SEED)
    qa_by_dataset: dict[str, list[dict]] = {}
    for ds, group in df.groupby("dataset"):
        pairs = [
            {"context_id": str(r["context_id"]), **{k: str(r[k])
             for k in ["question", "answer", "context", "dataset"]}}
            for _, r in group.iterrows()
        ]
        if len(pairs) > max_eval:
            pairs = rng.sample(pairs, max_eval)
        qa_by_dataset[ds] = pairs

    return all_docs, qa_by_dataset, n_contexts


# ── Chunk Statistics (RQ1a) ───────────────────────────────────────────────

def compute_chunk_stats(
    chunks: list[Document], n_contexts: int,
) -> dict:
    """Compute chunk statistics for RQ1a table.

    Args:
        chunks: list of chunked Documents (with metadata["context_id"]).
        n_contexts: number of unique contexts in original corpus.

    Returns:
        Dict with total, expansion, avg/min/max chunks per context,
        avg/median tokens per chunk (whitespace split).
    """
    ctx_chunks = Counter(
        doc.metadata.get("context_id", "unknown") for doc in chunks
    )
    chunks_per_ctx = list(ctx_chunks.values())

    # Word-level token count (whitespace split)
    tok_per_chunk = [len(doc.page_content.split()) for doc in chunks]
    char_per_chunk = [len(doc.page_content) for doc in chunks]

    return {
        "total_chunks": len(chunks),
        "n_contexts": n_contexts,
        "expansion": round(len(chunks) / max(n_contexts, 1), 2),
        "avg_chunks_per_ctx": round(float(np.mean(chunks_per_ctx)), 2),
        "min_chunks_per_ctx": int(min(chunks_per_ctx)),
        "max_chunks_per_ctx": int(max(chunks_per_ctx)),
        "std_chunks_per_ctx": round(float(np.std(chunks_per_ctx)), 2),
        "avg_words_per_chunk": round(float(np.mean(tok_per_chunk)), 1),
        "median_words_per_chunk": round(float(np.median(tok_per_chunk)), 1),
        "avg_chars_per_chunk": round(float(np.mean(char_per_chunk)), 1),
        "median_chars_per_chunk": round(float(np.median(char_per_chunk)), 1),
    }


# ── Core Benchmark ───────────────────────────────────────────────────────

def run_chunking_only(
    config: ChunkingConfig,
    all_docs: list[Document],
    n_contexts: int,
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Chunk only — no retrieval. For RQ1a stats."""
    result_file = output_dir / "results" / f"stats_{config.label}.json"
    if result_file.exists() and not force:
        print(f"  [SKIP] {config.label} (cached)")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    t0 = time.perf_counter()
    chunker = get_chunker(config.strategy, chunk_size=config.chunk_size,
                          chunk_overlap=config.chunk_overlap,
                          embed_model=config.embed_model)
    chunks = chunker.chunk(tqdm(all_docs, desc=f"Chunking [{config.label}]"))
    elapsed = round(time.perf_counter() - t0, 1)

    stats = compute_chunk_stats(chunks, n_contexts)

    result = {
        "chunking": asdict(config),
        "n_docs": len(all_docs),
        "n_chunks": len(chunks),
        "chunk_stats": stats,
        "elapsed_s": elapsed,
    }

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  {config.label}: {len(all_docs)} -> {len(chunks)} chunks "
          f"(exp={stats['expansion']}x, avg={stats['avg_words_per_chunk']} w/chunk, "
          f"{elapsed}s)")
    return result


def run_full_config(
    config: ChunkingConfig,
    all_docs: list[Document],
    qa_by_dataset: dict[str, list[dict]],
    n_contexts: int,
    output_dir: Path,
    embed_model,
    force: bool = False,
) -> dict:
    """Chunk + Index + Retrieve + Evaluate. For RQ1b/c/d."""
    result_file = output_dir / "results" / f"full_{config.label}.json"
    if result_file.exists() and not force:
        print(f"  [SKIP] {config.label} (cached, use --force to re-run)")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    t0 = time.perf_counter()

    # Chunk
    chunker = get_chunker(config.strategy, chunk_size=config.chunk_size,
                          chunk_overlap=config.chunk_overlap,
                          embed_model=config.embed_model)
    chunks = chunker.chunk(tqdm(all_docs, desc=f"Chunking [{config.label}]"))
    print(f"  {config.label}: {len(all_docs)} docs -> {len(chunks)} chunks")

    # Chunk stats
    stats = compute_chunk_stats(chunks, n_contexts)

    # Index
    rag_cfg = RagConfig(csv_path="unified", embed_model=EMBED_MODEL_KEY,
                        chroma_dir=str(output_dir / "chroma"),
                        force_reindex=force)
    vs = build_vectorstore(chunks, embed_model, rag_cfg,
                           dataset_name=f"rq1_{config.label}",
                           model_key=config.label)
    retriever = DenseRetriever(vs, top_k=TOP_K)

    # Pre-compute total relevant chunks per context_id
    ctx_counts: dict[str, int] = Counter(
        doc.metadata.get("context_id", "") for doc in chunks
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict)
    )

    # Evaluate per dataset
    per_dataset = {}
    for ds_name, qa_pairs in tqdm(qa_by_dataset.items(), desc="Evaluating"):
        questions = [qa["question"] for qa in qa_pairs]
        rr_list = retriever.batch_retrieve(questions)
        scores = [
            evaluate_retrieval(
                rr.documents, gold_context_id=qa["context_id"], k=TOP_K,
                n_relevant=ctx_counts.get(qa["context_id"], 1),
            )
            for qa, rr in zip(qa_pairs, rr_list)
        ]
        per_dataset[ds_name] = {
            k: round(float(np.mean([s[k] for s in scores if s.get(k) is not None])), 4)
            for k in METRIC_KEYS
        }
        m = per_dataset[ds_name]
        print(f"    {ds_name}: Hit={m['hit_rate']:.3f} MRR={m['mrr']:.3f} "
              f"NDCG@5={m['ndcg_at_5']:.3f}")

    # Aggregate
    overall = {k: round(float(np.mean([d[k] for d in per_dataset.values()])), 4)
               for k in METRIC_KEYS}
    elapsed = round(time.perf_counter() - t0, 1)

    result = {
        "chunking": asdict(config),
        "n_docs": len(all_docs),
        "n_chunks": len(chunks),
        "chunk_stats": stats,
        "per_dataset": per_dataset,
        "overall": overall,
        "elapsed_s": elapsed,
    }

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Overall: Hit={overall['hit_rate']:.3f} MRR={overall['mrr']:.3f} "
          f"NDCG@5={overall['ndcg_at_5']:.3f} ({elapsed}s)")
    return result


# ── Report Generation ─────────────────────────────────────────────────────

def generate_rq1a_report(results: list[dict], output_dir: Path) -> None:
    """RQ1a: Chunk Statistics Table."""
    path = output_dir / "rq1a_chunk_statistics.md"
    # Sort by expansion (ascending)
    results = sorted(results, key=lambda r: r["chunk_stats"]["expansion"])

    with open(path, "w", encoding="utf-8") as f:
        n_ctx = results[0]["chunk_stats"]["n_contexts"]
        f.write(f"# RQ1a: Chunk Statistics on Unified Corpus ({n_ctx:,} contexts)\n\n")
        f.write("| Strategy | Total Chunks | Avg/Ctx | Min | Max | Expansion | "
                "Avg Words/Chunk | Avg Chars/Chunk |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            s = r["chunk_stats"]
            c = r["chunking"]
            f.write(f"| {c['label']} | {s['total_chunks']:,} | "
                    f"{s['avg_chunks_per_ctx']:.2f} | "
                    f"{s['min_chunks_per_ctx']} | {s['max_chunks_per_ctx']} | "
                    f"{s['expansion']:.1f}× | "
                    f"{s['avg_words_per_chunk']:.0f} | "
                    f"{s['avg_chars_per_chunk']:.0f} |\n")

    print(f"  RQ1a report: {path}")


def generate_rq1b_report(results: list[dict], output_dir: Path) -> None:
    """RQ1b: Retrieval Performance (macro-averaged) Table."""
    path = output_dir / "rq1b_retrieval_performance.md"
    # Sort by NDCG@5 descending
    results = sorted(results, key=lambda r: r["overall"]["ndcg_at_5"], reverse=True)

    rq1b_metrics = ["mrr", "ndcg_at_5", "recall_at_1", "recall_at_10",
                    "map_at_5", "hit_rate"]
    labels = ["MRR", "NDCG@5", "R@1", "R@10", "MAP@5", "Hit@10"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# RQ1b: Chunking Strategy Comparison\n")
        f.write(f"Dense E5-Large, k={TOP_K}, macro-averaged over "
                f"{len(results[0]['per_dataset'])} datasets. "
                f"Sorted by NDCG@5 (desc). Best in **bold**.\n\n")

        # Find best values
        best = {k: max(r["overall"][k] for r in results) for k in rq1b_metrics}

        f.write("| Strategy | " + " | ".join(labels) + " |\n")
        f.write("|---" + "|---:" * len(labels) + "|\n")
        for r in results:
            row = [r["chunking"]["label"]]
            for mk in rq1b_metrics:
                val = r["overall"][mk] * 100
                cell = f"{val:.2f}"
                if r["overall"][mk] == best[mk]:
                    cell = f"**{cell}**"
                row.append(cell)
            f.write("| " + " | ".join(row) + " |\n")

    print(f"  RQ1b report: {path}")


def generate_rq1c_report(results: list[dict], output_dir: Path) -> None:
    """RQ1c: Per-Domain NDCG@5 Table."""
    path = output_dir / "rq1c_per_domain_ndcg.md"
    # Sort by overall NDCG@5 descending
    results = sorted(results, key=lambda r: r["overall"]["ndcg_at_5"], reverse=True)
    ds_names = sorted(results[0]["per_dataset"].keys())

    with open(path, "w", encoding="utf-8") as f:
        f.write("# RQ1c: Per-Domain NDCG@5\n")
        f.write(f"Dense E5-Large, k={TOP_K}. Best per-domain in **bold**.\n\n")

        # Find best per domain
        best_per_ds = {ds: max(r["per_dataset"][ds]["ndcg_at_5"] for r in results)
                       for ds in ds_names}

        # Short dataset names
        short_names = {d: d.replace("_v2", "").replace("_2", "")
                       for d in ds_names}

        header = "| Strategy | " + " | ".join(short_names[d] for d in ds_names) + " | Avg | Std |\n"
        sep = "|---" + "|---:" * (len(ds_names) + 2) + "|\n"
        f.write(header)
        f.write(sep)

        for r in results:
            vals = [r["per_dataset"][ds]["ndcg_at_5"] for ds in ds_names]
            avg = float(np.mean(vals))
            std = float(np.std(vals))
            row = [r["chunking"]["label"]]
            for ds in ds_names:
                v = r["per_dataset"][ds]["ndcg_at_5"] * 100
                cell = f"{v:.2f}"
                if r["per_dataset"][ds]["ndcg_at_5"] == best_per_ds[ds]:
                    cell = f"**{cell}**"
                row.append(cell)
            row.append(f"{avg * 100:.2f}")
            row.append(f"{std * 100:.2f}")
            f.write("| " + " | ".join(row) + " |\n")

    print(f"  RQ1c report: {path}")


def generate_rq1d_report(results: list[dict], output_dir: Path) -> None:
    """RQ1d: Overlap Sensitivity Table."""
    path = output_dir / "rq1d_overlap_sensitivity.md"
    # Sort by overlap (ascending)
    results = sorted(results, key=lambda r: r["chunking"]["chunk_overlap"])

    base_chunks = results[0]["n_chunks"] if results else 1
    rq1d_metrics = ["mrr", "ndcg_at_5", "recall_at_1", "recall_at_10",
                    "map_at_5"]
    labels = ["MRR", "NDCG@5", "R@1", "R@10", "MAP@5"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# RQ1d: Overlap Sensitivity\n")
        f.write(f"Strategy: {results[0]['chunking']['strategy']}-"
                f"{results[0]['chunking']['chunk_size']}, "
                f"Dense E5-Large, k={TOP_K}. Best in **bold**.\n\n")

        best = {k: max(r["overall"][k] for r in results) for k in rq1d_metrics}

        f.write("| Overlap | Chunks | " + " | ".join(labels) + " | Δ Index |\n")
        f.write("|---|---:" + "|---:" * len(labels) + "|---:|\n")
        for r in results:
            ov = r["chunking"]["chunk_overlap"]
            n = r["n_chunks"]
            delta = round(n / base_chunks, 2)
            row = [f"{ov} chars", f"{n:,}"]
            for mk in rq1d_metrics:
                val = r["overall"][mk] * 100
                cell = f"{val:.2f}"
                if r["overall"][mk] == best[mk]:
                    cell = f"**{cell}**"
                row.append(cell)
            row.append(f"{delta}×")
            f.write("| " + " | ".join(row) + " |\n")

    print(f"  RQ1d report: {path}")


def generate_combined_report(
    results: list[dict], output_dir: Path, tag: str,
) -> None:
    """Generate all applicable reports from results."""
    generate_rq1a_report(results, output_dir)
    if any("overall" in r for r in results):
        generate_rq1b_report(results, output_dir)
        generate_rq1c_report(results, output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="RQ1 Chunking Analysis — stats + retrieval evaluation")
    p.add_argument("--stats-only", action="store_true",
                   help="RQ1a only: compute chunk statistics without retrieval")
    p.add_argument("--experiment", nargs="+",
                   help="Experiment IDs: 1=Strategy Comparison, 2=Overlap, "
                        "all=everything (default: 1)")
    p.add_argument("--csv", default=BENCHMARK_CSV)
    p.add_argument("--datasets", nargs="+", help="Filter to specific datasets")
    p.add_argument("--max-samples", type=int, default=MAX_EVAL_SAMPLES)
    p.add_argument("--output-dir", default="outputs/rq1_analysis")
    p.add_argument("--force", action="store_true",
                   help="Ignore cache, rebuild everything")
    p.add_argument("--list", action="store_true",
                   help="List available experiments")
    args = p.parse_args()

    if args.list:
        print("\n[Experiments]")
        for eid, (name, configs) in EXPERIMENTS.items():
            labels = ", ".join(c.label for c in configs)
            print(f"  {eid}. {name}  [{labels}]")
        print()
        return

    # Load data
    print(f"Loading: {args.csv} (eval={args.max_samples}/dataset, seed={SEED})")
    all_docs, qa_by_dataset, n_contexts = load_benchmark_data(
        args.csv, args.max_samples)

    if args.datasets:
        qa_by_dataset = {k: v for k, v in qa_by_dataset.items()
                         if k in args.datasets}

    total_q = sum(len(v) for v in qa_by_dataset.values())
    print(f"  {len(all_docs)} docs ({n_contexts} unique contexts), "
          f"{len(qa_by_dataset)} datasets, {total_q} queries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stats-only mode (RQ1a) ────────────────────────────────────────
    if args.stats_only:
        configs = EXPERIMENTS[1][1]  # Use experiment 1 configs
        print(f"\n{'='*60}\nRQ1a: Chunk Statistics ({len(configs)} configs)\n{'='*60}")
        results = [run_chunking_only(c, all_docs, n_contexts, output_dir, args.force)
                   for c in configs]
        generate_rq1a_report(results, output_dir)
        return

    # ── Full experiment mode ──────────────────────────────────────────
    if args.experiment and "all" in args.experiment:
        exp_ids = list(EXPERIMENTS.keys())
    elif args.experiment:
        exp_ids = [int(e) for e in args.experiment]
    else:
        exp_ids = [1]

    print(f"\nLoading embedding model: {EMBED_MODEL_KEY}...")
    embed_model = get_embed_model(EMBED_MODEL_KEY)

    all_results = {}
    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            print(f"Unknown experiment: {eid}")
            continue
        name, configs = EXPERIMENTS[eid]
        print(f"\n{'='*60}\nExp {eid}: {name} ({len(configs)} configs)\n{'='*60}")

        results = []
        for c in configs:
            r = run_full_config(c, all_docs, qa_by_dataset, n_contexts,
                                output_dir, embed_model, args.force)
            results.append(r)

        all_results[eid] = results

        # Generate reports per experiment
        generate_rq1a_report(results, output_dir)
        if eid == 1:
            generate_rq1b_report(results, output_dir)
            generate_rq1c_report(results, output_dir)
        elif eid == 2:
            generate_rq1d_report(results, output_dir)

    # Save master results
    summary = output_dir / "rq1_summary.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nMaster results: {summary}")


if __name__ == "__main__":
    main()

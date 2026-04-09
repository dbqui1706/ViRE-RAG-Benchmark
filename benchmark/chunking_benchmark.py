"""Chunking Benchmark — Retrieval-only evaluation of chunking strategies.

Usage:
    python benchmark/chunking_benchmark.py --experiment 1
    python benchmark/chunking_benchmark.py --strategy recursive --chunk-size 768
    python benchmark/chunking_benchmark.py --strategy sentence --datasets UIT-ViQuAD2 ViNewsQA
    python benchmark/chunking_benchmark.py --list
"""

from __future__ import annotations
from yaml import warnings

import argparse
import json
import random
import sys
import time
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
SEED=42
BENCHMARK_CSV = "data/processed/benchmark.csv"
EMBED_MODEL_KEY = "multilingual-e5-large"
TOP_K = 10
MAX_EVAL_SAMPLES = 500
STRATEGIES = ["fixed", "sentence", "paragraph", "recursive", "semantic"]

METRIC_KEYS = [
    "hit_rate", "mrr", "context_precision", "context_recall",
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
]

# Experiment configs

@dataclass
class ChunkingConfig:
    label: str
    strategy: str
    chunk_size: int = 512
    chunk_overlap: int = 0


EXPERIMENTS = {
    1: ("Chunk Size Curve", [
        ChunkingConfig(f"C4-{sz}-50", "recursive", sz, 50)
        for sz in [256, 512, 1024]
    ]),
    2: ("Overlap Curve", [
        ChunkingConfig(f"C4-512-{ov}", "recursive", 512, ov)
        for ov in [0, 25, 50, 100, 200]
    ]),
    3: ("Method x Domain", [
        ChunkingConfig("C1-512", "fixed", 512, 0),
        ChunkingConfig("C2-sentence", "sentence"),
        ChunkingConfig("C3-paragraph", "paragraph"),
        ChunkingConfig("C4-512", "recursive", 512, 50),
        ChunkingConfig("C5-semantic", "semantic"),
    ]),
}


# Data loading

def load_benchmark_data(
    csv_path: str, max_eval: int,
) -> tuple[list[Document], dict[str, list[dict]]]:
    """Load CSV -> docs for indexing + QA pairs for eval (sampled, seed=42)."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["answer"] = df["answer"].fillna("")

    unique_ctx = df.drop_duplicates(subset=["context"])
    all_docs = [
        Document(page_content=str(r["context"]),
                 metadata={"dataset": str(r["dataset"])})
        for _, r in tqdm(unique_ctx.iterrows(), total=len(unique_ctx),
                         desc="Loading docs")
    ]

    rng = random.Random(SEED)
    qa_by_dataset: dict[str, list[dict]] = {}
    for ds, group in df.groupby("dataset"):
        pairs = [
            {k: str(r[k]) for k in ["question", "answer", "context", "dataset"]}
            for _, r in group.iterrows()
        ]
        if len(pairs) > max_eval:
            pairs = rng.sample(pairs, max_eval)
        qa_by_dataset[ds] = pairs

    return all_docs, qa_by_dataset


# Core benchmark 

def run_config(
    config: ChunkingConfig,
    all_docs: list[Document],
    qa_by_dataset: dict[str, list[dict]],
    output_dir: Path,
    embed_model,
    force: bool = False,
) -> dict:
    """Build index, evaluate retrieval, cache result to JSON."""
    result_file = output_dir / "results" / f"unified_{config.label}.json"
    if result_file.exists() and not force:
        print(f"  [SKIP] {config.label} (cached, use --force to re-run)")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    t0 = time.perf_counter()

    # Chunk
    chunker = get_chunker(config.strategy, chunk_size=config.chunk_size,
                          chunk_overlap=config.chunk_overlap)
    chunks = chunker.chunk(tqdm(all_docs, desc=f"Chunking [{config.label}]"))
    print(f"  {config.label}: {len(all_docs)} docs -> {len(chunks)} chunks")

    # Index
    rag_cfg = RagConfig(csv_path="unified", embed_model=EMBED_MODEL_KEY,
                        chroma_dir=str(output_dir / "chroma"),
                        force_reindex=force)
    vs = build_vectorstore(chunks, embed_model, rag_cfg,
                           dataset_name=f"unified_{config.label}",
                           model_key=config.label)
    retriever = DenseRetriever(vs, top_k=TOP_K)

    # Evaluate per dataset
    per_dataset = {}
    for ds_name, qa_pairs in tqdm(qa_by_dataset.items(), desc="Evaluating"):
        questions = [qa["question"] for qa in qa_pairs]
        rr_list = retriever.batch_retrieve(questions)
        scores = [evaluate_retrieval(rr.documents, qa["context"], k=TOP_K)
                  for qa, rr in zip(qa_pairs, rr_list)]
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

    result = {"chunking": asdict(config), "n_docs": len(all_docs),
              "n_chunks": len(chunks), "per_dataset": per_dataset,
              "overall": overall, "elapsed_s": elapsed}

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Overall: Hit={overall['hit_rate']:.3f} MRR={overall['mrr']:.3f} "
          f"NDCG@5={overall['ndcg_at_5']:.3f} ({elapsed}s)")
    return result


# ── Report ────────────────────────────────────────────────────────────────

def generate_report(results: list[dict], output_dir: Path, tag: str) -> None:
    """Write a markdown summary table to output_dir."""
    path = output_dir / f"exp{tag}_report.md"
    ds_names = list(dict.fromkeys(
        ds for r in results for ds in r.get("per_dataset", {})))

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Experiment {tag}\n\n")
        # Overall table
        f.write("## Overall\n\n")
        f.write("| Config | Chunks | Hit Rate | MRR | R@5 | NDCG@5 | Time |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            m = r["overall"]
            f.write(f"| {r['chunking']['label']} | {r['n_chunks']} "
                    f"| {m['hit_rate']:.4f} | {m['mrr']:.4f} "
                    f"| {m.get('recall_at_5', 0):.4f} "
                    f"| {m.get('ndcg_at_5', 0):.4f} | {r['elapsed_s']}s |\n")
        # Per-dataset breakdown
        for metric, key in [("Hit Rate", "hit_rate"), ("NDCG@5", "ndcg_at_5")]:
            f.write(f"\n## {metric} by Dataset\n\n")
            f.write("| Config | " + " | ".join(ds_names) + " |\n")
            f.write("|---" + "|---" * len(ds_names) + "|\n")
            for r in results:
                vals = [f"{r['per_dataset'].get(d, {}).get(key, 0):.4f}"
                        for d in ds_names]
                f.write(f"| {r['chunking']['label']} | " + " | ".join(vals) + " |\n")

    print(f"  Report: {path}")


# CLI 

def show_list(csv_path: str) -> None:
    """Print available strategies, experiments, and datasets."""
    print("\n[Strategies]")
    for s in STRATEGIES:
        print(f"  - {s}")

    print("\n[Experiments]")
    for eid, (name, configs) in EXPERIMENTS.items():
        labels = ", ".join(c.label for c in configs)
        print(f"  {eid}. {name}  [{labels}]")

    print("\n[Datasets]")
    try:
        df = pd.read_csv(csv_path, usecols=["dataset"], encoding="utf-8")
        for ds in sorted(df["dataset"].dropna().unique()):
            print(f"  - {ds}")
    except FileNotFoundError:
        print(f"  CSV not found: {csv_path}")
    print()


def main():
    p = argparse.ArgumentParser(
        description="Chunking benchmark for Vietnamese RAG")
    p.add_argument("--list", action="store_true",
                   help="List strategies, experiments, and datasets")
    p.add_argument("--experiment", type=int, nargs="+",
                   help="Experiment IDs: 1=ChunkSize, 2=Overlap, 3=Method (default: all)")
    p.add_argument("--strategy", choices=STRATEGIES,
                   help="Run single custom config instead of experiment suite")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--chunk-overlap", type=int, default=50)
    p.add_argument("--label", type=str, default=None,
                   help="Custom label (auto-generated if omitted)")
    p.add_argument("--csv", default=BENCHMARK_CSV)
    p.add_argument("--datasets", nargs="+", help="Filter to specific datasets")
    p.add_argument("--max-samples", type=int, default=MAX_EVAL_SAMPLES,
                   help=f"Eval samples per dataset (default: {MAX_EVAL_SAMPLES})")
    p.add_argument("--output-dir", default="outputs/chunking_benchmark")
    p.add_argument("--force", action="store_true",
                   help="Ignore cache, rebuild index")
    args = p.parse_args()

    if args.list:
        show_list(args.csv)
        return

    if args.strategy and args.experiment:
        p.error("Cannot use --strategy and --experiment together")

    # Load data
    print(f"Loading: {args.csv} (eval={args.max_samples}/dataset, seed=42)")
    all_docs, qa_by_dataset = load_benchmark_data(args.csv, args.max_samples)

    if args.datasets:
        qa_by_dataset = {k: v for k, v in qa_by_dataset.items()
                         if k in args.datasets}
        if not qa_by_dataset:
            sys.exit("ERROR: No matching datasets")

    total_q = sum(len(v) for v in qa_by_dataset.values())
    print(f"  {len(all_docs)} docs, {len(qa_by_dataset)} datasets, {total_q} queries")

    print(f"Loading embedding model: {EMBED_MODEL_KEY}...")
    embed_model = get_embed_model(EMBED_MODEL_KEY)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.strategy:
        # Custom single config
        label = args.label or (
            f"custom-{args.strategy}" if args.strategy in ("sentence", "paragraph")
            else f"custom-{args.strategy}-{args.chunk_size}-{args.chunk_overlap}")
        config = ChunkingConfig(label, args.strategy, args.chunk_size,
                                args.chunk_overlap)
        print(f"\n{'='*50}\nCustom: {config}\n{'='*50}")
        result = run_config(config, all_docs, qa_by_dataset, output_dir,
                            embed_model, args.force)
        generate_report([result], output_dir, f"custom-{label}")
    else:
        # Experiment suites
        exp_ids = args.experiment or [1, 2, 3]
        all_results = {}
        for eid in exp_ids:
            if eid not in EXPERIMENTS:
                print(f"Unknown experiment: {eid}")
                continue
            name, configs = EXPERIMENTS[eid]
            print(f"\n{'='*50}\nExp {eid}: {name} ({len(configs)} configs)\n{'='*50}")
            results = [run_config(c, all_docs, qa_by_dataset, output_dir,
                                  embed_model, args.force) for c in configs]
            all_results[eid] = results
            generate_report(results, output_dir, str(eid))

        summary = output_dir / "summary.json"
        with open(summary, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()

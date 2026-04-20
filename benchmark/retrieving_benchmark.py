"""Retrieval Benchmark — Evaluate retrieval strategies with fixed chunking.

Fixed parameters:
    Chunking: paragraph (best from chunking benchmark)
    Embedding: multilingual-e5-large
    BM25/TF-IDF: word-level (underthesea)

Usage:
    python benchmark/retrieving_benchmark.py --list
    python benchmark/retrieving_benchmark.py                    # run all strategies
    python benchmark/retrieving_benchmark.py --strategy R1-BM25 R3-Dense
    python benchmark/retrieving_benchmark.py --datasets ViNewsQA CSConDa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.documents import Document
from rag_bench.chunker import get_chunker
from rag_bench.config import RagConfig
from rag_bench.embeddings.registry import get_embed_model
from rag_bench.evaluator import evaluate_retrieval
from rag_bench.indexer import build_vectorstore

# Fixed parameters
SEED = 42
BENCHMARK_CSV = "data/processed/benchmark.csv"
EMBED_MODEL_KEY = "multilingual-e5-large" # namdp-ptit/ViDense, bkai-foundation-models/vietnamese-bi-encoder
CHUNK_STRATEGY = "paragraph"
CHUNK_SIZE = 0
CHUNK_OVERLAP = 0
TOP_K = 10
MAX_EVAL_SAMPLES = 500

METRIC_KEYS = [
    "hit_rate", "mrr", "precision",
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
    "map_at_1", "map_at_3", "map_at_5", "map_at_10",
]


# Retrieval configs

@dataclass
class RetrievalConfig:
    label: str
    strategy: str

STRATEGIES: dict[str, RetrievalConfig] = {
    "R1-BM25": RetrievalConfig("R1-BM25", "bm25_word"),
    "R2-TF-IDF": RetrievalConfig("R2-TF-IDF", "tfidf_word"),
    "R3-Dense": RetrievalConfig("R3-Dense", "dense"),
    "R4-Hybrid-RRF": RetrievalConfig("R4-Hybrid-RRF", "hybrid_rrf"),
    "R5-Hybrid-Weighted": RetrievalConfig("R5-Hybrid-Weighted", "hybrid_weighted"),
}


# Retriever factory

def create_retriever(config: RetrievalConfig, vectorstore, chunks):
    """Instantiate a retriever based on config strategy.

    All sparse strategies use word-level tokenization (underthesea).
    All hybrid strategies use Dense + BM25 word-level.
    """
    from rag_bench.retrievers.bm25 import BM25WordRetriever
    from rag_bench.retrievers.dense import DenseRetriever

    if config.strategy == "bm25_word":
        return BM25WordRetriever(documents=chunks, top_k=TOP_K)

    if config.strategy == "tfidf_word":
        from rag_bench.retrievers.tfidf import TfidfWordRetriever
        return TfidfWordRetriever(documents=chunks, top_k=TOP_K)

    if config.strategy == "dense":
        return DenseRetriever(vectorstore=vectorstore, top_k=TOP_K)

    if config.strategy == "hybrid_rrf":
        from rag_bench.retrievers.base import BaseRetriever
        from rag_bench.retrievers.hybrid import reciprocal_rank_fusion

        # RRF with BM25 word-level
        dense = DenseRetriever(vectorstore=vectorstore, top_k=TOP_K)
        sparse = BM25WordRetriever(documents=chunks, top_k=TOP_K)

        class HybridRRFWordRetriever(BaseRetriever):
            def __init__(self, dense_r, sparse_r, top_k):
                self._dense = dense_r
                self._sparse = sparse_r
                self._top_k = top_k

            def retrieve(self, query, **kwargs):
                d_docs = self._dense.retrieve(query)
                s_docs = self._sparse.retrieve(query)
                merged = reciprocal_rank_fusion([d_docs, s_docs])
                return merged[:self._top_k]

        return HybridRRFWordRetriever(dense, sparse, TOP_K)

    if config.strategy == "hybrid_weighted":
        from rag_bench.retrievers.hybrid_weighted import WeightedHybridRetriever
        return WeightedHybridRetriever(
            vectorstore=vectorstore, documents=chunks,
            top_k=TOP_K, alpha=0.3,
        )
    
    raise ValueError(f"Unknown retrieval strategy: {config.strategy}")


# Data loading

def load_benchmark_data(csv_path: str, max_eval: int):
    """Load CSV -> docs for indexing + QA pairs for eval (sampled, seed=42)."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["answer"] = df["answer"].fillna("")

    # Compute context_id: md5 hash of context text (dedup-safe)
    df["context_id"] = df["context"].apply(
        lambda c: hashlib.md5(str(c).encode("utf-8")).hexdigest()[:12]
    )

    unique_ctx = df.drop_duplicates(subset=["context"])
    all_docs = [
        Document(page_content=str(r["context"]),
                 metadata={"dataset": str(r["dataset"]),
                           "context_id": str(r["context_id"])})
        for _, r in tqdm(unique_ctx.iterrows(), total=len(unique_ctx),
                         desc="Loading docs")
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

    return all_docs, qa_by_dataset


# Core benchmark

def run_retrieval_config(
    config: RetrievalConfig,
    chunks: list[Document],
    vectorstore,
    qa_by_dataset: dict[str, list[dict]],
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Build retriever, evaluate across datasets, cache result."""
    result_file = output_dir / "results" / f"{config.label}.json"
    if result_file.exists() and not force:
        print(f"  [SKIP] {config.label} (cached, use --force to re-run)")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    t0 = time.perf_counter()

    # Create retriever
    print(f"\n  Building retriever: {config.label} ({config.strategy})...")
    retriever = create_retriever(config, vectorstore, chunks)

    # Pre-compute total relevant chunks per context_id (R for each query)
    from collections import Counter
    ctx_counts: dict[str, int] = Counter(
        doc.metadata.get("context_id", "") for doc in chunks
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict)
    )

    # Evaluate per dataset
    per_dataset = {}
    for ds_name, qa_pairs in tqdm(qa_by_dataset.items(), desc=f"Eval [{config.label}]"):
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
        "retrieval": asdict(config),
        "chunking": {"strategy": CHUNK_STRATEGY, "size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "embed_model": EMBED_MODEL_KEY,
        "n_chunks": len(chunks),
        "per_dataset": per_dataset,
        "overall": overall,
        "elapsed_s": elapsed,
    }

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  {config.label} Overall: Hit={overall['hit_rate']:.3f} "
          f"MRR={overall['mrr']:.3f} NDCG@5={overall['ndcg_at_5']:.3f} ({elapsed}s)")
    return result


# Report

def generate_report(results: list[dict], output_dir: Path) -> None:
    """Write markdown comparison report."""
    path = output_dir / "retrieval_report.md"
    ds_names = list(dict.fromkeys(
        ds for r in results for ds in r.get("per_dataset", {})))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Retrieval Benchmark Report\n\n")
        f.write(f"**Chunking:** {CHUNK_STRATEGY} ({CHUNK_SIZE}/{CHUNK_OVERLAP})\n")
        f.write(f"**Embedding:** {EMBED_MODEL_KEY}\n")
        f.write(f"**Top-K:** {TOP_K}\n\n")

        # Overall table
        f.write("## Overall Comparison\n\n")
        f.write("| Strategy | Hit Rate | MRR | R@1 | R@3 | R@5 | R@10 "
                "| NDCG@5 | NDCG@10 | Time |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            m = r["overall"]
            lbl = r["retrieval"]["label"]
            f.write(f"| {lbl} "
                    f"| {m['hit_rate']:.4f} | {m['mrr']:.4f} "
                    f"| {m.get('recall_at_1', 0):.4f} "
                    f"| {m.get('recall_at_3', 0):.4f} "
                    f"| {m.get('recall_at_5', 0):.4f} "
                    f"| {m.get('recall_at_10', 0):.4f} "
                    f"| {m.get('ndcg_at_5', 0):.4f} "
                    f"| {m.get('ndcg_at_10', 0):.4f} "
                    f"| {r['elapsed_s']}s |\n")

        # Per-dataset breakdowns
        for metric, key in [("Hit Rate", "hit_rate"), ("MRR", "mrr"),
                            ("NDCG@5", "ndcg_at_5")]:
            f.write(f"\n## {metric} by Dataset\n\n")
            f.write("| Strategy | " + " | ".join(ds_names) + " | Avg |\n")
            f.write("|---" + "|---" * len(ds_names) + "|---|\n")
            for r in results:
                vals = [f"{r['per_dataset'].get(d, {}).get(key, 0):.4f}"
                        for d in ds_names]
                avg = f"{r['overall'].get(key, 0):.4f}"
                f.write(f"| {r['retrieval']['label']} | "
                        + " | ".join(vals) + f" | {avg} |\n")

    print(f"  Report: {path}")


# CLI

def show_list(csv_path: str) -> None:
    print(f"\n[Fixed Config]")
    print(f"  Chunking: {CHUNK_STRATEGY} ({CHUNK_SIZE}/{CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBED_MODEL_KEY}")
    print(f"  Top-K: {TOP_K}")
    print(f"  Eval samples: {MAX_EVAL_SAMPLES}/dataset (seed={SEED})")

    print(f"\n[Retrieval Strategies]")
    for label, cfg in STRATEGIES.items():
        print(f"  {label}: {cfg.strategy}")

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
        description="Retrieval benchmark for Vietnamese RAG")
    p.add_argument("--list", action="store_true",
                   help="List strategies and datasets")
    p.add_argument("--strategy", nargs="+",
                   help="Specific strategies to run (e.g. R1-BM25 R3-Dense)")
    p.add_argument("--csv", default=BENCHMARK_CSV)
    p.add_argument("--datasets", nargs="+", help="Filter to specific datasets")
    p.add_argument("--max-samples", type=int, default=MAX_EVAL_SAMPLES)
    p.add_argument("--output-dir", default="outputs/retrieval_benchmark")
    p.add_argument("--force", action="store_true",
                   help="Ignore cache, rebuild index")
    args = p.parse_args()

    if args.list:
        show_list(args.csv)
        return

    # Validate strategy names
    if args.strategy:
        for s in args.strategy:
            if s not in STRATEGIES:
                avail = ", ".join(STRATEGIES.keys())
                p.error(f"Unknown strategy '{s}'. Available: {avail}")
        configs = [STRATEGIES[s] for s in args.strategy]
    else:
        configs = list(STRATEGIES.values())

    # Load data
    print(f"Loading: {args.csv} (eval={args.max_samples}/dataset, seed={SEED})")
    all_docs, qa_by_dataset = load_benchmark_data(args.csv, args.max_samples)

    if args.datasets:
        qa_by_dataset = {k: v for k, v in qa_by_dataset.items()
                         if k in args.datasets}
        if not qa_by_dataset:
            sys.exit("ERROR: No matching datasets")

    total_q = sum(len(v) for v in qa_by_dataset.values())
    print(f"  {len(all_docs)} docs, {len(qa_by_dataset)} datasets, {total_q} queries")

    # Chunk (fixed: recursive 512/50)
    print(f"\nChunking: {CHUNK_STRATEGY} ({CHUNK_SIZE}/{CHUNK_OVERLAP})...")
    chunker = get_chunker(CHUNK_STRATEGY, chunk_size=CHUNK_SIZE,
                          chunk_overlap=CHUNK_OVERLAP)
    chunks = chunker.chunk(tqdm(all_docs, desc="Chunking"))
    print(f"  {len(all_docs)} docs -> {len(chunks)} chunks")

    # Build vectorstore (shared across dense/hybrid strategies)
    print(f"\nLoading embedding model: {EMBED_MODEL_KEY}...")
    embed_model = get_embed_model(EMBED_MODEL_KEY)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rag_cfg = RagConfig(
        csv_path="unified", embed_model=EMBED_MODEL_KEY,
        chroma_dir=str(output_dir / "chroma"),
        force_reindex=args.force,
    )
    print("Building/loading vectorstore...")
    vs = build_vectorstore(
        chunks, embed_model, rag_cfg,
        dataset_name="retrieval_benchmark",
        model_key=f"{CHUNK_STRATEGY}-{EMBED_MODEL_KEY}",
    )

    # Run each retrieval strategy
    print(f"\n{'='*60}")
    print(f"Running {len(configs)} retrieval strategies")
    print(f"{'='*60}")

    results = []
    for config in configs:
        result = run_retrieval_config(
            config, chunks, vs, qa_by_dataset, output_dir, args.force
        )
        results.append(result)

    # Generate report
    print("\nGenerating report...")
    generate_report(results, output_dir)

    # Save summary
    summary = output_dir / "summary.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

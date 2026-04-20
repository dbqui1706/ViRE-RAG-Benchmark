"""Generation Benchmark — Evaluate LLM generation with fixed chunking & retrieval.

Fixed parameters:
    Chunking: paragraph
    Retrieval: hybrid_weighted (alpha=0.3)
    Embedding: multilingual-e5-large
    Top-K: 10

Metrics: F1, Exact Match, ROUGE-L

Usage:
    python benchmark/generation_benchmark.py --list
    python benchmark/generation_benchmark.py --strategy G1-GPT4o-mini
    python benchmark/generation_benchmark.py --strategy G1-GPT4o-mini --datasets ViNewsQA CSConDa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
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
from rag_bench.evaluator import evaluate_answer
from rag_bench.generator import OpenAIGenerator
from rag_bench.indexer import build_vectorstore

# ---------------------------------------------------------------------------
# Fixed retrieval parameters (best from retrieval benchmark)
# ---------------------------------------------------------------------------
SEED = 42
BENCHMARK_CSV = "data/processed/benchmark.csv"
EMBED_MODEL_KEY = "multilingual-e5-large"
CHUNK_STRATEGY = "paragraph"
CHUNK_SIZE = 0
CHUNK_OVERLAP = 0
TOP_K = 10
MAX_EVAL_SAMPLES = 500
FEW_SHOT_N = 3

METRIC_KEYS = ["f1", "exact_match", "rouge_l"]


# ---------------------------------------------------------------------------
# Generation configs
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    label: str
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.1


STRATEGIES: dict[str, GenerationConfig] = {
    "G1-GPT4o-mini": GenerationConfig(
        label="G1-GPT4o-mini",
        model="gpt-4o-mini",
    ),
    "G2-Llama-70B": GenerationConfig(
        label="G2-Llama-70B",
        model="Llama-3.3-70B-Instruct",
        base_url="https://mkp-api.fptcloud.com",
        api_key_env="FPT_API_KEY",
    ),
}


# ---------------------------------------------------------------------------
# Data loading (same as retrieval benchmark)
# ---------------------------------------------------------------------------

def load_benchmark_data(csv_path: str, max_eval: int):
    """Load CSV -> docs + QA pairs (sampled, seed=42)."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["answer"] = df["answer"].fillna("")
    df["context_id"] = df["context"].apply(
        lambda c: hashlib.md5(str(c).encode("utf-8")).hexdigest()[:12]
    )

    unique_ctx = df.drop_duplicates(subset=["context"])
    all_docs = [
        Document(
            page_content=str(r["context"]),
            metadata={"dataset": str(r["dataset"]),
                      "context_id": str(r["context_id"])},
        )
        for _, r in tqdm(unique_ctx.iterrows(), total=len(unique_ctx),
                         desc="Loading docs")
    ]

    rng = random.Random(SEED)
    qa_by_dataset: dict[str, list[dict]] = {}
    for ds, group in df.groupby("dataset"):
        pairs = [
            {"context_id": str(r["context_id"]),
             **{k: str(r[k]) for k in ["question", "answer", "context", "dataset"]}}
            for _, r in group.iterrows()
        ]
        if len(pairs) > max_eval:
            pairs = rng.sample(pairs, max_eval)
        qa_by_dataset[ds] = pairs

    return all_docs, qa_by_dataset


def split_few_shot(qa_pairs: list[dict], n_few_shot: int, rng: random.Random):
    """Split QA pairs into few-shot examples and eval set (no leakage)."""
    if n_few_shot <= 0 or len(qa_pairs) <= n_few_shot:
        return [], qa_pairs
    shuffled = list(qa_pairs)
    rng.shuffle(shuffled)
    return shuffled[:n_few_shot], shuffled[n_few_shot:]


# ---------------------------------------------------------------------------
# Retriever factory (fixed: hybrid_weighted)
# ---------------------------------------------------------------------------

def create_retriever(vectorstore, chunks):
    """Create hybrid_weighted retriever (fixed config)."""
    from rag_bench.retrievers.hybrid_weighted import WeightedHybridRetriever
    return WeightedHybridRetriever(
        vectorstore=vectorstore, documents=chunks,
        top_k=TOP_K, alpha=0.3,
    )


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_generation_config(
    config: GenerationConfig,
    retriever,
    qa_by_dataset: dict[str, list[dict]],
    output_dir: Path,
    max_workers: int = 2,
    force: bool = False,
) -> dict:
    """Retrieve -> Generate -> Evaluate for one LLM config."""
    result_file = output_dir / "results" / f"{config.label}.json"
    if result_file.exists() and not force:
        print(f"  [SKIP] {config.label} (cached, use --force to re-run)")
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)

    t0 = time.perf_counter()

    # Build generator
    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        print(f"  [WARN] {config.api_key_env} not set, generation may fail")

    rng = random.Random(SEED)

    per_dataset: dict[str, dict] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for ds_name, qa_pairs in tqdm(qa_by_dataset.items(), desc=f"Gen [{config.label}]"):
        # Split few-shot examples from eval set
        few_shot_examples, eval_pairs = split_few_shot(qa_pairs, FEW_SHOT_N, rng)

        generator = OpenAIGenerator(
            model=config.model,
            api_key=api_key,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            few_shot_examples=few_shot_examples,
        )

        # Retrieve for each question
        questions = [qa["question"] for qa in eval_pairs]
        rr_list = retriever.batch_retrieve(questions)

        # Build generation inputs: join top-K retrieved docs as context
        gen_items = []
        for qa, rr in zip(eval_pairs, rr_list, strict=True):
            context = "\n\n".join(doc.page_content for doc in rr.documents[:TOP_K])
            gen_items.append({"question": qa["question"], "context": context})

        # Batch generate
        gen_results = generator.batch_generate(gen_items, max_workers=max_workers)

        # Evaluate
        scores_list = []
        for qa, gr in zip(eval_pairs, gen_results, strict=True):
            scores = evaluate_answer(gr.text, qa["answer"], include_semantic=False)
            total_input_tokens += gr.input_tokens
            total_output_tokens += gr.output_tokens
            scores_list.append(scores)

        per_dataset[ds_name] = {
            k: round(float(np.mean([s[k] for s in scores_list])), 4)
            for k in METRIC_KEYS
        }
        m = per_dataset[ds_name]
        print(f"    {ds_name}: F1={m['f1']:.3f} EM={m['exact_match']:.3f} "
              f"ROUGE-L={m['rouge_l']:.3f} ({len(eval_pairs)} samples)")

    # Aggregate across datasets
    overall = {
        k: round(float(np.mean([d[k] for d in per_dataset.values()])), 4)
        for k in METRIC_KEYS
    }
    elapsed = round(time.perf_counter() - t0, 1)

    result = {
        "generation": {"label": config.label, "model": config.model},
        "retrieval": {"strategy": "hybrid_weighted", "alpha": 0.3},
        "chunking": {"strategy": CHUNK_STRATEGY},
        "embed_model": EMBED_MODEL_KEY,
        "top_k": TOP_K,
        "few_shot": FEW_SHOT_N,
        "per_dataset": per_dataset,
        "overall": overall,
        "tokens": {"input": total_input_tokens, "output": total_output_tokens},
        "elapsed_s": elapsed,
    }

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  {config.label} Overall: F1={overall['f1']:.3f} "
          f"EM={overall['exact_match']:.3f} ROUGE-L={overall['rouge_l']:.3f} ({elapsed}s)")
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], output_dir: Path) -> None:
    """Write markdown comparison report."""
    path = output_dir / "generation_report.md"
    ds_names = list(dict.fromkeys(
        ds for r in results for ds in r.get("per_dataset", {})))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Generation Benchmark Report\n\n")
        f.write(f"**Chunking:** {CHUNK_STRATEGY}\n")
        f.write("**Retrieval:** hybrid_weighted (alpha=0.3)\n")
        f.write(f"**Embedding:** {EMBED_MODEL_KEY}\n")
        f.write(f"**Top-K:** {TOP_K} | **Few-shot:** {FEW_SHOT_N}\n\n")

        # Overall table
        f.write("## Overall Comparison\n\n")
        f.write("| Model | F1 | Exact Match | ROUGE-L | Tokens (in/out) | Time |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            m = r["overall"]
            tok = r.get("tokens", {})
            f.write(f"| {r['generation']['label']} "
                    f"| {m['f1']:.4f} | {m['exact_match']:.4f} "
                    f"| {m['rouge_l']:.4f} "
                    f"| {tok.get('input', 0):,}/{tok.get('output', 0):,} "
                    f"| {r['elapsed_s']}s |\n")

        # Per-dataset breakdown
        for metric, key in [("F1", "f1"), ("Exact Match", "exact_match"),
                            ("ROUGE-L", "rouge_l")]:
            f.write(f"\n## {metric} by Dataset\n\n")
            f.write("| Model | " + " | ".join(ds_names) + " | Avg |\n")
            f.write("|---" + "|---" * len(ds_names) + "|---|\n")
            for r in results:
                vals = [f"{r['per_dataset'].get(d, {}).get(key, 0):.4f}"
                        for d in ds_names]
                avg = f"{r['overall'].get(key, 0):.4f}"
                f.write(f"| {r['generation']['label']} | "
                        + " | ".join(vals) + f" | {avg} |\n")

    print(f"  Report: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def show_list(csv_path: str) -> None:
    print("\n[Fixed Config]")
    print(f"  Chunking: {CHUNK_STRATEGY}")
    print("  Retrieval: hybrid_weighted (alpha=0.3)")
    print(f"  Embedding: {EMBED_MODEL_KEY}")
    print(f"  Top-K: {TOP_K}, Few-shot: {FEW_SHOT_N}")
    print(f"  Eval samples: {MAX_EVAL_SAMPLES}/dataset (seed={SEED})")

    print("\n[Generation Strategies]")
    for label, cfg in STRATEGIES.items():
        url = cfg.base_url or "OpenAI"
        print(f"  {label}: {cfg.model} ({url})")

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
        description="Generation benchmark for Vietnamese RAG")
    p.add_argument("--list", action="store_true",
                   help="List strategies and datasets")
    p.add_argument("--strategy", nargs="+",
                   help="Specific strategies (e.g. G1-GPT4o-mini G2-Llama-70B)")
    p.add_argument("--csv", default=BENCHMARK_CSV)
    p.add_argument("--datasets", nargs="+", help="Filter to specific datasets")
    p.add_argument("--max-samples", type=int, default=MAX_EVAL_SAMPLES)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--output-dir", default="outputs/generation_benchmark")
    p.add_argument("--force", action="store_true",
                   help="Ignore cache, re-run generation")
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

    # Chunk
    print(f"\nChunking: {CHUNK_STRATEGY}...")
    chunker = get_chunker(CHUNK_STRATEGY, chunk_size=CHUNK_SIZE,
                          chunk_overlap=CHUNK_OVERLAP)
    chunks = chunker.chunk(tqdm(all_docs, desc="Chunking"))
    print(f"  {len(all_docs)} docs -> {len(chunks)} chunks")

    # Build vectorstore
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
        dataset_name="generation_benchmark",
        model_key=f"{CHUNK_STRATEGY}-{EMBED_MODEL_KEY}",
    )

    # Build retriever (fixed: hybrid_weighted)
    print("Building retriever: hybrid_weighted (alpha=0.3)...")
    retriever = create_retriever(vs, chunks)

    # Run each generation strategy
    print(f"\n{'='*60}")
    print(f"Running {len(configs)} generation strategies")
    print(f"{'='*60}")

    results = []
    for config in configs:
        result = run_generation_config(
            config, retriever, qa_by_dataset, output_dir,
            max_workers=args.max_workers, force=args.force,
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

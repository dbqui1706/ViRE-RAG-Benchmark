"""End-to-end RAG pipeline — batch retrieve then batch generate."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

from .chunker import get_chunker
from .config import RagConfig
from .data_loader import load_dataset, sample_qa_pairs, split_few_shot_examples
from .embeddings.registry import get_embed_model
from .evaluator import evaluate_answer, evaluate_retrieval
from .generator import OpenAIGenerator
from .indexer import UNIFIED_DATASET_NAME, build_vectorstore
from .query_transforms import get_transformer
from .reporter import save_results
from .retrievers import RetrievalResult, get_retriever, list_strategies

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _build_retriever(config: RagConfig, vectorstore, docs: list):
    """Instantiate retriever via the registry based on ``config.search_type``.

    Supports optional two-stage reranking: when ``config.rerank`` is True,
    stage 1 over-retrieves by ``rerank_factor`` and stage 2 reranks to ``top_k``.
    """
    st = config.search_type

    # When reranking, stage-1 retrieves more candidates for the reranker
    stage1_k = config.top_k * config.rerank_factor if config.rerank else config.top_k

    if st in ("similarity", "mmr"):
        base_retriever = get_retriever("dense", vectorstore=vectorstore, top_k=stage1_k, search_type=st)
    elif st in ("bm25_syl", "bm25_word"):
        base_retriever = get_retriever(st, documents=docs, top_k=stage1_k)
    elif st == "hybrid":
        base_retriever = get_retriever("hybrid", vectorstore=vectorstore, documents=docs, top_k=stage1_k)
    elif st == "hybrid_weighted":
        base_retriever = get_retriever("hybrid_weighted", vectorstore=vectorstore, documents=docs, top_k=stage1_k)
    else:
        raise ValueError(f"Unknown search_type: '{st}'. Available: {list_strategies()}")

    # Wrap with query expansion if active
    if config.query_transform != "passthrough":
        transformer = get_transformer(
            config.query_transform,
            llm_model=config.transform_llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            n_variations=config.n_query_variations,
            max_sub_questions=config.max_sub_questions,
        )
        base_retriever = get_retriever(
            "expanded",
            base_retriever=base_retriever,
            transformer=transformer,
            top_k=stage1_k,
        )

    # Wrap with reranker if enabled
    if config.rerank:
        print(f"  Reranker: {config.rerank_model} (top_m={stage1_k} -> top_k={config.top_k})")
        base_retriever = get_retriever(
            "reranker",
            base_retriever=base_retriever,
            api_key=os.environ.get("FPT_API_KEY", ""),
            model=config.rerank_model,
            top_m=stage1_k,
            top_k=config.top_k,
        )

    # Wrap with corrective if enabled
    if config.corrective:
        print("  CRAG: Corrective Retrieval filtering enabled")
        base_retriever = get_retriever(
            "corrective",
            base_retriever=base_retriever,
            model=config.transform_llm_model,  # reuse transform model (gpt-4o-mini usually)
            api_key=config.llm_api_key,
            top_k=config.top_k,
        )

    # Wrap with compressor if enabled
    if config.compress:
        print(f"  Compressor: Contextual Compression enabled (max_tokens={config.compress_max_tokens})")
        base_retriever = get_retriever(
            "compressor",
            base_retriever=base_retriever,
            model=config.transform_llm_model,
            api_key=config.llm_api_key,
            top_k=config.top_k,
            compress_max_tokens=config.compress_max_tokens,
        )

    return base_retriever


# ---------------------------------------------------------------------------
# Shared pipeline steps (used by both run_pipeline and run_unified_pipeline)
# ---------------------------------------------------------------------------


def _prepare_qa(config: RagConfig, csv_path: str):
    """Load dataset, split few-shot, sample QA pairs.

    Returns:
        (qa_pairs, few_shot_examples)
    """
    _, all_qa = load_dataset(csv_path, prefer_unique=False)

    few_shot = None
    if config.prompt_strategy == "few_shot":
        few_shot, all_qa = split_few_shot_examples(
            all_qa, n_examples=config.n_few_shot, seed=config.sample_seed,
        )

    qa_pairs = sample_qa_pairs(all_qa, max_samples=config.max_samples, seed=config.sample_seed)
    return qa_pairs, few_shot


def _load_benchmark_qa(
    csv_path: str,
    max_samples: int = 500,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Load a unified CSV with ``dataset`` column and group QA pairs by dataset.

    Each dataset group is independently sampled to at most ``max_samples`` rows.
    This mirrors the approach used in ``benchmark/chunking_benchmark.py``.

    Args:
        csv_path: Path to a CSV containing columns: question, answer, context, dataset.
        max_samples: Max QA pairs per dataset.
        seed: Random seed for sampling.

    Returns:
        Dict mapping dataset name -> list of QA pair dicts.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["answer"] = df["answer"].fillna("")

    if "dataset" not in df.columns:
        raise ValueError(
            f"{csv_path} has no 'dataset' column. "
            "Use --datasets to specify individual CSV files instead."
        )

    rng = random.Random(seed)
    qa_by_dataset: dict[str, list[dict]] = {}
    for ds_name, group in df.groupby("dataset"):
        pairs = []
        for idx, (_, row) in enumerate(group.iterrows()):
            ctx = str(row["context"])
            context_id = hashlib.md5(ctx.encode("utf-8")).hexdigest()[:12]
            pairs.append({
                "qid": str(row.get("qid", idx)),
                "context_id": context_id,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "context": ctx,
            })
        if len(pairs) > max_samples:
            pairs = rng.sample(pairs, max_samples)
        qa_by_dataset[str(ds_name)] = pairs

    return qa_by_dataset


def _generate(config: RagConfig, retrieval_results: list[RetrievalResult],
              few_shot_examples=None, retriever=None):
    """Run batch LLM generation over retrieval results.

    When ``config.generation_strategy == "self_rag"``, uses the iterative
    SelfRAGGenerator instead of the standard single-pass OpenAIGenerator.

    Args:
        config: Experiment configuration.
        retrieval_results: Retrieved documents for each question.
        few_shot_examples: Optional few-shot examples.
        retriever: Required for self_rag strategy (needs re-retrieval access).

    Returns:
        list of generation result objects.
    """
    if config.generation_strategy == "self_rag":
        from .self_rag import SelfRAGGenerator
        if retriever is None:
            raise ValueError("Self-RAG strategy requires a retriever for re-retrieval")
        gen = SelfRAGGenerator(
            retriever=retriever,
            model=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url or None,
            max_iterations=config.self_rag_max_iter,
            few_shot_examples=few_shot_examples,
        )
        gen_items = [
            {
                "question": r.question,
                "context": "\n\n".join(doc.page_content for doc in r.documents),
            }
            for r in retrieval_results
        ]
        return gen.batch_generate(gen_items, retrieval_results)

    # Standard single-pass generation
    llm = OpenAIGenerator(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url or None,
        few_shot_examples=few_shot_examples,
    )
    gen_items = [
        {
            "question": r.question,
            "context": "\n\n".join(doc.page_content for doc in r.documents),
        }
        for r in retrieval_results
    ]
    return llm.batch_generate(gen_items, max_workers=config.max_workers)


def _build_generations(qa_pairs, retrieval_results, gen_results):
    """Combine QA pairs, retrieval results, and generation results into a
    serializable list of dicts (``generations.json`` format)."""
    generations = []
    for i, qa in enumerate(qa_pairs):
        ret = retrieval_results[i]
        gen = gen_results[i]
        entry = {
            "qid": qa["qid"],
            "context_id": qa.get("context_id", ""),
            "question": qa["question"],
            "gold_answer": qa["answer"],
            "predicted_answer": gen.text,
            "retrieved_contexts": [doc.page_content for doc in ret.documents],
            "retrieval_ms": ret.retrieval_ms,
            "generation_ms": gen.generation_ms,
            "total_ms": ret.retrieval_ms + gen.generation_ms,
            "input_tokens": gen.input_tokens,
            "output_tokens": gen.output_tokens,
        }
        # Self-RAG tracking fields (only present for self_rag strategy)
        if gen.iterations is not None:
            entry["self_rag_iterations"] = gen.iterations
            entry["self_rag_llm_calls"] = gen.total_llm_calls
        generations.append(entry)
    return generations


def _save_generations(generations: list[dict], out_dir: Path) -> Path:
    """Write generations.json and return the path."""
    path = out_dir / "generations.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)
    return path


def _evaluate(config: RagConfig, generations: list[dict],
              qa_pairs: list[dict], retrieval_results: list[RetrievalResult],
              dataset_name: str = "", chunks: list | None = None) -> dict:
    """Run evaluation (lexical + optional RAGAS) and return evaluations dict.

    Retrieval metrics use context_id-based index matching (aligned with ViRE).
    Each retrieved Document's ``metadata["context_id"]`` is compared against the
    gold context_id from the QA pair.

    When *chunks* is provided, n_relevant is pre-computed per context_id so that
    Recall, NDCG, and MAP use the true corpus-level R (matching
    ``retrieving_benchmark.py``).  Without chunks, R falls back to hits in top-K.
    """
    label = f"Evaluating {dataset_name}" if dataset_name else "Evaluating"

    # Pre-compute n_relevant per context_id from the chunked corpus
    ctx_counts: dict[str, int] = {}
    if chunks:
        from collections import Counter
        ctx_counts = Counter(
            doc.metadata.get("context_id", "")
            for doc in chunks
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict)
        )

    per_query_eval = []
    for i, g in enumerate(tqdm(generations, desc=label)):
        has_gold = bool(g["gold_answer"].strip())
        gen_scores = (
            evaluate_answer(
                g["predicted_answer"], g["gold_answer"],
                include_semantic=config.include_semantic,
            )
            if has_gold
            else {"exact_match": None, "f1": None, "rouge_l": None}
        )
        # Index-based retrieval evaluation: use Document objects with context_id
        ret_docs = retrieval_results[i].documents
        gold_ctx_id = g["context_id"]
        n_rel = ctx_counts.get(gold_ctx_id) if ctx_counts else None
        ret_scores = evaluate_retrieval(
            ret_docs, gold_context_id=gold_ctx_id, k=config.top_k, n_relevant=n_rel,
        )
        per_query_eval.append({
            "qid": g["qid"],
            "generation_scores": gen_scores,
            "retrieval_scores": ret_scores,
        })

    # Optional RAGAS
    ragas_metrics = {}
    if config.eval_faithfulness:
        ragas_data = [
            {
                "user_input": g["question"],
                "retrieved_contexts": g["retrieved_contexts"],
                "response": g["predicted_answer"],
                "reference": g["gold_answer"],
            }
            for g in generations
        ]
        ragas_client = AsyncOpenAI(api_key=config.llm_api_key)
        from .evaluator import run_ragas_evaluation
        ragas_metrics = asyncio.run(run_ragas_evaluation(
            ragas_data, model="gpt-4o-mini", client=ragas_client,
        ))

    # Aggregate
    def _avg(key, section):
        vals = [q[section][key] for q in per_query_eval if q[section].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    gen_keys = ["exact_match", "f1", "rouge_l"]
    if config.include_semantic:
        gen_keys += ["bert_score", "semantic_sim"]

    n = len(generations)
    return {
        "generation_metrics": {k: _avg(k, "generation_scores") for k in gen_keys},
        "retrieval_metrics": {
            k: _avg(k, "retrieval_scores")
            for k in [
                "precision", "mrr", "hit_rate",
                "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
                "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
                "map_at_1", "map_at_3", "map_at_5", "map_at_10",
                "mrr_at_1", "mrr_at_3", "mrr_at_5", "mrr_at_10",
                "first_relevant_rank",
            ]
        },
        "ragas_metrics": ragas_metrics,
        "latency": {
            "mean_retrieval_ms": sum(g["retrieval_ms"] for g in generations) / n,
            "mean_generation_ms": sum(g["generation_ms"] for g in generations) / n,
            "mean_total_ms": sum(g["total_ms"] for g in generations) / n,
            "total_input_tokens": sum(g["input_tokens"] for g in generations),
            "total_output_tokens": sum(g["output_tokens"] for g in generations),
        },
        "per_query": per_query_eval,
    }


def _save_evaluations(config: RagConfig, metrics: dict, out_dir: Path,
                      dataset_name: str, index_source: str = "per_dataset"):
    """Write evaluations.json + markdown report."""
    evaluations = {
        "config": {
            "dataset": dataset_name,
            "index_source": index_source,
            "search_type": config.search_type,
            "query_transform": config.query_transform,
            "transform_llm_model": config.transform_llm_model,
            "n_query_variations": config.n_query_variations,
            "embed_model": config.embed_model,
            "llm_model": config.llm_model,
            "chunk_strategy": config.chunk_strategy,
            "top_k": config.top_k,
            "max_samples": config.max_samples,
            "max_workers": config.max_workers,
            "include_semantic": config.include_semantic,
            "eval_faithfulness": config.eval_faithfulness,
            "rerank": config.rerank,
            "corrective": config.corrective,
            "compress": config.compress,
            "generation_strategy": config.generation_strategy,
        },
        **metrics,
    }
    path = out_dir / "evaluations.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)
    save_results(evaluations, out_dir)
    return evaluations


def _print_summary(metrics: dict):
    """Print a compact summary of generation + retrieval metrics."""
    gm = metrics["generation_metrics"]
    rm = metrics["retrieval_metrics"]
    lat = metrics["latency"]

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print(
        f"  Gen: EM={_fmt(gm.get('exact_match'))}  "
        f"F1={_fmt(gm.get('f1'))}  ROUGE-L={_fmt(gm.get('rouge_l'))}"
    )
    print(
        f"  Ret: P={_fmt(rm.get('precision'))}  "
        f"MRR={_fmt(rm.get('mrr'))}  Hit={_fmt(rm.get('hit_rate'))}  "
        f"R@5={_fmt(rm.get('recall_at_5'))}  NDCG@5={_fmt(rm.get('ndcg_at_5'))}  "
        f"MAP@5={_fmt(rm.get('map_at_5'))}"
    )
    print(f"  Latency: {lat['mean_total_ms']:.0f}ms avg")


# ---------------------------------------------------------------------------
# Shared infrastructure: load, chunk, index
# ---------------------------------------------------------------------------


def _build_index(config: RagConfig, csv_path: str, dataset_name: str):
    """Load → Chunk → Build vectorstore.

    Returns:
        (vectorstore, chunked_docs)
    """
    print(f"  Loading: {csv_path}")
    all_docs, _ = load_dataset(csv_path, prefer_unique=config.prefer_unique)
    print(f"  {len(all_docs)} documents loaded")

    chunker = get_chunker(
        config.chunk_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    docs = chunker.chunk(all_docs)
    print(f"  Chunking ({config.chunk_strategy}): {len(all_docs)} → {len(docs)} chunks")

    embed_model = get_embed_model(config.embed_model)
    print(f"  Building vectorstore with {config.embed_model} for {dataset_name}")
    vectorstore = build_vectorstore(docs, embed_model, config, dataset_name, config.embed_model)
    return vectorstore, docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _output_dir(config: RagConfig, dataset_name: str, suffix: str = "") -> Path:
    """Build output directory path."""
    name = f"{dataset_name}{suffix}"
    strategy = config.search_type
    if config.rerank:
        strategy += "+rerank"
    if config.corrective:
        strategy += "+corrective"
    if config.compress:
        strategy += "+compress"
    if config.generation_strategy == "self_rag":
        strategy += "+self_rag"
    out = Path(config.output_dir) / name / strategy / config.embed_model
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_pipeline(config: RagConfig) -> dict:
    """Run full RAG pipeline on a single dataset.

    Flow: Load -> Chunk -> Index -> Retrieve -> Generate -> Evaluate
    """
    dataset_name = Path(config.csv_path).stem
    out_dir = _output_dir(config, dataset_name)
    print(f"[Pipeline] {dataset_name} | {config.search_type} | {config.embed_model}")

    # Build index
    vectorstore, docs = _build_index(config, config.csv_path, dataset_name)

    # Build retriever
    retriever = _build_retriever(config, vectorstore=vectorstore, docs=docs)

    # Prepare QA
    qa_pairs, few_shot = _prepare_qa(config, config.csv_path)
    print(f"  {len(qa_pairs)} QA pairs sampled")

    # Retrieve + Generate
    questions = [qa["question"] for qa in qa_pairs]
    print(f"  Retrieving {len(questions)} queries ({config.search_type})...")
    retrieval_results = retriever.batch_retrieve(questions)

    print(f"  Generating {len(questions)} answers...")
    gen_results = _generate(config, retrieval_results, few_shot, retriever=retriever)

    # Save generations
    generations = _build_generations(qa_pairs, retrieval_results, gen_results)
    gen_path = _save_generations(generations, out_dir)
    print(f"  Generations -> {gen_path}")

    # Evaluate + Save
    metrics = _evaluate(
        config, generations, qa_pairs, retrieval_results, dataset_name, chunks=docs,
    )
    evaluations = _save_evaluations(config, metrics, out_dir, dataset_name)
    _print_summary(metrics)

    return evaluations


def run_unified_pipeline(config: RagConfig, dataset_csv_paths: list[str]) -> list[dict]:
    """Run RAG pipeline with ONE shared index across multiple datasets.

    Builds a single vectorstore from ``config.unified_index_csv``, then
    evaluates each dataset's queries against it independently.

    When ``dataset_csv_paths`` is empty, the pipeline auto-splits
    ``config.unified_index_csv`` by the ``dataset`` column and samples
    ``config.max_samples`` QA pairs per dataset group.  This is the
    recommended mode for ``data/processed/benchmark.csv``.
    """
    assert config.unified_index_csv, "unified_index_csv must be set"
    print(f"[Unified] Building shared index from {config.unified_index_csv}")

    # Build shared index once
    vectorstore, docs = _build_index(config, config.unified_index_csv, UNIFIED_DATASET_NAME)
    retriever = _build_retriever(config, vectorstore=vectorstore, docs=docs)
    print("[Unified] Shared index ready.\n")

    # ── Decide QA loading strategy ─────────────────────────────────
    if dataset_csv_paths:
        # Legacy mode: load from separate CSV files
        qa_groups = {
            Path(p).stem: _prepare_qa(config, p) for p in dataset_csv_paths
        }
    else:
        # Auto-split mode: group by 'dataset' column in benchmark.csv
        print(f"[Unified] Auto-split mode: grouping by 'dataset' column, "
              f"max_samples={config.max_samples}/dataset")
        benchmark_qa = _load_benchmark_qa(
            config.unified_index_csv,
            max_samples=config.max_samples,
            seed=config.sample_seed,
        )
        # Wrap each group's QA pairs with few-shot handling
        qa_groups = {}
        for ds_name, qa_pairs in benchmark_qa.items():
            few_shot = None
            if config.prompt_strategy == "few_shot":
                few_shot, qa_pairs = split_few_shot_examples(
                    qa_pairs, n_examples=config.n_few_shot, seed=config.sample_seed,
                )
            qa_groups[ds_name] = (qa_pairs, few_shot)

    # ── Evaluate each dataset ──────────────────────────────────────
    all_results = []
    for dataset_name, (qa_pairs, few_shot) in qa_groups.items():
        out_dir = _output_dir(config, dataset_name)
        print(f"[Unified] ── {dataset_name} ({len(qa_pairs)} samples) ──")

        # Retrieve + Generate
        questions = [qa["question"] for qa in qa_pairs]
        print(f"  Retrieving {len(questions)} queries ({config.search_type})...")
        retrieval_results = retriever.batch_retrieve(questions)
        gen_results = _generate(config, retrieval_results, few_shot, retriever=retriever)

        # Save generations
        generations = _build_generations(qa_pairs, retrieval_results, gen_results)
        gen_path = _save_generations(generations, out_dir)
        print(f"  Generations -> {gen_path}")

        # Evaluate + Save
        metrics = _evaluate(
            config, generations, qa_pairs, retrieval_results, dataset_name, chunks=docs,
        )
        evaluations = _save_evaluations(config, metrics, out_dir, dataset_name, "unified")
        _print_summary(metrics)
        all_results.append(evaluations)

    print(f"\n[Unified] All {len(all_results)} datasets evaluated.")
    return all_results

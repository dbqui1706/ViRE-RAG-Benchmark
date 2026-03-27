"""End-to-end Native RAG pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .chunker import PassthroughChunker
from .config import RagConfig
from .data_loader import load_and_sample
from .embeddings.registry import get_embed_model
from .evaluator import evaluate_answer
from .generator import FPTGenerator
from .indexer import build_index
from .reporter import save_results
from .retriever import query_with_timing
from .timer import aggregate_metrics


def run_pipeline(config: RagConfig) -> dict:
    """Run the full Native RAG pipeline.

    Args:
        config: Experiment configuration.

    Returns:
        Results dict with config, metrics, latency, and per-query details.
    """
    dataset_name = Path(config.csv_path).stem
    print(f"[Pipeline] Dataset: {dataset_name}, Embedding: {config.embed_model}")

    # 1. Load data
    print("[Pipeline] Loading data...")
    docs, qa_pairs = load_and_sample(
        config.csv_path,
        max_samples=config.max_samples,
        seed=config.sample_seed,
        prefer_unique=config.prefer_unique,
    )
    print(f"[Pipeline] Loaded {len(docs)} documents, {len(qa_pairs)} QA pairs")

    # 2. Chunk (passthrough for Native RAG)
    chunker = PassthroughChunker()
    docs = chunker.chunk(docs)

    # 3. Get embedding model
    print(f"[Pipeline] Loading embedding model: {config.embed_model}")
    embed_model = get_embed_model(config.embed_model)

    # 4. Build/load ChromaDB index
    print("[Pipeline] Building index...")
    index = build_index(docs, embed_model, config, dataset_name, config.embed_model)

    # 5. Create LLM + query engine
    llm = FPTGenerator(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=config.top_k,
    )

    # 6. Query + evaluate
    print(f"[Pipeline] Running {len(qa_pairs)} queries...")
    per_query = []
    all_metrics_timer = []

    for qa in tqdm(qa_pairs, desc="Querying"):
        result = query_with_timing(query_engine, qa["question"], llm)
        scores = evaluate_answer(result.answer, qa["answer"])

        per_query.append(
            {
                "qid": qa["qid"],
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "predicted_answer": result.answer,
                "scores": scores,
                "retrieval_ms": result.metrics.retrieval_ms,
                "generation_ms": result.metrics.generation_ms,
                "total_ms": result.metrics.total_ms,
                "input_tokens": result.metrics.input_tokens,
                "output_tokens": result.metrics.output_tokens,
            }
        )
        all_metrics_timer.append(result.metrics)

    # 7. Aggregate
    avg_scores = {}
    for key in ["em", "f1", "rouge_l"]:
        avg_scores[key] = sum(q["scores"][key] for q in per_query) / len(per_query)

    latency_agg = aggregate_metrics(all_metrics_timer)

    results = {
        "config": {
            "dataset": dataset_name,
            "embed_model": config.embed_model,
            "llm_model": config.llm_model,
            "top_k": config.top_k,
            "max_samples": config.max_samples,
        },
        "metrics": avg_scores,
        "latency": latency_agg,
        "per_query": per_query,
    }

    # 8. Save
    out_dir = Path(config.output_dir) / dataset_name / config.embed_model
    save_results(results, out_dir)
    print(f"[Pipeline] Results saved to {out_dir}")
    print(
        f"[Pipeline] EM={avg_scores['em']:.4f}  F1={avg_scores['f1']:.4f}  "
        f"ROUGE-L={avg_scores['rouge_l']:.4f}  "
        f"Avg latency={latency_agg['mean_total_ms']:.0f}ms"
    )

    return results

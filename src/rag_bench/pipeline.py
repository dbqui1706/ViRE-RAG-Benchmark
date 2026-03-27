"""End-to-end Native RAG pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .chunker import PassthroughChunker
from .config import RagConfig
from .data_loader import load_and_sample
from .embeddings.registry import get_embed_model
from .evaluator import evaluate_answer, evaluate_retrieval, evaluate_faithfulness
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

    # 5b. Create Judge LLM (if faithfulness evaluation enabled)
    judge_llm = None
    if config.eval_faithfulness:
        if not config.judge_model:
            print("[Pipeline] WARNING: --eval-faithfulness requires --judge-model. Skipping.")
        else:
            print(f"[Pipeline] Judge model: {config.judge_model}")
            judge_llm = FPTGenerator(
                model=config.judge_model,
                api_key=config.llm_api_key,
                base_url=config.llm_base_url,
            )

    # 6. Query + evaluate
    print(f"[Pipeline] Running {len(qa_pairs)} queries...")
    per_query = []
    all_metrics_timer = []

    for qa in tqdm(qa_pairs, desc="Querying"):
        result = query_with_timing(query_engine, qa["question"], llm)

        # Section 1: Generation Quality
        gen_scores = evaluate_answer(
            result.answer, qa["answer"],
            include_semantic=config.include_semantic,
        )

        # Section 2: Retrieval Quality
        ret_scores = evaluate_retrieval(
            result.source_nodes, qa["context"], k=config.top_k,
        )

        # Section 3: Faithfulness (optional)
        faith_scores = {}
        if judge_llm is not None:
            faith_scores = evaluate_faithfulness(
                qa["question"], result.answer,
                result.source_nodes, judge_llm,
            )

        per_query.append(
            {
                "qid": qa["qid"],
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "predicted_answer": result.answer,
                "generation_scores": gen_scores,
                "retrieval_scores": ret_scores,
                "faithfulness_scores": faith_scores,
                "retrieval_ms": result.metrics.retrieval_ms,
                "generation_ms": result.metrics.generation_ms,
                "total_ms": result.metrics.total_ms,
                "input_tokens": result.metrics.input_tokens,
                "output_tokens": result.metrics.output_tokens,
            }
        )
        all_metrics_timer.append(result.metrics)

    # 7. Aggregate metrics
    def _avg(key: str, section: str) -> float:
        values = [q[section][key] for q in per_query if key in q[section]]
        return sum(values) / len(values) if values else 0.0

    gen_keys = ["em", "f1", "rouge_l"]
    if config.include_semantic:
        gen_keys += ["bert_score", "semantic_sim"]

    avg_generation = {k: _avg(k, "generation_scores") for k in gen_keys}
    avg_retrieval = {k: _avg(k, "retrieval_scores")
                     for k in ["context_precision", "context_recall", "mrr", "hit_rate"]}

    avg_faithfulness = {}
    if judge_llm is not None:
        avg_faithfulness = {k: _avg(k, "faithfulness_scores")
                           for k in ["faithfulness", "answer_relevancy", "hallucination"]}

    latency_agg = aggregate_metrics(all_metrics_timer)

    results = {
        "config": {
            "dataset": dataset_name,
            "embed_model": config.embed_model,
            "llm_model": config.llm_model,
            "top_k": config.top_k,
            "max_samples": config.max_samples,
            "include_semantic": config.include_semantic,
            "eval_faithfulness": config.eval_faithfulness,
            "judge_model": config.judge_model,
        },
        "generation_metrics": avg_generation,
        "retrieval_metrics": avg_retrieval,
        "faithfulness_metrics": avg_faithfulness,
        "latency": latency_agg,
        "per_query": per_query,
    }

    # 8. Save
    out_dir = Path(config.output_dir) / dataset_name / config.embed_model
    save_results(results, out_dir)
    print(f"[Pipeline] Results saved to {out_dir}")
    print(
        f"[Pipeline] Generation: EM={avg_generation['em']:.4f}  F1={avg_generation['f1']:.4f}  "
        f"ROUGE-L={avg_generation['rouge_l']:.4f}"
    )
    print(
        f"[Pipeline] Retrieval: Precision={avg_retrieval['context_precision']:.4f}  "
        f"Recall={avg_retrieval['context_recall']:.4f}  "
        f"MRR={avg_retrieval['mrr']:.4f}  HitRate={avg_retrieval['hit_rate']:.4f}"
    )
    if avg_faithfulness:
        print(
            f"[Pipeline] Faithfulness={avg_faithfulness['faithfulness']:.4f}  "
            f"Relevancy={avg_faithfulness['answer_relevancy']:.4f}  "
            f"Hallucination={avg_faithfulness['hallucination']:.4f}"
        )
    print(f"[Pipeline] Avg latency={latency_agg['mean_total_ms']:.0f}ms")

    return results

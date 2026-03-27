"""End-to-end RAG pipeline — batch retrieve then batch generate."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .chunker import get_chunker
from .config import RagConfig
from .data_loader import load_and_sample
from .embeddings.registry import get_embed_model
from .evaluator import evaluate_answer, evaluate_retrieval, run_ragas_evaluation
from .generator import FPTGenerator
from .indexer import build_vectorstore
from .reporter import save_results
from .retriever import batch_retrieve


def run_pipeline(config: RagConfig) -> dict:
    """Run the full RAG pipeline with batch processing.

    Flow: Load → Chunk → Index → Batch Retrieve → Batch Generate → Evaluate → Report

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

    # 2. Chunk
    chunker = get_chunker(
        config.chunk_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    before_chunk = len(docs)
    docs = chunker.chunk(docs)
    print(f"[Pipeline] Chunking ({config.chunk_strategy}): {before_chunk} → {len(docs)} chunks")

    # 3. Get embedding model & build vector store
    print(f"[Pipeline] Loading embedding model: {config.embed_model}")
    embed_model = get_embed_model(config.embed_model)

    print("[Pipeline] Building vector store...")
    vectorstore = build_vectorstore(docs, embed_model, config, dataset_name, config.embed_model)

    # 4. Batch Retrieve
    questions = [qa["question"] for qa in qa_pairs]
    print(f"[Pipeline] Batch retrieving {len(questions)} queries...")
    retrieval_results = batch_retrieve(vectorstore, questions, k=config.top_k)

    # 5. Batch Generate
    llm = FPTGenerator(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )

    # Build generation items: question + combined retrieved context
    gen_items = []
    for ret_result in retrieval_results:
        combined_context = "\n\n".join(doc.page_content for doc in ret_result.documents)
        gen_items.append({"question": ret_result.question, "context": combined_context})

    print(f"[Pipeline] Batch generating {len(gen_items)} answers (workers={config.max_workers})...")
    gen_results = llm.batch_generate(gen_items, max_workers=config.max_workers)

    # 6. Evaluate
    print("[Pipeline] Evaluating...")
    per_query = []

    for i, qa in enumerate(tqdm(qa_pairs, desc="Evaluating")):
        ret = retrieval_results[i]
        gen = gen_results[i]

        # Section 1: Generation Quality
        gen_scores = evaluate_answer(
            gen.text, qa["answer"],
            include_semantic=config.include_semantic,
        )

        # Section 2: Retrieval Quality
        ret_scores = evaluate_retrieval(
            ret.documents, qa["context"], k=config.top_k,
        )

        per_query.append(
            {
                "qid": qa["qid"],
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "predicted_answer": gen.text,
                "generation_scores": gen_scores,
                "retrieval_scores": ret_scores,
                "faithfulness_scores": {},
                "retrieval_ms": ret.retrieval_ms,
                "generation_ms": gen.generation_ms,
                "total_ms": ret.retrieval_ms + gen.generation_ms,
                "input_tokens": gen.input_tokens,
                "output_tokens": gen.output_tokens,
            }
        )

    # 7. Aggregate metrics
    def _avg(key: str, section: str) -> float:
        values = [q[section][key] for q in per_query if key in q[section]]
        return sum(values) / len(values) if values else 0.0

    gen_keys = ["f1", "rouge_l"]
    if config.include_semantic:
        gen_keys += ["bert_score", "semantic_sim"]

    avg_generation = {k: _avg(k, "generation_scores") for k in gen_keys}
    avg_retrieval = {k: _avg(k, "retrieval_scores")
                     for k in ["context_precision", "context_recall", "mrr", "hit_rate"]}

    avg_latency = {
        "mean_retrieval_ms": sum(q["retrieval_ms"] for q in per_query) / len(per_query),
        "mean_generation_ms": sum(q["generation_ms"] for q in per_query) / len(per_query),
        "mean_total_ms": sum(q["total_ms"] for q in per_query) / len(per_query),
        "total_input_tokens": sum(q["input_tokens"] for q in per_query),
        "total_output_tokens": sum(q["output_tokens"] for q in per_query),
    }

    # 7b. RAGAS evaluation (LLM-based, optional)
    ragas_metrics = {}
    if config.eval_faithfulness:
        print("[Pipeline] Running RAGAS evaluation (LLMContextRecall, Faithfulness, FactualCorrectness)...")
        ragas_data = [
            {
                "user_input": per_query[i]["question"],
                "retrieved_contexts": [doc.page_content for doc in retrieval_results[i].documents],
                "response": per_query[i]["predicted_answer"],
                "reference": per_query[i]["gold_answer"],
            }
            for i in range(len(per_query))
        ]
        ragas_metrics = run_ragas_evaluation(ragas_data, llm.llm)
        print(f"[Pipeline] RAGAS: {ragas_metrics}")

    results = {
        "config": {
            "dataset": dataset_name,
            "embed_model": config.embed_model,
            "llm_model": config.llm_model,
            "chunk_strategy": config.chunk_strategy,
            "top_k": config.top_k,
            "max_samples": config.max_samples,
            "max_workers": config.max_workers,
            "include_semantic": config.include_semantic,
            "eval_faithfulness": config.eval_faithfulness,
        },
        "generation_metrics": avg_generation,
        "retrieval_metrics": avg_retrieval,
        "ragas_metrics": ragas_metrics,
        "latency": avg_latency,
        "per_query": per_query,
    }

    # 8. Save
    out_dir = Path(config.output_dir) / dataset_name / config.embed_model
    save_results(results, out_dir)
    print(f"[Pipeline] Results saved to {out_dir}")
    print(
        f"[Pipeline] Generation: F1={avg_generation['f1']:.4f}  "
        f"ROUGE-L={avg_generation['rouge_l']:.4f}"
    )
    print(
        f"[Pipeline] Retrieval: Precision={avg_retrieval['context_precision']:.4f}  "
        f"Recall={avg_retrieval['context_recall']:.4f}  "
        f"MRR={avg_retrieval['mrr']:.4f}  HitRate={avg_retrieval['hit_rate']:.4f}"
    )
    print(f"[Pipeline] Avg latency={avg_latency['mean_total_ms']:.0f}ms")

    return results

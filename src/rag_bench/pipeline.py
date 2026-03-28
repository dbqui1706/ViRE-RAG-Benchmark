"""End-to-end RAG pipeline — batch retrieve then batch generate."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from tqdm import tqdm
from openai import AsyncOpenAI

from .chunker import get_chunker
from .config import RagConfig
from .data_loader import load_dataset, sample_qa_pairs, split_few_shot_examples
from .embeddings.registry import get_embed_model
from .evaluator import evaluate_answer, evaluate_retrieval, run_ragas_evaluation
from .generator import OpenAIGenerator
from .indexer import build_vectorstore
from .reporter import save_results
from .retriever import batch_retrieve


def run_pipeline(config: RagConfig) -> dict:
    """Run the full RAG pipeline with batch processing.

    Flow:
        Load → Chunk → Index → Batch Retrieve → Batch Generate
        → Save generations.json
        → Evaluate → Save evaluations.json

    Args:
        config: Experiment configuration.

    Returns:
        Results dict with config, metrics, latency, and per-query details.
    """
    dataset_name = Path(config.csv_path).stem
    out_dir = Path(config.output_dir) / dataset_name / config.embed_model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Pipeline] Dataset: {dataset_name}, Embedding: {config.embed_model}")

    # 1. Load ALL data (for indexing)
    print("[Pipeline] Loading full dataset...")
    all_docs, all_qa_pairs = load_dataset(
        config.csv_path,
        prefer_unique=config.prefer_unique,
    )
    print(f"[Pipeline] Full dataset: {len(all_docs)} documents, {len(all_qa_pairs)} QA pairs")

    # Split few-shot examples (if enabled)
    few_shot_examples = None
    if config.prompt_strategy == "few_shot":
        few_shot_examples, all_qa_pairs = split_few_shot_examples(
            all_qa_pairs,
            n_examples=config.n_few_shot,
            seed=config.sample_seed,
        )
        print(f"[Pipeline] Few-shot: {len(few_shot_examples)} examples extracted from dataset")

    # Sample QA pairs for evaluation
    qa_pairs = sample_qa_pairs(
        all_qa_pairs,
        max_samples=config.max_samples,
        seed=config.sample_seed,
    )
    print(f"[Pipeline] Sampled {len(qa_pairs)} QA pairs for evaluation")

    # 2. Chunk
    chunker = get_chunker(
        config.chunk_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    before_chunk = len(all_docs)
    docs = chunker.chunk(all_docs)
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
    llm = OpenAIGenerator(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url or None,
        few_shot_examples=few_shot_examples,
    )

    # Build generation items: question + combined retrieved context
    gen_items = []
    for ret_result in retrieval_results:
        combined_context = "\n\n".join(doc.page_content for doc in ret_result.documents)
        gen_items.append({"question": ret_result.question, "context": combined_context})

    print(f"[Pipeline] Batch generating {len(gen_items)} answers (workers={config.max_workers})...")
    gen_results = llm.batch_generate(gen_items, max_workers=config.max_workers)

    # 6. Save generations.json 
    generations = []
    for i, qa in enumerate(qa_pairs):
        ret = retrieval_results[i]
        gen = gen_results[i]
        generations.append({
            "qid": qa["qid"],
            "question": qa["question"],
            "gold_answer": qa["answer"],
            "predicted_answer": gen.text,
            "retrieved_contexts": [doc.page_content for doc in ret.documents],
            "retrieval_ms": ret.retrieval_ms,
            "generation_ms": gen.generation_ms,
            "total_ms": ret.retrieval_ms + gen.generation_ms,
            "input_tokens": gen.input_tokens,
            "output_tokens": gen.output_tokens,
        })

    gen_path = out_dir / "generations.json"
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)
    print(f"[Pipeline] Generations saved → {gen_path}  ({len(generations)} queries)")

    # 7. Evaluate
    print("[Pipeline] Evaluating...")
    per_query_eval = []

    for g in tqdm(generations, desc="Evaluating"):
        # Section 1: Generation Quality
        gen_scores = evaluate_answer(
            g["predicted_answer"], g["gold_answer"],
            include_semantic=config.include_semantic,
        )

        # Section 2: Retrieval Quality
        # Need original qa context for overlap-based retrieval metrics
        qa_context = next(
            qa["context"] for qa in qa_pairs if qa["qid"] == g["qid"]
        )
        ret_scores = evaluate_retrieval(
            g["retrieved_contexts"], qa_context, k=config.top_k,
        )

        per_query_eval.append({
            "qid": g["qid"],
            "generation_scores": gen_scores,
            "retrieval_scores": ret_scores,
        })

    # RAGAS evaluation (LLM-based, optional)
    ragas_metrics = {}
    if config.eval_faithfulness:
        print("[Pipeline] Running RAGAS evaluation (Faithfulness, FactualCorrectness, ContextPrecision, ContextRecall)...")
        ragas_data = [
            {
                "user_input": g["question"],
                "retrieved_contexts": g["retrieved_contexts"],
                "response": g["predicted_answer"],
                "reference": g["gold_answer"],
            }
            for g in generations
        ]
        # AsyncOpenAI for RAGAS evaluation
        ragas_client = AsyncOpenAI(api_key=config.llm_api_key)
        ragas_metrics = asyncio.run(run_ragas_evaluation(
            ragas_data, model="gpt-4o-mini", client=ragas_client,
        ))
        print(f"[Pipeline] RAGAS: {ragas_metrics}")

    # Aggregate metrics
    def _avg(key: str, section: str) -> float:
        values = [q[section][key] for q in per_query_eval if key in q[section]]
        return sum(values) / len(values) if values else 0.0

    gen_keys = ["exact_match", "f1", "rouge_l"]
    if config.include_semantic:
        gen_keys += ["bert_score", "semantic_sim"]

    avg_generation = {k: _avg(k, "generation_scores") for k in gen_keys}
    avg_retrieval = {k: _avg(k, "retrieval_scores")
                     for k in ["context_precision", "context_recall", "mrr", "hit_rate"]}

    avg_latency = {
        "mean_retrieval_ms": sum(g["retrieval_ms"] for g in generations) / len(generations),
        "mean_generation_ms": sum(g["generation_ms"] for g in generations) / len(generations),
        "mean_total_ms": sum(g["total_ms"] for g in generations) / len(generations),
        "total_input_tokens": sum(g["input_tokens"] for g in generations),
        "total_output_tokens": sum(g["output_tokens"] for g in generations),
    }

    # 8. Save evaluations.json (scores only)
    evaluations = {
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
        "per_query": per_query_eval,
    }

    eval_path = out_dir / "evaluations.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)
    print(f"[Pipeline] Evaluations saved -> {eval_path}")

    # Also save markdown report
    save_results(evaluations, out_dir)

    # Print summary
    print(
        f"[Pipeline] Generation: EM={avg_generation['exact_match']:.4f}  "
        f"F1={avg_generation['f1']:.4f}  "
        f"ROUGE-L={avg_generation['rouge_l']:.4f}"
    )
    print(
        f"[Pipeline] Retrieval: Precision={avg_retrieval['context_precision']:.4f}  "
        f"Recall={avg_retrieval['context_recall']:.4f}  "
        f"MRR={avg_retrieval['mrr']:.4f}  HitRate={avg_retrieval['hit_rate']:.4f}"
    )
    print(f"[Pipeline] Avg latency={avg_latency['mean_total_ms']:.0f}ms")

    return evaluations

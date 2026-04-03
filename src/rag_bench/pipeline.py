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
from .indexer import build_vectorstore, UNIFIED_DATASET_NAME
from .reporter import save_results
from .reranker import FPTReranker
from .retriever import batch_advanced_retrieve, RetrievalResult
from .retrievers import get_retriever, list_strategies
import rag_bench.retrievers.dense   # noqa: F401 — register 'dense', 'mmr'
import rag_bench.retrievers.bm25    # noqa: F401 — register 'bm25_syl', 'bm25_word'


def _build_reranker(config: RagConfig):
    """Build FPT cross-encoder reranker if enabled in config.

    Args:
        config: Experiment configuration.

    Returns:
        ``FPTReranker`` instance, or ``None`` if ``config.rerank`` is False.
    """
    if not config.rerank:
        return None
    reranker = FPTReranker(
        api_key=os.environ.get("FPT_API_KEY", ""),
        model=config.rerank_model,
    )
    print(f"[Pipeline] Reranker: {config.rerank_model}")
    return reranker

def _build_retriever(config: "RagConfig", vectorstore, docs: list):
    """Instantiate the correct retriever using the registry.

    Dispatches by ``config.search_type``:

    - ``'similarity'`` / ``'mmr'`` → :class:`~rag_bench.retrievers.dense.DenseRetriever`
    - ``'bm25_syl'`` → :class:`~rag_bench.retrievers.bm25.BM25SylRetriever`
    - ``'bm25_word'`` → :class:`~rag_bench.retrievers.bm25.BM25WordRetriever`
    - ``'hybrid'`` → legacy BM25+Dense hybrid (kept for backward compatibility)

    Args:
        config: Experiment configuration.
        vectorstore: Built ChromaDB Chroma instance.
        docs: Chunked document list (needed for BM25 index building).

    Returns:
        A :class:`~rag_bench.retrievers.base.BaseRetriever` instance.

    Raises:
        ValueError: If ``config.search_type`` is not a known strategy.
    """
    st = config.search_type

    if st in ("similarity", "mmr"):
        return get_retriever("dense", vectorstore=vectorstore, top_k=config.top_k, search_type=st)

    if st == "bm25_syl":
        return get_retriever("bm25_syl", documents=docs, top_k=config.top_k)

    if st == "bm25_word":
        return get_retriever("bm25_word", documents=docs, top_k=config.top_k)

    if st == "hybrid":
        # Legacy hybrid: BM25 (langchain-community) + Dense via RRF
        # Kept for backward compatibility; will be migrated to hybrid.py in Phase 3
        from langchain_community.retrievers import BM25Retriever as _LCBm25
        bm25 = _LCBm25.from_documents(docs, k=config.top_k)
        print(f"[Pipeline] BM25 index built ({len(docs)} docs) for hybrid search")
        return bm25   # returned as-is; pipeline handles hybrid separately below

    available = list_strategies() + ["hybrid"]
    raise ValueError(
        f"Unknown search_type: '{st}'. Available: {available}"
    )


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
    strategy_suffix = config.retrieval_strategy
    if config.search_type != "similarity":
        strategy_suffix += f"+{config.search_type}"
    if config.rerank:
        strategy_suffix += "+rerank"
    out_dir = Path(config.output_dir) / dataset_name / strategy_suffix / config.embed_model
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

    # 4. Build reranker + registry retriever
    reranker = _build_reranker(config)
    retriever = _build_retriever(config, vectorstore=vectorstore, docs=docs)
    # For legacy hybrid search, retriever is a raw BM25Retriever (not BaseRetriever)
    is_legacy_hybrid = config.search_type == "hybrid"
    bm25_retriever = retriever if is_legacy_hybrid else None

    # 5. Batch Retrieve
    questions = [qa["question"] for qa in qa_pairs]
    print(
        f"[Pipeline] Batch retrieving {len(questions)} queries "
        f"(search={config.search_type})..."
    )
    if is_legacy_hybrid:
        retrieval_results = batch_advanced_retrieve(
            vectorstore, questions, k=config.top_k,
            reranker=reranker,
            rerank_factor=config.rerank_factor,
            search_type=config.search_type,
            bm25_retriever=bm25_retriever,
        )
    else:
        import time
        def _retrieve_one(q: str) -> RetrievalResult:
            t0 = time.perf_counter()
            docs_ret = retriever.retrieve(q)
            ms = (time.perf_counter() - t0) * 1000
            return RetrievalResult(question=q, documents=docs_ret, retrieval_ms=ms)
        retrieval_results = [_retrieve_one(q) for q in questions]

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


def run_unified_pipeline(config: RagConfig, dataset_csv_paths: list[str]) -> list[dict]:
    """Run RAG pipeline with a unified index shared across all datasets.

    Builds ONE vector store from config.unified_index_csv, then evaluates each
    dataset's queries (up to config.max_samples) against it independently.
    Output directory per dataset: <output_dir>/<dataset>_unified/<embed_model>/

    Args:
        config: Experiment config. config.unified_index_csv must be set.
        dataset_csv_paths: Per-dataset CSV paths for query sampling.

    Returns:
        List of evaluation dicts, one per dataset.
    """
    assert config.unified_index_csv, "unified_index_csv must be set in config"

    # 1. Load unified CSV for indexing 
    print(f"[UnifiedPipeline] Loading unified index source: {config.unified_index_csv}")
    all_docs, _ = load_dataset(config.unified_index_csv, prefer_unique=config.prefer_unique)
    print(f"[UnifiedPipeline] {len(all_docs)} documents from unified CSV")

    # 2. Chunk 
    chunker = get_chunker(
        config.chunk_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    docs = chunker.chunk(all_docs)
    print(f"[UnifiedPipeline] Chunking ({config.chunk_strategy}): {len(all_docs)} -> {len(docs)} chunks")

    # 3. Build shared vector store (once) 
    print(f"[UnifiedPipeline] Loading embedding model: {config.embed_model}")
    embed_model = get_embed_model(config.embed_model)
    print("[UnifiedPipeline] Building unified vector store (this may take a while)...")
    vectorstore = build_vectorstore(docs, embed_model, config, UNIFIED_DATASET_NAME, config.embed_model)
    print("[UnifiedPipeline] Unified vector store ready.")

    # 4. Build reranker + registry retriever (once for all datasets)
    reranker = _build_reranker(config)
    retriever = _build_retriever(config, vectorstore=vectorstore, docs=docs)
    is_legacy_hybrid = config.search_type == "hybrid"
    bm25_retriever = retriever if is_legacy_hybrid else None

    # 5. Evaluate each dataset against shared index
    all_results = []
    for csv_path in dataset_csv_paths:
        dataset_name = Path(csv_path).stem
        strategy_suffix = config.search_type
        if config.rerank:
            strategy_suffix += "+rerank"
        out_dir = Path(config.output_dir) / f"{dataset_name}_unified" / strategy_suffix / config.embed_model
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[UnifiedPipeline] Dataset: {dataset_name}")

        _, all_qa_pairs = load_dataset(csv_path, prefer_unique=False)

        few_shot_examples = None
        if config.prompt_strategy == "few_shot":
            few_shot_examples, all_qa_pairs = split_few_shot_examples(
                all_qa_pairs, n_examples=config.n_few_shot, seed=config.sample_seed,
            )
            print(f"  Few-shot: {len(few_shot_examples)} examples")

        qa_pairs = sample_qa_pairs(all_qa_pairs, max_samples=config.max_samples, seed=config.sample_seed)
        print(f"  Sampled {len(qa_pairs)} QA pairs")

        questions = [qa["question"] for qa in qa_pairs]
        if is_legacy_hybrid:
            retrieval_results = batch_advanced_retrieve(
                vectorstore, questions, k=config.top_k,
                transformer=transformer, reranker=reranker,
                rerank_factor=config.rerank_factor,
                search_type=config.search_type,
                bm25_retriever=bm25_retriever,
            )
        else:
            import time
            def _retrieve_one(q: str) -> RetrievalResult:
                t0 = time.perf_counter()
                docs_ret = retriever.retrieve(q)
                ms = (time.perf_counter() - t0) * 1000
                return RetrievalResult(question=q, documents=docs_ret, retrieval_ms=ms)
            retrieval_results = [_retrieve_one(q) for q in questions]

        llm = OpenAIGenerator(
            model=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url or None,
            few_shot_examples=few_shot_examples,
        )
        gen_items = [
            {"question": r.question, "context": "\n\n".join(doc.page_content for doc in r.documents)}
            for r in retrieval_results
        ]
        print(f"  Generating {len(gen_items)} answers (workers={config.max_workers})...")
        gen_results = llm.batch_generate(gen_items, max_workers=config.max_workers)

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
        print(f"  Generations saved -> {gen_path}")

        per_query_eval = []
        for g in tqdm(generations, desc=f"  Evaluating {dataset_name}"):
            has_gold = bool(g["gold_answer"].strip())
            gen_scores = (
                evaluate_answer(
                    g["predicted_answer"], g["gold_answer"],
                    include_semantic=config.include_semantic,
                )
                if has_gold
                else {"exact_match": None, "f1": None, "rouge_l": None}
            )
            qa_context = next(qa["context"] for qa in qa_pairs if qa["qid"] == g["qid"])
            ret_scores = evaluate_retrieval(g["retrieved_contexts"], qa_context, k=config.top_k)
            per_query_eval.append({
                "qid": g["qid"],
                "generation_scores": gen_scores,
                "retrieval_scores": ret_scores,
            })

        ragas_metrics = {}
        if config.eval_faithfulness:
            print("  Running RAGAS evaluation...")
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
            ragas_metrics = asyncio.run(run_ragas_evaluation(
                ragas_data, model="gpt-4o-mini", client=ragas_client,
            ))

        def _avg_non_none(key: str, section: str) -> float | None:
            values = [q[section][key] for q in per_query_eval if q[section].get(key) is not None]
            return sum(values) / len(values) if values else None

        gen_keys = ["exact_match", "f1", "rouge_l"]
        if config.include_semantic:
            gen_keys += ["bert_score", "semantic_sim"]

        avg_generation = {k: _avg_non_none(k, "generation_scores") for k in gen_keys}
        avg_retrieval = {
            k: _avg_non_none(k, "retrieval_scores")
            for k in ["context_precision", "context_recall", "mrr", "hit_rate"]
        }
        avg_latency = {
            "mean_retrieval_ms": sum(g["retrieval_ms"] for g in generations) / len(generations),
            "mean_generation_ms": sum(g["generation_ms"] for g in generations) / len(generations),
            "mean_total_ms": sum(g["total_ms"] for g in generations) / len(generations),
            "total_input_tokens": sum(g["input_tokens"] for g in generations),
            "total_output_tokens": sum(g["output_tokens"] for g in generations),
        }

        evaluations = {
            "config": {
                "dataset": dataset_name,
                "index_source": "unified",
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
        print(f"  Evaluations saved -> {eval_path}")
        save_results(evaluations, out_dir)
        all_results.append(evaluations)

    print("\n[UnifiedPipeline] All datasets evaluated.")
    return all_results

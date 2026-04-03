from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from .config import RagConfig
from .embeddings.registry import list_models
from .pipeline import run_pipeline, run_unified_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese RAG Benchmark — evaluate RAG pipelines.",
    )
    parser.add_argument("--csv", help="Path to dataset CSV (required for benchmark runs)")
    parser.add_argument(
        "--unified-csv",
        default="",
        help="Path to unified CSV (e.g. data/unified_vqa.csv). "
             "Triggers unified-index mode: one shared index, all --datasets evaluated against it.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Dataset CSV paths to evaluate in unified mode (e.g. data/ALQAC.csv data/ViNewsQA.csv).",
    )
    parser.add_argument(
        "--embed-model",
        default="bge-small-en-v1.5",
        help=f"Embedding model key or 'all'. Available: {', '.join(list_models())}",
    )
    parser.add_argument(
        "--llm-model", default="gpt-4o-mini", help="LLM model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--llm-base-url", default="",
        help="Custom API base URL (empty = OpenAI default; set for FPT/other)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-K documents to retrieve"
    )
    parser.add_argument(
        "--max-samples", type=int, default=200, help="Max samples per dataset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", default="outputs/rag", help="Output directory"
    )
    parser.add_argument(
        "--chroma-dir", default="outputs/chroma", help="ChromaDB storage dir"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild index"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available embedding models"
    )
    # Chunking options
    parser.add_argument(
        "--chunk-strategy", default="recursive",
        choices=["passthrough", "recursive"],
        help="Chunking strategy: passthrough (no chunking) or recursive (256 tokens)",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in tokens")
    # Batch generation
    parser.add_argument("--max-workers", type=int, default=5, help="Max concurrent API calls")
    # Prompt strategy
    parser.add_argument(
        "--prompt-strategy", default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Prompt strategy: zero_shot or few_shot (auto-selected from dataset)",
    )
    parser.add_argument(
        "--n-few-shot", type=int, default=3,
        help="Number of few-shot examples to auto-select from dataset (default: 3)",
    )
    # Evaluation options
    parser.add_argument(
        "--semantic", action="store_true",
        help="Include BERTScore and Semantic Similarity metrics",
    )
    parser.add_argument(
        "--eval-faithfulness", action="store_true",
        help="Evaluate faithfulness using LLM-as-Judge (doubles API cost)",
    )
    parser.add_argument(
        "--judge-model", default="",
        help="Model name for LLM-as-Judge (required with --eval-faithfulness)",
    )
    # Retrieval options
    parser.add_argument(
        "--search-type", default="similarity",
        choices=["similarity", "mmr", "hybrid", "bm25_syl", "bm25_word"],
        help=(
            "Retrieval method: similarity (dense), mmr (diverse dense), "
            "hybrid (BM25+dense RRF), bm25_syl (sparse syllable), "
            "bm25_word (sparse word via underthesea)"
        ),
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Enable cross-encoder reranking (FPT bge-reranker-v2-m3) post-retrieval",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # Load .env file if it exists
    load_dotenv()

    args = parse_args(argv)

    if args.list_models:
        print("Available embedding models:")
        for m in list_models():
            print(f"  - {m}")
        return

    models = list_models() if args.embed_model == "all" else [args.embed_model]

    #  Unified index mode
    if args.unified_csv:
        dataset_paths = args.datasets or ([args.csv] if args.csv else [])
        if not dataset_paths:
            print("ERROR: --unified-csv requires dataset paths via --datasets or --csv")
            return
        for model_key in models:
            config = RagConfig.from_env(
                csv_path=dataset_paths[0],  # placeholder; unified pipeline uses unified_index_csv
                embed_model=model_key,
                llm_model=args.llm_model,
                llm_base_url=args.llm_base_url,
                top_k=args.top_k,
                max_samples=args.max_samples,
                sample_seed=args.seed,
                output_dir=args.output_dir,
                chroma_dir=args.chroma_dir,
                force_reindex=args.force,
                unified_index_csv=args.unified_csv,
                chunk_strategy=args.chunk_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                max_workers=args.max_workers,
                prompt_strategy=args.prompt_strategy,
                n_few_shot=args.n_few_shot,
                include_semantic=args.semantic,
                eval_faithfulness=args.eval_faithfulness,
                judge_model=args.judge_model,
                rerank=args.rerank,
                search_type=args.search_type,
            )
            run_unified_pipeline(config, dataset_paths)
        return

    # ── Per-dataset mode (existing behaviour) ─────────────────────────────────
    if not args.csv:
        parse_args(["--help"])
        return

    for model_key in models:
        config = RagConfig.from_env(
            csv_path=args.csv,
            embed_model=model_key,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            top_k=args.top_k,
            max_samples=args.max_samples,
            sample_seed=args.seed,
            output_dir=args.output_dir,
            chroma_dir=args.chroma_dir,
            force_reindex=args.force,
            chunk_strategy=args.chunk_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_workers=args.max_workers,
            prompt_strategy=args.prompt_strategy,
            n_few_shot=args.n_few_shot,
            include_semantic=args.semantic,
            eval_faithfulness=args.eval_faithfulness,
            judge_model=args.judge_model,
            rerank=args.rerank,
            search_type=args.search_type,
        )
        run_pipeline(config)


if __name__ == "__main__":
    main()

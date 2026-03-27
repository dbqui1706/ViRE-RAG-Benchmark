from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from .config import RagConfig
from .embeddings.registry import list_models
from .pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese RAG Benchmark — evaluate RAG pipelines.",
    )
    parser.add_argument("--csv", help="Path to dataset CSV (required for benchmark runs)")
    parser.add_argument(
        "--embed-model",
        default="bge-small-en-v1.5",
        help=f"Embedding model key or 'all'. Available: {', '.join(list_models())}",
    )
    parser.add_argument("--llm-provider", default="fpt", help="LLM provider")
    parser.add_argument(
        "--llm-model", default="Qwen3-32B", help="LLM model name"
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
        "--chroma-dir", default="outputs/rag/chroma", help="ChromaDB storage dir"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild index"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available embedding models"
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

    if not args.csv:
        parse_args(["--help"])
        return

    models = list_models() if args.embed_model == "all" else [args.embed_model]

    for model_key in models:
        config = RagConfig.from_env(
            csv_path=args.csv,
            embed_model=model_key,
            llm_model=args.llm_model,
            top_k=args.top_k,
            max_samples=args.max_samples,
            sample_seed=args.seed,
            output_dir=args.output_dir,
            chroma_dir=args.chroma_dir,
            force_reindex=args.force,
            include_semantic=args.semantic,
            eval_faithfulness=args.eval_faithfulness,
            judge_model=args.judge_model,
        )
        run_pipeline(config)


if __name__ == "__main__":
    main()

"""
RAGAS evaluation script.

Usage:
    python scripts/evaluate_ragas.py --generations outputs/rag/ViRe4MRC_v2/vietnamese-v2/generations.json
    python scripts/evaluate_ragas.py --generations outputs/rag/ViRe4MRC_v2/vietnamese-v2/generations.json --answer-relevancy
    python scripts/evaluate_ragas.py --generations outputs/rag/ViRe4MRC_v2/vietnamese-v2/generations.json --max-samples 20

Datasets:
# CSConDa 
python scripts/evaluate_ragas.py --generations outputs/rag/CSConDa/vietnamese-v2/generations.json
# UIT-ViQuAD2
python scripts/evaluate_ragas.py --generations outputs/rag/UIT-ViQuAD2/vietnamese-v2/generations.json
# ViMedAQA_v2
python scripts/evaluate_ragas.py --generations outputs/rag/ViMedAQA_v2/vietnamese-v2/generations.json
# ViNewsQA
python scripts/evaluate_ragas.py --generations outputs/rag/ViNewsQA/vietnamese-v2/generations.json
# ViRe4MRC_v2
python scripts/evaluate_ragas.py --generations outputs/rag/ViRe4MRC_v2/vietnamese-v2/generations.json
# ViRHE4QA_v2
python scripts/evaluate_ragas.py --generations outputs/rag/ViRHE4QA_v2/vietnamese-v2/generations.json
# VlogQA_2
python scripts/evaluate_ragas.py --generations outputs/rag/VlogQA_2/vietnamese-v2/generations.json
# ZaloLegalQA
python scripts/evaluate_ragas.py --generations outputs/rag/ZaloLegalQA/vietnamese-v2/generations.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from rag_bench.evaluator import run_ragas_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on existing generations.json")
    parser.add_argument(
        "--generations", required=True,
        help="Path to generations.json file",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: same dir as generations, named ragas_scores.json)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model for RAGAS evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max samples to evaluate (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--answer-relevancy", action="store_true",
        help="Include AnswerRelevancy metric (requires embeddings, slower)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (default: from OPENAI_API_KEY env var)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    gen_path = Path(args.generations)
    if not gen_path.exists():
        print(f"ERROR: File not found: {gen_path}")
        sys.exit(1)

    with open(gen_path, "r", encoding="utf-8") as f:
        generations = json.load(f)

    print(f"[Evaluate] Loaded {len(generations)} samples from {gen_path}")

    # subsample if requested 
    if args.max_samples > 0:
        generations = generations[:args.max_samples]
        print(f"[Evaluate] Using first {len(generations)} samples")

    # build RAGAS input from generations
    ragas_data = [
        {
            "user_input": g["question"],
            "retrieved_contexts": g["retrieved_contexts"],
            "response": g["predicted_answer"],
            "reference": g["gold_answer"],
        }
        for g in generations
    ]

    # setup client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    # run RAGAS evaluation
    print(f"[Evaluate] Running RAGAS evaluation...")
    print(f"  Model: {args.model}")
    print(f"  Samples: {len(ragas_data)}")
    print(f"  AnswerRelevancy: {args.answer_relevancy}")
    print()

    scores = await run_ragas_evaluation(
        per_query_data=ragas_data,
        model=args.model,
        client=client,
        include_answer_relevancy=args.answer_relevancy,
    )

    # results
    print("\n" + "=" * 55)
    print("  RAGAS Evaluation Results")
    print("=" * 55)
    for metric, score in scores.items():
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"  {metric:25s} {bar} {score:.4f}")
    print("=" * 55)

    # save results
    output_path = args.output or (gen_path.parent / "ragas_scores.json")
    output_data = {
        "source": str(gen_path),
        "model": args.model,
        "n_samples": len(ragas_data),
        "include_answer_relevancy": args.answer_relevancy,
        "scores": scores,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Evaluate] Scores saved → {output_path}")

    return scores

if __name__ == "__main__":
    asyncio.run(main())

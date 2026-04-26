#!/bin/bash
# RQ3b Post-retrieval: Cross-Encoder ReRanking
# Over-retrieves (top_k * rerank_factor), then reranks with bge-reranker-v2-m3.
# Requires FPT_API_KEY environment variable.

set -euo pipefail

vi-rag-bench \
  --unified-csv data/processed/benchmark.csv \
  --chunk-strategy sentence \
  --chunk-size 512 \
  --chunk-overlap 128 \
  --search-type hybrid_weighted \
  --embed-model multilingual-e5-large \
  --llm-model gpt-4o-mini \
  --top-k 10 \
  --max-samples 5 \
  --seed 42 \
  --max-workers 1 \
  --rerank \
  --chroma-dir outputs/rq3_advanced/chroma \
  --output-dir outputs/rq3_advanced/05-reranker

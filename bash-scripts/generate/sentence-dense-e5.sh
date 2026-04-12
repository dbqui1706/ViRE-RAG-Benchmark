#!/usr/bin/env bash
# Dense Similarity — unified corpus, 500 samples/dataset
set -euo pipefail

echo "===== Chunking: SENTENCE | Dense: E5-LARGE ===== "

vi-rag-bench \
    --unified-csv data/processed/benchmark.csv \
    --embed-model multilingual-e5-large \
    --chunk-strategy sentence \
    --chunk-size 0 --chunk-overlap 0 \
    --search-type similarity \
    --top-k 10 \
    --max-samples 500 \
    --llm-model gpt-4o-mini \
    --output-dir outputs/generation_benchmark \
    --chroma-dir outputs/generation_benchmark/chroma
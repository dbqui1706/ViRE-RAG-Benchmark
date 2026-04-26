#!/bin/bash
# RQ3 Iterative: Self-RAG
# Retrieve → evaluate → regenerate loop (max 3 iterations).
# Paper ref: Asai et al., 2023

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
  --max-samples 10 \
  --seed 42 \
  --max-workers 1 \
  --generation-strategy self_rag \
  --self-rag-max-iter 3 \
  --chroma-dir outputs/rq3_advanced/chroma \
  --output-dir outputs/rq3_advanced/08-self-rag

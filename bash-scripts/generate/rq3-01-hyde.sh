#!/bin/bash
# RQ3a Pre-retrieval: HyDE (Hypothetical Document Embeddings)
# Generates a hypothetical answer, then uses it as the retrieval query.
# Paper ref: Gao et al., 2022

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
  --max-samples 500 \
  --seed 42 \
  --max-workers 2 \
  --query-transform hyde \
  --transform-llm-model gpt-4o-mini \
  --chroma-dir outputs/rq3_advanced/chroma \
  --output-dir outputs/rq3_advanced/01-hyde

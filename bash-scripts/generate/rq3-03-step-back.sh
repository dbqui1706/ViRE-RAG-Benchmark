#!/bin/bash
# RQ3a Pre-retrieval: Step-Back Prompting
# Abstracts the query to a higher-level concept before retrieval.
# Paper ref: Zheng et al., 2024

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
  --max-workers 3 \
  --query-transform step_back \
  --transform-llm-model gpt-4o-mini \
  --chroma-dir outputs/rq3_advanced/chroma \
  --output-dir outputs/rq3_advanced/03-step-back

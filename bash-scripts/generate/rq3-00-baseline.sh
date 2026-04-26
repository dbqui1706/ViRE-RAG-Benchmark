#!/bin/bash
# RQ3 Baseline: No advanced technique
# Pipeline: Sentence(512/128) + Hybrid-Weighted(E5-Large) + GPT-4o-mini + k=10
# Baseline: (c*, r*, g) = (Sentence, Hybrid-Weighted, GPT-4o-mini)

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
  --chroma-dir outputs/rq3_advanced/chroma \
  --output-dir outputs/rq3_advanced/00-baseline

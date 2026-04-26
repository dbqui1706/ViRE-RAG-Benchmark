#!/bin/bash
# RQ3b Post-retrieval: Corrective RAG (CRAG)
# LLM grades each retrieved chunk for relevance, filters irrelevant ones.
# Paper ref: Yan et al., 2024

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
  --corrective \
  --transform-llm-model gpt-4o-mini \
  --chroma-dir outputs/chroma \
  --output-dir outputs/rq3_advanced/06-crag

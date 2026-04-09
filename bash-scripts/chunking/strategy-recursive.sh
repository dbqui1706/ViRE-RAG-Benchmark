#!/usr/bin/env bash
# Chunking Benchmark — Recursive strategy (C4)
set -euo pipefail

echo "=== Strategy: Recursive (size=512, overlap=50) ==="
python benchmark/chunking_benchmark.py \
  --strategy recursive \
  --chunk-size 512 \
  --chunk-overlap 50

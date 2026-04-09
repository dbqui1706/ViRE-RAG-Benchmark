#!/usr/bin/env bash
# Chunking Benchmark — Fixed-size strategy (C1)
set -euo pipefail

echo "=== Strategy: Fixed (chunk_size=512) ==="
python benchmark/chunking_benchmark.py \
    --strategy fixed \
    --chunk-size 512 \
    --chunk-overlap 0

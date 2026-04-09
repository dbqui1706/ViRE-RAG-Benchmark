#!/usr/bin/env bash
# Chunking Benchmark — Sentence-based strategy (C2)
set -euo pipefail

echo "=== Strategy: Sentence ==="
python benchmark/chunking_benchmark.py \
    --strategy sentence

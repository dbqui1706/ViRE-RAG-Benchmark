#!/usr/bin/env bash
# Chunking Benchmark — Semantic strategy (C5)
set -euo pipefail

echo "=== Strategy: Semantic (Vietnamese_Embedding_v2) ==="
python benchmark/chunking_benchmark.py --strategy semantic

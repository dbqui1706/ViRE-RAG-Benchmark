#!/usr/bin/env bash
# Retrieval Benchmark — Run all 5 retrieval strategies
set -euo pipefail

echo "=== Retrieval Benchmark: All Strategies ==="
echo "  Chunking: recursive (512/50)"
echo "  Embedding: multilingual-e5-large"
echo ""

python benchmark/retrieving_benchmark.py

#!/usr/bin/env bash
# Retrieval Benchmark — Sparse only (BM25 + TF-IDF)
set -euo pipefail

echo "=== Retrieval Benchmark: Sparse Strategies ==="
python benchmark/retrieving_benchmark.py --strategy R1-BM25 R2-TF-IDF

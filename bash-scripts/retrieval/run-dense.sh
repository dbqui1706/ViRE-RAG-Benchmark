#!/usr/bin/env bash
# Retrieval Benchmark — Dense only
set -euo pipefail

echo "=== Retrieval Benchmark: Dense Strategy ==="
python benchmark/retrieving_benchmark.py --strategy R3-Dense

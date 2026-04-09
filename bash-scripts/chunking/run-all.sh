#!/usr/bin/env bash
# Chunking Benchmark — Run ALL experiments (1, 2, 3)
set -euo pipefail

echo "=== Chunking Benchmark: All Experiments ==="
python benchmark/chunking_benchmark.py --experiment 1 2 3

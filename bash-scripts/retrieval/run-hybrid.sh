#!/usr/bin/env bash
# Retrieval Benchmark — Hybrid strategies (RRF + Weighted Sum)
set -euo pipefail

echo "=== Retrieval Benchmark: Hybrid Strategies ==="
python benchmark/retrieving_benchmark.py --strategy R4-Hybrid-RRF R5-Hybrid-Weighted

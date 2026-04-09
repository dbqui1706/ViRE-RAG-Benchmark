#!/usr/bin/env bash
# Chunking Benchmark — Experiment 1: Chunk Size Curve
# Configs: C4-256-50, C4-512-50, C4-1024-50 (recursive, overlap=50)
set -euo pipefail

echo "=== Chunking Benchmark: Exp 1 — Chunk Size Curve ==="
python benchmark/chunking_benchmark.py --experiment 1

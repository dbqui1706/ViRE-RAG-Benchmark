#!/usr/bin/env bash
# Chunking Benchmark — Experiment 2: Overlap Curve
# Configs: C4-512-{0,25,50,100,200} (recursive, size=512)
set -euo pipefail

echo "=== Chunking Benchmark: Exp 2 — Overlap Curve ==="
python benchmark/chunking_benchmark.py --experiment 2

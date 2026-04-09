#!/usr/bin/env bash
# Chunking Benchmark — Experiment 3: Method x Domain
# Configs: C1-512 (fixed), C2-sentence, C3-paragraph, C4-512 (recursive)
set -euo pipefail

echo "=== Chunking Benchmark: Exp 3 — Method x Domain ==="
python benchmark/chunking_benchmark.py --experiment 3

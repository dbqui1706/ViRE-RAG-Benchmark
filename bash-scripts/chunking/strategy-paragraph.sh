#!/usr/bin/env bash
# Chunking Benchmark — Paragraph-based strategy (C3)
set -euo pipefail

echo "=== Strategy: Paragraph ==="
python benchmark/chunking_benchmark.py --strategy paragraph

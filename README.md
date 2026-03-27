# ViRE: Vietnamese Information Retrieval Evaluation Toolkit (RAG Benchmark)

A lightweight, LlamaIndex-based toolkit for benchmarking Native Retrieval-Augmented Generation (RAG) performance on Vietnamese datasets.

> This repository accompanies the paper:  
> **Which Works Best for Vietnamese? A Practical Study of Information Retrieval Methods across Domains**  
> Long S. T. Nguyen, Tho T. Quan. Accepted at _EACL 2026_.

---

## Overview

The `rag_bench` module evaluates Native RAG (and eventually Advanced RAG) strategies across 10 Vietnamese QA datasets. It uses **LlamaIndex** as the core orchestrator, **ChromaDB** for persistent vector storage, and a custom wrapper for **FPT AI Marketplace LLMs** (e.g., Llama-3.3-70B-Instruct).

### Core Features

- **End-to-End Evaluation:** Evaluates both retrieval and generation stages (Exact Match, Token F1, ROUGE-L).
- **Extensible Embedding Registry:** Currently supports `bge-small-en-v1.5`, `vietnamese-v2`, `jina-v3`, `bge-m3`, and `snowflake-v2`.
- **FPT API Integration:** Built-in `CustomLLM` for routing queries to FPT's models with detailed latency and token usage tracking.
- **Reporting:** Generates detailed JSON statistics and Markdown summaries for each run, ensuring reproducibility and easy cross-model comparisons.

---

## Installation

```bash
git clone https://github.com/longstnguyen/ViRE.git
cd ViRE

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -e "."
```

---

## Setup

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
2. Add your FPT AI Marketplace API Key to `.env`:
```env
FPT_API_KEY=sk-...
FPT_BASE_URL=https://mkp-api.fptcloud.com
```

---

## Quickstart

Run a benchmark on a dataset using the CLI:

```bash
# Evaluate using the default embedding model (bge-small-en-v1.5)
vi-rag-bench --csv data/CSConDa.csv --embed-model bge-small-en-v1.5 --max-samples 200

# Evaluate using ALL registered embedding models sequentially
vi-rag-bench --csv data/CSConDa.csv --embed-model all --max-samples 200

# List available embedding models
vi-rag-bench --list-models
```

### Outputs

Results are saved to `outputs/rag/<dataset_name>/<embed_model>/`.
Each run directory contains:
- `metrics_summary.json`: Detailed evaluation metrics (EM, Token F1, ROUGE-L) and timing/cost estimations.
- `report.md`: A readable markdown summary of the benchmark run.
- `results.json`: Query-by-query breakdown of the answers and scores.

---

## Citation

```bibtex
@inproceedings{nguyen-quan-2026-works,
    title = "{Which Works Best for Vietnamese? A Practical Study of Information Retrieval Methods across Domains}",
    author = "Nguyen, Long S. T.  and
      Quan, Tho",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.110/",
    pages = "2098--2119",
    ISBN = "979-8-89176-386-9"
}
```

## References

[1] T. P. P. Do et al., "R2GQA: retriever-reader-generator question answering system to support students understanding legal regulations in higher education", _Artificial Intelligence and Law_, 2025.

[2] K. Van Nguyen et al., "New Vietnamese Corpus for Machine Reading Comprehension of Health News Articles", _ACM TALIP_, 2022.

[3] M.-N. Tran et al., "ViMedAQA: A Vietnamese Medical Abstractive Question-Answering Dataset", in _ACL 2024 Student Research Workshop_, 2024.

[4] T. Ngo et al., "VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension", in _EACL 2024_, 2024.

[5] T. P. P. Do et al., "Machine Reading Comprehension for Vietnamese Customer Reviews", in _PACLIC 37_, 2023.

[6] K. Van Nguyen et al., "A Vietnamese Dataset for Evaluating Machine Reading Comprehension", in _COLING 2020_, 2020.

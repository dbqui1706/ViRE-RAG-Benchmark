# ViRE: Vietnamese RAG Evaluation Benchmark

A toolkit for benchmarking Retrieval-Augmented Generation (RAG) pipelines on Vietnamese datasets.

> This repository accompanies the paper:  
> **Which Works Best for Vietnamese? A Practical Study of Information Retrieval Methods across Domains**  
> Long S. T. Nguyen, Tho T. Quan. Accepted at _EACL 2026_.

---

## Overview

The `rag_bench` module evaluates RAG pipelines across 9 Vietnamese QA datasets. It uses **LangChain** as the core orchestrator, **ChromaDB** for persistent vector storage, and supports any **OpenAI-compatible LLM** (OpenAI, FPT AI, etc.).

### Pipeline Flow

```
Load Dataset → Chunk → Index → Batch Retrieve → Batch Generate
  → Save generations.json
  → Evaluate → Save evaluations.json + report.md
```

### Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Generation** | Exact Match, Token F1, ROUGE-L |
| **Retrieval** | Context Precision@K, Context Recall, MRR, Hit Rate |
| **Faithfulness** (optional) | RAGAS ContextRecall, Faithfulness, FactualCorrectness |
| **Semantic** (optional) | BERTScore, Semantic Similarity |

### Embedding Models

| Key | Model |
|-----|-------|
| `bge-small-en-v1.5` | BAAI/bge-small-en-v1.5 |
| `vietnamese-v2` | AITeamVN/Vietnamese_Embedding_v2 |
| `jina-v3` | jinaai/jina-embeddings-v3 |
| `bge-m3` | BAAI/bge-m3 |

### Datasets

| Dataset | Domain | Size |
|---------|--------|------|
| ALQAC | Legal QA | 804 KB |
| CSConDa | Customer Service | 12 MB |
| UIT-ViQuAD2 | Wikipedia QA | 36 MB |
| ViMedAQA_v2 | Medical QA | 911 KB |
| ViNewsQA | News QA | 56 MB |
| ViRHE4QA_v2 | Higher Education | 2.5 MB |
| ViRe4MRC_v2 | Customer Reviews | 612 KB |
| VlogQA_2 | Spoken QA | 21 MB |
| ZaloLegalQA | Legal QA | 14 MB |

---

## Installation

```bash
git clone https://github.com/dbqui1706/ViRE-RAG-Benchmark.git
cd ViRE-RAG-Benchmark

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e "."
```

---

## Setup

Create a `.env` file in the project root:

```env
# Required: OpenAI API key (used for generation + RAGAS evaluation)
OPENAI_API_KEY=sk-proj-...

# Optional: HuggingFace token (for gated models)
HF_TOKEN=hf_...

# Optional: Custom LLM endpoint (FPT, Azure, etc.)
LLM_BASE_URL=https://mkp-api.fptcloud.com
```

---

## Usage

### Quick Start (10 samples)

```bash
vi-rag-bench \
    --csv data/ViRe4MRC_v2.csv \
    --embed-model bge-m3 \
    --llm-model gpt-4o-mini \
    --max-samples 10 \
    --top-k 3
```

### Full Run with All Options

```bash
vi-rag-bench \
    --csv data/ViRe4MRC_v2.csv \
    --embed-model bge-m3 \
    --llm-model gpt-4o-mini \
    --llm-base-url "" \
    --top-k 3 \
    --max-samples 200 \
    --seed 42 \
    --output-dir outputs/rag \
    --chroma-dir outputs/rag/chroma \
    --chunk-strategy recursive \
    --chunk-size 512 \
    --chunk-overlap 128 \
    --max-workers 5 \
    --prompt-strategy zero_shot \
    --n-few-shot 3 \
    --eval-faithfulness \
    --judge-model gpt-4o-mini \
    --force
```

### Using FPT AI Endpoint

```bash
vi-rag-bench \
    --csv data/CSConDa.csv \
    --embed-model vietnamese-v2 \
    --llm-model Qwen3-32B \
    --llm-base-url "https://mkp-api.fptcloud.com" \
    --max-samples 200
```

### Run All Embedding Models

```bash
vi-rag-bench --csv data/CSConDa.csv --embed-model all --max-samples 200
```

### List Available Models

```bash
vi-rag-bench --list-models
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | _(required)_ | Path to dataset CSV |
| `--embed-model` | `bge-small-en-v1.5` | Embedding model key or `all` |
| `--llm-model` | `gpt-4o-mini` | LLM model name |
| `--llm-base-url` | `""` | Custom API base URL (empty = OpenAI) |
| `--top-k` | `5` | Top-K documents to retrieve |
| `--max-samples` | `200` | Max QA pairs to evaluate |
| `--seed` | `42` | Random seed for sampling |
| `--output-dir` | `outputs/rag` | Output directory |
| `--chroma-dir` | `outputs/rag/chroma` | ChromaDB storage directory |
| `--chunk-strategy` | `recursive` | `passthrough` or `recursive` |
| `--chunk-size` | `256` | Chunk size in tokens |
| `--chunk-overlap` | `50` | Chunk overlap in tokens |
| `--max-workers` | `5` | Max concurrent API calls |
| `--prompt-strategy` | `zero_shot` | `zero_shot` or `few_shot` |
| `--n-few-shot` | `3` | Number of few-shot examples |
| `--semantic` | `false` | Include BERTScore + Semantic Similarity |
| `--eval-faithfulness` | `false` | Run RAGAS LLM-based evaluation |
| `--judge-model` | `""` | Model for LLM-as-Judge |
| `--force` | `false` | Force rebuild vector index |

---

## Output Structure

Results are saved to `outputs/rag/<dataset>/<embed_model>/`:

```
outputs/rag/ViRe4MRC_v2/bge-m3/
├── generations.json      # Raw data: qid, question, gold_answer, predicted_answer, retrieved_contexts
├── evaluations.json      # Scores: per-query + aggregate metrics
├── metrics_summary.json  # Full results (config + metrics + per-query)
└── report.md             # Human-readable markdown report
```

### generations.json

Saved immediately after retrieval + generation (no evaluation scores):

```json
[
  {
    "qid": "1824",
    "question": "...",
    "gold_answer": "...",
    "predicted_answer": "...",
    "retrieved_contexts": ["chunk1...", "chunk2...", "chunk3..."],
    "retrieval_ms": 191.5,
    "generation_ms": 256.7,
    "total_ms": 448.2,
    "input_tokens": 958,
    "output_tokens": 25
  }
]
```

### evaluations.json

Saved after all evaluation completes:

```json
{
  "config": { "dataset": "ViRe4MRC_v2", "embed_model": "bge-m3", "..." },
  "generation_metrics": { "exact_match": 0.02, "f1": 0.23, "rouge_l": 0.28 },
  "retrieval_metrics": { "context_precision": 0.30, "context_recall": 0.72, "mrr": 0.55, "hit_rate": 0.85 },
  "ragas_metrics": { "context_recall": 0.90, "faithfulness": 0.75, "factual_correctness": 0.68 },
  "latency": { "mean_total_ms": 350.0 },
  "per_query": [ { "qid": "1824", "generation_scores": {}, "retrieval_scores": {} } ]
}
```

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

# AGENTS.md — ViRE Project Guide

> This file helps AI coding agents (Jules, Gemini, etc.) understand the project
> conventions and work effectively with the codebase.

## Project Overview

**ViRE (Vietnamese RAG Evaluation)** is a benchmarking toolkit for evaluating
Retrieval-Augmented Generation pipelines on Vietnamese QA datasets.

- **Paper:** "Which Works Best for Vietnamese?" — EACL 2026
- **Stack:** Python 3.10+, LangChain, ChromaDB, OpenAI-compatible LLMs
- **Package:** `vi-rag-bench` (installed via `pip install -e "."`)

## Repository Structure

```
src/rag_bench/           # Main package
├── cli.py               # CLI entry point (argparse → main())
├── config.py            # RagConfig dataclass
├── pipeline.py          # run_pipeline() + run_unified_pipeline()
├── data_loader.py       # CSV loading, QA sampling, few-shot splitting
├── chunker.py           # Passthrough + RecursiveCharacterTextSplitter
├── indexer.py           # ChromaDB vector store build/load
├── retriever.py         # Similarity, MMR, Hybrid (BM25+Dense+RRF)
├── generator.py         # OpenAIGenerator (LCEL chain)
├── evaluator.py         # EM, F1, ROUGE-L, BERTScore, RAGAS
├── reporter.py          # JSON + Markdown report generation
├── reranker.py          # FPT bge-reranker-v2-m3 API client
├── timer.py             # Latency + cost tracking
├── embeddings/
│   └── registry.py      # Embedding model registry (@register decorator)
└── query_transforms/
    ├── base.py           # QueryTransformer ABC + registry
    ├── multi_query.py    # Multi-query (LLM generates variants)
    └── passthrough.py    # No-op transformer

tests/                   # Pytest test suite
data/                    # CSV datasets (DO NOT modify)
outputs/                 # Experiment results (gitignored)
scripts/                 # Utility scripts
```

## Setup & Development

```bash
# Install (editable mode)
pip install -e "."

# Install with dev dependencies
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[dev,semantic,evaluation]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/ tests/
```

## Testing Conventions

- Framework: **pytest** + **pytest-mock**
- Fixtures: defined in `tests/conftest.py`
- Mock external services (ChromaDB, OpenAI API) — never call real APIs in tests
- Test files follow `test_<module>.py` naming convention
- Use `tmp_path` fixture for temporary files
- Tests MUST be runnable without API keys or GPU

## Code Style

- **Type hints:** Required on all public function signatures
- **Docstrings:** Google-style (Args, Returns, Raises, Example)
- **Imports:** Use `from __future__ import annotations` at top of every module
- **Line length:** 100 characters max
- **Formatting:** Follow PEP 8 conventions
- **Vietnamese:** Comments in Vietnamese are acceptable; code and docstrings in English

## Architecture Patterns

### Registry Pattern
Both embedding models and query transformers use a decorator-based registry:
```python
@register("model-key")
def _factory():
    return SomeModel(...)
```
When adding new models/transformers, follow this pattern.

### LCEL Chain
The generator uses LangChain Expression Language:
```python
chain = ChatPromptTemplate | ChatOpenAI | StrOutputParser
```
Batch processing uses `chain.batch()` with `max_concurrency`.

### Data Flow
```
CSV → load_dataset() → Documents + QA pairs
  → get_chunker().chunk() → chunked Documents
  → build_vectorstore() → ChromaDB
  → batch_advanced_retrieve() → RetrievalResult[]
  → OpenAIGenerator.batch_generate() → GenerationResult[]
  → evaluate_answer() + evaluate_retrieval() → scores
  → save_results() → JSON + Markdown
```

## Environment Variables

```bash
OPENAI_API_KEY=...         # Required for generation + RAGAS
HF_TOKEN=...               # Optional: gated HuggingFace models
LLM_BASE_URL=...           # Optional: custom LLM endpoint
FPT_API_KEY=...            # Optional: FPT AI Marketplace
FPT_BASE_URL=...           # Optional: FPT endpoint
TRANSFORM_LLM_MODEL=...   # Optional: query transform LLM
```

## Important Notes

- **data/ directory** contains research datasets — DO NOT modify or delete
- **outputs/** is gitignored — experiment results live here
- ChromaDB collections use naming: `{dataset}_{model_key}` (max 63 chars)
- The `_clean()` method in generator.py strips `<think>` tags from reasoning LLMs
- ZaloLegalQA has no gold answers — handle empty `answer` column gracefully
- Few-shot examples are split from eval set to prevent data leakage

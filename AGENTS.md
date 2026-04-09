# AGENTS.md - ViRE Workspace Instructions

## Project Snapshot

ViRE (Vietnamese RAG Evaluation) benchmarks retrieval-augmented generation pipelines on Vietnamese QA datasets.

- Stack: Python 3.10+, LangChain, ChromaDB, OpenAI-compatible endpoints
- Package entry point: `vi-rag-bench`
- Core code: `src/rag_bench/`

## Build And Test

Use these commands by default:

```bash
# Install
pip install -e "."
pip install -e ".[dev]"
pip install -e ".[dev,semantic,evaluation,vietnamese]"

# Lint
ruff check src/ tests/

# Test
pytest tests/ -v
pytest tests/ -v -m "not slow"
pytest tests/ -v -m "not integration"
```

## Architecture

Primary pipeline flow:

```text
CSV -> load_dataset() -> chunk() -> build_vectorstore()
-> batch_advanced_retrieve() -> batch_generate()
-> evaluate_answer()/evaluate_retrieval() -> save_results()
```

Module boundaries in `src/rag_bench/`:

- `cli.py`: argparse entry point and mode selection
- `config.py`: `RagConfig` and environment-backed settings
- `data_loader.py`: dataset normalization, sampling, few-shot splitting
- `chunker.py`: passthrough and recursive chunking strategies
- `indexer.py`: ChromaDB build/load lifecycle
- `retrievers/`: retrieval strategy implementations and registry-based selection
- `query_transforms/`: query transform registry and implementations
- `generator.py`: LCEL generation chain and batched execution
- `evaluator.py`: generation, retrieval, semantic, and optional faithfulness metrics
- `reporter.py`: JSON/Markdown output artifacts
- `pipeline.py`: orchestration (`run_pipeline`, `run_unified_pipeline`)

## Code Style

- Type hints are required on public function signatures.
- Use Google-style docstrings for public APIs.
- Keep `from __future__ import annotations` at module top.
- Target max line length: 100.
- Keep code/docstrings in English; Vietnamese comments are acceptable.

## Conventions

- Follow decorator-based registries for extensibility (embeddings, retrievers, query transforms).
- Preserve LCEL-style generator composition (`Prompt | ChatModel | OutputParser`).
- Mock external services (LLM endpoints, vector stores) in unit tests.
- Tests should run without API keys or GPU unless explicitly marked integration.

## Project Gotchas

- Do not modify files under `data/`.
- `outputs/` is experiment output and is gitignored.
- ZaloLegalQA may have empty gold answers; handle missing/empty `answer` safely.
- Chroma collection names are constrained to `{dataset}_{model_key}` with <= 63 characters.
- Few-shot examples must be split from eval samples to avoid leakage.

## Reference Docs (Link, Do Not Duplicate)

- Setup, CLI usage, dataset overview: `README.md`
- Contribution workflow and commit/test standards: `CONTRIBUTING.md`
- Retrieval architecture details: `docs/superpowers/specs/2026-04-03-retrieval-architecture-design.md`
- Evaluation metric definitions: `docs/superpowers/specs/2026-03-27-rag-evaluation-metrics-design.md`
- Query expansion and transform design: `docs/superpowers/specs/2026-04-08-query-expansion-specification.md`
- Advanced RAG foundation context: `docs/superpowers/specs/2026-03-30-advanced-rag-foundation-phase1.md`

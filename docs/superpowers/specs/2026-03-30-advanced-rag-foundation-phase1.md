# Advanced RAG — Foundation + Phase 1: Multi-Query Expansion

## Goal

Establish a pluggable retrieval strategy framework and implement the first technique (Multi-Query Expansion) for benchmarking against the baseline.

## Background

Current pipeline uses naive `similarity_search(question, k)`. This spec adds:
1. **Foundation**: Strategy Pattern + Registry for query transformers, reranker skeleton, updated output paths
2. **Phase 1**: Multi-Query Expansion — LLM generates query variants to improve recall

## Decisions

| Item | Decision |
|------|----------|
| Architecture | Strategy Pattern + Registry (same as `embeddings/registry.py`) |
| Transform LLM | FPT Marketplace (free) via `FPT_API_KEY` / `FPT_BASE_URL` in `.env` |
| Reranker | FPT API `bge-reranker-v2-m3` (skeleton now, full impl Phase 6) |
| Output path | `outputs/rag/<dataset>/<strategy>/<embed_model>/` |

> [!WARNING]
> Output path changes from `outputs/rag/<dataset>/<embed>/` to `outputs/rag/<dataset>/<strategy>/<embed>/`.
> Existing results should be manually migrated or re-generated. 

---

## Proposed Changes

### Query Transforms Module

#### [NEW] `src/rag_bench/query_transforms/__init__.py`

Re-export registry functions.

#### [NEW] `src/rag_bench/query_transforms/base.py`

Abstract base class + registry:

```python
class QueryTransformer(ABC):
    def __init__(self, llm=None, **kwargs):
        self.llm = llm

    @abstractmethod
    def transform(self, question: str) -> list[str]:
        """Return list of queries to retrieve against."""
        ...
```

#### [NEW] `src/rag_bench/query_transforms/passthrough.py`

Baseline — returns query unchanged.

#### [NEW] `src/rag_bench/query_transforms/multi_query.py`

LLM generates N alternative phrasings. Returns `[original] + variants`.

---

### Reranker Module

#### [NEW] `src/rag_bench/reranker.py`

FPT `bge-reranker-v2-m3` client. `POST /v1/rerank` with query + documents + top_n.

---

### Core Changes

#### [MODIFY] `src/rag_bench/config.py`

Add: `retrieval_strategy`, `rerank`, `rerank_model`, `rerank_factor`, `transform_llm_model/api_key/base_url`.

#### [MODIFY] `src/rag_bench/retriever.py`

Add `advanced_retrieve()`: transform → retrieve per query → dedup → optional rerank → top-k.

#### [MODIFY] `src/rag_bench/pipeline.py`

Update output path to include strategy. Wire transformer + reranker into pipeline flow.

#### [MODIFY] `src/rag_bench/cli.py`

Add flags: `--retrieval-strategy`, `--rerank`, `--transform-llm`.

---

## Verification

1. Unit test passthrough and multi_query transformers
2. Smoke test both strategies on 1 dataset
3. Verify output paths are correct and non-conflicting

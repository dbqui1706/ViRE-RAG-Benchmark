# Foundation + Phase 1: Multi-Query Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pluggable query transformation framework + Multi-Query Expansion strategy to the RAG benchmark.
**Architecture:** Strategy Pattern with class registry (`QueryTransformer`).
**Tech Stack:** LangChain `ChatOpenAI` (FPT Marketplace), ChromaDB, `requests`.

---

### Task 1: Create query_transforms base module
**Files:**
- Create: `src/rag_bench/query_transforms/__init__.py`
- Create: `src/rag_bench/query_transforms/base.py`

- [ ] **Step 1: Write `base.py`**
Implement `QueryTransformer(ABC)` with an `abstractmethod(transform)` and a registry dict (`_REGISTRY`), `register()` decorator, `get_query_transformer()`, and `list_transformers()`.

- [ ] **Step 2: Write `__init__.py`**
Re-export `QueryTransformer`, `get_query_transformer`, `list_transformers`. Also import `passthrough` and `multi_query` to trigger registration.

---

### Task 2: Create passthrough (baseline) transformer
**Files:**
- Create: `src/rag_bench/query_transforms/passthrough.py`

- [ ] **Step 1: Write `passthrough.py`**
Implement `PassthroughTransformer` (registered as "baseline") that inherits from `QueryTransformer` and returns `[question]`.

---

### Task 3: Create multi_query transformer
**Files:**
- Create: `src/rag_bench/query_transforms/multi_query.py`

- [ ] **Step 1: Write `multi_query.py`**
Implement `MultiQueryTransformer` (registered as "multi_query"). Prompt LLM to generate `n_queries` alternative phrasings. `transform()` returns `[question] + variants`.

---

### Task 4: Create FPTReranker skeleton
**Files:**
- Create: `src/rag_bench/reranker.py`

- [ ] **Step 1: Write `reranker.py`**
Define `FPTReranker` class. Method `rerank(query, documents, top_n)` calls `POST mkp-api.fptcloud.com/v1/rerank` using `requests` and returns a list of `RerankResult(index, relevance_score)`.

---

### Task 5: Update rag_bench configuration
**Files:**
- Modify: `src/rag_bench/config.py`

- [ ] **Step 1: Add new fields to `RagConfig`**
Add: `retrieval_strategy="baseline"`, `rerank=False`, `rerank_model="bge-reranker-v2-m3"`, `rerank_factor=3`
Add: `transform_llm_model=""`, `transform_llm_api_key=""`, `transform_llm_base_url=""`.

- [ ] **Step 2: Update `from_env`**
Set `transform_llm_api_key` to `os.environ.get("FPT_API_KEY", "")`, etc.

---

### Task 6: Extend retriever for advanced retrieval
**Files:**
- Modify: `src/rag_bench/retriever.py`

- [ ] **Step 1: Implement `advanced_retrieve()`**
1. Get queries from `transformer.transform()`.
2. Determine `effective_k = k * rerank_factor` if `reranker` else `k`.
3. Call `vectorstore.similarity_search` for all queries.
4. Merge and deduplicate documents by `page_content`.
5. If `reranker`, call `reranker.rerank()`, grab top indices, keeping `top_k`.
6. Return `RetrievalResult`.

- [ ] **Step 2: Implement `batch_advanced_retrieve()`**
List comprehension wrapper around `advanced_retrieve()`.

---

### Task 7: Update pipeline
**Files:**
- Modify: `src/rag_bench/pipeline.py`

- [ ] **Step 1: Add Component Builder**
Create `_build_transform_components(config)` returning `(transformer, reranker)`. Initialize `ChatOpenAI` and `FPTReranker` appropriately.

- [ ] **Step 2: Update `run_pipeline` Output Path**
Change `out_dir` to insert `config.retrieval_strategy` (appended with `+rerank` if activated) before `config.embed_model`.

- [ ] **Step 3: Wire Retriever into `run_pipeline`**
Build components before retrieval, and call `batch_advanced_retrieve`.

- [ ] **Step 4: Update `run_unified_pipeline`**
Apply the same changes (output path and `batch_advanced_retrieve` wiring) to the unified pipeline flow.

---

### Task 8: Update CLI options
**Files:**
- Modify: `src/rag_bench/cli.py`

- [ ] **Step 1: Add CLI arguments**
Add `--retrieval-strategy`, `--rerank`, and `--transform-llm`.

- [ ] **Step 2: Map to config**
Populate these into `RagConfig.from_env()`.

---

### Task 9: Verification
- [ ] Run `vi-rag-bench --retrieval-strategy baseline`
- [ ] Run `vi-rag-bench --retrieval-strategy multi_query`
- [ ] Check output directories confirm isolation properties.

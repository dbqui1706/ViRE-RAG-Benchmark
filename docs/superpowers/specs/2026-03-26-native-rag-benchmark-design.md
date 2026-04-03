# Native RAG Benchmark — Design Spec

> **Goal:** Build a standalone RAG benchmark using **LlamaIndex** + **ChromaDB** + **FPT 8B LLM** to evaluate Native RAG performance, inference time, and cost across multiple embedding models on the 10 ViRE Vietnamese QA datasets. Results feed a survey paper.

## Scope

**In scope (Phase 1 — Native RAG):**
- LlamaIndex pipeline: load CSV → Documents → embed → ChromaDB VectorStoreIndex → QueryEngine → FPT LLM generate → evaluate → report
- Multiple embedding models as independent configurations
- 200-sample subsets per dataset, seed=42
- Metrics: answer quality (EM, F1, ROUGE-L), latency (retrieval/generation/total), cost

**Out of scope (Phase 2 — Advanced RAG, future):**
- Query rewriting, HyDE, re-ranking, iterative retrieval
- LlamaIndex natively supports these as `QueryTransform`, `Reranker` modules — easy to plug in later

---

## Architecture

```
src/rag_bench/
├── __init__.py
├── cli.py                 # CLI entry point: vi-rag-bench
├── config.py              # RagConfig dataclass
├── pipeline.py            # Orchestrates the LlamaIndex RAG flow
├── data_loader.py         # CSV → LlamaIndex Documents
├── chunker.py             # Chunking strategies (passthrough for Native RAG)
├── indexer.py             # ChromaDB VectorStoreIndex builder
├── retriever.py           # VectorIndexRetriever wrapper + timing
├── generator.py           # FPT marketplace CustomLLM for LlamaIndex
├── evaluator.py           # Answer quality metrics (EM, F1, ROUGE-L)
├── timer.py               # Latency & cost tracking
├── reporter.py            # JSON + Markdown report generation
└── embeddings/
    ├── __init__.py
    └── registry.py        # Embedding model registry → LlamaIndex embed_model
```

### Pipeline Flow

```
CSV → data_loader → list[Document]
       ↓
  chunker (passthrough)
       ↓
  VectorStoreIndex(ChromaVectorStore, embed_model)
       ↓
  index.as_query_engine(llm=FPTGenerator, similarity_top_k=5)
       ↓
  query_engine.query(question) → Response
       ↓
  evaluator.evaluate(response.response, gold_answer)
       ↓
  reporter.save(metrics, latencies, costs)
```

---

## Components

### `config.py` — Experiment Configuration

```python
@dataclass
class RagConfig:
    csv_path: str
    embed_model: str             # Registry key or HuggingFace model ID
    llm_provider: str = "fpt"
    llm_model: str = "llama-3.1-8b-instruct"
    llm_api_key: str = ""        # from env: FPT_API_KEY
    llm_base_url: str = ""       # from env: FPT_BASE_URL
    top_k: int = 5
    max_samples: int = 200
    sample_seed: int = 42
    chroma_dir: str = "outputs/rag/chroma"
    output_dir: str = "outputs/rag"
    prefer_unique: bool = True
    force_reindex: bool = False
```

### `data_loader.py` — CSV → LlamaIndex Documents

- Load CSV, normalize columns to `(qid, question, context, answer)`
- Convert each row to a `llama_index.core.Document`:
  ```python
  Document(
      text=row["context"],
      metadata={"qid": row["qid"], "source": dataset_name},
      doc_id=f"{dataset}_{row['qid']}"
  )
  ```
- Separate questions + gold answers for evaluation
- Random sample of 200 rows with `seed=42`

### `chunker.py` — Chunking Strategies

Native RAG: passthrough — each context = one Document (no splitting).

```python
class PassthroughNodeParser(NodeParser):
    """Each Document becomes one TextNode — no chunking."""
    def _parse_nodes(self, nodes, **kwargs):
        return nodes  # Documents are already atomic chunks
```

Future: `SentenceSplitter`, `SemanticSplitter` for Advanced RAG.

### `indexer.py` — ChromaDB VectorStoreIndex

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

def build_index(documents, embed_model, config):
    client = chromadb.PersistentClient(path=config.chroma_dir)
    collection = client.get_or_create_collection(f"{dataset}_{model_key}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,       # From registry
        show_progress=True,
    )
    return index
```

### `embeddings/registry.py` — Embedding Models

Maps short keys to LlamaIndex embedding objects:

| Key | Model | LlamaIndex Class |
|-----|-------|-----------------|
| `vietnamese-v2` | `AITeamVN/Vietnamese_Embedding_v2` | `HuggingFaceEmbedding` |
| `jina-v3` | `jinaai/jina-embeddings-v3` | `HuggingFaceEmbedding` |
| `bge-m3` | `BAAI/bge-m3` | `HuggingFaceEmbedding` |
| `snowflake-v2` | `Snowflake/snowflake-arctic-embed-l-v2.0` | `HuggingFaceEmbedding` |
| `default` | `BAAI/bge-small-en-v1.5` | LlamaIndex default |

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

REGISTRY = {
    "vietnamese-v2": lambda: HuggingFaceEmbedding(model_name="AITeamVN/Vietnamese_Embedding_v2"),
    "jina-v3": lambda: HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v3"),
    ...
}
```

### `generator.py` — FPT LLM as Custom LlamaIndex LLM

```python
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata

class FPTGenerator(CustomLLM):
    """FPT marketplace API as a LlamaIndex-compatible LLM."""
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model, ...)
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # POST to FPT API endpoint
        # Track: latency, input_tokens, output_tokens
        return CompletionResponse(text=answer, raw=response_json)
```

### `retriever.py` — Query with Timing

```python
def query_with_timing(query_engine, question):
    t0 = time.perf_counter()
    response = query_engine.query(question)
    total_ms = (time.perf_counter() - t0) * 1000
    return response, total_ms
```

### `evaluator.py` — Answer Quality Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | 1 if normalized generated == normalized gold |
| **Token F1** | Word-level precision/recall/F1 |
| **ROUGE-L** | Longest common subsequence F1 |

Normalization: lowercase, strip punctuation/whitespace.

### `timer.py` — Latency & Cost

```python
@dataclass
class QueryMetrics:
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
```

Aggregates: mean, median, p95, p99, total cost.

### `reporter.py` — Results Output

Per configuration:
```
outputs/rag/{dataset}/{embed_model}/
├── results.json            # Full per-query results
├── metrics_summary.json    # Aggregated metrics
└── report.md               # Human-readable summary
```

### `cli.py` — Entry Point

```bash
# Single run
vi-rag-bench --csv data/CSConDa.csv \
  --embed-model vietnamese-v2 \
  --llm-provider fpt --llm-model llama-3.1-8b-instruct \
  --top-k 5 --max-samples 200 --output-dir outputs/rag

# All embedding models × one dataset
vi-rag-bench --csv data/CSConDa.csv --embed-model all \
  --llm-provider fpt --llm-model llama-3.1-8b-instruct \
  --max-samples 200 --output-dir outputs/rag
```

---

## Environment Variables

```
FPT_API_KEY=...           # FPT marketplace API key
FPT_BASE_URL=...          # FPT API base URL
```

---

## Dependencies

```toml
[project]
dependencies = [
    "llama-index-core>=0.12",
    "llama-index-vector-stores-chroma>=0.4",
    "llama-index-embeddings-huggingface>=0.5",
    "chromadb>=1.0",
    "rouge-score>=0.1",
    "requests>=2.31",
    "pandas>=2.0",
    "tqdm>=4.67",
]
```

---

## Why LlamaIndex

- **Native RAG abstractions:** `VectorStoreIndex → QueryEngine → Response` maps directly to the paper's pipeline
- **Built-in Advanced RAG:** Phase 2 only needs adding `QueryTransform`, `Reranker` — no architecture changes
- **ChromaDB integration:** first-class via `llama-index-vector-stores-chroma`
- **Custom LLM support:** `CustomLLM` base class makes FPT integration clean

---

## Verification Plan

### Automated Tests
1. Unit test each module with mock data (10-row sample)
2. Integration test: full pipeline with `default` embedding + mock LLM
3. End-to-end: CSConDa × one embedding × FPT API (200 samples)

### Manual Verification
- Confirm generated answers are Vietnamese and contextually relevant
- Cross-check metrics against manual spot-checks
- Verify latency/cost tracking matches API response headers

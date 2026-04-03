# Native RAG Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LlamaIndex-based RAG benchmark that evaluates Native RAG performance, inference time, and cost across multiple embedding models on Vietnamese QA datasets.

**Architecture:** Standalone `src/rag_bench/` module using LlamaIndex for orchestration, ChromaDB for vector storage, and FPT marketplace 8B model for generation. Each (dataset × embedding model) pair is one configuration. CLI entry point `vi-rag-bench` drives single and batch runs.

**Tech Stack:** LlamaIndex, ChromaDB, HuggingFace Embeddings, FPT API, pandas, rouge-score, pytest

**Spec:** [2026-03-26-native-rag-benchmark-design.md](file:///d:/projects/ViRE/docs/superpowers/specs/2026-03-26-native-rag-benchmark-design.md)

---

## File Structure

```
src/rag_bench/
├── __init__.py               # Package init
├── config.py                 # RagConfig dataclass
├── data_loader.py            # CSV → LlamaIndex Documents + QA pairs
├── chunker.py                # Passthrough chunker (Native RAG)
├── embeddings/
│   ├── __init__.py
│   └── registry.py           # Embedding model registry
├── indexer.py                # ChromaDB VectorStoreIndex builder
├── retriever.py              # Query + timing wrapper
├── generator.py              # FPT CustomLLM for LlamaIndex
├── evaluator.py              # EM, F1, ROUGE-L metrics
├── timer.py                  # QueryMetrics + aggregation
├── reporter.py               # JSON + Markdown output
├── pipeline.py               # End-to-end Native RAG orchestrator
└── cli.py                    # argparse CLI entry point
tests/
├── conftest.py               # Shared fixtures (sample data, mock LLM)
├── test_config.py
├── test_data_loader.py
├── test_evaluator.py
├── test_timer.py
├── test_reporter.py
└── test_pipeline.py          # Integration test with mocks
```

---

### Task 1: Project Setup & Dependencies

**Files:**
- Modify: `d:\projects\ViRE\pyproject.toml`
- Modify: `d:\projects\ViRE\.gitignore`
- Create: `d:\projects\ViRE\.env.example`
- Create: `d:\projects\ViRE\src\rag_bench\__init__.py`

- [ ] **Step 1: Rewrite `pyproject.toml` for the RAG benchmark**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vi-rag-bench"
version = "0.1.0"
description = "Vietnamese RAG Benchmark: evaluating Native/Advanced RAG pipelines across embedding models and LLMs on Vietnamese QA datasets"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "llama-index-core>=0.12",
    "llama-index-vector-stores-chroma>=0.4",
    "llama-index-embeddings-huggingface>=0.5",
    "chromadb>=1.0",
    "rouge-score>=0.1",
    "requests>=2.31",
    "pandas>=2.0",
    "numpy>=2.0",
    "tqdm>=4.67",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.14"]

[project.scripts]
vi-rag-bench = "rag_bench.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Update `.gitignore`**

Append these lines:
```
outputs/rag/
*.chroma/
```

- [ ] **Step 3: Create `.env.example`**

```
FPT_API_KEY=your-fpt-api-key-here
FPT_BASE_URL=https://api.fpt.ai/v1
```

- [ ] **Step 4: Create package init**

```python
# src/rag_bench/__init__.py
"""Vietnamese RAG Benchmark — LlamaIndex + ChromaDB + FPT LLM."""
__version__ = "0.1.0"
```

- [ ] **Step 5: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: installs without errors

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/rag_bench/__init__.py
git commit -m "chore: setup rag_bench project with LlamaIndex dependencies"
```

---

### Task 2: Config & Timer (foundation dataclasses)

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\config.py`
- Create: `d:\projects\ViRE\src\rag_bench\timer.py`
- Create: `d:\projects\ViRE\tests\test_config.py`
- Create: `d:\projects\ViRE\tests\test_timer.py`

- [ ] **Step 1: Write failing test for `RagConfig`**

```python
# tests/test_config.py
from rag_bench.config import RagConfig

def test_config_defaults():
    cfg = RagConfig(csv_path="data/CSConDa.csv", embed_model="default")
    assert cfg.top_k == 5
    assert cfg.max_samples == 200
    assert cfg.sample_seed == 42

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("FPT_API_KEY", "test-key")
    monkeypatch.setenv("FPT_BASE_URL", "https://test.api")
    cfg = RagConfig.from_env(csv_path="data/CSConDa.csv", embed_model="default")
    assert cfg.llm_api_key == "test-key"
    assert cfg.llm_base_url == "https://test.api"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `config.py`**

```python
# src/rag_bench/config.py
"""Experiment configuration."""
from __future__ import annotations
import os
from dataclasses import dataclass, field

@dataclass
class RagConfig:
    csv_path: str
    embed_model: str
    llm_provider: str = "fpt"
    llm_model: str = "llama-3.1-8b-instruct"
    llm_api_key: str = ""
    llm_base_url: str = ""
    top_k: int = 5
    max_samples: int = 200
    sample_seed: int = 42
    chroma_dir: str = "outputs/rag/chroma"
    output_dir: str = "outputs/rag"
    prefer_unique: bool = True
    force_reindex: bool = False

    @classmethod
    def from_env(cls, **kwargs) -> RagConfig:
        kwargs.setdefault("llm_api_key", os.environ.get("FPT_API_KEY", ""))
        kwargs.setdefault("llm_base_url", os.environ.get("FPT_BASE_URL", ""))
        return cls(**kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Write failing test for `QueryMetrics`**

```python
# tests/test_timer.py
from rag_bench.timer import QueryMetrics, aggregate_metrics

def test_query_metrics():
    m = QueryMetrics(retrieval_ms=10.0, generation_ms=50.0, input_tokens=100, output_tokens=30)
    assert m.total_ms == 60.0
    assert m.estimated_cost_usd > 0

def test_aggregate_metrics():
    metrics = [
        QueryMetrics(retrieval_ms=10, generation_ms=50, input_tokens=100, output_tokens=30),
        QueryMetrics(retrieval_ms=20, generation_ms=40, input_tokens=150, output_tokens=40),
    ]
    agg = aggregate_metrics(metrics)
    assert agg["mean_total_ms"] == 65.0
    assert agg["total_queries"] == 2
```

- [ ] **Step 6: Implement `timer.py`**

```python
# src/rag_bench/timer.py
"""Latency and cost tracking."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# FPT pricing estimate (USD per 1K tokens) — update with actual pricing
INPUT_COST_PER_1K = 0.0003
OUTPUT_COST_PER_1K = 0.0006

@dataclass
class QueryMetrics:
    retrieval_ms: float
    generation_ms: float
    input_tokens: int
    output_tokens: int

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms

    @property
    def estimated_cost_usd(self) -> float:
        return (self.input_tokens / 1000 * INPUT_COST_PER_1K
                + self.output_tokens / 1000 * OUTPUT_COST_PER_1K)

def aggregate_metrics(metrics: list[QueryMetrics]) -> dict:
    totals = [m.total_ms for m in metrics]
    return {
        "total_queries": len(metrics),
        "mean_total_ms": float(np.mean(totals)),
        "median_total_ms": float(np.median(totals)),
        "p95_total_ms": float(np.percentile(totals, 95)),
        "p99_total_ms": float(np.percentile(totals, 99)),
        "mean_retrieval_ms": float(np.mean([m.retrieval_ms for m in metrics])),
        "mean_generation_ms": float(np.mean([m.generation_ms for m in metrics])),
        "total_cost_usd": sum(m.estimated_cost_usd for m in metrics),
    }
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_config.py tests/test_timer.py -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add src/rag_bench/config.py src/rag_bench/timer.py tests/test_config.py tests/test_timer.py
git commit -m "feat: add RagConfig and QueryMetrics with tests"
```

---

### Task 3: Data Loader

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\data_loader.py`
- Create: `d:\projects\ViRE\tests\test_data_loader.py`
- Create: `d:\projects\ViRE\tests\conftest.py`

- [ ] **Step 1: Create shared test fixtures**

```python
# tests/conftest.py
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_csv(tmp_path):
    """Create a small CSV matching CSConDa schema."""
    df = pd.DataFrame({
        "qid": range(1, 11),
        "question": [f"Question {i}?" for i in range(1, 11)],
        "context": [f"Context text for document {i}. " * 20 for i in range(1, 11)],
        "answer": [f"Answer {i}" for i in range(1, 11)],
    })
    path = tmp_path / "test_dataset.csv"
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 2: Write failing test for data loader**

```python
# tests/test_data_loader.py
from rag_bench.data_loader import load_and_sample

def test_load_and_sample(sample_csv):
    docs, qa_pairs = load_and_sample(sample_csv, max_samples=5, seed=42)
    assert len(docs) == 5
    assert len(qa_pairs) == 5
    assert hasattr(docs[0], "text")
    assert "qid" in docs[0].metadata
    assert "question" in qa_pairs[0]
    assert "answer" in qa_pairs[0]

def test_load_full_dataset(sample_csv):
    docs, qa_pairs = load_and_sample(sample_csv, max_samples=None, seed=42)
    assert len(docs) == 10

def test_unique_contexts(sample_csv):
    docs, _ = load_and_sample(sample_csv, max_samples=5, seed=42, prefer_unique=True)
    texts = [d.text for d in docs]
    assert len(set(texts)) == len(texts)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_data_loader.py -v`
Expected: FAIL

- [ ] **Step 4: Implement `data_loader.py`**

```python
# src/rag_bench/data_loader.py
"""Load CSV datasets and convert to LlamaIndex Documents."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from llama_index.core import Document

# Column name mappings for different dataset schemas
_COLUMN_MAP = {
    "id": "qid", "idx": "qid", "index": "qid",
    "extractive answer": "answer", "abstractive answer": "answer",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _COLUMN_MAP.get(c.lower(), c.lower()) for c in df.columns})
    if "qid" not in df.columns:
        df["qid"] = range(len(df))
    return df

def load_and_sample(
    csv_path: str | Path,
    max_samples: int | None = 200,
    seed: int = 42,
    prefer_unique: bool = True,
) -> tuple[list[Document], list[dict]]:
    """Load a CSV and return LlamaIndex Documents + QA pairs.

    Args:
        csv_path: Path to the CSV file.
        max_samples: Number of samples (None = all).
        seed: Random seed for sampling.
        prefer_unique: If True, prefer rows with unique contexts.

    Returns:
        (documents, qa_pairs) where documents have context as text
        and qa_pairs are dicts with {qid, question, answer}.
    """
    path = Path(csv_path)
    dataset_name = path.stem
    df = pd.read_csv(path, encoding="utf-8")
    df = _normalize_columns(df)

    required = {"question", "context", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if prefer_unique:
        df = df.drop_duplicates(subset=["context"])

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)

    documents = []
    qa_pairs = []
    for _, row in df.iterrows():
        doc = Document(
            text=str(row["context"]),
            metadata={"qid": str(row["qid"]), "source": dataset_name},
            doc_id=f"{dataset_name}_{row['qid']}",
        )
        documents.append(doc)
        qa_pairs.append({
            "qid": str(row["qid"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
        })

    return documents, qa_pairs
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_data_loader.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/rag_bench/data_loader.py tests/conftest.py tests/test_data_loader.py
git commit -m "feat: add data_loader with CSV normalization and sampling"
```

---

### Task 4: Evaluator (answer quality metrics)

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\evaluator.py`
- Create: `d:\projects\ViRE\tests\test_evaluator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluator.py
from rag_bench.evaluator import exact_match, token_f1, rouge_l, evaluate_answer

def test_exact_match_identical():
    assert exact_match("Hello world", "hello world") == 1.0

def test_exact_match_different():
    assert exact_match("Hello", "World") == 0.0

def test_token_f1_perfect():
    assert token_f1("the cat sat", "the cat sat") == 1.0

def test_token_f1_partial():
    f1 = token_f1("the cat", "the cat sat on mat")
    assert 0.0 < f1 < 1.0

def test_rouge_l():
    score = rouge_l("the cat sat on the mat", "the cat on the mat")
    assert 0.0 < score <= 1.0

def test_evaluate_answer():
    result = evaluate_answer("Paris", "paris is the capital")
    assert "em" in result
    assert "f1" in result
    assert "rouge_l" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `evaluator.py`**

```python
# src/rag_bench/evaluator.py
"""Answer quality metrics: EM, Token F1, ROUGE-L."""
from __future__ import annotations
import re
import string
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0

def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def rouge_l(prediction: str, gold: str) -> float:
    scores = _scorer.score(gold, prediction)
    return scores["rougeL"].fmeasure

def evaluate_answer(prediction: str, gold: str) -> dict:
    return {
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "rouge_l": rouge_l(prediction, gold),
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_evaluator.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_bench/evaluator.py tests/test_evaluator.py
git commit -m "feat: add evaluator with EM, F1, ROUGE-L metrics"
```

---

### Task 5: Embedding Registry

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\embeddings\__init__.py`
- Create: `d:\projects\ViRE\src\rag_bench\embeddings\registry.py`

- [ ] **Step 1: Implement `embeddings/__init__.py`**

```python
# src/rag_bench/embeddings/__init__.py
```

- [ ] **Step 2: Implement `registry.py`**

```python
# src/rag_bench/embeddings/registry.py
"""Embedding model registry for LlamaIndex."""
from __future__ import annotations
from typing import Callable
from llama_index.core.embeddings import BaseEmbedding

_REGISTRY: dict[str, Callable[[], BaseEmbedding]] = {}

def register(key: str):
    def decorator(factory: Callable[[], BaseEmbedding]):
        _REGISTRY[key] = factory
        return factory
    return decorator

def get_embed_model(key: str) -> BaseEmbedding:
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown embedding model '{key}'. Available: {available}")
    return _REGISTRY[key]()

def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())

# --- Registered models ---

@register("default")
def _default():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

@register("vietnamese-v2")
def _vietnamese_v2():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name="AITeamVN/Vietnamese_Embedding_v2")

@register("jina-v3")
def _jina_v3():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v3")

@register("bge-m3")
def _bge_m3():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name="BAAI/bge-m3")

@register("snowflake-v2")
def _snowflake_v2():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name="Snowflake/snowflake-arctic-embed-l-v2.0")
```

- [ ] **Step 3: Verify registry loads**

Run: `python -c "from rag_bench.embeddings.registry import list_models; print(list_models())"`
Expected: `['bge-m3', 'default', 'jina-v3', 'snowflake-v2', 'vietnamese-v2']`

- [ ] **Step 4: Commit**

```bash
git add src/rag_bench/embeddings/
git commit -m "feat: add embedding model registry with 5 models"
```

---

### Task 6: Chunker & Indexer

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\chunker.py`
- Create: `d:\projects\ViRE\src\rag_bench\indexer.py`

- [ ] **Step 1: Implement `chunker.py`**

```python
# src/rag_bench/chunker.py
"""Chunking strategies for RAG pipeline."""
from __future__ import annotations
from llama_index.core import Document

class PassthroughChunker:
    """No chunking — each Document stays as-is (Native RAG)."""
    def chunk(self, documents: list[Document]) -> list[Document]:
        return documents
```

- [ ] **Step 2: Implement `indexer.py`**

```python
# src/rag_bench/indexer.py
"""ChromaDB VectorStoreIndex management."""
from __future__ import annotations
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import RagConfig

def _collection_name(dataset: str, model_key: str) -> str:
    name = f"{dataset}_{model_key}".replace("/", "_").replace("-", "_")
    # ChromaDB collection names: 3-63 chars, alphanumeric + underscores
    return name[:63]

def build_index(
    documents: list[Document],
    embed_model: BaseEmbedding,
    config: RagConfig,
    dataset_name: str,
    model_key: str,
) -> VectorStoreIndex:
    """Build or load a ChromaDB-backed VectorStoreIndex.

    Args:
        documents: LlamaIndex Documents to index.
        embed_model: The embedding model to use.
        config: Experiment configuration.
        dataset_name: Name of the dataset (for collection naming).
        model_key: Short key for the embedding model.

    Returns:
        A VectorStoreIndex backed by ChromaDB.
    """
    chroma_path = str(Path(config.chroma_dir) / dataset_name / model_key)
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_path)
    col_name = _collection_name(dataset_name, model_key)

    if config.force_reindex:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(col_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0 and not config.force_reindex:
        # Load existing index
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model,
        )
    else:
        # Build new index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

    return index
```

- [ ] **Step 3: Commit**

```bash
git add src/rag_bench/chunker.py src/rag_bench/indexer.py
git commit -m "feat: add passthrough chunker and ChromaDB indexer"
```

---

### Task 7: FPT Generator (CustomLLM)

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\generator.py`

- [ ] **Step 1: Implement `generator.py`**

```python
# src/rag_bench/generator.py
"""FPT marketplace LLM as a LlamaIndex CustomLLM."""
from __future__ import annotations
import time
from typing import Any, Sequence

import requests
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms import ChatMessage
from pydantic import Field

from .timer import QueryMetrics

class FPTGenerator(CustomLLM):
    """FPT marketplace API wrapper compatible with LlamaIndex."""

    model: str = Field(default="llama-3.1-8b-instruct")
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.1)

    # Track last query metrics
    _last_metrics: dict = {}

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=4096,
            num_output=self.max_tokens,
            is_chat_model=True,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        t0 = time.perf_counter()

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful Vietnamese QA assistant. Answer the question based only on the provided context. Answer in Vietnamese."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        gen_ms = (time.perf_counter() - t0) * 1000

        self._last_metrics = {
            "generation_ms": gen_ms,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

        return CompletionResponse(text=answer, raw=data)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Streaming not supported for benchmarking")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return self.complete(prompt, **kwargs)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        raise NotImplementedError("Streaming not supported for benchmarking")

    def get_last_metrics(self) -> dict:
        return self._last_metrics.copy()
```

- [ ] **Step 2: Commit**

```bash
git add src/rag_bench/generator.py
git commit -m "feat: add FPT marketplace CustomLLM with timing"
```

---

### Task 8: Retriever (query + timing)

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\retriever.py`

- [ ] **Step 1: Implement `retriever.py`**

```python
# src/rag_bench/retriever.py
"""Query engine wrapper with latency tracking."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import Response

from .timer import QueryMetrics

@dataclass
class QueryResult:
    question: str
    answer: str
    source_nodes: list[Any]
    metrics: QueryMetrics

def query_with_timing(
    query_engine: Any,
    question: str,
    llm: Any,
) -> QueryResult:
    """Run a query and track retrieval + generation latency.

    Args:
        query_engine: LlamaIndex QueryEngine.
        question: The question to ask.
        llm: The FPTGenerator (to extract last_metrics).

    Returns:
        QueryResult with answer, sources, and timing.
    """
    t0 = time.perf_counter()
    response = query_engine.query(question)
    total_ms = (time.perf_counter() - t0) * 1000

    llm_metrics = llm.get_last_metrics() if hasattr(llm, "get_last_metrics") else {}
    gen_ms = llm_metrics.get("generation_ms", 0.0)
    retrieval_ms = total_ms - gen_ms

    metrics = QueryMetrics(
        retrieval_ms=max(retrieval_ms, 0.0),
        generation_ms=gen_ms,
        input_tokens=llm_metrics.get("input_tokens", 0),
        output_tokens=llm_metrics.get("output_tokens", 0),
    )

    return QueryResult(
        question=question,
        answer=str(response),
        source_nodes=response.source_nodes if hasattr(response, "source_nodes") else [],
        metrics=metrics,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/rag_bench/retriever.py
git commit -m "feat: add retriever with latency tracking"
```

---

### Task 9: Reporter (JSON + Markdown output)

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\reporter.py`
- Create: `d:\projects\ViRE\tests\test_reporter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_reporter.py
import json
from pathlib import Path
from rag_bench.reporter import save_results

def test_save_results(tmp_path):
    results = {
        "config": {"dataset": "CSConDa", "embed_model": "default"},
        "metrics": {"em": 0.5, "f1": 0.7, "rouge_l": 0.65},
        "latency": {"mean_total_ms": 120.0},
    }
    save_results(results, tmp_path)
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "report.md").exists()
    data = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert data["metrics"]["em"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `reporter.py`**

```python
# src/rag_bench/reporter.py
"""Result reporting — JSON + Markdown."""
from __future__ import annotations
import json
from pathlib import Path

def save_results(results: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Full per-query results (if present)
    if "per_query" in results:
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results["per_query"], f, indent=2, ensure_ascii=False)

    # Markdown report
    cfg = results.get("config", {})
    metrics = results.get("metrics", {})
    latency = results.get("latency", {})

    md = [
        f"# RAG Benchmark Report",
        f"",
        f"**Dataset:** {cfg.get('dataset', 'N/A')}",
        f"**Embedding:** {cfg.get('embed_model', 'N/A')}",
        f"**LLM:** {cfg.get('llm_model', 'N/A')}",
        f"**Samples:** {cfg.get('max_samples', 'N/A')}",
        f"",
        f"## Answer Quality",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
    ]
    for k, v in metrics.items():
        md.append(f"| {k.upper()} | {v:.4f} |")

    md.extend([
        f"",
        f"## Latency",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ])
    for k, v in latency.items():
        unit = "ms" if "ms" in k else ("USD" if "cost" in k else "")
        md.append(f"| {k} | {v:.2f} {unit} |")

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_reporter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_bench/reporter.py tests/test_reporter.py
git commit -m "feat: add reporter with JSON and Markdown output"
```

---

### Task 10: Pipeline (end-to-end orchestrator)

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\pipeline.py`
- Create: `d:\projects\ViRE\tests\test_pipeline.py`

- [ ] **Step 1: Implement `pipeline.py`**

```python
# src/rag_bench/pipeline.py
"""End-to-end Native RAG pipeline orchestrator."""
from __future__ import annotations
from pathlib import Path

from tqdm import tqdm

from .config import RagConfig
from .data_loader import load_and_sample
from .chunker import PassthroughChunker
from .embeddings.registry import get_embed_model
from .indexer import build_index
from .generator import FPTGenerator
from .retriever import query_with_timing
from .evaluator import evaluate_answer
from .timer import aggregate_metrics
from .reporter import save_results

def run_pipeline(config: RagConfig) -> dict:
    """Run the full Native RAG pipeline.

    Args:
        config: Experiment configuration.

    Returns:
        Results dictionary with metrics, latency, and per-query details.
    """
    dataset_name = Path(config.csv_path).stem
    print(f"[Pipeline] Dataset: {dataset_name}, Embedding: {config.embed_model}")

    # 1. Load data
    print("[Pipeline] Loading data...")
    docs, qa_pairs = load_and_sample(
        config.csv_path,
        max_samples=config.max_samples,
        seed=config.sample_seed,
        prefer_unique=config.prefer_unique,
    )
    print(f"[Pipeline] Loaded {len(docs)} documents, {len(qa_pairs)} QA pairs")

    # 2. Chunk (passthrough for Native RAG)
    chunker = PassthroughChunker()
    docs = chunker.chunk(docs)

    # 3. Get embedding model
    print(f"[Pipeline] Loading embedding model: {config.embed_model}")
    embed_model = get_embed_model(config.embed_model)

    # 4. Build/load ChromaDB index
    print("[Pipeline] Building index...")
    index = build_index(docs, embed_model, config, dataset_name, config.embed_model)

    # 5. Create LLM + query engine
    llm = FPTGenerator(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=config.top_k,
    )

    # 6. Query + evaluate
    print(f"[Pipeline] Running {len(qa_pairs)} queries...")
    per_query = []
    all_metrics_timer = []

    for qa in tqdm(qa_pairs, desc="Querying"):
        result = query_with_timing(query_engine, qa["question"], llm)
        scores = evaluate_answer(result.answer, qa["answer"])

        per_query.append({
            "qid": qa["qid"],
            "question": qa["question"],
            "gold_answer": qa["answer"],
            "predicted_answer": result.answer,
            "scores": scores,
            "retrieval_ms": result.metrics.retrieval_ms,
            "generation_ms": result.metrics.generation_ms,
            "total_ms": result.metrics.total_ms,
            "input_tokens": result.metrics.input_tokens,
            "output_tokens": result.metrics.output_tokens,
        })
        all_metrics_timer.append(result.metrics)

    # 7. Aggregate
    avg_scores = {}
    for key in ["em", "f1", "rouge_l"]:
        avg_scores[key] = sum(q["scores"][key] for q in per_query) / len(per_query)

    latency_agg = aggregate_metrics(all_metrics_timer)

    results = {
        "config": {
            "dataset": dataset_name,
            "embed_model": config.embed_model,
            "llm_model": config.llm_model,
            "top_k": config.top_k,
            "max_samples": config.max_samples,
        },
        "metrics": avg_scores,
        "latency": latency_agg,
        "per_query": per_query,
    }

    # 8. Save
    out_dir = Path(config.output_dir) / dataset_name / config.embed_model
    save_results(results, out_dir)
    print(f"[Pipeline] Results saved to {out_dir}")
    print(f"[Pipeline] EM={avg_scores['em']:.4f}  F1={avg_scores['f1']:.4f}  "
          f"ROUGE-L={avg_scores['rouge_l']:.4f}  "
          f"Avg latency={latency_agg['mean_total_ms']:.0f}ms")

    return results
```

- [ ] **Step 2: Write integration test with mock LLM**

```python
# tests/test_pipeline.py
"""Integration test for the full pipeline with a mock LLM."""
from unittest.mock import patch, MagicMock
from rag_bench.config import RagConfig
from rag_bench.pipeline import run_pipeline

def test_pipeline_with_mock(sample_csv, tmp_path):
    """Verify the pipeline runs end-to-end with a mock LLM."""
    config = RagConfig(
        csv_path=str(sample_csv),
        embed_model="default",
        llm_api_key="fake-key",
        llm_base_url="https://fake.api",
        max_samples=5,
        top_k=2,
        chroma_dir=str(tmp_path / "chroma"),
        output_dir=str(tmp_path / "output"),
    )

    # Mock the FPT API call
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Mock answer"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("rag_bench.generator.requests.post", return_value=mock_response):
        results = run_pipeline(config)

    assert results["config"]["dataset"] == "test_dataset"
    assert "em" in results["metrics"]
    assert "f1" in results["metrics"]
    assert len(results["per_query"]) == 5
    assert (tmp_path / "output" / "test_dataset" / "default" / "report.md").exists()
```

- [ ] **Step 3: Run integration test**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/rag_bench/pipeline.py tests/test_pipeline.py
git commit -m "feat: add end-to-end RAG pipeline with mock integration test"
```

---

### Task 11: CLI Entry Point

**Files:**
- Create: `d:\projects\ViRE\src\rag_bench\cli.py`

- [ ] **Step 1: Implement `cli.py`**

```python
# src/rag_bench/cli.py
"""CLI entry point for vi-rag-bench."""
from __future__ import annotations
import argparse
import sys

from .config import RagConfig
from .embeddings.registry import list_models
from .pipeline import run_pipeline

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese RAG Benchmark — evaluate RAG pipelines across embedding models and datasets.",
    )
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--embed-model", default="default",
                        help=f"Embedding model key or 'all'. Available: {', '.join(list_models())}")
    parser.add_argument("--llm-provider", default="fpt", help="LLM provider")
    parser.add_argument("--llm-model", default="llama-3.1-8b-instruct", help="LLM model name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K documents to retrieve")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs/rag", help="Output directory")
    parser.add_argument("--chroma-dir", default="outputs/rag/chroma", help="ChromaDB storage dir")
    parser.add_argument("--force", action="store_true", help="Force rebuild index")
    parser.add_argument("--list-models", action="store_true", help="List available embedding models")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_models:
        print("Available embedding models:")
        for m in list_models():
            print(f"  - {m}")
        return

    models = list_models() if args.embed_model == "all" else [args.embed_model]

    for model_key in models:
        config = RagConfig.from_env(
            csv_path=args.csv,
            embed_model=model_key,
            llm_model=args.llm_model,
            top_k=args.top_k,
            max_samples=args.max_samples,
            sample_seed=args.seed,
            output_dir=args.output_dir,
            chroma_dir=args.chroma_dir,
            force_reindex=args.force,
        )
        run_pipeline(config)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `python -m rag_bench.cli --help`
Expected: prints usage with all arguments

- [ ] **Step 3: Verify `--list-models` works**

Run: `python -m rag_bench.cli --list-models`
Expected: lists 5 embedding models

- [ ] **Step 4: Commit**

```bash
git add src/rag_bench/cli.py
git commit -m "feat: add CLI entry point with single and batch run support"
```

---

### Task 12: End-to-End Verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 2: Test real run with CSConDa + default embedding + FPT API**

Run:
```bash
vi-rag-bench --csv data/CSConDa.csv \
  --embed-model default \
  --llm-model llama-3.1-8b-instruct \
  --top-k 5 --max-samples 10 \
  --output-dir outputs/rag
```
Expected: 10 queries processed, results saved to `outputs/rag/CSConDa/default/`

- [ ] **Step 3: Inspect output files**

Check:
- `outputs/rag/CSConDa/default/metrics_summary.json` — has `metrics`, `latency`, `config`
- `outputs/rag/CSConDa/default/report.md` — human-readable table
- `outputs/rag/CSConDa/default/results.json` — per-query details

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: complete Native RAG benchmark v0.1.0"
```

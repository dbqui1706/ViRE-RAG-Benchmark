# RAG Evaluation Metrics — Design Spec

Expand `rag_bench` with comprehensive metrics covering **Retrieval Quality**, **Generation Quality**, and **Faithfulness & Hallucination**. Currently, only basic generation metrics exist (EM, F1, ROUGE-L).

---

## 1. Retrieval Quality

Evaluate how well the retriever finds relevant context before generation.

### Data Available

From `retriever.py`, `QueryResult.source_nodes` already returns the retrieved nodes (contexts). Each QA pair has a gold `context` field from the CSV. This gives us the ground truth for retrieval scoring.

### Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Context Precision@K** | `relevant_in_top_k / k` | Fraction of retrieved docs that match gold context |
| **Context Recall** | `1.0 if gold_context found in retrieved set, else 0.0` | Whether the gold answer's source context was retrieved |
| **MRR (Mean Reciprocal Rank)** | `1 / rank_of_first_relevant` | How high the first relevant doc ranks |
| **Hit Rate@K** | `1.0 if any retrieved doc contains gold context` | Binary: was the right context retrieved at all? |

### Matching Strategy

Vietnamese text matching requires fuzzy matching — exact string match won't work due to chunking, whitespace, encoding differences. Use **normalized substring containment**:

```python
def context_match(retrieved_text: str, gold_context: str, threshold: float = 0.8) -> bool:
    """Check if gold_context is substantially contained in retrieved_text."""
    # Normalize both texts (lowercase, strip punctuation, collapse whitespace)
    # Compute token overlap ratio: |intersection| / |gold_tokens|
    # Return True if ratio >= threshold
```

All retrieval metrics are added directly to `evaluator.py` (Section 2).

---

## 2. Generation Quality (Expanded)

Add semantic-level metrics beyond token overlap.

### New Metrics

| Metric | Library | Description |
|--------|---------|-------------|
| **BERTScore** | `bert-score` | Cosine similarity of contextual embeddings between prediction and gold |
| **Semantic Similarity** | `sentence-transformers` | Embedding cosine similarity using a multilingual model |

### Integration

Add to existing `evaluator.py` with lazy imports (these models are heavy):

```python
def evaluate_answer(prediction, gold, include_semantic=False) -> dict:
    scores = {"em": ..., "f1": ..., "rouge_l": ...}
    if include_semantic:
        scores["bert_score"] = compute_bert_score(prediction, gold)
        scores["semantic_sim"] = compute_semantic_similarity(prediction, gold)
    return scores
```

`include_semantic` defaults to `False` for backward compatibility — enabled via `--semantic` CLI flag.

### Model Choice for BERTScore

Use `bert-base-multilingual-cased` (default) or allow override via `--bert-model` flag. For Vietnamese-specific, `vinai/phobert-base` is an option.

---

## 3. Faithfulness & Hallucination

Uses **LLM-as-Judge** to evaluate whether the generated answer is faithful to the retrieved context and relevant to the question.

### Metrics

| Metric | What it Measures | Scoring |
|--------|-----------------|---------|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? | 0.0–1.0 (fraction of faithful claims) |
| **Answer Relevancy** | Does the answer actually address the question? | 0.0–1.0 |
| **Hallucination Score** | `1 - Faithfulness` | Higher = more hallucination |

### Prompt Design

Two structured prompts sent to the Judge LLM:

**Faithfulness Prompt:**
```
Given the following context and answer, evaluate whether each claim in the answer
is supported by the context.

Context: {retrieved_context}
Answer: {generated_answer}

For each claim, respond with SUPPORTED or UNSUPPORTED.
Then provide a faithfulness score from 0.0 to 1.0.

Respond in JSON: {"claims": [...], "score": float}
```

**Answer Relevancy Prompt:**
```
Given the question and answer, evaluate whether the answer addresses the question.

Question: {question}
Answer: {generated_answer}

Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

Respond in JSON: {"score": float, "reasoning": str}
```

### Judge Configuration

- Judge model is configurable via `--judge-model` CLI flag (default: TBD by user)
- Reuses `FPTGenerator` class — same API, different model name
- Judge calls are tracked separately in latency/cost reporting
- Judge evaluation is **optional** — enabled via `--eval-faithfulness` CLI flag (off by default since it doubles API costs)

Faithfulness evaluation is added directly to `evaluator.py` (Section 3).

---

## Architecture Summary

```
evaluator.py  ← ALL metrics in one file, organized by sections:
                Section 1: Generation Quality (EM, F1, ROUGE-L, BERTScore, Semantic Sim)
                Section 2: Retrieval Quality (Context Precision/Recall, MRR, Hit Rate)
                Section 3: Faithfulness & Hallucination (LLM-as-Judge)
pipeline.py   ← updated: calls evaluator functions, aggregates scores
reporter.py   ← updated: 3 report sections
config.py     ← updated: judge_model, eval flags
cli.py        ← updated: --semantic, --eval-faithfulness, --judge-model flags
```

### Data Flow

```
query_with_timing() → QueryResult
    ├── .source_nodes → evaluator.evaluate_retrieval(source_nodes, gold_context)
    ├── .answer       → evaluator.evaluate_answer(answer, gold, include_semantic)
    └── .answer + .source_nodes → evaluator.evaluate_faithfulness(
                                      question, answer, source_nodes, judge_llm)
```

---

## Dependencies

| Package | Purpose | New? |
|---------|---------|------|
| `bert-score` | BERTScore metric | ✅ New |
| `sentence-transformers` | Semantic similarity | ✅ New |
| Existing FPT API | Judge LLM calls | Reuse |

---

## CLI Changes

```bash
# Basic run (same as before — no new metrics)
vi-rag-bench --csv data/CSConDa.csv --embed-model bge-small-en-v1.5

# With semantic generation metrics
vi-rag-bench --csv data/CSConDa.csv --embed-model bge-small-en-v1.5 --semantic

# With faithfulness evaluation (requires Judge LLM)
vi-rag-bench --csv data/CSConDa.csv --embed-model bge-small-en-v1.5 \
  --eval-faithfulness --judge-model DeepSeek-V3.2-Speciale

# Full evaluation (all metrics)
vi-rag-bench --csv data/CSConDa.csv --embed-model bge-small-en-v1.5 \
  --semantic --eval-faithfulness --judge-model DeepSeek-V3.2-Speciale
```

---

## Verification Plan

### Automated Tests
- `test_eval_retrieval.py`: Unit tests for context matching, precision, recall, MRR, hit rate
- `test_eval_faithfulness.py`: Mock Judge LLM responses, test JSON parsing and scoring
- `test_evaluator.py`: Expand existing tests for BERTScore and semantic similarity
- `test_pipeline.py`: Update integration test with new metric flags

### Manual Verification
- Run on CSConDa with `--max-samples 5 --semantic --eval-faithfulness` and inspect outputs

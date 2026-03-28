"""RAG evaluation metrics — Generation, Retrieval, and Faithfulness."""

from __future__ import annotations

import re
import string
from typing import Any
from collections import Counter

from rouge_score import rouge_scorer
from sentence_transformers.util import cos_sim

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

# Helpers

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


# Section 1: Generation Quality

def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalized texts are identical, else 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Word-level F1 between prediction and gold."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F-measure."""
    scores = _scorer.score(gold, prediction)
    return scores["rougeL"].fmeasure


# Semantic metrics (lazy-loaded)

_bert_scorer = None
_st_model = None


def compute_bert_score(prediction: str, gold: str, model_type: str = "bert-base-multilingual-cased") -> float:
    """BERTScore F1 using contextual embeddings (lazy-loaded)."""
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(model_type=model_type, lang="vi", rescale_with_baseline=False)

    P, R, F1 = _bert_scorer.score([prediction], [gold])
    return F1.item()


def compute_semantic_similarity(prediction: str, gold: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> float:
    """Cosine similarity of sentence embeddings (lazy-loaded)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(model_name)

    embeddings = _st_model.encode([prediction, gold])
    return cos_sim(embeddings[0], embeddings[1]).item()


def evaluate_answer(prediction: str, gold: str, include_semantic: bool = False) -> dict:
    """Compute generation quality metrics."""
    scores = {
        "exact_match": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "rouge_l": rouge_l(prediction, gold),
    }
    if include_semantic:
        scores["bert_score"] = compute_bert_score(prediction, gold)
        scores["semantic_sim"] = compute_semantic_similarity(prediction, gold)
    return scores


# Section 2: Retrieval Quality

def context_overlap(retrieved_text: str, gold_context: str) -> float:
    """Compute bidirectional token overlap between retrieved text and gold context.

    Uses max of both directions to handle chunked retrieval:
    - forward:  sum(min_counts) / total_gold_tokens  (gold coverage)
    - backward: sum(min_counts) / total_ret_tokens   (chunk precision)

    Returns:
        Max overlap ratio (0.0 to 1.0)
    """
    ret_tokens = _normalize(retrieved_text).split()
    gold_tokens = _normalize(gold_context).split()
    if not gold_tokens or not ret_tokens:
        return 0.0
    common = Counter(ret_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    forward = num_common / len(gold_tokens)
    backward = num_common / len(ret_tokens)
    return max(forward, backward)


def context_match(retrieved_text: str, gold_context: str, threshold: float = 0.5) -> bool:
    """Check if gold context is substantially contained in retrieved text.

    Args:
        retrieved_text: Text from a retrieved document.
        gold_context: Gold-standard context from dataset.
        threshold: Minimum overlap ratio to consider a match (default: 0.5).

    Returns:
        True if overlap ratio >= threshold.
    """
    return context_overlap(retrieved_text, gold_context) >= threshold


def evaluate_retrieval(documents: list[Any], gold_context: str, k: int = 5) -> dict:
    """Evaluate retrieval quality by comparing retrieved documents to gold context.

    Args:
        documents: List of LangChain Documents.
        gold_context: The gold-standard context from the dataset.
        k: Top-K to evaluate against.

    Returns:
        Dict with context_precision, context_recall, mrr, hit_rate, best_overlap.
    """
    texts = []
    for doc in documents[:k]:
        if hasattr(doc, "page_content"):
            texts.append(doc.page_content)
        else:
            texts.append(str(doc))

    if not texts:
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
            "best_overlap": 0.0,
            "combined_overlap": 0.0,
        }

    # Per-chunk bidirectional overlap (Counter-based)
    overlaps = [context_overlap(text, gold_context) for text in texts]
    matches = [o >= 0.5 for o in overlaps]
 
    # Combined overlap: concatenate all chunks → multiset overlap with gold
    combined_text = " ".join(texts)
    combined_ret_counter = Counter(_normalize(combined_text).split())
    gold_counter = Counter(_normalize(gold_context).split())
    gold_total = sum(gold_counter.values())
    if gold_total > 0:
        common = combined_ret_counter & gold_counter
        combined_overlap = sum(common.values()) / gold_total
    else:
        combined_overlap = 0.0
 
    # Context Precision@K: fraction of retrieved docs that match
    context_precision = sum(matches) / len(matches)
 
    # Context Recall (continuous): token-level coverage of gold by all chunks
    context_recall = combined_overlap
 
    # MRR: 1 / rank of first relevant (1-indexed)
    mrr = 0.0
    for i, m in enumerate(matches):
        if m:
            mrr = 1.0 / (i + 1)
            break
 
    # Hit Rate@K: 1.0 if any chunk matches, else 0.0 (binary)
    hit_rate = 1.0 if any(matches) else 0.0
 
    # Best overlap: highest overlap score among individual chunks
    best_overlap = max(overlaps)
 
    return {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "mrr": mrr,
        "hit_rate": hit_rate,
        "best_overlap": best_overlap,
        "combined_overlap": combined_overlap,
    }


# Section 3: RAGAS Evaluation (LLM-based metrics)

def run_ragas_evaluation(
    per_query_data: list[dict],
    *,
    model: str = "gpt-4o-mini",
    client=None,
) -> dict:
    """Run RAGAS evaluation on per-query results.

    Args:
        per_query_data: List of dicts, each with:
            - user_input: question string
            - retrieved_contexts: list[str] of retrieved context texts
            - response: predicted answer string
            - reference: gold answer string
        model: Model name for llm_factory (default: "gpt-4o-mini").
        client: An OpenAI() or AsyncOpenAI() client instance.

    Returns:
        Dict with aggregate RAGAS scores:
            - context_recall (float)
            - faithfulness (float)
            - factual_correctness (float)
    """
    import numpy as np
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, Faithfulness, FactualCorrectness

    if client is None:
        raise ValueError("client (OpenAI client instance) is required.")

    evaluator_llm = llm_factory(model, client=client, max_tokens=2048)
    metrics = [
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
    ]

    # Each metric's ascore() accepts different kwargs, so filter per metric.
    import inspect
    scores: dict[str, float] = {}
    for metric in metrics:
        params = set(inspect.signature(metric.ascore).parameters) - {"self"}
        filtered = [{k: v for k, v in d.items() if k in params} for d in per_query_data]
        results = metric.batch_score(filtered)
        values = [r.value for r in results]
        scores[metric.name] = float(np.nanmean(values))

    return scores
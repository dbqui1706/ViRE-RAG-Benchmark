"""RAG evaluation metrics — Generation, Retrieval, and Faithfulness.
"""

from __future__ import annotations

import re
import string
from typing import Any

from rouge_score import rouge_scorer
from sentence_transformers.util import cos_sim

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F-measure."""
    scores = _scorer.score(gold, prediction)
    return scores["rougeL"].fmeasure


# --- Semantic metrics (lazy-loaded) ---

_bert_scorer = None
_st_model = None


def compute_bert_score(prediction: str, gold: str, model_type: str = "bert-base-multilingual-cased") -> float:
    """BERTScore F1 using contextual embeddings (lazy-loaded).

    Args:
        prediction: Generated answer.
        gold: Reference answer.
        model_type: HuggingFace model for BERTScore.

    Returns:
        BERTScore F1 (float).
    """
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(model_type=model_type, lang="vi", rescale_with_baseline=False)

    P, R, F1 = _bert_scorer.score([prediction], [gold])
    return F1.item()


def compute_semantic_similarity(prediction: str, gold: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> float:
    """Cosine similarity of sentence embeddings (lazy-loaded).

    Args:
        prediction: Generated answer.
        gold: Reference answer.
        model_name: Sentence-transformer model name.

    Returns:
        Cosine similarity (float, -1 to 1).
    """
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        
        _st_model = SentenceTransformer(model_name)

    embeddings = _st_model.encode([prediction, gold])
    return cos_sim(embeddings[0], embeddings[1]).item()


def evaluate_answer(prediction: str, gold: str, include_semantic: bool = False) -> dict:
    """Compute generation quality metrics.

    Args:
        prediction: Generated answer text.
        gold: Reference (gold) answer text.
        include_semantic: If True, also compute BERTScore and Semantic Similarity.

    Returns:
        Dict with metric scores.
    """
    scores = {
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "rouge_l": rouge_l(prediction, gold),
    }
    if include_semantic:
        scores["bert_score"] = compute_bert_score(prediction, gold)
        scores["semantic_sim"] = compute_semantic_similarity(prediction, gold)
    return scores


# Section 2: Retrieval Quality

def context_match(retrieved_text: str, gold_context: str, threshold: float = 0.8) -> bool:
    """Check if gold context is substantially contained in retrieved text.

    Uses normalized token overlap: |intersection(retrieved, gold)| / |gold_tokens|.

    Args:
        retrieved_text: Text from a retrieved node.
        gold_context: Gold-standard context from dataset.
        threshold: Minimum overlap ratio to consider a match.

    Returns:
        True if overlap ratio >= threshold.
    """
    ret_tokens = set(_normalize(retrieved_text).split())
    gold_tokens = set(_normalize(gold_context).split())
    if not gold_tokens:
        return False
    overlap = len(ret_tokens & gold_tokens) / len(gold_tokens)
    return overlap >= threshold


def evaluate_retrieval(source_nodes: list[Any], gold_context: str, k: int = 5) -> dict:
    """Evaluate retrieval quality by comparing retrieved nodes to gold context.

    Args:
        source_nodes: List of LlamaIndex NodeWithScore objects.
        gold_context: The gold-standard context from the dataset.
        k: Top-K to evaluate against.

    Returns:
        Dict with context_precision, context_recall, mrr, hit_rate.
    """
    # Extract text from source nodes
    texts = []
    for node in source_nodes[:k]:
        if hasattr(node, "text"):
            texts.append(node.text)
        elif hasattr(node, "node") and hasattr(node.node, "text"):
            texts.append(node.node.text)
        elif hasattr(node, "get_content"):
            texts.append(node.get_content())
        else:
            texts.append(str(node))

    if not texts:
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
        }

    # Check each retrieved node for match
    matches = [context_match(text, gold_context) for text in texts]

    # Context Precision@K: fraction of retrieved docs that match
    context_precision = sum(matches) / len(matches)

    # Context Recall: 1.0 if any match found
    context_recall = 1.0 if any(matches) else 0.0

    # MRR: 1 / rank of first relevant (1-indexed)
    mrr = 0.0
    for i, m in enumerate(matches):
        if m:
            mrr = 1.0 / (i + 1)
            break

    # Hit Rate@K: same as context_recall for single-gold
    hit_rate = context_recall

    return {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "mrr": mrr,
        "hit_rate": hit_rate,
    }


# Section 3: Faithfulness & Hallucination (LLM-as-Judge)
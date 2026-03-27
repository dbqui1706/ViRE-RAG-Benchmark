"""RAG evaluation metrics — Generation, Retrieval, and Faithfulness.

Organized in three sections:
  Section 1: Generation Quality (EM, F1, ROUGE-L, BERTScore, Semantic Sim)
  Section 2: Retrieval Quality (Context Precision/Recall, MRR, Hit Rate)
  Section 3: Faithfulness & Hallucination (LLM-as-Judge)
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

from rouge_score import rouge_scorer

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


# ===========================================================================
# Section 1: Generation Quality
# ===========================================================================


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
    from numpy import dot
    from numpy.linalg import norm
    cos_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    return float(cos_sim)


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


# ===========================================================================
# Section 2: Retrieval Quality
# ===========================================================================


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


# ===========================================================================
# Section 3: Faithfulness & Hallucination (LLM-as-Judge)
# ===========================================================================

_FAITHFULNESS_PROMPT = """Given the following context and answer, evaluate whether each claim in the answer is supported by the context.

Context:
{context}

Answer:
{answer}

Instructions:
1. List each distinct claim in the answer
2. For each claim, determine if it is SUPPORTED or UNSUPPORTED by the context
3. Calculate a faithfulness score = (number of SUPPORTED claims) / (total claims)

Respond ONLY in valid JSON format:
{{"claims": [{{"claim": "...", "verdict": "SUPPORTED"}}, {{"claim": "...", "verdict": "UNSUPPORTED"}}], "score": 0.0}}"""

_RELEVANCY_PROMPT = """Given the question and answer below, evaluate whether the answer addresses the question.

Question:
{question}

Answer:
{answer}

Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

Respond ONLY in valid JSON format:
{{"score": 0.0, "reasoning": "..."}}"""


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def evaluate_faithfulness(
    question: str,
    answer: str,
    source_nodes: list[Any],
    judge_llm: Any,
) -> dict:
    """Evaluate faithfulness and answer relevancy using LLM-as-Judge.

    Args:
        question: The original question.
        answer: The generated answer.
        source_nodes: Retrieved context nodes.
        judge_llm: An LLM instance (e.g., FPTGenerator) to use as judge.

    Returns:
        Dict with faithfulness, answer_relevancy, hallucination scores.
    """
    # Build combined context from source nodes
    contexts = []
    for node in source_nodes:
        if hasattr(node, "text"):
            contexts.append(node.text)
        elif hasattr(node, "node") and hasattr(node.node, "text"):
            contexts.append(node.node.text)
        elif hasattr(node, "get_content"):
            contexts.append(node.get_content())
        else:
            contexts.append(str(node))
    combined_context = "\n\n".join(contexts) if contexts else "(no context retrieved)"

    # --- Faithfulness ---
    faithfulness_score = 0.0
    try:
        faith_prompt = _FAITHFULNESS_PROMPT.format(
            context=combined_context, answer=answer
        )
        faith_response = judge_llm.complete(faith_prompt)
        faith_text = faith_response.text if hasattr(faith_response, "text") else str(faith_response)
        faith_data = _parse_json_response(faith_text)
        faithfulness_score = float(faith_data.get("score", 0.0))
        faithfulness_score = max(0.0, min(1.0, faithfulness_score))
    except Exception:
        faithfulness_score = 0.0

    # --- Answer Relevancy ---
    relevancy_score = 0.0
    try:
        rel_prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
        rel_response = judge_llm.complete(rel_prompt)
        rel_text = rel_response.text if hasattr(rel_response, "text") else str(rel_response)
        rel_data = _parse_json_response(rel_text)
        relevancy_score = float(rel_data.get("score", 0.0))
        relevancy_score = max(0.0, min(1.0, relevancy_score))
    except Exception:
        relevancy_score = 0.0

    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "hallucination": 1.0 - faithfulness_score,
    }

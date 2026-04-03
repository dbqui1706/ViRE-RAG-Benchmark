"""RAG evaluation metrics — Generation, Retrieval, and Faithfulness."""

from __future__ import annotations

import asyncio
import re
import string
from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel
from ragas import EvaluationDataset, SingleTurnSample, experiment
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)
from rouge_score import rouge_scorer
from sentence_transformers.util import cos_sim
from tqdm.asyncio import tqdm

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

    _P, _R, F1 = _bert_scorer.score([prediction], [gold])
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

class ExperimentResult(BaseModel):
    faithfulness: float
    factual_correctness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float | None = None


def build_eval_dataset(per_query_data: list[dict]) -> EvaluationDataset:
    samples = []
    for d in per_query_data:
        samples.append(SingleTurnSample(
            user_input=d["user_input"],
            retrieved_contexts=d["retrieved_contexts"],
            response=d["response"],
            reference=d["reference"],
        ))
    return EvaluationDataset(samples=samples)


async def run_ragas_evaluation(
    per_query_data: list[dict],
    *,
    model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    client=None,
    include_answer_relevancy: bool = False,
) -> dict:
    if client is None:
        raise ValueError("client (AsyncOpenAI client instance) is required.")

    llm = llm_factory(model, client=client, max_tokens=2048)

    # Define metrics
    faithfulness = Faithfulness(llm=llm)
    factual_correctness = FactualCorrectness(llm=llm)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    answer_relevancy = None
    if include_answer_relevancy:
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import AnswerRelevancy

        evaluator_embeddings = embedding_factory(
            "openai", model=embedding_model, client=client,
        )
        answer_relevancy = AnswerRelevancy(llm=llm, embeddings=evaluator_embeddings)

    @experiment(ExperimentResult)
    async def run_evaluation(row):
        # Base tasks — always run
        tasks = [
            faithfulness.ascore(
                user_input=row.user_input,
                response=row.response,
                retrieved_contexts=row.retrieved_contexts,
            ),
            factual_correctness.ascore(
                response=row.response,
                reference=row.reference,
            ),
            context_precision.ascore(
                user_input=row.user_input,
                reference=row.reference,
                retrieved_contexts=row.retrieved_contexts,
            ),
            context_recall.ascore(
                user_input=row.user_input,
                retrieved_contexts=row.retrieved_contexts,
                reference=row.reference,
            ),
        ]

        if answer_relevancy is not None:
            tasks.append(answer_relevancy.ascore(
                user_input=row.user_input,
                response=row.response,
            ))

        results = await asyncio.gather(*tasks)

        return ExperimentResult(
            faithfulness=results[0].value,
            factual_correctness=results[1].value,
            context_precision=results[2].value,
            context_recall=results[3].value,
            answer_relevancy=results[4].value if len(results) > 4 else None,
        )

    # Build dataset + run with bounded concurrency
    eval_dataset = build_eval_dataset(per_query_data)
    n = len(eval_dataset)
    max_concurrent = 5  # limit concurrent samples to avoid rate limits
    print(f"  [RAGAS] {n} samples, include_answer_relevancy={include_answer_relevancy}, concurrency={max_concurrent}")

    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=n, desc="  [RAGAS] Evaluating")

    async def _bounded(sample):
        async with semaphore:
            result = await run_evaluation(sample)
            pbar.update(1)
            return result

    results = await asyncio.gather(*[_bounded(s) for s in eval_dataset])
    pbar.close()

    # Aggregate scores
    scores = {}
    for field in ExperimentResult.model_fields:
        values = [getattr(r, field) for r in results if getattr(r, field) is not None]
        if values:
            scores[field] = float(np.nanmean(values))

    return scores

"""RAG evaluation metrics — Generation, Retrieval, and Faithfulness."""

from __future__ import annotations

import asyncio
import math
import re
import string
from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel
from rouge_score import rouge_scorer

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

    _, _, F1 = _bert_scorer.score([prediction], [gold])
    return F1.item()


def compute_semantic_similarity(prediction: str, gold: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> float:
    """Cosine similarity of sentence embeddings (lazy-loaded)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(model_name)

    from sentence_transformers.util import cos_sim

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

# Multi-K evaluation depths
_EVAL_K_VALUES = [1, 3, 5, 10]


def recall_at_k(matches: list[bool], k: int, n_relevant: int = 1) -> float:
    """Recall@K: fraction of relevant documents found in top-K.

    Formula: Recall@K = |relevant ∩ top-K| / R

    Args:
        matches: Boolean list — True if chunk at that rank is relevant.
        k: Evaluation depth.
        n_relevant: Total number of relevant documents (R).

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if n_relevant <= 0:
        return 0.0
    hits = sum(1 for m in matches[:k] if m)
    return hits / n_relevant


def ndcg_at_k(matches: list[bool], k: int, n_relevant: int = 1) -> float:
    """NDCG@K with binary relevance.

    IDCG is computed using the total number of relevant documents (R),
    not just those found within top-K.  This properly penalises rankings
    that miss relevant documents.

    Formula: NDCG@k = DCG@k / IDCG@k
    DCG@k  = sum_{i=1}^{k} (rel_i / log2(i+1))
    IDCG@k = sum_{i=1}^{min(R,k)} (1 / log2(i+1))

    Args:
        matches: Boolean list — True if chunk at that rank is relevant.
        k: Evaluation depth.
        n_relevant: Total number of relevant documents (R) in the corpus.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if n_relevant <= 0:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, m in enumerate(matches[:k]) if m
    )
    if dcg == 0.0:
        return 0.0
    # IDCG: ideal ranking puts min(R, k) relevant docs at top positions
    ideal_count = min(n_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg


def map_at_k(matches: list[bool], k: int, n_relevant: int = 1) -> float:
    """Average Precision@K.

    AP@K = (1 / min(R, K)) * sum_{i=1}^{K} (rel_i * Precision@i)

    Args:
        matches: Boolean list — True if chunk at that rank is relevant.
        k: Evaluation depth.
        n_relevant: Total number of relevant documents (R).

    Returns:
        AP score between 0.0 and 1.0.
    """
    if n_relevant <= 0 or k <= 0:
        return 0.0
    denom = min(n_relevant, k)
    running_hits = 0.0
    ap_sum = 0.0
    for i, m in enumerate(matches[:k]):
        if m:
            running_hits += 1.0
            ap_sum += running_hits / (i + 1)
    return ap_sum / denom


def evaluate_retrieval(
    documents: list[Any],
    gold_context_id: str,
    k: int = 5,
    n_relevant: int | None = None,
) -> dict:
    """Evaluate retrieval quality using index-based matching (aligned with ViRE).

    Relevance is determined by comparing each retrieved chunk's
    ``context_id`` metadata against ``gold_context_id``.  This avoids the
    noise of text-overlap heuristics and is consistent with the original
    ViRE evaluation methodology which uses known document indices.

    Metrics:
    - Precision@K: fraction of relevant in top-K
    - Recall@K: continuous, hits_in_topk / R
    - NDCG@K: uses R for IDCG (total relevant in corpus)
    - MAP@K: Average Precision at K, denominator min(R, K)
    - MRR@K: K-aware (0 if first relevant rank > K)
    - HitRate@K: binary indicator
    - first_relevant_rank: 1-indexed rank of first relevant doc

    Args:
        documents: List of LangChain Documents (with ``metadata["context_id"]``).
        gold_context_id: The context_id of the gold-standard context.
        k: Top-K to evaluate against.
        n_relevant: Total number of relevant chunks in the *entire* corpus
            for this query.  When provided, Recall/NDCG/MAP use this as R;
            when ``None``, R is counted from the retrieved top-K only.

    Returns:
        Dict with precision, mrr, hit_rate, first_relevant_rank,
        and recall_at_k / ndcg_at_k / map_at_k / mrr_at_k for k in {1,3,5,10}.
    """
    if not documents:
        zero_dict: dict[str, float] = {
            "precision": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
            "first_relevant_rank": float("inf"),
        }
        for k_val in _EVAL_K_VALUES:
            zero_dict[f"recall_at_{k_val}"] = 0.0
            zero_dict[f"ndcg_at_{k_val}"] = 0.0
            zero_dict[f"map_at_{k_val}"] = 0.0
            zero_dict[f"mrr_at_{k_val}"] = 0.0
        return zero_dict

    # Determine relevance by matching context_id metadata
    docs_top_k = documents[:k]
    matches: list[bool] = []
    for doc in docs_top_k:
        doc_ctx_id = None
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            doc_ctx_id = str(doc.metadata.get("context_id", ""))
        matches.append(doc_ctx_id == gold_context_id)

    # R = total relevant in corpus (if given), else fall back to top-K count
    hits_in_topk = sum(matches)
    R = n_relevant if n_relevant is not None else hits_in_topk

    # First relevant rank (1-indexed); inf if no match found
    first_rel_rank = float("inf")
    for i, m in enumerate(matches):
        if m:
            first_rel_rank = float(i + 1)
            break

    # MRR (backward-compatible): 1 / first_relevant_rank
    mrr = (1.0 / first_rel_rank) if math.isfinite(first_rel_rank) else 0.0

    # Precision@K: fraction of retrieved docs that are relevant
    precision = hits_in_topk / len(matches) if matches else 0.0

    # Hit Rate@K: 1.0 if any chunk matches, else 0.0
    hit_rate = 1.0 if hits_in_topk > 0 else 0.0

    result_dict: dict[str, float] = {
        "precision": precision,
        "mrr": mrr,
        "hit_rate": hit_rate,
        "first_relevant_rank": first_rel_rank,
    }

    # Multi-K metrics (pad matches to handle k_val > len(matches))
    for k_val in _EVAL_K_VALUES:
        padded = matches + [False] * max(0, k_val - len(matches))
        result_dict[f"recall_at_{k_val}"] = recall_at_k(padded, k_val, R)
        result_dict[f"ndcg_at_{k_val}"] = ndcg_at_k(padded, k_val, R)
        result_dict[f"map_at_{k_val}"] = map_at_k(padded, k_val, R)

        # MRR@K: K-aware — 0 if first relevant rank > K
        kk = min(k_val, len(padded))
        mrr_k = (1.0 / first_rel_rank) if (
            math.isfinite(first_rel_rank) and first_rel_rank <= kk
        ) else 0.0
        result_dict[f"mrr_at_{k_val}"] = mrr_k

    return result_dict



# Section 3: RAGAS Evaluation (LLM-based metrics)

class ExperimentResult(BaseModel):
    faithfulness: float
    factual_correctness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float | None = None


def build_eval_dataset(per_query_data: list[dict]):
    from ragas import EvaluationDataset, SingleTurnSample

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
    from ragas import experiment
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )
    from tqdm.asyncio import tqdm

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

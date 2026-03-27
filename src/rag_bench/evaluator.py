"""Answer quality metrics: EM, Token F1, ROUGE-L."""

from __future__ import annotations

import re
import string

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


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


def evaluate_answer(prediction: str, gold: str) -> dict:
    """Compute all metrics for a single prediction/gold pair."""
    return {
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "rouge_l": rouge_l(prediction, gold),
    }

"""Load CSV datasets and convert to LangChain Documents."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

# Column name mappings for different dataset schemas
_COLUMN_MAP = {
    "id": "qid",
    "idx": "qid",
    "index": "qid",
    "extractive answer": "answer",
    "abstractive answer": "answer",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard schema."""
    df = df.rename(
        columns={c: _COLUMN_MAP.get(c.lower(), c.lower()) for c in df.columns}
    )
    if "qid" not in df.columns:
        df["qid"] = range(len(df))
    return df


def load_dataset(
    csv_path: str | Path,
    prefer_unique: bool = True,
) -> tuple[list[Document], list[dict]]:
    """Load ALL data from CSV — for full indexing.

    Args:
        csv_path: Path to the CSV file.
        prefer_unique: If True, deduplicate by context.

    Returns:
        (all_documents, all_qa_pairs)
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

    documents = []
    qa_pairs = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=str(row["context"]),
            metadata={"qid": str(row["qid"]), "source": dataset_name},
        )
        documents.append(doc)
        qa_pairs.append(
            {
                "qid": str(row["qid"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "context": str(row["context"]),
            }
        )

    return documents, qa_pairs


def sample_qa_pairs(
    qa_pairs: list[dict],
    max_samples: int | None = 200,
    seed: int = 42,
) -> list[dict]:
    """Sample a subset of QA pairs for evaluation.

    Args:
        qa_pairs: Full list of QA pairs.
        max_samples: Number of samples (None = all).
        seed: Random seed.

    Returns:
        Sampled QA pairs.
    """
    if max_samples is None or len(qa_pairs) <= max_samples:
        return qa_pairs

    import random
    rng = random.Random(seed)
    return rng.sample(qa_pairs, max_samples)


def split_few_shot_examples(
    qa_pairs: list[dict],
    n_examples: int = 3,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split QA pairs into few-shot examples and evaluation set.

    Selects N diverse examples with concise answers (30-200 chars)
    from the dataset. These are removed from the eval set to avoid
    data leakage.

    Args:
        qa_pairs: Full list of QA pairs.
        n_examples: Number of few-shot examples to select.
        seed: Random seed.

    Returns:
        (few_shot_examples, remaining_qa_pairs)
    """
    import random
    rng = random.Random(seed)

    # Prefer examples with concise, informative answers
    candidates = [
        q for q in qa_pairs
        if 30 <= len(q["answer"]) <= 200 and len(q["question"]) > 15
    ]

    # Fallback to all if not enough candidates
    if len(candidates) < n_examples:
        candidates = qa_pairs

    n_examples = min(n_examples, len(candidates))
    examples = rng.sample(candidates, n_examples)
    example_qids = {ex["qid"] for ex in examples}

    # Remove few-shot examples from eval set
    remaining = [q for q in qa_pairs if q["qid"] not in example_qids]

    return examples, remaining

# Backward-compatible alias
def load_and_sample(
    csv_path: str | Path,
    max_samples: int | None = 200,
    seed: int = 42,
    prefer_unique: bool = True,
) -> tuple[list[Document], list[dict]]:
    """Load CSV and sample — backward-compatible wrapper.

    Returns:
        (all_documents, sampled_qa_pairs)
    """
    docs, qa_pairs = load_dataset(csv_path, prefer_unique=prefer_unique)
    sampled = sample_qa_pairs(qa_pairs, max_samples=max_samples, seed=seed)
    return docs, sampled

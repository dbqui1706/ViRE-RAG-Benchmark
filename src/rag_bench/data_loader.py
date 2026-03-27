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


def load_and_sample(
    csv_path: str | Path,
    max_samples: int | None = 200,
    seed: int = 42,
    prefer_unique: bool = True,
) -> tuple[list[Document], list[dict]]:
    """Load a CSV and return LangChain Documents + QA pairs.

    Args:
        csv_path: Path to the CSV file.
        max_samples: Number of samples (None = all).
        seed: Random seed for sampling.
        prefer_unique: If True, prefer rows with unique contexts.

    Returns:
        (documents, qa_pairs) where documents have context as page_content
        and qa_pairs are dicts with {qid, question, answer, context}.
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

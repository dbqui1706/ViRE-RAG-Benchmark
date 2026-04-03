"""Shared EDA utility functions for ViRE datasets.

Provides reusable analysis and plotting helpers so that per-dataset
notebooks contain minimal boilerplate.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Colour palette — accessible, publication-friendly
PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
    "#CCB974",  # olive
    "#64B5CD",  # cyan
]

DPI = 150


def set_plot_style() -> None:
    """Configure matplotlib/seaborn defaults for clean, readable charts."""
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        font_scale=1.1,
        rc={
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.figsize": (10, 5),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )
    warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and print a quick overview.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with utf-8 encoding.
    """
    path = Path(csv_path)
    df = pd.read_csv(path, encoding="utf-8")

    print(f"[Dataset] : {path.stem}")
    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {list(df.columns)}")
    print(f"   Dtypes  :")
    for col in df.columns:
        print(f"      {col:20s} {str(df[col].dtype):10s}  "
              f"(missing: {df[col].isnull().sum():,})")
    return df


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _word_count(text: str) -> int:
    """Count words by splitting on whitespace."""
    if not isinstance(text, str):
        return 0
    return len(_WHITESPACE_RE.split(text.strip()))


def _char_count(text: str) -> int:
    """Count characters (excluding leading/trailing whitespace)."""
    if not isinstance(text, str):
        return 0
    return len(text.strip())


def _sentence_count(text: str) -> int:
    """Rough sentence count by splitting on sentence-ending punctuation."""
    if not isinstance(text, str):
        return 0
    # Split on . ! ? followed by space or end-of-string
    parts = re.split(r"[.!?]+(?:\s|$)", text.strip())
    return max(len([p for p in parts if p.strip()]), 1)


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------


def basic_stats(df: pd.DataFrame, text_cols: Sequence[str]) -> pd.DataFrame:
    """Compute char/word/sentence statistics for text columns.

    Args:
        df: Dataset DataFrame.
        text_cols: Columns to analyse.

    Returns:
        Summary DataFrame with rows per column × metric.
    """
    records = []
    for col in text_cols:
        if col not in df.columns:
            continue
        series = df[col].fillna("")
        chars = series.map(_char_count)
        words = series.map(_word_count)
        sents = series.map(_sentence_count)

        for metric_name, metric_series in [
            ("char_count", chars),
            ("word_count", words),
            ("sentence_count", sents),
        ]:
            records.append({
                "column": col,
                "metric": metric_name,
                "min": int(metric_series.min()),
                "max": int(metric_series.max()),
                "mean": round(metric_series.mean(), 1),
                "median": round(metric_series.median(), 1),
                "std": round(metric_series.std(), 1),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting — distributions
# ---------------------------------------------------------------------------


def plot_length_distribution(
    df: pd.DataFrame,
    col: str,
    length_type: str = "word",
    title: str | None = None,
    ax: plt.Axes | None = None,
    color: str | None = None,
) -> plt.Axes:
    """Histogram + KDE for text length of a single column.

    Args:
        df: Dataset DataFrame.
        col: Text column name.
        length_type: "word" or "char".
        title: Optional plot title.
        ax: Matplotlib axes (created if None).
        color: Bar colour.

    Returns:
        The axes object.
    """
    counter = _word_count if length_type == "word" else _char_count
    lengths = df[col].fillna("").map(counter)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    sns.histplot(
        lengths,
        bins=50,
        kde=True,
        ax=ax,
        color=color or PALETTE[0],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel(f"{length_type.title()} count")
    ax.set_ylabel("Frequency")
    ax.set_title(title or f"{col} — {length_type} length distribution")
    ax.axvline(lengths.mean(), color=PALETTE[3], ls="--", lw=1.5,
               label=f"mean = {lengths.mean():.0f}")
    ax.axvline(lengths.median(), color=PALETTE[4], ls=":", lw=1.5,
               label=f"median = {lengths.median():.0f}")
    ax.legend(fontsize=9)
    return ax


def plot_length_boxplots(
    df: pd.DataFrame,
    cols: Sequence[str],
    length_type: str = "word",
    title: str = "Text Length Comparison",
) -> plt.Figure:
    """Side-by-side box plots comparing text lengths across columns.

    Args:
        df: Dataset DataFrame.
        cols: Text columns to compare.
        length_type: "word" or "char".
        title: Plot title.

    Returns:
        The figure object.
    """
    counter = _word_count if length_type == "word" else _char_count

    data = []
    for col in cols:
        if col not in df.columns:
            continue
        lengths = df[col].fillna("").map(counter)
        for val in lengths:
            data.append({"Column": col, f"{length_type}_count": val})

    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=plot_df,
        x="Column",
        y=f"{length_type}_count",
        palette=PALETTE,
        ax=ax,
        fliersize=2,
    )
    ax.set_title(title)
    ax.set_ylabel(f"{length_type.title()} count")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotting — word frequency
# ---------------------------------------------------------------------------

# Simple Vietnamese stopwords (high-frequency, low-information words)
_VI_STOPWORDS = frozenset(
    "và của là có được cho các một này trong không với những "
    "đã từ như đến về cũng để hay hoặc rằng thì còn nếu khi "
    "ra vào lại trên đó bị nhưng nên do sẽ mà tại vì theo "
    "bởi hơn nhiều rất nào đây vậy ở gì ai".split()
)


def plot_word_frequency(
    df: pd.DataFrame,
    col: str,
    top_n: int = 20,
    title: str | None = None,
    remove_stopwords: bool = True,
) -> plt.Figure:
    """Bar chart of the most frequent words in a text column.

    Args:
        df: Dataset DataFrame.
        col: Text column.
        top_n: Number of words to display.
        title: Plot title.
        remove_stopwords: Whether to remove Vietnamese stopwords.

    Returns:
        The figure object.
    """
    counter: Counter[str] = Counter()
    for text in df[col].fillna(""):
        words = _WHITESPACE_RE.split(text.strip().lower())
        if remove_stopwords:
            words = [w for w in words if w and w not in _VI_STOPWORDS]
        else:
            words = [w for w in words if w]
        counter.update(words)

    most_common = counter.most_common(top_n)
    words, counts = zip(*most_common) if most_common else ([], [])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(words)), counts, color=PALETTE[0], edgecolor="white")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(title or f"Top {top_n} words in '{col}'")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Duplicate analysis
# ---------------------------------------------------------------------------


def compute_duplicates(
    df: pd.DataFrame,
    col: str,
) -> dict:
    """Count exact duplicates in a text column.

    Args:
        df: Dataset DataFrame.
        col: Column to check.

    Returns:
        Dictionary with total, unique, duplicate counts and ratio.
    """
    total = len(df)
    series = df[col].fillna("")
    unique = series.nunique()
    duplicated = series.duplicated().sum()

    return {
        "column": col,
        "total_rows": total,
        "unique_values": unique,
        "duplicate_rows": int(duplicated),
        "unique_ratio": round(unique / total, 4) if total else 0,
    }


# ---------------------------------------------------------------------------
# Vocabulary & retrieval-relevant analysis
# ---------------------------------------------------------------------------


def compute_vocabulary_stats(
    df: pd.DataFrame,
    col: str,
) -> dict:
    """Compute vocabulary statistics for a text column.

    Args:
        df: Dataset DataFrame.
        col: Text column.

    Returns:
        Dictionary with vocab size, total tokens, TTR, hapax count.
    """
    all_words: list[str] = []
    for text in df[col].fillna(""):
        tokens = _WHITESPACE_RE.split(text.strip().lower())
        all_words.extend(t for t in tokens if t)

    counter = Counter(all_words)
    total_tokens = len(all_words)
    vocab_size = len(counter)
    hapax = sum(1 for count in counter.values() if count == 1)

    return {
        "column": col,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "type_token_ratio": round(vocab_size / total_tokens, 4) if total_tokens else 0,
        "hapax_legomena": hapax,
        "hapax_ratio": round(hapax / vocab_size, 4) if vocab_size else 0,
    }


def compute_overlap(
    df: pd.DataFrame,
    q_col: str,
    c_col: str,
) -> np.ndarray:
    """Compute per-row Jaccard overlap between question and context words.

    Args:
        df: Dataset DataFrame.
        q_col: Question column.
        c_col: Context column.

    Returns:
        Array of Jaccard similarity scores (0–1).
    """
    overlaps = []
    for _, row in df.iterrows():
        q_words = set(_WHITESPACE_RE.split(str(row.get(q_col, "")).strip().lower()))
        c_words = set(_WHITESPACE_RE.split(str(row.get(c_col, "")).strip().lower()))
        q_words.discard("")
        c_words.discard("")

        if not q_words or not c_words:
            overlaps.append(0.0)
        else:
            intersection = q_words & c_words
            union = q_words | c_words
            overlaps.append(len(intersection) / len(union))

    return np.array(overlaps)


def plot_overlap_distribution(
    overlaps: np.ndarray,
    title: str = "Question–Context Jaccard Overlap",
) -> plt.Figure:
    """Histogram of overlap scores.

    Args:
        overlaps: Array of Jaccard similarity scores.
        title: Plot title.

    Returns:
        The figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(overlaps, bins=50, kde=True, ax=ax,
                 color=PALETTE[2], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Jaccard Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.axvline(overlaps.mean(), color=PALETTE[3], ls="--", lw=1.5,
               label=f"mean = {overlaps.mean():.3f}")
    ax.axvline(np.median(overlaps), color=PALETTE[4], ls=":", lw=1.5,
               label=f"median = {np.median(overlaps):.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def summary_table(
    df: pd.DataFrame,
    text_cols: Sequence[str],
    dataset_name: str = "",
) -> pd.DataFrame:
    """Generate a consolidated summary DataFrame.

    Args:
        df: Dataset DataFrame.
        text_cols: Columns to include in summary.
        dataset_name: Name for the dataset.

    Returns:
        Summary DataFrame.
    """
    records = []
    for col in text_cols:
        if col not in df.columns:
            continue
        series = df[col].fillna("")
        words = series.map(_word_count)
        dup_info = compute_duplicates(df, col)

        records.append({
            "dataset": dataset_name or "—",
            "column": col,
            "total_rows": len(df),
            "unique_values": dup_info["unique_values"],
            "unique_ratio": dup_info["unique_ratio"],
            "avg_words": round(words.mean(), 1),
            "median_words": round(words.median(), 1),
            "max_words": int(words.max()),
        })

    return pd.DataFrame(records)

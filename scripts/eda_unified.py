"""EDA for the unified Vietnamese QA dataset.

Generates statistics, distributions, and visualizations saved to outputs/eda/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Config ───────────────────────────────────────────────────────────────────
INPUT_PATH = Path("data/unified_vqa.csv")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Vietnamese-friendly font fallback
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 10,
})

COLORS = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0",
          "#06D6A0", "#FFD166", "#EF476F"]


def load_data() -> pd.DataFrame:
    """Load unified dataset and add derived length columns."""
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    df["q_len"] = df["question"].astype(str).str.len()
    df["a_len"] = df["answer"].astype(str).str.len()
    df["c_len"] = df["context"].astype(str).str.len()
    df["q_words"] = df["question"].astype(str).str.split().str.len()
    df["a_words"] = df["answer"].astype(str).str.split().str.len()
    df["c_words"] = df["context"].astype(str).str.split().str.len()
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset summary statistics."""
    stats = df.groupby("dataset").agg(
        count=("question", "size"),
        q_len_mean=("q_len", "mean"),
        q_len_median=("q_len", "median"),
        a_len_mean=("a_len", "mean"),
        a_len_median=("a_len", "median"),
        c_len_mean=("c_len", "mean"),
        c_len_median=("c_len", "median"),
        q_words_mean=("q_words", "mean"),
        a_words_mean=("a_words", "mean"),
        c_words_mean=("c_words", "mean"),
    ).round(1)
    return stats


def plot_dataset_distribution(df: pd.DataFrame) -> None:
    """Bar chart: number of samples per dataset."""
    counts = df["dataset"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(counts.index, counts.values, color=COLORS[:len(counts)])
    ax.bar_label(bars, fmt="{:,.0f}", padding=5, fontsize=10)
    ax.set_xlabel("Number of QA Pairs")
    ax.set_title("Dataset Size Distribution")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_dataset_distribution.png")
    plt.close()
    print("  ✅ 01_dataset_distribution.png")


def plot_length_distributions(df: pd.DataFrame) -> None:
    """Box plots: question, answer, context word counts per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, col, title in zip(
        axes,
        ["q_words", "a_words", "c_words"],
        ["Question Length (words)", "Answer Length (words)", "Context Length (words)"],
    ):
        datasets = df["dataset"].unique()
        data_groups = [df[df["dataset"] == d][col].values for d in datasets]

        bp = ax.boxplot(data_groups, labels=datasets, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], COLORS[:len(datasets)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title)
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_length_distributions.png")
    plt.close()
    print("  ✅ 02_length_distributions.png")


def plot_length_histograms(df: pd.DataFrame) -> None:
    """Histograms: distribution of question/answer/context lengths across all data."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, col, title, xlim in zip(
        axes,
        ["q_words", "a_words", "c_words"],
        ["Question Length", "Answer Length", "Context Length"],
        [100, 200, 1500],
    ):
        data = df[col].clip(upper=xlim)
        ax.hist(data, bins=50, color="#4361EE", alpha=0.7, edgecolor="white")
        ax.axvline(data.median(), color="#F72585", linestyle="--", linewidth=2,
                   label=f"Median: {data.median():.0f}")
        ax.axvline(data.mean(), color="#06D6A0", linestyle="--", linewidth=2,
                   label=f"Mean: {data.mean():.0f}")
        ax.set_title(f"{title} (words)")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_length_histograms.png")
    plt.close()
    print("  ✅ 03_length_histograms.png")


def plot_answer_context_ratio(df: pd.DataFrame) -> None:
    """Scatter: Answer length vs Context length, colored by dataset."""
    fig, ax = plt.subplots(figsize=(12, 8))

    datasets = df["dataset"].unique()
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].sample(min(500, len(df[df["dataset"] == ds])),
                                              random_state=42)
        ax.scatter(sub["c_words"], sub["a_words"],
                   alpha=0.4, s=10, color=COLORS[i % len(COLORS)], label=ds)

    ax.set_xlabel("Context Length (words)")
    ax.set_ylabel("Answer Length (words)")
    ax.set_title("Answer vs Context Length by Dataset")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 500)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_answer_context_ratio.png")
    plt.close()
    print("  ✅ 04_answer_context_ratio.png")


def plot_cross_dataset_comparison(df: pd.DataFrame) -> None:
    """Grouped bar chart: mean word counts per dataset."""
    stats = df.groupby("dataset")[["q_words", "a_words", "c_words"]].mean()
    stats = stats.sort_values("c_words", ascending=True)

    x = np.arange(len(stats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(x - width, stats["q_words"], width, label="Question", color="#4361EE")
    ax.barh(x, stats["a_words"], width, label="Answer", color="#F72585")
    ax.barh(x + width, stats["c_words"], width, label="Context", color="#06D6A0")

    ax.set_yticks(x)
    ax.set_yticklabels(stats.index)
    ax.set_xlabel("Mean Word Count")
    ax.set_title("Mean Lengths by Dataset")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_cross_dataset_comparison.png")
    plt.close()
    print("  ✅ 05_cross_dataset_comparison.png")


def generate_markdown_report(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Write a markdown EDA report."""
    md = [
        "# Vietnamese QA Datasets — Exploratory Data Analysis",
        "",
        f"**Total QA Pairs:** {len(df):,}",
        f"**Datasets:** {df['dataset'].nunique()}",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Dataset Summary",
        "",
        stats.to_markdown(),
        "",
        "## Visualizations",
        "",
        "### 1. Dataset Size Distribution",
        "![Dataset Distribution](01_dataset_distribution.png)",
        "",
        "### 2. Length Distributions (Box Plot)",
        "![Length Distributions](02_length_distributions.png)",
        "",
        "### 3. Length Histograms",
        "![Length Histograms](03_length_histograms.png)",
        "",
        "### 4. Answer vs Context Length",
        "![Answer Context Ratio](04_answer_context_ratio.png)",
        "",
        "### 5. Cross-Dataset Comparison",
        "![Cross Dataset Comparison](05_cross_dataset_comparison.png)",
        "",
        "## Key Observations",
        "",
        "| Observation | Detail |",
        "|-------------|--------|",
        f"| Largest dataset | {df['dataset'].value_counts().index[0]} ({df['dataset'].value_counts().values[0]:,} rows) |",
        f"| Smallest dataset | {df['dataset'].value_counts().index[-1]} ({df['dataset'].value_counts().values[-1]:,} rows) |",
        f"| Longest avg context | {stats['c_words_mean'].idxmax()} ({stats['c_words_mean'].max():.0f} words) |",
        f"| Shortest avg context | {stats['c_words_mean'].idxmin()} ({stats['c_words_mean'].min():.0f} words) |",
        f"| Longest avg answer | {stats['a_words_mean'].idxmax()} ({stats['a_words_mean'].max():.0f} words) |",
        f"| Shortest avg answer | {stats['a_words_mean'].idxmin()} ({stats['a_words_mean'].min():.0f} words) |",
        "",
        "> **Note:** ZaloLegalQA excluded (no answer column).",
        "",
    ]

    report_path = OUTPUT_DIR / "eda_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  ✅ EDA report: {report_path}")


def main() -> None:
    print("Running EDA on unified Vietnamese QA dataset...")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}\n")

    df = load_data()
    print(f"Loaded {len(df):,} rows from {df['dataset'].nunique()} datasets\n")

    stats = summary_stats(df)
    print("Per-dataset statistics:")
    print(stats.to_string())
    print()

    # Save stats CSV
    stats.to_csv(OUTPUT_DIR / "summary_stats.csv")
    print("  ✅ summary_stats.csv")

    # Generate plots
    print("\nGenerating plots...")
    plot_dataset_distribution(df)
    plot_length_distributions(df)
    plot_length_histograms(df)
    plot_answer_context_ratio(df)
    plot_cross_dataset_comparison(df)

    # Generate report
    print("\nGenerating report...")
    generate_markdown_report(df, stats)

    print("\n✅ EDA complete!")


if __name__ == "__main__":
    main()

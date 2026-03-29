"""Consolidate all Vietnamese QA datasets into one unified CSV.

Output schema: dataset, question, answer, context
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "unified_vqa.csv"

# ── Column mappings per dataset ──────────────────────────────────────────────
# Each entry: { "answer_col": str, "extra_drops": list[str] }
# For most datasets, the answer column is "answer".

DATASET_CONFIG: dict[str, dict] = {
    "ALQAC": {},
    "CSConDa": {},
    "UIT-ViQuAD2": {},
    "ViMedAQA_v2": {},
    "ViRe4MRC_v2": {},
    "ViNewsQA": {},
    "VlogQA_2": {},
    "ViRHE4QA_v2": {"answer_col": "extractive answer"},
    "ZaloLegalQA": {"answer_col": None},  # No answer column -> fill with ""
}


def load_dataset(name: str, config: dict) -> pd.DataFrame:
    """Load a single CSV and normalize to (dataset, question, answer, context)."""
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path, encoding="utf-8")

    # Normalize column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename / fill answer column
    answer_col = config.get("answer_col", "answer")
    if answer_col is None:
        # Dataset has no answer column — fill with empty string
        df["answer"] = ""
    elif answer_col.lower() != "answer":
        df = df.rename(columns={answer_col.lower(): "answer"})

    # Keep only required columns
    required = ["question", "answer", "context"]
    missing = set(required) - set(df.columns)
    if missing:
        print(f"  ⚠️  {name}: missing columns {missing}, skipping.")
        return pd.DataFrame()

    df = df[required].copy()
    df["dataset"] = name

    # Drop rows with NaN in question/context only — allow empty answer
    before = len(df)
    df = df.dropna(subset=["question", "context"])
    after = len(df)
    if before != after:
        print(f"  ℹ️  {name}: dropped {before - after} rows with NaN in question/context")

    # Fill remaining NaN answers with empty string
    df["answer"] = df["answer"].fillna("")

    return df[["dataset", "question", "answer", "context"]]


def main() -> None:
    """Consolidate all datasets."""
    print("Consolidating Vietnamese QA datasets...")
    print(f"Source directory: {DATA_DIR.resolve()}")
    print()

    frames = []
    for name, config in DATASET_CONFIG.items():
        df = load_dataset(name, config)
        if not df.empty:
            print(f"  ✅ {name:20s} → {len(df):,} rows")
            frames.append(df)

    if not frames:
        print("No datasets loaded!")
        return

    unified = pd.concat(frames, ignore_index=True)

    # Summary
    print(f"\n{'='*50}")
    print(f"Total rows: {len(unified):,}")
    print(f"Datasets:   {unified['dataset'].nunique()}")
    print(f"\nPer-dataset breakdown:")
    print(unified["dataset"].value_counts().to_string())

    # Save
    unified.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n💾 Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

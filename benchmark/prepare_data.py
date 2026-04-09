"""Prepare processed benchmark dataset from unified_vqa.csv.

Filters out excluded datasets and samples up to N QA pairs per dataset.
Saves a single benchmark.csv to data/processed/.

At runtime, the benchmark script samples eval subsets with a fixed seed
for reproducibility — no need for a separate eval file.

Usage:
    python benchmark/prepare_data.py
    python benchmark/prepare_data.py --max-pairs 1000
    python benchmark/prepare_data.py --exclude ZaloLegalQA ALQAC --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

UNIFIED_CSV = "data/unified_vqa.csv"
OUTPUT_DIR = "data/processed"
EXCLUDE_DATASETS = ["ZaloLegalQA", "ALQAC"]
MAX_PAIRS_PER_DATASET = 1000
SEED = 42


def prepare_data(
    csv_path: str,
    output_dir: str,
    exclude: list[str],
    max_pairs: int,
    seed: int,
    dry_run: bool = False,
) -> None:
    """Process unified CSV into a single benchmark-ready dataset.

    For each included dataset, sample up to `max_pairs` QA pairs.

    Outputs:
      - data/processed/benchmark.csv   (sampled QA pairs)
      - data/processed/summary.txt     (human-readable summary)

    Args:
        csv_path: Path to unified_vqa.csv.
        output_dir: Output directory.
        exclude: Dataset names to exclude.
        max_pairs: Max QA pairs per dataset.
        seed: Random seed for reproducibility.
        dry_run: If True, only print summary without writing files.
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["answer"] = df["answer"].fillna("")

    print(f"  Total rows: {len(df)}")
    print(f"  Datasets found: {sorted(df['dataset'].unique())}")

    # Filter out excluded datasets
    exclude_set = set(exclude)
    df_filtered = df[~df["dataset"].isin(exclude_set)]

    print(f"\n  Excluded: {exclude}")
    for ds in exclude:
        count = len(df[df["dataset"] == ds])
        print(f"    - {ds}: {count} rows removed")

    remaining_datasets = sorted(df_filtered["dataset"].unique())
    print(f"\n  Remaining datasets ({len(remaining_datasets)}): {remaining_datasets}")

    # Sample per dataset
    sampled_frames = []
    summary_lines = []

    summary_lines.append("Chunking Benchmark -- Processed Dataset Summary")
    summary_lines.append("=" * 55)
    summary_lines.append(f"Source: {csv_path}")
    summary_lines.append(f"Excluded: {exclude}")
    summary_lines.append(f"Max pairs per dataset: {max_pairs}")
    summary_lines.append(f"Seed: {seed}")
    summary_lines.append("")
    summary_lines.append(
        f"{'Dataset':<20s} {'Original':>10s} {'Sampled':>10s}"
    )
    summary_lines.append("-" * 45)

    for ds_name in remaining_datasets:
        group = df_filtered[df_filtered["dataset"] == ds_name]
        original_count = len(group)

        if original_count > max_pairs:
            sampled = group.sample(n=max_pairs, random_state=seed)
        else:
            sampled = group.copy()

        sampled_frames.append(sampled)

        summary_lines.append(
            f"{ds_name:<20s} {original_count:>10d} {len(sampled):>10d}"
        )
        print(f"  {ds_name}: {original_count} -> {len(sampled)} sampled")

    df_out = pd.concat(sampled_frames, ignore_index=True)

    summary_lines.append("-" * 45)
    summary_lines.append(
        f"{'TOTAL':<20s} {len(df_filtered):>10d} {len(df_out):>10d}"
    )

    print(f"\n  Result: {len(df_out)} total rows")

    if dry_run:
        print("\n  [DRY RUN] No files written.")
        print("\n" + "\n".join(summary_lines))
        return

    # Write outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_out_path = out / "benchmark.csv"
    summary_path = out / "summary.txt"

    df_out.to_csv(csv_out_path, index=False, encoding="utf-8")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\n  Saved:")
    print(f"    {csv_out_path}  ({len(df_out)} rows)")
    print(f"    {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare processed benchmark dataset from unified_vqa.csv"
    )
    parser.add_argument(
        "--csv", default=UNIFIED_CSV,
        help=f"Path to unified CSV (default: {UNIFIED_CSV})",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--exclude", type=str, nargs="+", default=EXCLUDE_DATASETS,
        help=f"Datasets to exclude (default: {EXCLUDE_DATASETS})",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=MAX_PAIRS_PER_DATASET,
        help=f"Max QA pairs per dataset (default: {MAX_PAIRS_PER_DATASET})",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary only, don't write files",
    )
    args = parser.parse_args()

    prepare_data(
        csv_path=args.csv,
        output_dir=args.output_dir,
        exclude=args.exclude,
        max_pairs=args.max_pairs,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

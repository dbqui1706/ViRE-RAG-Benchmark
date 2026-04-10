"""Chunking Statistics Report — Compare C1-C5 strategies across all datasets.

Outputs a markdown table with chunk count, avg/min/max/median size,
and expansion ratio for each strategy.

Usage:
    python scripts/chunking_stats.py [--dataset unified_vqa] [--sample 0]
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Chunking strategy definitions
# ---------------------------------------------------------------------------

def _fixed_size_chunker(chunk_size: int, overlap: int = 0):
    """C1: Fixed-size character chunking (no intelligent splitting)."""
    return CharacterTextSplitter(
        separator="",  # pure character-level splitting
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )

def _sentence_chunker():
    """C2: Vietnamese sentence-based chunking using underthesea."""
    from underthesea import sent_tokenize

    class SentenceChunker:
        def split_documents(self, docs: list[Document]) -> list[Document]:
            results = []
            for doc in docs:
                sentences = sent_tokenize(doc.page_content)
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        results.append(Document(
                            page_content=sent,
                            metadata=doc.metadata.copy(),
                        ))
            return results
    return SentenceChunker()

def _paragraph_chunker():
    """C3: Paragraph-based chunking — split by double newlines."""
    class ParagraphChunker:
        def split_documents(self, docs: list[Document]) -> list[Document]:
            results = []
            for doc in docs:
                paragraphs = re.split(r'\n\s*\n', doc.page_content)
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        results.append(Document(
                            page_content=para,
                            metadata=doc.metadata.copy(),
                        ))
            return results
    return ParagraphChunker()

def _recursive_chunker(chunk_size: int = 256, overlap: int = 50):
    """C4: LangChain Recursive Character Text Splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

def _semantic_chunker():
    """C5: Semantic chunking based on embedding similarity between sentences."""
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch
    embed = HuggingFaceEmbeddings(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return SemanticChunker(embed, breakpoint_threshold_type="percentile")


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
STRATEGIES: dict[str, dict] = {
    "C1-256":  {"label": "Fixed-size (256)",      "factory": lambda: _fixed_size_chunker(256, 0)},
    "C1-512":  {"label": "Fixed-size (512)",      "factory": lambda: _fixed_size_chunker(512, 0)},
    "C1-1024": {"label": "Fixed-size (1024)",     "factory": lambda: _fixed_size_chunker(1024, 0)},
    "C2":      {"label": "Sentence (underthesea)","factory": _sentence_chunker},
    "C3":      {"label": "Paragraph (\\n\\n)",    "factory": _paragraph_chunker},
    "C4-256":  {"label": "Recursive (256/50)",    "factory": lambda: _recursive_chunker(256, 50)},
    "C4-512":  {"label": "Recursive (512/100)",   "factory": lambda: _recursive_chunker(512, 100)},
    "C5":      {"label": "Semantic (Vietnamese_Embedding_v2)",     "factory": _semantic_chunker},
}


# ---------------------------------------------------------------------------
# Data loading (minimal, no rag_bench dependency needed)
# ---------------------------------------------------------------------------

def load_documents(csv_path: str, sample: int = 0) -> list[Document]:
    """Load documents from CSV, optionally sampling."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    # Normalize column names
    col_map = {c.lower(): c for c in df.columns}
    context_col = col_map.get("context", df.columns[0])

    df = df.drop_duplicates(subset=[context_col])
    df = df.dropna(subset=[context_col])

    if sample > 0:
        df = df.head(sample)

    docs = [
        Document(page_content=str(row[context_col]), metadata={"idx": i})
        for i, (_, row) in enumerate(df.iterrows())
    ]
    return docs


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_stats(docs_before: list[Document], docs_after: list[Document], elapsed: float) -> dict:
    """Compute chunking statistics."""
    sizes = [len(d.page_content) for d in docs_after]
    n_before = len(docs_before)
    n_after = len(docs_after)

    return {
        "n_docs": n_before,
        "n_chunks": n_after,
        "ratio": f"{n_after / n_before:.1f}x" if n_before > 0 else "N/A",
        "avg_chars": int(statistics.mean(sizes)) if sizes else 0,
        "median_chars": int(statistics.median(sizes)) if sizes else 0,
        "min_chars": min(sizes) if sizes else 0,
        "max_chars": max(sizes) if sizes else 0,
        "std_chars": int(statistics.stdev(sizes)) if len(sizes) > 1 else 0,
        "time_s": f"{elapsed:.1f}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chunking statistics report")
    parser.add_argument("--data-dir", default="data", help="Directory containing CSVs")
    parser.add_argument("--dataset", default="unified_vqa", help="Dataset CSV name (without .csv)")
    parser.add_argument("--sample", type=int, default=0, help="Sample N documents (0=all)")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES.keys()),
                        help="Strategy IDs to run")
    parser.add_argument("--output", default="", help="Output markdown file path")
    args = parser.parse_args()

    csv_path = Path(args.data_dir) / f"{args.dataset}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    docs = load_documents(str(csv_path), sample=args.sample)
    print(f"  {len(docs)} unique documents loaded" + (f" (sampled {args.sample})" if args.sample else ""))

    results = []
    for sid in args.strategies:
        if sid not in STRATEGIES:
            print(f"  SKIP: unknown strategy '{sid}'")
            continue

        info = STRATEGIES[sid]
        print(f"\n[{sid}] {info['label']}...")

        try:
            t0 = time.perf_counter()
            splitter = info["factory"]()
            chunks = splitter.split_documents(docs)
            elapsed = time.perf_counter() - t0

            stats = compute_stats(docs, chunks, elapsed)
            stats["id"] = sid
            stats["label"] = info["label"]
            results.append(stats)

            print(f"  {stats['n_docs']} -> {stats['n_chunks']} chunks "
                  f"({stats['ratio']}) | avg={stats['avg_chars']} chars | {stats['time_s']}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": sid, "label": info["label"],
                "n_docs": len(docs), "n_chunks": "ERROR", "ratio": "-",
                "avg_chars": "-", "median_chars": "-",
                "min_chars": "-", "max_chars": "-",
                "std_chars": "-", "time_s": "-",
            })

    # Output markdown table
    print("\n" + "=" * 80)
    header = "| ID | Strategy | Docs | Chunks | Ratio | Avg | Median | Min | Max | Std | Time |"
    sep =    "|:---|:---------|-----:|-------:|:------|----:|-------:|----:|----:|----:|-----:|"
    rows = [header, sep]
    for r in results:
        rows.append(
            f"| {r['id']} | {r['label']} | {r['n_docs']} | {r['n_chunks']} | "
            f"{r['ratio']} | {r['avg_chars']} | {r['median_chars']} | "
            f"{r['min_chars']} | {r['max_chars']} | {r['std_chars']} | {r['time_s']}s |"
        )

    table = "\n".join(rows)
    print(f"\n## Chunking Statistics — {args.dataset}\n")
    print(table)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Chunking Statistics — {args.dataset}\n\n")
            f.write(f"- Source: `{csv_path}`\n")
            f.write(f"- Documents: {len(docs)}\n")
            f.write(f"- Sample: {'all' if not args.sample else args.sample}\n\n")
            f.write(table + "\n")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

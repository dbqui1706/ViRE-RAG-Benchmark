from rag_bench.data_loader import load_and_sample, load_dataset, sample_qa_pairs


def test_load_dataset(sample_csv):
    """Load ALL documents from CSV."""
    docs, qa_pairs = load_dataset(sample_csv)
    assert len(docs) == 10  # All unique contexts
    assert len(qa_pairs) == 10
    assert hasattr(docs[0], "page_content")
    assert "qid" in docs[0].metadata
    assert "question" in qa_pairs[0]
    assert "answer" in qa_pairs[0]


def test_sample_qa_pairs(sample_csv):
    """Sample a subset of QA pairs."""
    _, qa_pairs = load_dataset(sample_csv)
    sampled = sample_qa_pairs(qa_pairs, max_samples=5, seed=42)
    assert len(sampled) == 5


def test_load_and_sample_compat(sample_csv):
    """Backward-compatible wrapper returns all docs + sampled QA."""
    docs, qa_pairs = load_and_sample(sample_csv, max_samples=5, seed=42)
    assert len(docs) == 10  # ALL docs returned for indexing
    assert len(qa_pairs) == 5  # Only QA pairs are sampled


def test_unique_contexts(sample_csv):
    docs, _ = load_dataset(sample_csv, prefer_unique=True)
    texts = [d.page_content for d in docs]
    assert len(set(texts)) == len(texts)

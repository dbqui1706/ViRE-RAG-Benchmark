import pytest
from langchain_core.documents import Document

from rag_bench.chunker import get_chunker


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Đây là câu thứ nhất. Đây là câu thứ hai.\n\nĐoạn văn thứ hai ở đây.",
            metadata={"source": "test"},
        ),
    ]


# C1: Fixed-size
def test_fixed_size_chunker(sample_docs):
    chunker = get_chunker("fixed", chunk_size=20, chunk_overlap=0)
    chunks = chunker.chunk(sample_docs)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.page_content) <= 20
        assert c.metadata["source"] == "test"


def test_fixed_size_preserves_metadata(sample_docs):
    chunker = get_chunker("fixed", chunk_size=50, chunk_overlap=0)
    chunks = chunker.chunk(sample_docs)
    assert all(c.metadata.get("source") == "test" for c in chunks)


# C2: Sentence
def test_sentence_chunker(sample_docs):
    chunker = get_chunker("sentence")
    chunks = chunker.chunk(sample_docs)
    assert len(chunks) >= 2  # at least 2 sentences
    for c in chunks:
        assert len(c.page_content.strip()) > 0
        assert c.metadata["source"] == "test"


# C3: Paragraph
def test_paragraph_chunker(sample_docs):
    chunker = get_chunker("paragraph")
    chunks = chunker.chunk(sample_docs)
    assert len(chunks) == 2  # two paragraphs separated by \n\n
    assert "câu thứ nhất" in chunks[0].page_content
    assert "Đoạn văn thứ hai" in chunks[1].page_content


def test_paragraph_single_paragraph():
    """Document without double newlines -> 1 chunk."""
    docs = [Document(page_content="No paragraph breaks here.", metadata={})]
    chunker = get_chunker("paragraph")
    chunks = chunker.chunk(docs)
    assert len(chunks) == 1


# Passthrough and recursive still work
def test_passthrough_unchanged(sample_docs):
    chunker = get_chunker("passthrough")
    assert chunker.chunk(sample_docs) == sample_docs


def test_recursive_chunker(sample_docs):
    chunker = get_chunker("recursive", chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(sample_docs)
    assert len(chunks) > 1


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("nonexistent")

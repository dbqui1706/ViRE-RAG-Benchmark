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


# C1: Token
def test_token_chunker(sample_docs):
    chunker = get_chunker("token", chunk_size=20, chunk_overlap=0)
    chunks = chunker.chunk(sample_docs)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source"] == "test"


def test_token_preserves_metadata(sample_docs):
    chunker = get_chunker("token", chunk_size=50, chunk_overlap=0)
    chunks = chunker.chunk(sample_docs)
    assert all(c.metadata.get("source") == "test" for c in chunks)


# C2: Sentence (underthesea)
def test_sentence_chunker(sample_docs):
    chunker = get_chunker("sentence", chunk_size=512)
    chunks = chunker.chunk(sample_docs)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.page_content.strip()) > 0
        assert c.metadata["source"] == "test"


# C3: Paragraph
def test_paragraph_chunker(sample_docs):
    chunker = get_chunker("paragraph", chunk_size=40)
    chunks = chunker.chunk(sample_docs)
    assert len(chunks) == 2  # two paragraphs separated by \n\n
    assert "câu thứ nhất" in chunks[0].page_content
    assert "Đoạn văn thứ hai" in chunks[1].page_content


def test_paragraph_single_paragraph():
    """Document without double newlines -> 1 chunk."""
    docs = [Document(page_content="No paragraph breaks here.", metadata={})]
    chunker = get_chunker("paragraph", chunk_size=512)
    chunks = chunker.chunk(docs)
    assert len(chunks) == 1


# C4: Recursive
def test_recursive_chunker(sample_docs):
    chunker = get_chunker("recursive", chunk_size=20)
    chunks = chunker.chunk(sample_docs)
    assert len(chunks) > 1


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("nonexistent")

"""Chunking strategies for RAG pipeline."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PassthroughChunker:
    """No chunking — each Document stays as-is (Native RAG baseline)."""

    def chunk(self, documents: list[Document]) -> list[Document]:
        return documents


class RecursiveSplitter:
    """Recursive character text splitter.

    Splits by paragraph → sentence → word boundaries.
    Default: 256 chars chunk size, 50 chars overlap.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks."""
        return self.splitter.split_documents(documents)


def get_chunker(strategy: str = "passthrough", **kwargs):
    """Factory function to get a chunker by strategy name.

    Args:
        strategy: "passthrough" or "recursive"
        **kwargs: chunk_size, chunk_overlap for recursive strategy

    Returns:
        Chunker instance.
    """
    if strategy == "passthrough":
        return PassthroughChunker()
    elif strategy == "recursive":
        return RecursiveSplitter(
            chunk_size=kwargs.get("chunk_size", 256),
            chunk_overlap=kwargs.get("chunk_overlap", 50),
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

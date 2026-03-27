"""Chunking strategies for RAG pipeline."""

from __future__ import annotations

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class PassthroughChunker:
    """No chunking — each Document stays as-is (Native RAG)."""

    def chunk(self, documents: list[Document]) -> list[Document]:
        return documents


class RecursiveSplitter:
    """Recursive character splitter using LlamaIndex SentenceSplitter.

    Splits by paragraph → sentence → word boundaries.
    Default: 256 tokens chunk size, 50 tokens overlap.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks."""
        nodes = self.splitter.get_nodes_from_documents(documents)
        # Convert nodes back to Documents for indexer compatibility
        chunked = []
        for node in nodes:
            doc = Document(
                text=node.get_content(),
                metadata=node.metadata,
            )
            chunked.append(doc)
        return chunked


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

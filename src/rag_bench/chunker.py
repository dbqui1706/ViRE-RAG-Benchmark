"""Chunking strategies for RAG pipeline."""

from __future__ import annotations

from llama_index.core import Document


class PassthroughChunker:
    """No chunking — each Document stays as-is (Native RAG)."""

    def chunk(self, documents: list[Document]) -> list[Document]:
        return documents

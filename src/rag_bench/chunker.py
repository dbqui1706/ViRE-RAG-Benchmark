"""Chunking strategies for RAG pipeline."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class PassthroughChunker:
    """No chunking — each Document stays as-is (Native RAG baseline)."""

    def chunk(self, documents: list[Document]) -> list[Document]:
        return documents


class FixedSizeChunker:
    """C1: Fixed-size character chunking (no intelligent splitting).

    Splits purely by character count, ignoring word/sentence boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        self.splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)


class SentenceChunker:
    """C2: Vietnamese sentence-based chunking using underthesea."""

    def chunk(self, documents: list[Document]) -> list[Document]:
        from underthesea import sent_tokenize

        results = []
        for doc in documents:
            sentences = sent_tokenize(doc.page_content)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    results.append(Document(
                        page_content=sent,
                        metadata=doc.metadata.copy(),
                    ))
        return results


class ParagraphChunker:
    """C3: Paragraph-based chunking — split by double newlines."""

    def chunk(self, documents: list[Document]) -> list[Document]:
        results = []
        for doc in documents:
            paragraphs = re.split(r'\n\s*\n', doc.page_content)
            for para in paragraphs:
                para = para.strip()
                if para:
                    results.append(Document(
                        page_content=para,
                        metadata=doc.metadata.copy(),
                    ))
        return results


class RecursiveSplitter:
    """C4: Recursive character text splitter.

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


class SemanticChunker:
    """C5: Semantic chunking based on embedding similarity between sentences."""

    def __init__(self, embed_model: str = "AITeamVN/Vietnamese_Embedding_v2",
                 breakpoint_threshold_type: str = "percentile"):
        from langchain_experimental.text_splitter import SemanticChunker as _SC
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch

        embed = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        self.splitter = _SC(embed, breakpoint_threshold_type=breakpoint_threshold_type)

    def chunk(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(list(documents))


def get_chunker(strategy: str = "passthrough", **kwargs):
    """Factory function to get a chunker by strategy name.

    Args:
        strategy: "passthrough", "fixed", "sentence", "paragraph",
                  "recursive", or "semantic"
        **kwargs: chunk_size, chunk_overlap for fixed/recursive strategies

    Returns:
        Chunker instance.
    """
    if strategy == "passthrough":
        return PassthroughChunker()
    elif strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=kwargs.get("chunk_size", 512),
            chunk_overlap=kwargs.get("chunk_overlap", 0),
        )
    elif strategy == "sentence":
        return SentenceChunker()
    elif strategy == "paragraph":
        return ParagraphChunker()
    elif strategy == "recursive":
        return RecursiveSplitter(
            chunk_size=kwargs.get("chunk_size", 256),
            chunk_overlap=kwargs.get("chunk_overlap", 50),
        )
    elif strategy == "semantic":
        return SemanticChunker(
            embed_model=kwargs.get("embed_model", "AITeamVN/Vietnamese_Embedding_v2"),
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

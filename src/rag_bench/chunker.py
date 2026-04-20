"""Chunking strategies for RAG pipeline — powered by Chonkie."""

from __future__ import annotations

from langchain_core.documents import Document


class ChonkieAdapter:
    """Base adapter: wraps a Chonkie chunker for LangChain Document I/O.

    Chonkie operates on raw ``str`` and returns ``Chunk`` dataclasses.
    This adapter extracts ``page_content`` from each Document, feeds it to
    the underlying Chonkie chunker, and converts results back to Documents
    with metadata copied from the source document.
    """

    def __init__(self, chonkie_chunker):
        self._chunker = chonkie_chunker

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents using the wrapped Chonkie chunker."""
        results: list[Document] = []
        for doc in documents:
            chunks = self._chunker.chunk(doc.page_content)
            for c in chunks:
                results.append(Document(
                    page_content=c.text,
                    metadata=doc.metadata.copy(),
                ))
        return results


class VietnameseSentenceChunker:
    """Vietnamese sentence chunking: underthesea NLP + Chonkie token counting.

    Uses ``underthesea.sent_tokenize`` for linguistically-aware Vietnamese
    sentence boundary detection and Chonkie's ``Tokenizer`` for consistent
    token counting when grouping sentences into chunks.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
    ):
        from chonkie import CharacterTokenizer
        from underthesea import sent_tokenize

        self._sent_tokenize = sent_tokenize
        self._tokenizer = CharacterTokenizer()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into sentence-based chunks."""
        results: list[Document] = []
        for doc in documents:
            sentences = self._sent_tokenize(doc.page_content)
            chunks = self._group_sentences(sentences)
            for chunk_text in chunks:
                results.append(Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy(),
                ))
        return results

    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """Group sentences into chunks respecting chunk_size."""
        grouped: list[str] = []
        current_sents: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = self._tokenizer.count_tokens(sent)

            if current_sents and current_tokens + sent_tokens > self._chunk_size:
                grouped.append(" ".join(current_sents))

                # Handle overlap: keep trailing sentences
                if self._chunk_overlap > 0:
                    overlap_sents: list[str] = []
                    overlap_tokens = 0
                    for s in reversed(current_sents):
                        s_tok = self._tokenizer.count_tokens(s)
                        if overlap_tokens + s_tok > self._chunk_overlap:
                            break
                        overlap_sents.insert(0, s)
                        overlap_tokens += s_tok
                    current_sents = overlap_sents
                    current_tokens = overlap_tokens
                else:
                    current_sents = []
                    current_tokens = 0

            current_sents.append(sent)
            current_tokens += sent_tokens

        if current_sents:
            grouped.append(" ".join(current_sents))

        return grouped


class ParagraphChunker:
    """Paragraph-based chunking — split by double newlines.

    Each paragraph (text between ``\\n\\n`` boundaries) becomes a separate chunk.
    Consecutive small paragraphs whose combined token count is under ``chunk_size``
    are merged to avoid excessively small chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        import re

        from chonkie import CharacterTokenizer

        self._split_re = re.compile(r"\n\s*\n")
        self._tokenizer = CharacterTokenizer()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents at paragraph boundaries."""
        results: list[Document] = []
        for doc in documents:
            paragraphs = [p.strip() for p in self._split_re.split(doc.page_content) if p.strip()]
            chunks = self._group_paragraphs(paragraphs)
            for chunk_text in chunks:
                results.append(Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy(),
                ))
        return results

    def _group_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Group small paragraphs together up to chunk_size."""
        grouped: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._tokenizer.count_tokens(para)

            if current_parts and current_tokens + para_tokens > self._chunk_size:
                grouped.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            grouped.append("\n\n".join(current_parts))

        return grouped


def get_chunker(strategy: str = "recursive", **kwargs):
    """Factory function to get a chunker by strategy name.

    Args:
        strategy: "token", "sentence", "paragraph", "recursive",
                  "semantic", or "neural"
        **kwargs: chunk_size, chunk_overlap, embed_model, neural_model

    Returns:
        Chunker instance with a ``.chunk(documents)`` method.
    """
    if strategy == "token":
        from chonkie import TokenChunker
        return ChonkieAdapter(TokenChunker(
            tokenizer="character",
            chunk_size=kwargs.get("chunk_size", 512),
            chunk_overlap=kwargs.get("chunk_overlap", 128),
        ))
    elif strategy == "sentence":
        return VietnameseSentenceChunker(
            chunk_size=kwargs.get("chunk_size", 512),
            chunk_overlap=kwargs.get("chunk_overlap", 128),
        )
    elif strategy == "paragraph":
        return ParagraphChunker(
            chunk_size=kwargs.get("chunk_size", 512),
            chunk_overlap=kwargs.get("chunk_overlap", 0),
        )
    elif strategy == "recursive":
        from chonkie import RecursiveChunker
        return ChonkieAdapter(RecursiveChunker(
            tokenizer="character",
            chunk_size=kwargs.get("chunk_size", 256),
        ))
    elif strategy == "semantic":
        from chonkie import SemanticChunker
        from chonkie.embeddings import SentenceTransformerEmbeddings
        from sentence_transformers import SentenceTransformer
        model_name = kwargs.get("embed_model", "intfloat/multilingual-e5-large")
        st_model = SentenceTransformer(model_name)
        embeddings = SentenceTransformerEmbeddings(st_model)
        return ChonkieAdapter(SemanticChunker(
            embedding_model=embeddings,
            chunk_size=kwargs.get("chunk_size", 512),
            threshold=kwargs.get("threshold", 0.5),
        ))
    elif strategy == "neural":
        import torch
        from chonkie import NeuralChunker
        return ChonkieAdapter(NeuralChunker(
            model=kwargs.get("neural_model", "mirth/chonky_modernbert_base_1"),
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        ))
    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available: token, sentence, paragraph, recursive, semantic, neural"
        )

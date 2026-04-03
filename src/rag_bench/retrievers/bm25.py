"""BM25 sparse retrieval strategies for ViRAG-Bench.

Provides two retriever variants registered under the keys:

- ``'bm25_syl'``: Syllable-level tokenization (whitespace split). No NLP
  dependency required. Suitable as a fast baseline.
- ``'bm25_word'``: Word-level tokenization using ``underthesea.word_tokenize``.
  Handles Vietnamese compound words (e.g. ``"thủ tục"`` → ``"thủ_tục"``),
  improving sparse retrieval quality for legal and medical domains.

Both variants use ``rank_bm25.BM25Okapi`` under the hood and are built from
a list of ``langchain_core.documents.Document`` objects at construction time.
"""
from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

from . import register
from .base import BaseRetriever


# ---------------------------------------------------------------------------
# Tokenizers (module-level, easy to mock in tests)
# ---------------------------------------------------------------------------


def _tokenize_syllable(text: str) -> list[str]:
    """Tokenize Vietnamese text at the syllable (whitespace) level.

    Args:
        text: Raw Vietnamese text.

    Returns:
        A list of syllable tokens (unicode-aware split on spaces).
    """
    return text.lower().split()


def _tokenize_word(text: str) -> list[str]:
    """Tokenize Vietnamese text at the word level using underthesea.

    Compound words are joined with underscores by ``underthesea``
    (e.g. ``"thủ tục"`` → ``"thủ_tục"``).

    Args:
        text: Raw Vietnamese text.

    Returns:
        A list of word tokens, lowercased.
    """
    return [t.lower() for t in word_tokenize(text)]


# ---------------------------------------------------------------------------
# Base BM25 retriever (shared logic)
# ---------------------------------------------------------------------------


class _BM25BaseRetriever(BaseRetriever):
    """Internal base class for BM25 variants.

    Subclasses only need to define ``_tokenize``.
    """

    def __init__(self, documents: list[Document], top_k: int = 5) -> None:
        self._documents = documents
        self._top_k = top_k
        self._index = self._build_index(documents)

    def _tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def _build_index(self, documents: list[Document]) -> BM25Okapi | None:
        """Build the BM25 index from document page content.

        Args:
            documents: Corpus documents.

        Returns:
            A ``BM25Okapi`` index, or ``None`` if the corpus is empty.
        """
        if not documents:
            return None
        corpus = [self._tokenize(doc.page_content) for doc in documents]
        return BM25Okapi(corpus)

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Retrieve top-k documents using BM25 scoring.

        Args:
            query: The search query string.
            **kwargs: Ignored.

        Returns:
            A list of up to ``top_k`` Document objects, ordered by BM25 score.
        """
        if self._index is None or not self._documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Pair each doc with its score, sort descending, take top_k
        ranked = sorted(
            zip(scores, self._documents), key=lambda x: x[0], reverse=True
        )
        return [doc for _, doc in ranked[: self._top_k]]


# ---------------------------------------------------------------------------
# Concrete retrievers
# ---------------------------------------------------------------------------


@register("bm25_syl")
class BM25SylRetriever(_BM25BaseRetriever):
    """BM25 retriever with syllable-level (whitespace) tokenization.

    Best used as a fast sparse baseline. No external NLP tool required.

    Args:
        documents: Corpus of documents to index.
        top_k: Number of documents to return per query.

    Example:
        >>> r = BM25SylRetriever(documents=docs, top_k=5)
        >>> results = r.retrieve("thủ tục hành chính")
    """

    def _tokenize(self, text: str) -> list[str]:
        return _tokenize_syllable(text)


@register("bm25_word")
class BM25WordRetriever(_BM25BaseRetriever):
    """BM25 retriever with word-level tokenization via ``underthesea``.

    Handles Vietnamese compound words correctly, which significantly
    improves exact-match quality for legal, medical, and technical domains.
    Requires ``underthesea`` (install with ``pip install -e ".[vietnamese]"``).

    Args:
        documents: Corpus of documents to index.
        top_k: Number of documents to return per query.

    Example:
        >>> r = BM25WordRetriever(documents=docs, top_k=5)
        >>> results = r.retrieve("thủ tục hành chính")
    """

    def _tokenize(self, text: str) -> list[str]:
        return _tokenize_word(text)

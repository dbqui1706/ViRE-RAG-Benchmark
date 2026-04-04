"""BM25 sparse retrieval strategies for ViRAG-Bench.

Provides two retriever variants registered under the keys:

- ``'bm25_syl'``: Syllable-level tokenization (whitespace split). No NLP
  dependency required. Suitable as a fast baseline.
- ``'bm25_word'``: Word-level tokenization using ``underthesea.word_tokenize``.
  Handles Vietnamese compound words (e.g. ``"thủ tục"`` → ``"thủ_tục"``),
  improving sparse retrieval quality for legal and medical domains.

Both variants use ``bm25s`` under the hood and are built from
a list of ``langchain_core.documents.Document`` objects at construction time.
"""
from __future__ import annotations

import time
from tqdm import tqdm

from langchain_core.documents import Document
import bm25s
from underthesea import word_tokenize

from . import register
from .base import BaseRetriever, RetrievalResult

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
    """Internal base class for BM25 variants using bm25s.

    Subclasses only need to define ``_tokenize``.
    """

    def __init__(self, documents: list[Document], top_k: int = 5) -> None:
        self._documents = documents
        self._top_k = top_k
        self._index = self._build_index(documents)

    def _tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def _build_index(self, documents: list[Document]) -> bm25s.BM25 | None:
        """Build the BM25 index from document page content.

        Args:
            documents: Corpus documents.

        Returns:
            A ``bm25s.BM25`` index, or ``None`` if the corpus is empty.
        """
        if not documents:
            return None
            
        corpus_tokens = [
            self._tokenize(doc.page_content) 
            for doc in tqdm(documents, desc="Tokenizing BM25 corpus")
        ]
        
        # bm25s optimizes indexing in C/NumPy
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        return retriever

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
        k = min(self._top_k, len(self._documents))
        
        # bm25s.retrieve expects a list of queries (so we wrap in list)
        results, scores = self._index.retrieve([tokenized_query], corpus=self._documents, k=k)
        
        # results[0] contains the Document objects for the first (and only) query
        return list(results[0])

    def batch_retrieve(self, queries: list[str], **kwargs) -> list[RetrievalResult]:
        """Retrieve documents for multiple queries simultaneously.
        
        Leverages bm25s optimized C/NumPy vectorized retrieval.
        
        Args:
            queries: List of search query strings.
            **kwargs: Ignored.
            
        Returns:
            A list of ``RetrievalResult``, one per query.
        """
        if self._index is None or not self._documents:
            return [RetrievalResult(question=q, documents=[], retrieval_ms=0.0) for q in queries]
            
        k = min(self._top_k, len(self._documents))
        
        t0 = time.perf_counter()
        
        # Tokenize all queries
        tokenized_queries = [self._tokenize(q) for q in queries]
        
        # Vectorized batch retrieval
        results, scores = self._index.retrieve(tokenized_queries, corpus=self._documents, k=k)
        
        ms = (time.perf_counter() - t0) * 1000
        ms_per_query = ms / len(queries)
        
        ret_results = []
        for i, q in enumerate(queries):
            docs = list(results[i])
            ret_results.append(RetrievalResult(question=q, documents=docs, retrieval_ms=ms_per_query))
            
        return ret_results


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

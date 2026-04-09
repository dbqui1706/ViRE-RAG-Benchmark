"""TF-IDF sparse retrieval strategies for ViRAG-Bench."""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from . import register
from .base import BaseRetriever, RetrievalResult
from .bm25 import _tokenize_syllable, _tokenize_word

# ---------------------------------------------------------------------------
# Identity analyzer for pre-tokenized text
# ---------------------------------------------------------------------------


def _identity(tokens: list[str]) -> list[str]:
    """Identity function to bypass TfidfVectorizer's default tokenization."""
    return tokens


# ---------------------------------------------------------------------------
# Base TF-IDF retriever
# ---------------------------------------------------------------------------


class _TfidfBaseRetriever(BaseRetriever):
    """Internal base class for TF-IDF variants."""

    def __init__(self, documents: list[Document], top_k: int = 5) -> None:
        self._documents = documents
        self._top_k = top_k
        self._vectorizer, self._doc_matrix = self._build_index(documents)

    @property
    def _tokenizer_func(self):
        """Return the module-level tokenization callable."""
        raise NotImplementedError

    def _build_index(self, documents: list[Document]):
        """Build the TF-IDF matrix from document page content.

        Returns:
            Tuple of (TfidfVectorizer, scipy.sparse.csr_matrix).
            Returns (None, None) if corpus is empty.
        """
        if not documents:
            return None, None

        cache_dir = Path("outputs/.tfidf_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        md5 = hashlib.md5()
        for doc in documents:
            md5.update(doc.page_content.encode('utf-8'))
        corpus_hash = md5.hexdigest()

        strategy = self.__class__.__name__
        cache_file = cache_dir / f"{strategy}_{corpus_hash}.pkl"

        if cache_file.exists():
            print(f"  [Cache] Loading cached TF-IDF models from {cache_file.name}")
            with open(cache_file, "rb") as f:
                vectorizer, matrix = pickle.load(f)
            return vectorizer, matrix

        tokenizer_func = self._tokenizer_func
        corpus_tokens = [
            tokenizer_func(doc.page_content)
            for doc in tqdm(documents, desc=f"Tokenizing TF-IDF corpus ({strategy})")
        ]

        vectorizer = TfidfVectorizer(
            analyzer=_identity,
            token_pattern=None,
            lowercase=False
        )
        print(f"  Fitting TfidfVectorizer for {len(corpus_tokens)} docs...")
        matrix = vectorizer.fit_transform(corpus_tokens)

        with open(cache_file, "wb") as f:
            pickle.dump((vectorizer, matrix), f)

        return vectorizer, matrix

    def retrieve(self, query: str, **kwargs) -> list[Document]:
        """Retrieve top-k documents using TF-IDF cosine similarity."""
        if self._vectorizer is None or self._doc_matrix is None or not self._documents:
            return []

        tokenized_query = self._tokenizer_func(query)
        query_vec = self._vectorizer.transform([tokenized_query])
        
        # doc_matrix is (N_docs, N_vocab), query_vec is (1, N_vocab)
        # Cosine similarity is just dot product since vectors are L2-normalized
        similarities = query_vec.dot(self._doc_matrix.T).toarray().flatten()
        
        k = min(self._top_k, len(self._documents))
        
        # argpartition is faster than argosrt for getting top K
        if len(similarities) > k:
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            # argpartition doesn't guarantee sorted order, so sort the top k
            top_k_docs_sorted_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]
        else:
            top_k_docs_sorted_idx = np.argsort(-similarities)
            
        # Return docs with positive similarity score
        results = [
            self._documents[idx] 
            for idx in top_k_docs_sorted_idx 
            if similarities[idx] > 0
        ]
        
        return results

    def batch_retrieve(self, queries: list[str], **kwargs) -> list[RetrievalResult]:
        """Retrieve documents for multiple queries simultaneously."""
        if self._vectorizer is None or self._doc_matrix is None or not self._documents:
            return [RetrievalResult(question=q, documents=[], retrieval_ms=0.0) for q in queries]

        k = min(self._top_k, len(self._documents))
        t0 = time.perf_counter()

        tokenizer_func = self._tokenizer_func
        tokenized_queries = [tokenizer_func(q) for q in queries]
        
        queries_matrix = self._vectorizer.transform(tokenized_queries)
        # Shape: (N_queries, N_docs)
        sim_matrix = queries_matrix.dot(self._doc_matrix.T).toarray()
        
        ms = (time.perf_counter() - t0) * 1000
        ms_per_query = ms / max(len(queries), 1)
        
        ret_results = []
        for i, q in enumerate(queries):
            similarities = sim_matrix[i]
            
            if len(similarities) > k:
                top_k_idx = np.argpartition(similarities, -k)[-k:]
                top_k_docs_sorted_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]
            else:
                top_k_docs_sorted_idx = np.argsort(-similarities)
                
            docs = [
                self._documents[idx] 
                for idx in top_k_docs_sorted_idx 
                if similarities[idx] > 0
            ]
            
            ret_results.append(RetrievalResult(question=q, documents=docs, retrieval_ms=ms_per_query))

        return ret_results


# ---------------------------------------------------------------------------
# Concrete retrievers
# ---------------------------------------------------------------------------


@register("tfidf_syl")
class TfidfSylRetriever(_TfidfBaseRetriever):
    """TF-IDF retriever with syllable-level (whitespace) tokenization."""

    @property
    def _tokenizer_func(self):
        return _tokenize_syllable


@register("tfidf_word")
class TfidfWordRetriever(_TfidfBaseRetriever):
    """TF-IDF retriever with word-level tokenization via ``underthesea``."""

    @property
    def _tokenizer_func(self):
        return _tokenize_word

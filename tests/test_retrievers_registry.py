"""Tests for the retrievers registry and base interface."""
from __future__ import annotations

import pytest


def test_registry_registration():
    """@register should add a class to the internal registry."""
    from rag_bench.retrievers import get_retriever, register
    from rag_bench.retrievers.base import BaseRetriever

    @register("_test_mock")
    class MockRetriever(BaseRetriever):
        def retrieve(self, query: str, **kwargs):
            return []

    instance = get_retriever("_test_mock")
    assert isinstance(instance, MockRetriever)


def test_registry_unknown_key_raises():
    """get_retriever should raise KeyError for unknown strategy."""
    from rag_bench.retrievers import get_retriever

    with pytest.raises(KeyError, match="Unknown retrieval strategy"):
        get_retriever("__definitely_does_not_exist__")


def test_list_strategies_includes_registered():
    """list_strategies should return all registered strategy names."""
    from rag_bench.retrievers import list_strategies, register
    from rag_bench.retrievers.base import BaseRetriever

    @register("_test_another")
    class AnotherRetriever(BaseRetriever):
        def retrieve(self, query: str, **kwargs):
            return []

    strategies = list_strategies()
    assert "_test_another" in strategies


def test_base_retriever_is_abstract():
    """BaseRetriever cannot be instantiated directly."""
    from rag_bench.retrievers.base import BaseRetriever

    with pytest.raises(TypeError):
        BaseRetriever()  # type: ignore[abstract]

"""Query transformation strategies for Advanced RAG."""

from .base import (
    QueryTransformer,
    get_transformer,
    list_strategies,
    register,
    dedup_documents,
)

# Import submodules to trigger @register decorators
from . import passthrough  # noqa: F401
from . import multi_query  # noqa: F401

__all__ = [
    "QueryTransformer",
    "get_transformer",
    "list_strategies",
    "register",
    "dedup_documents",
]
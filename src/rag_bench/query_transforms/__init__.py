"""Query transformation strategies for Advanced RAG."""

# Import submodules to trigger @register decorators
from . import (
    multi_query,  # noqa: F401
    passthrough,  # noqa: F401
)
from .base import (
    QueryTransformer,
    dedup_documents,
    get_transformer,
    list_strategies,
    register,
)

__all__ = [
    "QueryTransformer",
    "dedup_documents",
    "get_transformer",
    "list_strategies",
    "register",
]

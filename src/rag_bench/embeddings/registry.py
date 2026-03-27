"""Embedding model registry for LangChain."""

from __future__ import annotations

from typing import Callable

from langchain_huggingface import HuggingFaceEmbeddings

_REGISTRY: dict[str, Callable[[], HuggingFaceEmbeddings]] = {}


def register(key: str):
    """Decorator to register an embedding model factory."""

    def decorator(factory: Callable[[], HuggingFaceEmbeddings]):
        _REGISTRY[key] = factory
        return factory

    return decorator


def get_embed_model(key: str) -> HuggingFaceEmbeddings:
    """Get an embedding model by registry key."""
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown embedding model '{key}'. Available: {available}")
    return _REGISTRY[key]()


def list_models() -> list[str]:
    """List all registered model keys."""
    return sorted(_REGISTRY.keys())


# --- Registered models ---


@register("bge-small-en-v1.5")
def _bge_small_en_v1_5():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@register("vietnamese-v2")
def _vietnamese_v2():
    return HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding_v2")


@register("jina-v3")
def _jina_v3():
    return HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3")


@register("bge-m3")
def _bge_m3():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


@register("snowflake-v2")
def _snowflake_v2():
    return HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l-v2.0")

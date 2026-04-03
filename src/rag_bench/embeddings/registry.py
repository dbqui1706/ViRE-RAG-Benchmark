"""Embedding model registry for LangChain."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

_REGISTRY: dict[str, Callable[[], Any]] = {}

def register(key: str):
    """Decorator to register an embedding model factory."""

    def decorator(factory: Callable[[], Any]):
        _REGISTRY[key] = factory
        return factory

    return decorator


def get_embed_model(key: str) -> Any:
    """Get an embedding model by registry key."""
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown embedding model '{key}'. Available: {available}")
    return _REGISTRY[key]()


def list_models() -> list[str]:
    """List all registered model keys."""
    return sorted(_REGISTRY.keys())


def _get_model_kwargs() -> dict:
    try:
        import torch
        device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return {"device": device}

# Registered models

@register("bge-small-en-v1.5")
def _bge_small_en_v1_5():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs=_get_model_kwargs()
    )


@register("vietnamese-v2")
def _vietnamese_v2():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        model_kwargs=_get_model_kwargs()
    )


@register("jina-v3")
def _jina_v3():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs=_get_model_kwargs()
    )


@register("bge-m3")
def _bge_m3():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=_get_model_kwargs()
    )

"""Embedding model registry for LangChain."""

from __future__ import annotations

import os
from typing import Callable

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

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


import torch

def _get_model_kwargs() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"device": device}

# Registered models

@register("bge-small-en-v1.5")
def _bge_small_en_v1_5():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs=_get_model_kwargs()
    )


@register("vietnamese-v2")
def _vietnamese_v2():
    return HuggingFaceEmbeddings(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        model_kwargs=_get_model_kwargs()
    )


@register("jina-v3")
def _jina_v3():
    return HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs=_get_model_kwargs()
    )


@register("bge-m3")
def _bge_m3():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=_get_model_kwargs()
    )

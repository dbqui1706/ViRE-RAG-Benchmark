"""Experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RagConfig:
    """Configuration for a single RAG benchmark experiment."""

    csv_path: str
    embed_model: str
    llm_provider: str = "fpt"
    llm_model: str = "Llama-3.3-70B-Instruct"
    llm_api_key: str = ""
    llm_base_url: str = ""
    top_k: int = 5
    max_samples: int = 200
    sample_seed: int = 42
    chroma_dir: str = "outputs/rag/chroma"
    output_dir: str = "outputs/rag"
    prefer_unique: bool = True
    force_reindex: bool = False

    @classmethod
    def from_env(cls, **kwargs) -> RagConfig:
        """Create config, filling API credentials from environment."""
        kwargs.setdefault("llm_api_key", os.environ.get("FPT_API_KEY", ""))
        kwargs.setdefault("llm_base_url", os.environ.get("FPT_BASE_URL", ""))
        return cls(**kwargs)

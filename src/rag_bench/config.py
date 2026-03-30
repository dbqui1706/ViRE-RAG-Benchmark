"""Experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RagConfig:
    """Configuration for a single RAG benchmark experiment."""

    csv_path: str
    embed_model: str
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""  # Empty = use OpenAI default; set for FPT/custom
    top_k: int = 5
    max_samples: int = 200
    sample_seed: int = 42
    chroma_dir: str = "outputs/rag/chroma"
    output_dir: str = "outputs/rag"
    prefer_unique: bool = True
    force_reindex: bool = False
    unified_index_csv: str = ""  # If set, build index from this CSV; queries still per dataset
    # Chunking options
    chunk_strategy: str = "recursive"
    chunk_size: int = 256
    chunk_overlap: int = 50
    # Batch generation
    max_workers: int = 5
    # Prompt strategy
    prompt_strategy: str = "zero_shot"  # "zero_shot" or "few_shot"
    n_few_shot: int = 3  # Number of few-shot examples (auto-selected from dataset)
    # Evaluation options
    include_semantic: bool = False
    eval_faithfulness: bool = False
    judge_model: str = ""
    # Advanced retrieval
    retrieval_strategy: str = "baseline"       # baseline | multi_query
    rerank: bool = False                       # Enable reranking post-retrieval
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_factor: int = 3                     # Over-retrieve k*factor, rerank to k
    # Transform LLM (separate from generation LLM, defaults to FPT)
    transform_llm_model: str = ""              # Empty = use TRANSFORM_LLM_MODEL env
    transform_llm_api_key: str = ""            # Empty = use FPT_API_KEY
    transform_llm_base_url: str = ""           # Empty = use FPT_BASE_URL

    @classmethod
    def from_env(cls, **kwargs) -> RagConfig:
        """Create config, filling API credentials from environment.

        Uses OPENAI_API_KEY by default.
        Set FPT_BASE_URL + FPT_API_KEY to use FPT endpoint instead.
        """
        kwargs.setdefault("llm_api_key", os.environ.get("OPENAI_API_KEY", ""))
        kwargs.setdefault("llm_base_url", os.environ.get("LLM_BASE_URL", ""))
        # Transform LLM defaults (FPT Marketplace)
        kwargs.setdefault("transform_llm_api_key", os.environ.get("FPT_API_KEY", ""))
        kwargs.setdefault("transform_llm_base_url", os.environ.get("FPT_BASE_URL", ""))
        kwargs.setdefault("transform_llm_model", os.environ.get("TRANSFORM_LLM_MODEL", ""))
        return cls(**kwargs)

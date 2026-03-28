"""ChromaDB vector store management"""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import RagConfig


def _collection_name(dataset: str, model_key: str) -> str:
    """Generate a valid ChromaDB collection name."""
    name = f"{dataset}_{model_key}".replace("/", "_").replace("-", "_")
    # ChromaDB collection names: 3-63 chars, alphanumeric + underscores
    return name[:63]


def build_vectorstore(
    documents: list[Document],
    embed_model: HuggingFaceEmbeddings,
    config: RagConfig,
    dataset_name: str,
    model_key: str,
) -> Chroma:
    """Build or load a ChromaDB-backed vector store.

    Args:
        documents: LangChain Documents to index.
        embed_model: The HuggingFace embedding model.
        config: Experiment configuration.
        dataset_name: Name of the dataset (for collection naming).
        model_key: Short key for the embedding model.

    Returns:
        A Chroma vector store.
    """
    chroma_path = str(Path(config.chroma_dir) / dataset_name / model_key)
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    col_name = _collection_name(dataset_name, model_key)

    client = chromadb.PersistentClient(path=chroma_path)

    if config.force_reindex:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(col_name)

    if collection.count() > 0 and not config.force_reindex:
        # Load existing
        vectorstore = Chroma(
            client=client,
            collection_name=col_name,
            embedding_function=embed_model,
        )
    else:
        # Build new index
        vectorstore = Chroma.from_documents(
            documents,
            embedding=embed_model,
            client=client,
            collection_name=col_name,
        )

    return vectorstore

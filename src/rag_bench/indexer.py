"""ChromaDB VectorStoreIndex management."""

from __future__ import annotations

from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import RagConfig


def _collection_name(dataset: str, model_key: str) -> str:
    """Generate a valid ChromaDB collection name."""
    name = f"{dataset}_{model_key}".replace("/", "_").replace("-", "_")
    # ChromaDB collection names: 3-63 chars, alphanumeric + underscores
    return name[:63]


def build_index(
    documents: list[Document],
    embed_model: BaseEmbedding,
    config: RagConfig,
    dataset_name: str,
    model_key: str,
) -> VectorStoreIndex:
    """Build or load a ChromaDB-backed VectorStoreIndex.

    Args:
        documents: LlamaIndex Documents to index.
        embed_model: The embedding model to use.
        config: Experiment configuration.
        dataset_name: Name of the dataset (for collection naming).
        model_key: Short key for the embedding model.

    Returns:
        A VectorStoreIndex backed by ChromaDB.
    """
    chroma_path = str(Path(config.chroma_dir) / dataset_name / model_key)
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_path)
    col_name = _collection_name(dataset_name, model_key)

    if config.force_reindex:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(col_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0 and not config.force_reindex:
        # Load existing index
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
    else:
        # Build new index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

    return index

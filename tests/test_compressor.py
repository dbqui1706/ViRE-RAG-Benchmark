"""Tests for A6 Contextual Compression — ContextualCompressor."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag_bench.retrievers import get_retriever
from rag_bench.retrievers.base import BaseRetriever


class MockCompressResponse:
    """Mimics the Pydantic CompressedChunk model."""
    def __init__(self, compressed_text: str):
        self.compressed_text = compressed_text


class MockBaseRetriever(BaseRetriever):
    """A dummy base retriever for testing decorators."""
    def __init__(self, docs: list[Document]):
        self.docs = docs

    def retrieve(self, query: str) -> list[Document]:
        return self.docs

    def batch_retrieve(self, queries: list[str]) -> list[list[Document]]:
        return [self.docs for _ in queries]


class TestCompressorRegistry:
    """Verify the retriever is registered."""

    def test_registry_key_exists(self):
        with patch("rag_bench.retrievers.compressor.ChatOpenAI"):
            base = MockBaseRetriever([])
            retriever = get_retriever("compressor", base_retriever=base, top_k=5)
            assert retriever is not None


class TestCompressorBehavior:
    """Core behavioral tests for Contextual Compression."""

    @pytest.fixture
    def docs(self):
        return [
            Document(page_content="Hà Nội là thủ đô của Việt Nam. Thời tiết hôm nay đẹp.", metadata={"id": "0"}),
            Document(page_content="Python là ngôn ngữ lập trình phổ biến. Được tạo bởi Guido.", metadata={"id": "1"}),
            Document(page_content="Dân số Việt Nam khoảng 100 triệu người. Kinh tế đang phát triển.", metadata={"id": "2"}),
        ]

    @pytest.fixture
    def compressor(self, docs):
        with patch("rag_bench.retrievers.compressor.ChatOpenAI"):
            base = MockBaseRetriever(docs)
            r = get_retriever("compressor", base_retriever=base, top_k=5)
            r.chain = MagicMock()
            return r

    def test_compresses_chunks_to_relevant_content(self, compressor):
        """Each chunk should be compressed to only the relevant portion."""
        compressor.chain.invoke.side_effect = [
            MockCompressResponse(compressed_text="Hà Nội là thủ đô của Việt Nam."),
            MockCompressResponse(compressed_text=""),  # irrelevant
            MockCompressResponse(compressed_text="Dân số Việt Nam khoảng 100 triệu người."),
        ]

        result = compressor.retrieve("Thông tin về Việt Nam?")

        assert len(result) == 2
        assert result[0].page_content == "Hà Nội là thủ đô của Việt Nam."
        assert result[1].page_content == "Dân số Việt Nam khoảng 100 triệu người."

    def test_empty_compression_filtered_out(self, compressor):
        """Chunks that compress to empty string should be dropped."""
        compressor.chain.invoke.side_effect = [
            MockCompressResponse(compressed_text=""),
            MockCompressResponse(compressed_text="   "),
            MockCompressResponse(compressed_text=""),
        ]

        result = compressor.retrieve("Irrelevant query")

        assert len(result) == 0

    def test_metadata_preserved(self, compressor):
        """Original metadata should be preserved on compressed documents."""
        compressor.chain.invoke.side_effect = [
            MockCompressResponse(compressed_text="Compressed text 0"),
            MockCompressResponse(compressed_text="Compressed text 1"),
            MockCompressResponse(compressed_text="Compressed text 2"),
        ]

        result = compressor.retrieve("test query")

        assert len(result) == 3
        assert result[0].metadata == {"id": "0"}
        assert result[1].metadata == {"id": "1"}
        assert result[2].metadata == {"id": "2"}

    def test_empty_base_retrieval_skips_llm(self, compressor):
        """If base retriever returns nothing, don't call LLM."""
        compressor.base.docs = []

        result = compressor.retrieve("test query")

        assert len(result) == 0
        compressor.chain.invoke.assert_not_called()

    def test_batch_retrieve(self, compressor):
        """batch_retrieve processes each query independently."""
        compressor.chain.invoke.side_effect = [
            # Query 1: 3 docs
            MockCompressResponse(compressed_text="Relevant 1"),
            MockCompressResponse(compressed_text=""),
            MockCompressResponse(compressed_text="Relevant 3"),
            # Query 2: 3 docs
            MockCompressResponse(compressed_text="Another relevant"),
            MockCompressResponse(compressed_text="More relevant"),
            MockCompressResponse(compressed_text=""),
        ]

        result = compressor.batch_retrieve(["query1", "query2"])

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

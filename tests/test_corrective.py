"""Tests for A9 Corrective RAG — CorrectiveRetriever."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag_bench.retrievers import get_retriever
from rag_bench.retrievers.base import BaseRetriever


# Because corrective retriever uses structured output, we mock the Pydantic model response
class MockGrade:
    def __init__(self, chunk_id: int, label: str):
        self.chunk_id = chunk_id
        self.label = label

class MockGradeResponse:
    def __init__(self, grades: list[MockGrade]):
        self.grades = grades


class MockBaseRetriever(BaseRetriever):
    """A dummy base retriever for testing decorators."""
    def __init__(self, docs: list[Document]):
        self.docs = docs

    def retrieve(self, query: str) -> list[Document]:
        return self.docs

    def batch_retrieve(self, queries: list[str]) -> list[list[Document]]:
        return [self.docs for _ in queries]


class TestCorrectiveRetrieverRegistry:
    """Verify the retriever is registered."""

    def test_registry_key_exists(self):
        with patch("rag_bench.retrievers.corrective.ChatOpenAI"):
            base = MockBaseRetriever([])
            retriever = get_retriever("corrective", base_retriever=base, top_k=5)
            assert retriever is not None


class TestCorrectiveRetrieverBehavior:
    """Core behavioral tests for Corrective RAG."""

    @pytest.fixture
    def docs(self):
        return [
            Document(page_content="Context 0", metadata={"id": "0"}),
            Document(page_content="Context 1", metadata={"id": "1"}),
            Document(page_content="Context 2", metadata={"id": "2"}),
        ]

    @pytest.fixture
    def corrective(self, docs):
        with patch("rag_bench.retrievers.corrective.ChatOpenAI"):
            base = MockBaseRetriever(docs)
            r = get_retriever("corrective", base_retriever=base, top_k=5)
            r.chain = MagicMock()
            return r

    def test_filters_incorrect_chunks(self, corrective):
        """Only chunks graded as CORRECT should be returned."""
        # 0 is CORRECT, 1 is INCORRECT, 2 is CORRECT
        corrective.chain.invoke.return_value = MockGradeResponse(
            grades=[
                MockGrade(0, "CORRECT"),
                MockGrade(1, "INCORRECT"),
                MockGrade(2, "CORRECT")
            ]
        )
        
        result = corrective.retrieve("test query")
        
        assert len(result) == 2
        assert result[0].page_content == "Context 0"
        assert result[1].page_content == "Context 2"

    def test_all_incorrect_returns_fallback(self, corrective):
        """If all chunks are INCORRECT, return a single fallback Document."""
        corrective.chain.invoke.return_value = MockGradeResponse(
            grades=[
                MockGrade(0, "INCORRECT"),
                MockGrade(1, "INCORRECT"),
                MockGrade(2, "INCORRECT")
            ]
        )
        
        result = corrective.retrieve("test query")
        
        assert len(result) == 1
        assert "Không đủ thông tin để trả lời" in result[0].page_content

    def test_empty_base_retrieval_returns_fallback(self, corrective):
        """If base retriever returns nothing, short-circuit to fallback."""
        corrective.base.docs = []  # Empty base retrieval
        
        result = corrective.retrieve("test query")
        
        assert len(result) == 1
        assert "Không đủ thông tin để trả lời" in result[0].page_content
        # Ensure LLM was not called to save costs
        corrective.chain.invoke.assert_not_called()

    def test_missing_grades_fallback_to_incorrect(self, corrective):
        """If the LLM fails to output a grade for a chunk, treat it as INCORRECT."""
        # Missing grade for "1" and "2"
        corrective.chain.invoke.return_value = MockGradeResponse(
            grades=[MockGrade(0, "CORRECT")]
        )
        
        result = corrective.retrieve("test query")
        
        assert len(result) == 1
        assert result[0].page_content == "Context 0"

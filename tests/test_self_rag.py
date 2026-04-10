"""Tests for A8 Self-RAG — SelfRAGGenerator."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag_bench.generator import GenerationResult
from rag_bench.retrievers.base import BaseRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockBaseRetriever(BaseRetriever):
    """A dummy retriever for testing Self-RAG."""
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.retrieve_count = 0

    def retrieve(self, query: str) -> list[Document]:
        self.retrieve_count += 1
        return self.docs


class MockSelfEvaluation:
    """Mimics the Pydantic SelfEvaluation model."""
    def __init__(self, verdict: str, reasoning: str = ""):
        self.verdict = verdict
        self.reasoning = reasoning


class MockRefinedQuery:
    """Mimics the Pydantic RefinedQuery model."""
    def __init__(self, query: str):
        self.query = query


# ---------------------------------------------------------------------------
# Task 1: GenerationResult extension
# ---------------------------------------------------------------------------


class TestGenerationResultExtension:
    """Verify GenerationResult supports optional Self-RAG fields."""

    def test_standard_result_has_none_tracking(self):
        """Standard results should have None for Self-RAG tracking fields."""
        r = GenerationResult(text="answer", generation_ms=100.0, input_tokens=10, output_tokens=5)
        assert r.iterations is None
        assert r.total_llm_calls is None

    def test_self_rag_result_has_tracking(self):
        """Self-RAG results should carry iteration count and LLM call count."""
        r = GenerationResult(
            text="answer", generation_ms=100.0, input_tokens=10, output_tokens=5,
            iterations=2, total_llm_calls=5,
        )
        assert r.iterations == 2
        assert r.total_llm_calls == 5


# ---------------------------------------------------------------------------
# Task 2: Config fields
# ---------------------------------------------------------------------------

from rag_bench.config import RagConfig


class TestSelfRAGConfig:
    """Verify config supports Self-RAG fields."""

    def test_default_generation_strategy_is_standard(self):
        cfg = RagConfig(csv_path="test.csv", embed_model="bge-small-en-v1.5")
        assert cfg.generation_strategy == "standard"

    def test_default_max_iter_is_3(self):
        cfg = RagConfig(csv_path="test.csv", embed_model="bge-small-en-v1.5")
        assert cfg.self_rag_max_iter == 3

    def test_can_set_self_rag_strategy(self):
        cfg = RagConfig(
            csv_path="test.csv", embed_model="bge-small-en-v1.5",
            generation_strategy="self_rag", self_rag_max_iter=5,
        )
        assert cfg.generation_strategy == "self_rag"
        assert cfg.self_rag_max_iter == 5


# ---------------------------------------------------------------------------
# Task 3: SelfRAGGenerator behavior
# ---------------------------------------------------------------------------


class TestSelfRAGSupported:
    """Tests for when the first draft is already SUPPORTED."""

    @pytest.fixture
    def self_rag(self):
        with patch("rag_bench.self_rag.ChatOpenAI"):
            from rag_bench.self_rag import SelfRAGGenerator
            docs = [Document(page_content="Hà Nội là thủ đô của Việt Nam.")]
            retriever = MockBaseRetriever(docs)
            gen = SelfRAGGenerator(
                retriever=retriever,
                model="gpt-4o-mini",
                api_key="test-key",
                max_iterations=3,
            )
            # Mock the chains
            gen.gen_chain = MagicMock()
            gen.eval_chain = MagicMock()
            gen.refine_chain = MagicMock()
            return gen

    def test_supported_on_first_try_returns_early(self, self_rag):
        """If eval says SUPPORTED on iteration 1, return immediately."""
        self_rag.gen_chain.invoke.return_value = "Hà Nội là thủ đô."
        self_rag.eval_chain.invoke.return_value = MockSelfEvaluation("SUPPORTED")

        result = self_rag.generate("Thủ đô Việt Nam?", [
            Document(page_content="Hà Nội là thủ đô của Việt Nam.")
        ])

        assert result.text == "Hà Nội là thủ đô."
        assert result.iterations == 1
        assert result.total_llm_calls == 2  # gen + eval
        self_rag.refine_chain.invoke.assert_not_called()


class TestSelfRAGNotSupported:
    """Tests for iterative refinement when NOT_SUPPORTED."""

    @pytest.fixture
    def self_rag(self):
        with patch("rag_bench.self_rag.ChatOpenAI"):
            from rag_bench.self_rag import SelfRAGGenerator
            docs = [Document(page_content="Some context")]
            retriever = MockBaseRetriever(docs)
            gen = SelfRAGGenerator(
                retriever=retriever,
                model="gpt-4o-mini",
                api_key="test-key",
                max_iterations=3,
            )
            gen.gen_chain = MagicMock()
            gen.eval_chain = MagicMock()
            gen.refine_chain = MagicMock()
            return gen

    def test_not_supported_then_supported(self, self_rag):
        """Iter 1: NOT_SUPPORTED -> refine -> re-retrieve -> Iter 2: SUPPORTED."""
        self_rag.gen_chain.invoke.side_effect = ["Draft 1", "Draft 2"]
        self_rag.eval_chain.invoke.side_effect = [
            MockSelfEvaluation("NOT_SUPPORTED"),
            MockSelfEvaluation("SUPPORTED"),
        ]
        self_rag.refine_chain.invoke.return_value = MockRefinedQuery("refined query")

        result = self_rag.generate("Test?", [Document(page_content="Initial")])

        assert result.text == "Draft 2"
        assert result.iterations == 2
        assert result.total_llm_calls == 5  # gen+eval+refine + gen+eval


class TestSelfRAGMaxIterations:
    """Tests for max iteration capping."""

    @pytest.fixture
    def self_rag(self):
        with patch("rag_bench.self_rag.ChatOpenAI"):
            from rag_bench.self_rag import SelfRAGGenerator
            docs = [Document(page_content="Some context")]
            retriever = MockBaseRetriever(docs)
            gen = SelfRAGGenerator(
                retriever=retriever,
                model="gpt-4o-mini",
                api_key="test-key",
                max_iterations=2,  # Only 2 iterations allowed
            )
            gen.gen_chain = MagicMock()
            gen.eval_chain = MagicMock()
            gen.refine_chain = MagicMock()
            return gen

    def test_caps_at_max_iterations(self, self_rag):
        """If never SUPPORTED, return last draft after max_iterations."""
        self_rag.gen_chain.invoke.side_effect = ["Draft 1", "Draft 2"]
        self_rag.eval_chain.invoke.side_effect = [
            MockSelfEvaluation("NOT_SUPPORTED"),
            MockSelfEvaluation("NOT_SUPPORTED"),
        ]
        self_rag.refine_chain.invoke.return_value = MockRefinedQuery("refined")

        result = self_rag.generate("Hard question", [Document(page_content="ctx")])

        assert result.text == "Draft 2"
        assert result.iterations == 2
        # Iter 1: gen+eval+refine=3, Iter 2: gen+eval+refine=3 → total=6
        assert result.total_llm_calls == 6


class TestSelfRAGContextMerge:
    """Tests for context accumulation and deduplication."""

    @pytest.fixture
    def self_rag(self):
        with patch("rag_bench.self_rag.ChatOpenAI"):
            from rag_bench.self_rag import SelfRAGGenerator
            retriever = MockBaseRetriever([
                Document(page_content="New doc"),
                Document(page_content="Initial doc"),  # duplicate of initial
            ])
            gen = SelfRAGGenerator(
                retriever=retriever,
                model="gpt-4o-mini",
                api_key="test-key",
                max_iterations=2,
            )
            gen.gen_chain = MagicMock()
            gen.eval_chain = MagicMock()
            gen.refine_chain = MagicMock()
            return gen

    def test_deduplicates_by_page_content(self, self_rag):
        """Re-retrieved docs with same page_content should be deduplicated."""
        self_rag.gen_chain.invoke.side_effect = ["Draft 1", "Draft 2"]
        self_rag.eval_chain.invoke.side_effect = [
            MockSelfEvaluation("NOT_SUPPORTED"),
            MockSelfEvaluation("SUPPORTED"),
        ]
        self_rag.refine_chain.invoke.return_value = MockRefinedQuery("refined")

        initial_docs = [Document(page_content="Initial doc")]
        result = self_rag.generate("Query?", initial_docs)

        # Verify gen_chain was called with merged (deduplicated) context on iter 2
        second_call_args = self_rag.gen_chain.invoke.call_args_list[1]
        context_text = second_call_args[0][0]["context"]
        # "Initial doc" should appear only once, "New doc" should be added
        assert context_text.count("Initial doc") == 1
        assert "New doc" in context_text


class TestSelfRAGEmptyDocs:
    """Tests for edge cases with empty documents."""

    @pytest.fixture
    def self_rag(self):
        with patch("rag_bench.self_rag.ChatOpenAI"):
            from rag_bench.self_rag import SelfRAGGenerator
            retriever = MockBaseRetriever([])  # Returns empty docs
            gen = SelfRAGGenerator(
                retriever=retriever,
                model="gpt-4o-mini",
                api_key="test-key",
                max_iterations=3,
            )
            gen.gen_chain = MagicMock()
            gen.eval_chain = MagicMock()
            gen.refine_chain = MagicMock()
            return gen

    def test_empty_initial_docs_still_generates(self, self_rag):
        """Even with empty initial docs, should try to generate."""
        self_rag.gen_chain.invoke.return_value = "No info available"
        self_rag.eval_chain.invoke.return_value = MockSelfEvaluation("SUPPORTED")

        result = self_rag.generate("Unknown?", [])

        assert result.text == "No info available"
        assert result.iterations == 1


class TestSelfRAGBatchGenerate:
    """Tests for batch_generate method."""

    @pytest.fixture
    def self_rag(self):
        with patch("rag_bench.self_rag.ChatOpenAI"):
            from rag_bench.self_rag import SelfRAGGenerator
            retriever = MockBaseRetriever([Document(page_content="ctx")])
            gen = SelfRAGGenerator(
                retriever=retriever,
                model="gpt-4o-mini",
                api_key="test-key",
                max_iterations=3,
            )
            gen.gen_chain = MagicMock()
            gen.eval_chain = MagicMock()
            gen.refine_chain = MagicMock()
            return gen

    def test_batch_generate_processes_sequentially(self, self_rag):
        """batch_generate should process each item independently."""
        self_rag.gen_chain.invoke.side_effect = ["Answer 1", "Answer 2"]
        self_rag.eval_chain.invoke.return_value = MockSelfEvaluation("SUPPORTED")

        items = [
            {"question": "Q1", "context": "C1"},
            {"question": "Q2", "context": "C2"},
        ]
        retrieval_results = [
            RetrievalResult(question="Q1", documents=[Document(page_content="C1")], retrieval_ms=10),
            RetrievalResult(question="Q2", documents=[Document(page_content="C2")], retrieval_ms=10),
        ]

        results = self_rag.batch_generate(items, retrieval_results)

        assert len(results) == 2
        assert results[0].text == "Answer 1"
        assert results[1].text == "Answer 2"


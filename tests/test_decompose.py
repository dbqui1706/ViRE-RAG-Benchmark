"""Tests for A12 Multi-hop RAG — DecomposeTransformer."""
import pytest
from unittest.mock import MagicMock, patch
from rag_bench.query_transforms.decompose import DecomposedQuestions
from rag_bench.query_transforms import get_transformer


class TestDecomposeTransformerRegistry:
    """Verify the transformer is registered and constructable."""

    def test_registry_key_exists(self):
        with patch("rag_bench.query_transforms.decompose.ChatOpenAI"):
            transformer = get_transformer("decompose", llm_model="gpt-4o-mini")
            assert transformer is not None


class TestDecomposeTransformerBehavior:
    """Core behavioral tests for question decomposition."""

    @pytest.fixture
    def transformer(self):
        with patch("rag_bench.query_transforms.decompose.ChatOpenAI"):
            t = get_transformer("decompose", llm_model="gpt-4o-mini")
            t.chain = MagicMock()
            return t

    def test_complex_question_decomposed_into_sub_questions(self, transformer):
        """A complex multi-hop question should be split into sub-questions."""
        transformer.chain.invoke.return_value = DecomposedQuestions(
            questions=["Dân số Hà Nội là bao nhiêu?", "Dân số TP.HCM là bao nhiêu?"]
        )
        result = transformer.batch_transform(["So sánh dân số Hà Nội và TP.HCM?"])

        assert len(result) == 1
        queries = result[0]
        # Original question is included, plus sub-questions
        assert queries[0] == "So sánh dân số Hà Nội và TP.HCM?"
        assert len(queries) >= 3  # original + 2 sub-questions

    def test_simple_question_returns_original_only(self, transformer):
        """A simple single-hop question: LLM returns just itself."""
        transformer.chain.invoke.return_value = DecomposedQuestions(
            questions=["Thủ đô của Việt Nam là gì?"]
        )
        result = transformer.batch_transform(["Thủ đô của Việt Nam là gì?"])

        assert len(result) == 1
        queries = result[0]
        # Original + the LLM echoed the same question
        assert queries[0] == "Thủ đô của Việt Nam là gì?"
        assert len(queries) >= 1

    def test_sub_questions_limited_to_max(self, transformer):
        """Sub-questions should be capped at max_sub_questions."""
        transformer.chain.invoke.return_value = DecomposedQuestions(
            questions=["Sub 1?", "Sub 2?", "Sub 3?", "Sub 4?", "Sub 5?", "Sub 6?"]
        )
        result = transformer.batch_transform(["Complex question?"])

        queries = result[0]
        # original + at most max_sub_questions (default 3)
        assert len(queries) <= 4  # 1 original + 3 max

    def test_empty_lines_filtered(self, transformer):
        """Blank lines in LLM response should not become queries."""
        transformer.chain.invoke.return_value = DecomposedQuestions(
            questions=["Sub 1?", " ", "", "Sub 2?", ""]
        )
        result = transformer.batch_transform(["Question?"])

        queries = result[0]
        for q in queries:
            assert q.strip() != ""

    def test_batch_transform_multiple_queries(self, transformer):
        """batch_transform handles multiple input queries independently."""
        transformer.chain.invoke.side_effect = [
            DecomposedQuestions(questions=["Sub A1?", "Sub A2?"]),
            DecomposedQuestions(questions=["Sub B1?"]),
        ]
        result = transformer.batch_transform(["Query A?", "Query B?"])

        assert len(result) == 2
        assert result[0][0] == "Query A?"
        assert result[1][0] == "Query B?"

    def test_max_tokens_is_limited(self, transformer):
        """LLM should be configured with bounded max_tokens."""
        # Access the underlying LLM config — just check it was constructed
        # The real validation is that ChatOpenAI was called with max_tokens
        with patch("rag_bench.query_transforms.decompose.ChatOpenAI") as mock_cls:
            get_transformer("decompose", llm_model="test")
            call_kwargs = mock_cls.call_args
            assert call_kwargs is not None
            # max_tokens should be present in the constructor kwargs
            all_kwargs = call_kwargs[1] if call_kwargs[1] else call_kwargs[0]
            assert "max_tokens" in all_kwargs

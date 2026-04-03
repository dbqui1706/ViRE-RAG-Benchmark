"""Multi-Query Expansion — generate N alternative phrasings via LLM."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .base import QueryTransformer, register

_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant that generates alternative search queries.\n"
    "Given a user question, generate {n} alternative versions of the question "
    "that capture different phrasings, synonyms, or angles.\n"
    "The alternative queries should help retrieve relevant documents that "
    "the original query might miss.\n\n"
    "Keep the same language as the original question.\n"
    "Return ONLY the alternative queries, one per line. "
    "Do NOT include numbering or prefixes.\n\n"
    "Original question: {question}"
)


@register("multi_query")
class MultiQueryTransformer(QueryTransformer):
    """Multi-query expansion via LLM.

    Generates N alternative phrasings of the original query.
    Returns [original_question] + [variant_1, ..., variant_n].
    """

    def __init__(self, llm, n_queries: int = 3, **kwargs):
        super().__init__(llm=llm, **kwargs)
        if self.llm is None:
            raise ValueError("MultiQueryTransformer requires an LLM instance.")
        self.n_queries = n_queries
        self._chain = _PROMPT | self.llm | StrOutputParser()

    def transform(self, question: str) -> list[str]:
        """Generate N alternative phrasings, return original + variants."""
        raw = self._chain.invoke({"question": question, "n": self.n_queries})
        variants = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        # Always include original query + up to n_queries variants
        return [question, *variants[:self.n_queries]]

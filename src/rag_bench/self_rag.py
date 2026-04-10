"""A8 Self-RAG — Iterative retrieve-evaluate-regenerate generation strategy."""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from .generator import GenerationResult, build_prompt
from .retrievers.base import BaseRetriever, RetrievalResult

# ---------------------------------------------------------------------------
# Pydantic Structured Output Models
# ---------------------------------------------------------------------------


class SelfEvaluation(BaseModel):
    """LLM self-evaluation of whether the answer is supported by context."""

    verdict: str = Field(
        description="'SUPPORTED' if the answer is fully supported by the context, "
        "'NOT_SUPPORTED' if key information is missing or contradicted."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why the answer is or is not supported.",
    )


class RefinedQuery(BaseModel):
    """A refined search query to find missing information."""

    query: str = Field(
        description="A new search query designed to find the specific information "
        "that was missing from the original context."
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EVAL_SYSTEM = (
    "Bạn là một chuyên gia đánh giá chất lượng câu trả lời. "
    "Hãy đánh giá xem câu trả lời có được HỖ TRỢ ĐẦY ĐỦ bởi context được cung cấp hay không.\n\n"
    "Trả lời SUPPORTED nếu tất cả thông tin trong câu trả lời đều có trong context.\n"
    "Trả lời NOT_SUPPORTED nếu câu trả lời chứa thông tin không có trong context "
    "hoặc context không đủ để trả lời câu hỏi."
)

EVAL_HUMAN = (
    "Câu hỏi: {question}\n\n"
    "Context:\n{context}\n\n"
    "Câu trả lời: {answer}\n\n"
    "Đánh giá:"
)

REFINE_SYSTEM = (
    "Bạn là một chuyên gia tìm kiếm thông tin. "
    "Dựa vào câu hỏi gốc và bản nháp câu trả lời (chưa đủ thông tin), "
    "hãy tạo một câu truy vấn mới để tìm kiếm thông tin còn thiếu.\n\n"
    "Câu truy vấn mới phải cụ thể và tập trung vào phần thông tin bị thiếu."
)

REFINE_HUMAN = (
    "Câu hỏi gốc: {question}\n\n"
    "Bản nháp câu trả lời: {draft_answer}\n\n"
    "Sinh câu truy vấn tìm kiếm mới:"
)


# ---------------------------------------------------------------------------
# SelfRAGGenerator
# ---------------------------------------------------------------------------


class SelfRAGGenerator:
    """A8 Self-RAG: Iterative retrieve-evaluate-regenerate generation.

    Uses 3 LCEL chains:
        1. gen_chain: Generate answer from question + context
        2. eval_chain: Evaluate if answer is SUPPORTED by context
        3. refine_chain: Generate refined query for missing info

    Loop:
        generate → evaluate → (if NOT_SUPPORTED) refine → re-retrieve → repeat
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str | None = None,
        max_iterations: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 512,
        few_shot_examples: list[dict] | None = None,
    ):
        self.retriever = retriever
        self.max_iterations = max_iterations

        # Shared LLM kwargs
        llm_kwargs: dict[str, Any] = {
            "model": model,
            "openai_api_key": api_key,
            "temperature": temperature,
        }
        if base_url:
            llm_kwargs["openai_api_base"] = base_url

        # ① Generation chain — reuse build_prompt() from generator.py
        gen_llm = ChatOpenAI(**llm_kwargs, max_tokens=max_tokens)
        gen_prompt = build_prompt(few_shot_examples)
        self.gen_chain = gen_prompt | gen_llm | StrOutputParser()

        # ② Evaluation chain — structured output
        eval_llm = ChatOpenAI(**llm_kwargs, max_tokens=128).with_structured_output(
            SelfEvaluation
        )
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", EVAL_SYSTEM),
            ("human", EVAL_HUMAN),
        ])
        self.eval_chain = eval_prompt | eval_llm

        # ③ Refine chain — structured output
        refine_llm = ChatOpenAI(**llm_kwargs, max_tokens=128).with_structured_output(
            RefinedQuery
        )
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", REFINE_SYSTEM),
            ("human", REFINE_HUMAN),
        ])
        self.refine_chain = refine_prompt | refine_llm

    @staticmethod
    def _clean(text: str) -> str:
        """Strip <think> tags from reasoning model outputs."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _merge_docs(
        existing: list[Document], new_docs: list[Document]
    ) -> list[Document]:
        """Merge new documents into existing, deduplicating by page_content."""
        seen = {doc.page_content for doc in existing}
        merged = list(existing)
        for doc in new_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        return merged

    @staticmethod
    def _join_context(docs: list[Document]) -> str:
        """Join document contents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def generate(
        self, question: str, initial_docs: list[Document]
    ) -> GenerationResult:
        """Generate an answer with iterative self-evaluation.

        Args:
            question: The user's question.
            initial_docs: Documents from initial retrieval.

        Returns:
            GenerationResult with text, timing, and Self-RAG tracking fields.
        """
        context_docs = list(initial_docs)
        llm_calls = 0
        draft = ""
        t0 = time.perf_counter()

        for iteration in range(1, self.max_iterations + 1):
            # ① Generate draft answer
            context_text = self._join_context(context_docs)
            draft = self._clean(
                self.gen_chain.invoke({"question": question, "context": context_text})
            )
            llm_calls += 1

            # ② Self-evaluate
            eval_resp = self.eval_chain.invoke({
                "question": question,
                "context": context_text,
                "answer": draft,
            })
            llm_calls += 1

            if eval_resp.verdict == "SUPPORTED":
                gen_ms = (time.perf_counter() - t0) * 1000
                return GenerationResult(
                    text=draft,
                    generation_ms=gen_ms,
                    input_tokens=0,  # Token tracking not available per-chain
                    output_tokens=0,
                    iterations=iteration,
                    total_llm_calls=llm_calls,
                )

            # ③ Refine query
            refine_resp = self.refine_chain.invoke({
                "question": question,
                "draft_answer": draft,
            })
            llm_calls += 1

            # ④ Re-retrieve and merge context
            new_docs = self.retriever.retrieve(refine_resp.query)
            context_docs = self._merge_docs(context_docs, new_docs)

        # Max iterations reached — return last draft
        gen_ms = (time.perf_counter() - t0) * 1000
        return GenerationResult(
            text=draft,
            generation_ms=gen_ms,
            input_tokens=0,
            output_tokens=0,
            iterations=self.max_iterations,
            total_llm_calls=llm_calls,
        )

    def batch_generate(
        self,
        items: list[dict],
        retrieval_results: list[RetrievalResult],
    ) -> list[GenerationResult]:
        """Generate answers for multiple questions sequentially.

        Self-RAG is inherently sequential (each question has its own
        iterative loop), so no concurrency is possible.

        Args:
            items: List of {"question": str, "context": str} dicts.
            retrieval_results: Corresponding RetrievalResult objects.

        Returns:
            List of GenerationResult, one per item.
        """
        results: list[GenerationResult] = []
        for item, ret in tqdm(
            zip(items, retrieval_results, strict=True),
            total=len(items),
            desc="Self-RAG Generating",
        ):
            result = self.generate(item["question"], ret.documents)
            results.append(result)
        return results

"""Multi-hop RAG — Decompose complex questions into sub-questions."""
from __future__ import annotations
from .base import QueryTransformer, register
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tqdm import tqdm
import os


@register("decompose")
def _factory(**kwargs) -> DecomposeTransformer:
    llm_model = kwargs.get("llm_model", "gpt-4o-mini")
    base_url = kwargs.get("base_url") or ""
    api_key = kwargs.get("api_key", "")
    max_sub_questions = kwargs.get("max_sub_questions", 3)

    if not api_key:
        print("Warning: API Key is not set.")

    return DecomposeTransformer(
        llm_model=llm_model, base_url=base_url, api_key=api_key,
        max_sub_questions=max_sub_questions,
    )


class DecomposedQuestions(BaseModel):
    questions: list[str] = Field(
        description="Danh sách các câu hỏi con đơn giản. Nếu câu hỏi gốc đã đơn giản, trả về danh sách chứa chính câu hỏi đó."
    )

class DecomposeTransformer(QueryTransformer):
    """Decompose a complex question into simpler sub-questions for multi-hop retrieval."""

    def __init__(
        self, llm_model: str, base_url: str | None, api_key: str | None,
        max_sub_questions: int = 3,
    ):
        print(f"LLM Model (Decompose): {llm_model}, Base URL: {base_url}")

        chat_kwargs = {
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": 512,
        }
        if api_key:
            chat_kwargs["openai_api_key"] = api_key
        if base_url:
            chat_kwargs["openai_api_base"] = base_url

        self.llm = ChatOpenAI(**chat_kwargs).with_structured_output(DecomposedQuestions)
        self.max_sub = max_sub_questions
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str], max_concurrency: int = 5) -> list[list[str]]:
        from .base import BatchProgressCallback
        cb = BatchProgressCallback(len(queries), desc="Decomposing queries")
        inputs = [{"question": q, "max_sub": self.max_sub} for q in queries]
        responses = self.chain.batch(
            inputs,
            config={"max_concurrency": max_concurrency, "callbacks": [cb]},
        )
        cb.close()
        results = []
        for q, resp in zip(queries, responses):
            valid_questions = [sq.strip() for sq in resp.questions if sq.strip()]
            sub_questions = valid_questions[:self.max_sub]
            results.append([q] + sub_questions)
        return results

    def _build_prompt(self) -> ChatPromptTemplate:
        from langchain_core.prompts import (
            SystemMessagePromptTemplate, HumanMessagePromptTemplate,
        )

        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "Bạn là một chuyên gia phân tích câu hỏi. Nhiệm vụ của bạn là phân tách câu hỏi phức tạp "
            "thành tối đa {max_sub} câu hỏi con đơn giản hơn.\n\n"
            "Nếu câu hỏi đã đơn giản và không cần phân tách, chỉ trả lại chính câu hỏi đó."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template("{question}")

        return ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

"""Corrective RAG (CRAG) — Post-retrieval chunks grading."""
from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from . import register
from .base import BaseRetriever


class Grade(BaseModel):
    chunk_id: int = Field(description="The numeric index of the chunk being graded (e.g., 0)")
    label: str = Field(description="'CORRECT' or 'INCORRECT'")

class GradeResponse(BaseModel):
    grades: list[Grade] = Field(
        description="A list of grades corresponding to each chunk evaluated"
    )


@register("corrective")
def _factory(base_retriever: BaseRetriever, **kwargs) -> CorrectiveRetriever:
    llm_model = kwargs.get("model", "gpt-4o-mini")
    base_url = kwargs.get("base_url") or ""
    api_key = kwargs.get("api_key", "")
    top_k = kwargs.get("top_k", 5)

    if not api_key:
        print("Warning: API Key is not set. Please set FPT_API_KEY environment variable.")

    return CorrectiveRetriever(
        base_retriever=base_retriever,
        llm_model=llm_model,
        base_url=base_url,
        api_key=api_key,
        top_k=top_k,
    )


class CorrectiveRetriever(BaseRetriever):
    """Filter retrieved chunks by grading their relevance using an LLM."""

    def __init__(
        self, base_retriever: BaseRetriever, llm_model: str, 
        base_url: str | None, api_key: str | None, top_k: int = 5
    ):
        self.base = base_retriever
        self._top_k = top_k

        print(f"LLM Model (CRAG): {llm_model}, Base URL: {base_url}")

        chat_kwargs = {
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": 128,  # Enough for a JSON dict with a few keys
        }
        if api_key:
            chat_kwargs["openai_api_key"] = api_key
        if base_url:
            chat_kwargs["openai_api_base"] = base_url

        self.llm = ChatOpenAI(**chat_kwargs).with_structured_output(GradeResponse)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

    def _build_prompt(self) -> ChatPromptTemplate:
        from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "Bạn là một chuyên gia đánh giá thông tin. Hãy đánh giá các đoạn văn bản (chunks) được cung cấp "
            "có chứa thông tin liên quan và đủ khả năng để trả lời cho câu hỏi tương ứng hay không.\n\n"
            "Dán nhãn 'CORRECT' nếu đoạn văn chứa thông tin trả lời được câu hỏi.\n"
            "Dán nhãn 'INCORRECT' nếu đoạn văn không liên quan hoặc nằm ngoài ngữ cảnh câu hỏi."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template(
            "Câu hỏi: {question}\n\nCác đoạn văn bản:\n{chunks}"
        )

        return ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

    def _get_fallback_docs(self) -> list[Document]:
        """Return the constant fallback document when everything is INCORRECT."""
        return [Document(page_content="Không đủ thông tin để trả lời")]

    def retrieve(self, query: str) -> list[Document]:
        docs = self.base.retrieve(query)
        
        if not docs:
            # Short-circuit if base retriever returns nothing to save LLM cost
            return self._get_fallback_docs()

        # Format the chunks for the prompt
        chunks_text = "\n".join(
            f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)
        )

        resp = self.chain.invoke({
            "question": query,
            "chunks": chunks_text
        })

        graded_docs = []
        grades_list = getattr(resp, "grades", [])
        grades_map = {g.chunk_id: g.label for g in grades_list}
        
        for i, doc in enumerate(docs):
            label = grades_map.get(i, "INCORRECT").upper()
            if label == "CORRECT":
                graded_docs.append(doc)

        if not graded_docs:
            return self._get_fallback_docs()

        return graded_docs[:self._top_k]


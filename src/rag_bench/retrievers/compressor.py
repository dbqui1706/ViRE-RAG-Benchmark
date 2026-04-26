"""A6 Contextual Compression — Post-retrieval chunk compression."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel, Field

from . import register
from .base import BaseRetriever


class CompressedChunk(BaseModel):
    compressed_text: str = Field(
        description="Phần nội dung liên quan đến câu hỏi được trích xuất từ đoạn văn. "
                    "Trả về chuỗi rỗng nếu đoạn văn hoàn toàn không liên quan."
    )


@register("compressor")
def _factory(base_retriever: BaseRetriever, **kwargs) -> ContextualCompressor:
    llm_model = kwargs.get("model", "gpt-4o-mini")
    base_url = kwargs.get("base_url") or ""
    api_key = kwargs.get("api_key", "")
    top_k = kwargs.get("top_k", 5)
    compress_max_tokens = kwargs.get("compress_max_tokens", 128)

    if not api_key:
        print("Warning: API Key is not set. Please set FPT_API_KEY environment variable.")

    return ContextualCompressor(
        base_retriever=base_retriever,
        llm_model=llm_model,
        base_url=base_url,
        api_key=api_key,
        top_k=top_k,
        max_tokens=compress_max_tokens,
    )


class ContextualCompressor(BaseRetriever):
    """Compress each retrieved chunk to only the query-relevant portion using an LLM."""

    def __init__(
        self, base_retriever: BaseRetriever, llm_model: str,
        base_url: str | None, api_key: str | None,
        top_k: int = 5, max_tokens: int = 128,
    ):
        self.base = base_retriever
        self._top_k = top_k
        self._warned_length_limit = False

        print(f"LLM Model (Compressor): {llm_model}, Base URL: {base_url}")

        chat_kwargs = {
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if api_key:
            chat_kwargs["openai_api_key"] = api_key
        if base_url:
            chat_kwargs["openai_api_base"] = base_url

        self.llm = ChatOpenAI(**chat_kwargs).with_structured_output(CompressedChunk)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

    def _compress_doc(self, query: str, doc: Document) -> Document | None:
        """Compress one document and fall back gracefully on truncated output."""
        try:
            resp = self.chain.invoke({
                "question": query,
                "passage": doc.page_content,
            })
        except LengthFinishReasonError:
            if not self._warned_length_limit:
                print(
                    "Warning: Compressor output hit max_tokens; "
                    "falling back to original chunks for truncated responses. "
                    "Increase --compress-max-tokens to reduce this."
                )
                self._warned_length_limit = True
            return Document(page_content=doc.page_content, metadata=doc.metadata)

        text = getattr(resp, "compressed_text", "").strip()
        if not text:
            return None
        return Document(page_content=text, metadata=doc.metadata)

    def _build_prompt(self) -> ChatPromptTemplate:
        from langchain_core.prompts import (
            HumanMessagePromptTemplate,
            SystemMessagePromptTemplate,
        )

        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "Bạn là một chuyên gia trích xuất thông tin. "
            "Nhiệm vụ của bạn là trích xuất ĐÚNG các câu liên quan đến câu hỏi từ đoạn văn được cung cấp.\n\n"
            "Nếu đoạn văn hoàn toàn không liên quan đến câu hỏi, trả về chuỗi rỗng."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template(
            "Câu hỏi: {question}\n\nĐoạn văn:\n{passage}"
        )

        return ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

    def retrieve(self, query: str) -> list[Document]:
        docs = self.base.retrieve(query)

        if not docs:
            return []

        compressed = []
        for doc in docs:
            compressed_doc = self._compress_doc(query, doc)
            if compressed_doc is not None:
                compressed.append(compressed_doc)

        return compressed[:self._top_k]

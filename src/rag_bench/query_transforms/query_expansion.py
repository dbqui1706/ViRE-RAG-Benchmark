from __future__ import annotations
from .base import QueryTransformer, register
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tqdm import tqdm
from dotenv import load_dotenv
import os

@register("query_expansion")
def _factory(**kwargs) -> QueryExpansionTransformer:
    # Use config overrides or defaults
    llm_model = kwargs.get("llm_model", "gpt-4o-mini")
    base_url = kwargs.get("base_url") or ""
    api_key = kwargs.get("api_key", "")
    n_variations = kwargs.get("n_variations", 3)
    
    if not api_key:
        print("Warning: API Key is not set.")
        
    return QueryExpansionTransformer(llm_model=llm_model, base_url=base_url, api_key=api_key)

class ExpansionResponse(BaseModel):
    keywords: str = Field(description="Các từ khóa, từ đồng nghĩa hoặc khái niệm liên quan trọng yếu nhất (tối đa 4-5 từ/cụm từ), phân cách bằng dấu phẩy.")

class QueryExpansionTransformer(QueryTransformer):
    def __init__(self, llm_model: str, api_key: str | None, base_url: str | None):
        print(f"LLM Model (Expansion): {llm_model}, Base URL: {base_url}")
        
        chat_kwargs = {
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": 256,
        }
        if api_key:
            chat_kwargs["openai_api_key"] = api_key
        if base_url:
            chat_kwargs["openai_api_base"] = base_url
        self.llm = ChatOpenAI(**chat_kwargs).with_structured_output(ExpansionResponse)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str], max_concurrency: int = 5) -> list[list[str]]:
        from .base import BatchProgressCallback
        cb = BatchProgressCallback(len(queries), desc="Expanding queries")
        inputs = [{"question": q} for q in queries]
        responses = self.chain.batch(
            inputs,
            config={"max_concurrency": max_concurrency, "callbacks": [cb]},
        )
        cb.close()
        return [
            [f"{q} {resp.keywords.strip()}"]
            for q, resp in zip(queries, responses)
        ]

    def _build_prompt(self) -> ChatPromptTemplate:
        from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
        
        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "Bạn là một chuyên gia ngôn ngữ học. Nhiệm vụ của bạn là mở rộng câu hỏi của người dùng "
            "bằng cách trích xuất các từ khóa chính, sau đó cung cấp thêm các từ đồng nghĩa (synonyms) "
            "hoặc các thuật ngữ/khái niệm liên quan trọng yếu nhất (tối đa 4-5 từ/cụm từ)."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template("{question}") 
        
        return ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

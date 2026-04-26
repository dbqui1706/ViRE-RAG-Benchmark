from __future__ import annotations
from .base import QueryTransformer, register
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm
import os

@register("hyde")
def _factory(**kwargs) -> HydeTransformer:
    llm_model = kwargs.get("llm_model", "gpt-4o-mini")
    base_url = kwargs.get("base_url") or "https://openrouter.ai/api/v1"
    api_key = kwargs.get("api_key", "")
    
    if not api_key:
        print("Warning: API Key is not set.")
        
    return HydeTransformer(llm_model=llm_model, base_url=base_url, api_key=api_key)

class HydeResponse(BaseModel):
    document: str = Field(description="Đoạn văn bản giả định bằng tiếng Việt để trả lời câu hỏi.")

class HydeTransformer(QueryTransformer):
    def __init__(self, llm_model: str, base_url: str | None, api_key: str | None):
        
        chat_kwargs = {
            "model": llm_model,
            "temperature": 0.0,
        }
        if api_key:
            chat_kwargs["openai_api_key"] = api_key
        if base_url:
            chat_kwargs["openai_api_base"] = base_url
            
        self.llm = ChatOpenAI(**chat_kwargs).with_structured_output(HydeResponse)
        
        from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "Bạn là một trợ lý AI. Hãy viết một đoạn văn bản giả định bằng tiếng Việt để trả lời câu hỏi dưới đây."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template("{question}")
        
        self.prompt = ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str], max_concurrency: int = 5) -> list[list[str]]:
        from .base import BatchProgressCallback
        cb = BatchProgressCallback(len(queries), desc="HyDE generating")
        inputs = [{"question": q} for q in queries]
        responses = self.chain.batch(
            inputs,
            config={"max_concurrency": max_concurrency, "callbacks": [cb]},
        )
        cb.close()
        return [[resp.document.strip()] for resp in responses]

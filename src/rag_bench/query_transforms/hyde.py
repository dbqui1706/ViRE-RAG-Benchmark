from __future__ import annotations
from .base import QueryTransformer, register
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os

@register("hyde")
def _factory(**kwargs) -> HydeTransformer:
    llm_model = kwargs.get("llm_model", "gpt-4o-mini")
    base_url = kwargs.get("base_url") or os.environ.get("LLM_BASE_URL", "")
    api_key = kwargs.get("api_key") or os.environ.get("FPT_API_KEY", "")
    return HydeTransformer(llm_model=llm_model, base_url=base_url, api_key=api_key)

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
            
        self.llm = ChatOpenAI(**chat_kwargs)
        
        from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "Bạn là một trợ lý AI. Hãy viết một đoạn văn bản giả định bằng tiếng Việt để trả lời câu hỏi dưới đây. Không cần giải thích thêm, chỉ viết nội dung của đoạn văn bản."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template("{question}")
        
        self.prompt = ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        results = []
        for q in queries:
            resp = self.chain.invoke({"question": q})
            doc = str(resp.content).strip()
            results.append([doc])
        return results

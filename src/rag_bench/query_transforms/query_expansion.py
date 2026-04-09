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
    base_url = os.environ.get("FPT_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://mkp-api.fptcloud.com")
    
    # Priority: FPT_API_KEY (direct from env) > api_key (from kwargs)
    api_key = os.environ.get("FPT_API_KEY") or kwargs.get("api_key", "")
    n_variations = kwargs.get("n_variations", 3)
    
    if not api_key:
        print("Warning: API Key is not set. Please set FPT_API_KEY environment variable.")
        
    return QueryExpansionTransformer(llm_model=llm_model, base_url=base_url, api_key=api_key)

class ExpansionResponse(BaseModel):
    keywords: str = Field(description="Các từ khóa, từ đồng nghĩa hoặc khái niệm liên quan trọng yếu nhất (tối đa 4-5 từ/cụm từ), phân cách bằng dấu phẩy.")

class QueryExpansionTransformer(QueryTransformer):
    def __init__(self, llm_model: str, base_url: str | None, api_key: str | None):
        print(f"LLM Model (Expansion): {llm_model}, Base URL: {base_url}")
        
        chat_kwargs = {
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": 64,
        }
        if api_key:
            chat_kwargs["openai_api_key"] = api_key
        if base_url:
            chat_kwargs["openai_api_base"] = base_url
            
        self.llm = ChatOpenAI(**chat_kwargs).with_structured_output(ExpansionResponse)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        results = []
        for q in tqdm(queries, desc="Expanding queries"):
            resp = self.chain.invoke({"question": q})
            keywords = resp.keywords.strip()
            
            # Combine the original query with the generated keywords
            expanded_query = f"{q} {keywords}"
            results.append([expanded_query])
            
        return results

    def _build_prompt(self) -> ChatPromptTemplate:
        from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
        
        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "Bạn là một chuyên gia ngôn ngữ học. Nhiệm vụ của bạn là mở rộng câu hỏi của người dùng "
            "bằng cách trích xuất các từ khóa chính, sau đó cung cấp thêm các từ đồng nghĩa (synonyms) "
            "hoặc các thuật ngữ/khái niệm liên quan trọng yếu nhất (tối đa 4-5 từ/cụm từ)."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template("{question}") 
        
        return ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

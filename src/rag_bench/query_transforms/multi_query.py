from __future__ import annotations
from .base import QueryTransformer, register
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from tqdm import tqdm
from dotenv import load_dotenv
import os

@register("multi_query")
def _factory(**kwargs) -> MultiQueryTransformer:
    # Use config overrides or defaults
    llm_model = kwargs.get("llm_model", "gpt-4o-mini")
    base_url = os.environ.get("FPT_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://mkp-api.fptcloud.com")
    
    # Priority: FPT_API_KEY (direct from env) > api_key (from kwargs)
    api_key = os.environ.get("FPT_API_KEY") or kwargs.get("api_key", "")
    n_variations = kwargs.get("n_variations", 3)
    
    if not api_key:
        print("Warning: API Key is not set. Please set FPT_API_KEY environment variable.")
        
    return MultiQueryTransformer(llm_model=llm_model, base_url=base_url, api_key=api_key, n_variations=n_variations)


class MultiQueryTransformer(QueryTransformer):
    def __init__(self, llm_model: str, base_url: str | None, api_key: str | None, n_variations: int):
        print(f"LLM Model: {llm_model}, Base URL: {base_url}, N Variations: {n_variations}")
        
        chat_kwargs = {
            "model": llm_model,
            "temperature": 0.0,
        }
        if api_key:
            chat_kwargs["openai_api_key"] = api_key
        if base_url:
            chat_kwargs["openai_api_base"] = base_url
            
        self.llm = ChatOpenAI(**chat_kwargs)
        self.n_variations = n_variations
        self.prompt = self._build_prompt(n_variations)
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        results = []
        for q in tqdm(queries, desc="Generating query variations"):
            resp = self.chain.invoke({"n_variations": self.n_variations, "question": q})
            variations = [line.strip() for line in str(resp.content).split("\n") if line.strip()]
            results.append([q] + variations[:self.n_variations])
        return results

    def _build_prompt(self, n_variations: int) -> ChatPromptTemplate:
        from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
        
        system_prompt_template = SystemMessagePromptTemplate.from_template(
            f"Bạn là một trợ lý AI. Nhiệm vụ của bạn là tạo ra {n_variations} cách viết khác nhau cho câu hỏi của người dùng bằng tiếng Việt, giúp tối ưu hóa việc tìm kiếm thông tin. Mỗi câu hỏi phải ở trên một dòng riêng biệt, không có số thứ tự hay ký tự gạch đầu dòng."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template("{question}") 
        
        return ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

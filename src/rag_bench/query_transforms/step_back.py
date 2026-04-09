from __future__ import annotations
from .base import QueryTransformer, register
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tqdm import tqdm
from dotenv import load_dotenv
import os

@register("step_back")
def _factory(**kwargs) -> StepBackTransformer:
    # Use config overrides or defaults
    llm_model = kwargs.get("llm_model", "gpt-4o-mini")
    base_url = os.environ.get("FPT_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://mkp-api.fptcloud.com")
    
    # Priority: FPT_API_KEY (direct from env) > api_key (from kwargs)
    api_key = os.environ.get("FPT_API_KEY") or kwargs.get("api_key", "")
    
    if not api_key:
        print("Warning: API Key is not set. Please set FPT_API_KEY environment variable.")
        
    return StepBackTransformer(llm_model=llm_model, base_url=base_url, api_key=api_key)

class StepBackResponse(BaseModel):
    question: str = Field(description="Câu hỏi mang tính nguyên lý hoặc tổng quát hơn.")

class StepBackTransformer(QueryTransformer):
    """Generates a step-back (generalized) question and returns both original and step-back question."""
    
    def __init__(self, llm_model: str, base_url: str | None, api_key: str | None):
        print(f"LLM Model (Step-Back): {llm_model}, Base URL: {base_url}")
        
        chat_kwargs = {
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": 64,
        }
        if api_key:
            chat_kwargs["openai_api_key"] = api_key
        if base_url:
            chat_kwargs["openai_api_base"] = base_url
            
        self.llm = ChatOpenAI(**chat_kwargs).with_structured_output(StepBackResponse)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        results = []
        for q in tqdm(queries, desc="Generating step-back queries"):
            resp = self.chain.invoke({"question": q})
            step_back_q = resp.question.strip()
            
            # Return both the original query and the step-back query for RRF fusion
            # ExpandedRetriever will retrieve both independently and merge them
            results.append([q, step_back_q])
            
        return results

    def _build_prompt(self) -> ChatPromptTemplate:
        from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
        
        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "Bạn là một chuyên gia. Nhiệm vụ của bạn là chuyển đổi câu hỏi cụ thể của người dùng "
            "thành một câu hỏi mang tính nguyên lý hoặc có phạm vi tổng quát hơn (Step-back question). "
            "Điều này giúp truy xuất được những tài liệu chứa thông tin nền tảng quan trọng trước khi đi vào chi tiết."
        )
        human_prompt_template = HumanMessagePromptTemplate.from_template("{question}") 
        
        return ChatPromptTemplate.from_messages([system_prompt_template, human_prompt_template])

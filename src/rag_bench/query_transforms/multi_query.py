from __future__ import annotations
from .base import QueryTransformer, register
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

@register("multi_query")
def _factory(**kwargs) -> MultiQueryTransformer:
    # Use config overrides or defaults
    llm_model = kwargs.get("llm_model", "gpt-4o-mini")
    base_url = kwargs.get("base_url")
    n_variations = kwargs.get("n_variations", 3)
    return MultiQueryTransformer(llm_model=llm_model, base_url=base_url, n_variations=n_variations)

class MultiQueryTransformer(QueryTransformer):
    def __init__(self, llm_model: str, base_url: str | None, n_variations: int):
        self.llm = ChatOpenAI(model=llm_model, base_url=base_url)
        self.n_variations = n_variations
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý AI. Nhiệm vụ của bạn là tạo ra {n} cách viết khác nhau cho câu hỏi của người dùng bằng tiếng Việt, giúp tối ưu hóa việc tìm kiếm thông tin. Mỗi câu hỏi phải ở trên một dòng riêng biệt, không có số thứ tự hay ký tự gạch đầu dòng."),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        results = []
        for q in queries:
            resp = self.chain.invoke({"n": self.n_variations, "question": q})
            variations = [line.strip() for line in str(resp.content).split("\n") if line.strip()]
            results.append([q] + variations[:self.n_variations])
        return results

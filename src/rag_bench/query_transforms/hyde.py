from __future__ import annotations
from .base import QueryTransformer, register
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

@register("hyde")
def _factory(**kwargs) -> HydeTransformer:
    llm_model = kwargs.get("llm_model", "gpt-4o-mini")
    base_url = kwargs.get("base_url")
    return HydeTransformer(llm_model=llm_model, base_url=base_url)

class HydeTransformer(QueryTransformer):
    def __init__(self, llm_model: str, base_url: str | None):
        self.llm = ChatOpenAI(model=llm_model, base_url=base_url)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý AI. Hãy viết một đoạn văn bản giả định bằng tiếng Việt để trả lời câu hỏi dưới đây. Không cần giải thích thêm, chỉ viết nội dung của đoạn văn bản."),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.llm

    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        results = []
        for q in queries:
            resp = self.chain.invoke({"question": q})
            doc = str(resp.content).strip()
            results.append([doc])
        return results

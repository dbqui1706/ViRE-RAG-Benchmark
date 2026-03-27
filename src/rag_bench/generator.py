"""FPT LLM generation using LangChain LCEL chain."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class GenerationResult:
    """Result of a single generation call."""

    text: str
    generation_ms: float
    input_tokens: int
    output_tokens: int


# PromptTemplate — reusable across all generations
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful Vietnamese QA assistant. "
     "Answer the question based only on the provided context. "
     "Answer in Vietnamese. /no_think"),
    ("human",
     "Context:\n{context}\n\n"
     "Question: {question}\n\n"
     "Answer:"),
])


class FPTGenerator:
    """FPT API client using LangChain LCEL chain.

    FPT API is OpenAI-compatible, so we use ChatOpenAI directly.
    """

    def __init__(
        self,
        model: str = "Qwen3-32B",
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # LCEL chain: prompt → llm → parse string
        self.chain = PROMPT | self.llm | StrOutputParser()

    def _clean(self, text: str) -> str:
        """Strip <think> tags from LLM output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(self, question: str, context: str) -> GenerationResult:
        """Single generation."""
        t0 = time.perf_counter()
        response = self.chain.invoke({"question": question, "context": context})
        gen_ms = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            text=self._clean(response),
            generation_ms=gen_ms,
            input_tokens=0,
            output_tokens=0,
        )

    def batch_generate(
        self,
        items: list[dict],
        max_workers: int = 5,
    ) -> list[GenerationResult]:
        """Batch generation using LangChain .batch() with concurrency control."""
        inputs = [
            {"question": item["question"], "context": item["context"]}
            for item in items
        ]

        t0 = time.perf_counter()
        responses = self.chain.batch(
            inputs,
            config={"max_concurrency": max_workers},
        )
        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / len(responses) if responses else 0

        return [
            GenerationResult(
                text=self._clean(r),
                generation_ms=avg_ms,
                input_tokens=0,
                output_tokens=0,
            )
            for r in responses
        ]

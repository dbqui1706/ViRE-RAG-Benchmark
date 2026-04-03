"""OpenAI-compatible LLM generation using LangChain LCEL chain."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm


class BatchCallback(BaseCallbackHandler):
    """Callback for batch progress bar and token tracking."""

    def __init__(self, total: int):
        super().__init__()
        self.progress_bar = tqdm(total=total, desc="Generating")
        self.token_usage: list[dict] = []

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        """Called after every LLM response — track progress and tokens."""
        self.progress_bar.update(1)
        # Extract token usage from OpenAI-compatible response
        usage = {}
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
        self.token_usage.append({
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        })

    def close(self):
        self.progress_bar.close()

@dataclass
class GenerationResult:
    """Result of a single generation call."""

    text: str
    generation_ms: float
    input_tokens: int
    output_tokens: int

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are a helpful Vietnamese QA assistant. "
    "Answer the question based ONLY on the provided context. "
    "Be concise and direct — do NOT add information not in the context. "
    "Answer in Vietnamese."
)


def build_prompt(few_shot_examples: list[dict] | None = None) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate, optionally with few-shot examples.

    Args:
        few_shot_examples: List of dicts with 'question' and 'answer' keys.
            If None or empty, returns a zero-shot prompt.

    Returns:
        ChatPromptTemplate with {context} and {question} input variables.
    """
    if not few_shot_examples:
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MSG),
            ("human",
             "Context:\n{context}\n\n"
             "Question: {question}\n\n"
             "Answer:"),
        ])

    # Build few-shot block from examples
    examples_text = "Here are examples of good answers — concise, grounded in context:\n\n---\n"
    for i, ex in enumerate(few_shot_examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Question: {ex['question']}\n"
        examples_text += f"Answer: {ex['answer']}\n\n"
    examples_text += "---"

    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG + "\n\n" + examples_text),
        ("human",
         "Context:\n{context}\n\n"
         "Question: {question}\n\n"
         "Answer:"),
    ])


class OpenAIGenerator:
    """OpenAI-compatible LLM generator using LangChain LCEL chain.

    Works with OpenAI, FPT, and any OpenAI-compatible API via base_url.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        few_shot_examples: list[dict] | None = None,
    ):
        kwargs = dict(
            model=model,
            openai_api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if base_url:
            kwargs["openai_api_base"] = base_url
        self.llm = ChatOpenAI(**kwargs)
        prompt = build_prompt(few_shot_examples)
        # LCEL chain: prompt → llm → parse string
        self.chain = prompt | self.llm | StrOutputParser()

    def _clean(self, text: str) -> str:
        """Strip <think> tags from LLM output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(self, question: str, context: str) -> GenerationResult:
        """Single generation."""
        cb = BatchCallback(total=1)
        t0 = time.perf_counter()
        response = self.chain.invoke(
            {"question": question, "context": context},
            config={"callbacks": [cb]},
        )
        gen_ms = (time.perf_counter() - t0) * 1000
        cb.close()

        tokens = cb.token_usage[0] if cb.token_usage else {}
        return GenerationResult(
            text=self._clean(response),
            generation_ms=gen_ms,
            input_tokens=tokens.get("input_tokens", 0),
            output_tokens=tokens.get("output_tokens", 0),
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

        cb = BatchCallback(total=len(inputs))
        t0 = time.perf_counter()
        responses = self.chain.batch(
            inputs,
            config={
                "max_concurrency": max_workers,
                "callbacks": [cb],
            },
        )
        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / len(responses) if responses else 0
        cb.close()

        results = []
        for i, r in enumerate(responses):
            tokens = cb.token_usage[i] if i < len(cb.token_usage) else {}
            results.append(GenerationResult(
                text=self._clean(r),
                generation_ms=avg_ms,
                input_tokens=tokens.get("input_tokens", 0),
                output_tokens=tokens.get("output_tokens", 0),
            ))
        return results


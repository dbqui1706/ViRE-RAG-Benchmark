"""FPT marketplace LLM — standalone HTTP client with batch support."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class GenerationResult:
    """Result of a single generation call."""

    text: str
    generation_ms: float
    input_tokens: int
    output_tokens: int


class FPTGenerator:
    """FPT marketplace API client for text generation.

    No framework dependency — plain HTTP requests with timing.
    """

    def __init__(
        self,
        model: str = "Qwen3-32B",
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, question: str, context: str) -> GenerationResult:
        """Generate an answer for a question given context.

        Args:
            question: The user question.
            context: Retrieved context to ground the answer.

        Returns:
            GenerationResult with text, timing, and token counts.
        """
        t0 = time.perf_counter()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Vietnamese QA assistant. "
                        "Answer the question based only on the provided context. "
                        "Answer in Vietnamese. /no_think"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Answer:"
                    ),
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=60)

        if resp.status_code != 200:
            error_detail = resp.text[:500] if resp.text else "No response body"
            raise RuntimeError(
                f"FPT API error {resp.status_code} at {url}\n"
                f"Response: {error_detail}\n"
                f"Check: 1) API key is valid  2) Model name '{self.model}' exists  "
                f"3) Base URL is correct"
            )

        data = resp.json()

        answer = data["choices"][0]["message"]["content"]
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        usage = data.get("usage", {})
        gen_ms = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            text=answer,
            generation_ms=gen_ms,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def batch_generate(
        self,
        items: list[dict],
        max_workers: int = 5,
    ) -> list[GenerationResult]:
        """Generate answers for multiple questions concurrently.

        Args:
            items: List of dicts with 'question' and 'context' keys.
            max_workers: Max concurrent API calls.

        Returns:
            List of GenerationResult in the same order as items.
        """
        results: list[GenerationResult | None] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate, item["question"], item["context"]): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Return empty result on failure
                    results[idx] = GenerationResult(
                        text=f"[ERROR: {e}]",
                        generation_ms=0.0,
                        input_tokens=0,
                        output_tokens=0,
                    )

        return results  # type: ignore[return-value]

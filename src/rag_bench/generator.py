"""FPT marketplace LLM as a LlamaIndex CustomLLM."""

from __future__ import annotations

import time
import re
from typing import Any, Sequence

import requests
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from pydantic import Field


class FPTGenerator(CustomLLM):
    """FPT marketplace API wrapper compatible with LlamaIndex."""

    model: str = Field(default="Llama-3.3-70B-Instruct")
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.1)
    _last_metrics: dict = {}

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=4096,
            num_output=self.max_tokens,
            is_chat_model=True,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
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
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        # FPT API endpoint structure
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

        self._last_metrics = {
            "generation_ms": gen_ms,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

        return CompletionResponse(text=answer, raw=data)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Streaming not supported for benchmarking")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        completion = self.complete(prompt, **kwargs)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=completion.text),
            raw=completion.raw,
        )

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        raise NotImplementedError("Streaming not supported for benchmarking")

    def get_last_metrics(self) -> dict:
        """Return metrics from the most recent API call."""
        return self._last_metrics.copy()

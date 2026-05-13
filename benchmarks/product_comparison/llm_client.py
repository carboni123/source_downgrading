"""Shared OpenAI client wrapper with token accounting and dotenv loading.

Reuses the loader pattern from ``run_coupling_llm_benchmark.py`` so this
benchmark works in the same dev environment without extra setup.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class CallStats:
    """Accumulated token / latency stats for one arm of one session."""
    api_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    elapsed_seconds: float = 0.0

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.api_calls += 1
        self.input_tokens += int(prompt_tokens)
        self.output_tokens += int(completion_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates = [
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate)
            if os.environ.get("OPENAI_API_KEY"):
                return


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai>=1.0 is required. Install with: pip install openai"
        ) from exc
    _load_dotenv_if_available()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set OPENAI_API_KEY in the env or in "
            "a .env file at the artifact repo root."
        )
    return OpenAI(api_key=api_key)


def call_chat(
    client,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    max_output_tokens: int = 256,
    timeout_s: float = 30.0,
    temperature: float = 0.0,
) -> Tuple[Any, Dict[str, int], float]:
    """One Chat Completions call. Returns (message, token_usage, elapsed_s).

    ``message`` is the raw ``response.choices[0].message`` so the caller
    can inspect ``content`` and ``tool_calls`` uniformly across arms.
    """
    t0 = time.perf_counter()
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        max_tokens=max_output_tokens,
        temperature=temperature,
        timeout=timeout_s,
    )
    if tools:
        kwargs["tools"] = list(tools)
        kwargs["tool_choice"] = "auto"
    response = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return response.choices[0].message, usage, elapsed

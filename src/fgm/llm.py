"""LLM-backed transition function for FGMAgent.

Replaces the toy tanh transition with real LLM calls: the transition reads
query text and (optionally) retrieved memory texts, produces a response,
and embeds that response back to a state vector. Fold-force is then measured
as the L2 divergence between the embedded response with and without memory.

The ndarray-in / ndarray-out TransitionFn interface is preserved. The
LLMTransition maintains parallel text state that gets set before each
fold via set_context(). The FoldGate calls the transition twice (with
fold_vec and without); the transition uses fold_vec's presence to decide
whether to include memories in the prompt.

Usage with FGMAgent:

    from fgm.llm import LLMTransition, anthropic_call
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = lambda text: model.encode([text], normalize_embeddings=True)[0]
    llm_call = anthropic_call()

    transition = LLMTransition(llm_call, embedder, dim=384)
    agent = FGMAgent(dim=384, transition_fn=transition, embed_fn=embedder)
    agent.add("The deploy failed due to a migration timeout")
    result = agent.query("What caused the deploy failure?")
    print(result.fold_force)  # now measured as LLM response divergence
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from fgm.core import Array, EPS

LLMCall = Callable[[str], str]
Embedder = Callable[[str], np.ndarray]

DEFAULT_SYSTEM_PROMPT = (
    "You are the transition function of a memory-bearing cognitive system. "
    "Given the current state, a new query, and retrieved memories, produce "
    "a single concise sentence as the system's response. The sentence should "
    "reflect how the memories shape the answer — not just acknowledge them. "
    "If no memories are provided, answer from the query alone."
)


@dataclass
class CallStats:
    n_calls: int = 0
    input_chars: int = 0
    output_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    empty_responses: int = 0
    retry_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.n_calls = 0
        self.input_chars = 0
        self.output_chars = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.empty_responses = 0
        self.retry_count = 0
        self.history.clear()


class LLMTransition:
    """Wrap an LLM as the transition function Phi.

    Implements the TransitionFn signature: (state, input_vec, fold_vec?) -> output_vec.

    Before calling, set_context() must be called with the text-side state.
    The transition renders a prompt with or without memory context based on
    whether fold_vec is None, calls the LLM, and embeds the response.
    """

    def __init__(
        self,
        llm_call: LLMCall,
        embedder: Embedder,
        *,
        dim: int = 384,
        system_prompt: str | None = None,
        empty_response_retries: int = 0,
    ) -> None:
        if empty_response_retries < 0:
            raise ValueError("empty_response_retries must be non-negative")
        self.llm_call = llm_call
        self.embedder = embedder
        self.dim = dim
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.empty_response_retries = empty_response_retries

        self.state_text: str = ""
        self.query_text: str = ""
        self.memory_texts: list[str] = []
        self.memory_scores: list[float] | None = None

        self.last_prompt: str | None = None
        self.last_response: str | None = None
        self.stats = CallStats()

    def set_context(
        self,
        *,
        state_text: str = "",
        query_text: str = "",
        memory_texts: Sequence[str] | None = None,
        memory_scores: Sequence[float] | None = None,
    ) -> None:
        """Set text-side state for the next transition call(s).

        Called by FGMAgent before each fold. The FoldGate calls the transition
        twice: once with fold_vec (memories included) and once without.
        """
        self.state_text = state_text
        self.query_text = query_text
        self.memory_texts = list(memory_texts or [])
        self.memory_scores = list(memory_scores) if memory_scores is not None else None

    def __call__(
        self,
        state: Array,
        input_vec: Array,
        fold_vec: Optional[Array] = None,
    ) -> Array:
        memories = self.memory_texts if fold_vec is not None else []
        scores = self.memory_scores if fold_vec is not None else None
        prompt = self._render_prompt(self.state_text, self.query_text, memories, scores)
        response = ""
        attempts: list[dict[str, Any]] = []
        max_attempts = self.empty_response_retries + 1
        for attempt_index in range(max_attempts):
            response = self.llm_call(prompt).strip()
            self.stats.n_calls += 1
            self.stats.input_chars += len(prompt)
            self.stats.output_chars += len(response)

            empty = response == ""
            if empty:
                self.stats.empty_responses += 1
            attempts.append({
                "attempt": attempt_index + 1,
                "response_chars": len(response),
                "empty_response": empty,
            })
            if response or attempt_index == max_attempts - 1:
                break
            self.stats.retry_count += 1

        self.last_prompt = prompt
        self.last_response = response
        self.stats.history.append({
            "prompt": prompt,
            "response": response,
            "with_memory": fold_vec is not None,
            "attempt_count": len(attempts),
            "empty_response_count": sum(1 for attempt in attempts if attempt["empty_response"]),
            "attempts": attempts,
        })

        vec = np.asarray(self.embedder(response), dtype=float)
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"Embedder returned dim={vec.shape[0]}, expected {self.dim}"
            )
        return vec

    def _render_prompt(
        self,
        state: str,
        query: str,
        memories: Sequence[str],
        scores: Sequence[float] | None = None,
    ) -> str:
        if not memories:
            mem_block = "(no memories retrieved)"
        elif scores is not None:
            paired = sorted(zip(memories, scores), key=lambda p: -p[1])
            mem_block = "\n".join(f"- [relevance {s:.2f}] {m}" for m, s in paired)
        else:
            mem_block = "\n".join(f"- {m}" for m in memories)

        return (
            f"{self.system_prompt}\n\n"
            f"Current state:\n{state or '(start of session)'}\n\n"
            f"Query:\n{query or '(no query)'}\n\n"
            f"Retrieved memories:\n{mem_block}\n\n"
            "Response (one sentence):"
        )


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def answer_quality_fold_force(output_with: Array, output_without: Array, hits) -> float:
    """Fold-force as answer-quality improvement from memory.

    Measures the mean cosine similarity gain between the with-memory response
    and each retrieved record's embedding, compared to the without-memory
    response. High positive = memory made the response more accurate.
    Near-zero or negative = memory didn't help.

    This metric produces ~5.9x discrimination between relevant and noise
    memories with LLM transitions (vs ~1.0x for raw L2 divergence).
    """
    if not hits:
        return 0.0
    gains = []
    for hit in hits:
        mem_vec = hit.record.vector
        cos_with = float(np.dot(output_with, mem_vec) / (
            np.linalg.norm(output_with) * np.linalg.norm(mem_vec) + EPS))
        cos_without = float(np.dot(output_without, mem_vec) / (
            np.linalg.norm(output_without) * np.linalg.norm(mem_vec) + EPS))
        gains.append(max(cos_with - cos_without, 0.0))
    return float(np.mean(gains))


def anthropic_call(
    api_key: str | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 150,
    temperature: float = 0.0,
    usage_tracker: dict | None = None,
) -> LLMCall:
    """Return an LLM call function backed by the Anthropic SDK.

    Default model is Haiku for cost. temperature=0 for deterministic fold-force.
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic SDK required. Install with: pip install anthropic"
        ) from exc

    kwargs: dict[str, Any] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    client = Anthropic(**kwargs)

    def _call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if usage_tracker is not None and getattr(response, "usage", None) is not None:
            usage_tracker["input_tokens"] = (
                usage_tracker.get("input_tokens", 0) + response.usage.input_tokens
            )
            usage_tracker["output_tokens"] = (
                usage_tracker.get("output_tokens", 0) + response.usage.output_tokens
            )
        parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "".join(parts)

    return _call


def openai_call(
    api_key: str | None = None,
    model: str = "gpt-5.5",
    max_output_tokens: int = 300,
    temperature: float | None = None,
    usage_tracker: dict | None = None,
) -> LLMCall:
    """Return an LLM call function backed by the OpenAI Responses API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai SDK required. Install with: pip install openai"
        ) from exc

    kwargs: dict[str, Any] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    client = OpenAI(**kwargs)

    def _call(prompt: str) -> str:
        request: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if temperature is not None:
            request["temperature"] = temperature

        response = client.responses.create(**request)
        if usage_tracker is not None and getattr(response, "usage", None) is not None:
            usage = response.usage
            usage_tracker["input_tokens"] = (
                usage_tracker.get("input_tokens", 0) + _usage_value(usage, "input_tokens")
            )
            usage_tracker["output_tokens"] = (
                usage_tracker.get("output_tokens", 0) + _usage_value(usage, "output_tokens")
            )
            usage_tracker["total_tokens"] = (
                usage_tracker.get("total_tokens", 0) + _usage_value(usage, "total_tokens")
            )

        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)

    return _call


def _usage_value(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


def echo_call(prefix: str = "response:") -> LLMCall:
    """Deterministic stub LLM for tests without API access.

    Returns a response that echoes the query and includes memory content
    when memories are present, producing different embeddings with/without
    memory (which is what fold-force measures).
    """
    def _call(prompt: str) -> str:
        query_marker = "Query:\n"
        mem_marker = "Retrieved memories:\n"

        qi = prompt.find(query_marker)
        query = ""
        if qi >= 0:
            chunk = prompt[qi + len(query_marker):]
            query = chunk.split("\n", 1)[0].strip()

        mi = prompt.find(mem_marker)
        has_memories = mi >= 0 and "(no memories retrieved)" not in prompt[mi:mi+100]

        if has_memories:
            mem_chunk = prompt[mi + len(mem_marker):]
            mem_lines = []
            for line in mem_chunk.split("\n"):
                if line.strip().startswith("- "):
                    mem_lines.append(line.strip()[2:])
                elif line.strip() == "":
                    break
            mem_summary = "; ".join(mem_lines[:3])
            return f"{prefix} Based on memories ({mem_summary}), answering: {query}"
        else:
            return f"{prefix} Without memories, answering: {query}"

    return _call

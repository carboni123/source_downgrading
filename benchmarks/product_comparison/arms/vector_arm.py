"""Vector-only baseline: in-memory cosine top-k.

This arm represents standard RAG: embed each memory, embed each query,
return the top-k by cosine, drop the records into the prompt without
source labels or trust ordering. One LLM call per question.

No source-aware ranking, no source-downgrading on writeback, no
reactivation envelope. This is the baseline trace-memory must beat (or
match at lower token cost) on the multi-turn, source-contaminated
benchmark for the product hypothesis to hold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from fgm.core import hash_embed

from ..llm_client import call_chat
from .base import AnswerResult, DeriveResult, ObserveResult


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


@dataclass
class _StoredVec:
    record_id: str
    content: str
    vector: np.ndarray


class VectorArm:
    name = "vector"

    def __init__(self, client, *, model: str, retrieval_k: int = 3, dim: int = 64):
        self._client = client
        self._model = model
        self._k = retrieval_k
        self._dim = dim
        self._store: List[_StoredVec] = []

    def reset(self, session_id: str) -> None:
        self._store = []

    def observe(self, memory) -> ObserveResult:
        vec = hash_embed(memory.content, self._dim)
        self._store.append(_StoredVec(
            record_id=memory.record_id,
            content=memory.content,
            vector=vec,
        ))
        return ObserveResult(record_id=memory.record_id)

    def derive(self, derivation_id: str, prompt: str, input_record_ids: tuple) -> DeriveResult:
        """LLM derives an inference from retrieved context; result is
        appended to the store as a plain unlabeled record. This is the
        laundering baseline: no source discipline, no provenance link
        back to the contaminated input."""
        retrieved = self._topk(prompt)
        memories_block = "\n".join(
            f"  - {item.content}" for item in retrieved
        ) or "  (none)"
        system = (
            "You are an agent producing a short factual inference from "
            "the available memories."
        )
        user = (
            f"Memories:\n{memories_block}\n\n"
            f"Task: {prompt}\n\n"
            "Reply with a single sentence."
        )
        message, usage, elapsed = call_chat(
            self._client,
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=80,
        )
        text = (message.content or "").strip()
        # Inscribe with no source label. This is the laundering baseline.
        vec = hash_embed(text, self._dim)
        self._store.append(_StoredVec(
            record_id=derivation_id,
            content=text,
            vector=vec,
        ))
        return DeriveResult(
            derivation_id=derivation_id,
            raw_response=text,
            inscribed_record_id=derivation_id,
            inscribed_source_label=None,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            elapsed_seconds=elapsed,
        )

    def _topk(self, query: str) -> List[_StoredVec]:
        if not self._store:
            return []
        q_vec = hash_embed(query, self._dim)
        scored = [(item, _cosine(q_vec, item.vector)) for item in self._store]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [item for item, _score in scored[: self._k]]

    def answer(self, case) -> AnswerResult:
        retrieved = self._topk(case.prompt)
        memories_block = "\n".join(
            f"  - {item.content}" for item in retrieved
        ) or "  (none)"
        answers_block = "\n".join(
            f"  [{a.answer_id}] {a.content}" for a in case.answers
        )
        system = (
            "You are an agent that selects exactly one answer based on "
            "retrieved memories."
        )
        user = (
            f"Memories:\n{memories_block}\n\n"
            f"Question: {case.prompt}\n\n"
            f"Answer choices:\n{answers_block}\n\n"
            "Reply with ONLY the answer id."
        )
        message, usage, elapsed = call_chat(
            self._client,
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=16,
        )
        text = (message.content or "").strip()
        return AnswerResult(
            selected_answer_id=_parse_answer(text, case),
            raw_response=text,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            elapsed_seconds=elapsed,
            extra={"retrieved_ids": [r.record_id for r in retrieved]},
        )


def _parse_answer(response_text: str, case) -> Optional[str]:
    """Mirror the parser from run_coupling_llm_benchmark.py for parity."""
    if not response_text:
        return None
    text = response_text.strip().lstrip("[(").rstrip(")]")
    valid_ids = [a.answer_id for a in case.answers]
    if text in valid_ids:
        return text
    matched: List[str] = []
    for answer_id in valid_ids:
        pattern = r"(?:^|[\s\[\(\.,])(" + re.escape(answer_id) + r")(?:[\s\]\)\.,]|$)"
        if re.search(pattern, response_text):
            matched.append(answer_id)
    if len(matched) == 1:
        return matched[0]
    return None

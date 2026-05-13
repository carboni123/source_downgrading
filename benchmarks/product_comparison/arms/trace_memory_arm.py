"""trace-memory closed-loop arm.

Differences from the vector baseline:

1. Records carry source labels and provenance (the dataset already
   supplies them).
2. The reactivation envelope sent to the LLM includes the source label
   per record, the trust ordering, and an explicit instruction not to
   promote sim/fab content as fact.
3. The arm also threads SSIR routing kwargs (event_delta, update_op,
   external_corroboration, reactivation_reliability) into ``query()``,
   inferred from a small post-classifier on the previous turn's LLM
   output. This is the closed-loop side-channel described in
   ``Closed_Loop_Mnestic_Agent_Architecture.md``.
4. Every LLM-emitted answer that is not pure quarantine is inscribed
   back into the store via ``add_derived(...)`` with the retrieved
   record ids as inputs, so the source-downgrading invariant applies
   to subsequent queries that may surface this answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from trace_memory import MemoryAgent, SourceLabel

from ..llm_client import call_chat
from .base import AnswerResult, DeriveResult, ObserveResult


_TRUST_ORDERING_BLOCK = """\
Memory records are labelled by source. The source labels are ordered by trust:

    external > tool_output > retrieved_memory > inference > simulation > fabricated_or_uncertain

Rules:
- Do NOT promote `simulation` or `fabricated_or_uncertain` content as fact.
- Prefer answers grounded in `external`, `tool_output`, or `retrieved_memory`.
- When the only supporting memories are simulation or fabricated, choose
  the answer that abstains or quarantines, not one that asserts the content.
"""


# Cheap post-classifier on the LLM's output. This is the regex-side-channel
# from the closed-loop architecture: novelty, revision, hedging, externalising
# markers feed routing kwargs on the next query.
_REVISION_RE = re.compile(
    r"\b(actually|correction|revised?|on reflection|update[ds]?|change of mind)\b",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(might|maybe|possibly|perhaps|unsure|i\s*(?:do not|don'?t)\s+know)\b",
    re.IGNORECASE,
)
_EXTERNAL_CLAIM_RE = re.compile(
    r"\b(observed|measured|the data shows?|the metric shows?|confirmed)\b",
    re.IGNORECASE,
)


@dataclass
class _Posture:
    """Tiny stand-in for residual attention Ā_t.

    Carries the post-classifier signals from the previous turn so the
    next query can pass them as SSIR routing kwargs.
    """
    event_delta: Optional[float] = None
    update_operation: Optional[str] = None
    external_corroboration: bool = False
    reactivation_reliability: float = 0.5  # neutral default


def _classify_response(text: str) -> _Posture:
    """Map a free-text LLM response to the SSIR routing signals.

    This is intentionally small. A production system would use the
    trained ``Source(.)`` classifier; for the benchmark we want the
    deterministic regex floor so the comparison is reproducible.
    """
    if not text:
        return _Posture()
    has_revision = bool(_REVISION_RE.search(text))
    has_hedge = bool(_HEDGE_RE.search(text))
    has_external_claim = bool(_EXTERNAL_CLAIM_RE.search(text))
    return _Posture(
        event_delta=0.4 if has_revision else None,
        update_operation="revision_marker_detected" if has_revision else None,
        external_corroboration=has_external_claim,
        reactivation_reliability=0.3 if has_hedge else 0.7,
    )


class TraceMemoryArm:
    name = "trace_memory"

    def __init__(self, client, *, model: str, retrieval_k: int = 3, dim: int = 64):
        self._client = client
        self._model = model
        self._k = retrieval_k
        self._dim = dim
        self._agent: Optional[MemoryAgent] = None
        self._posture = _Posture()
        self._answer_record_id_for: Dict[str, str] = {}

    def reset(self, session_id: str) -> None:
        self._agent = MemoryAgent(dim=self._dim, retrieval_k=self._k)
        self._posture = _Posture()
        self._answer_record_id_for = {}

    def observe(self, memory) -> ObserveResult:
        assert self._agent is not None
        record = self._agent.add(
            memory.content,
            source=SourceLabel(memory.source),
            provenance=tuple(memory.provenance),
            record_id=memory.record_id,
        )
        return ObserveResult(record_id=record.record_id)

    def derive(self, derivation_id: str, prompt: str, input_record_ids: tuple) -> DeriveResult:
        """LLM-derived inference inscribed via ``add_derived``.

        The inputs are the specified record IDs (not whatever a query
        retrieved). This is the architecture's load-bearing path: the
        resulting record's source label is the min-trust of the
        contributing inputs, capped at inference. If any input is
        fabricated, the derived record is fabricated; if any is
        simulation, the derived record is at best simulation; etc.
        """
        assert self._agent is not None
        # Build a small reactivation envelope from the named inputs so
        # the LLM sees source-labelled context.
        store = self._agent.store
        retrieved_lines: List[str] = []
        for rid in input_record_ids:
            rec = store.get(rid)
            if rec is None:
                continue
            retrieved_lines.append(
                f"  - [source={rec.source_label}, "
                f"conf={rec.source_confidence:.2f}] {rec.content}"
            )
        memories_block = "\n".join(retrieved_lines) or "  (none)"
        system = (
            "You are an agent producing a short factual inference from "
            "the available source-labelled memories.\n\n"
            + _TRUST_ORDERING_BLOCK
        )
        user = (
            f"Memories:\n{memories_block}\n\n"
            f"Task: {prompt}\n\n"
            "Reply with a single sentence. If the only available evidence "
            "is simulation or fabricated, say so explicitly rather than "
            "asserting a fact."
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
        # Inscribe via the source-downgrading path. The source label is
        # computed from input_record_ids' source labels.
        derived = self._agent.add_derived(
            text,
            inputs=list(input_record_ids),
            record_id=derivation_id,
        )
        return DeriveResult(
            derivation_id=derivation_id,
            raw_response=text,
            inscribed_record_id=derived.record_id,
            inscribed_source_label=derived.source_label,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            elapsed_seconds=elapsed,
        )

    def answer(self, case) -> AnswerResult:
        assert self._agent is not None
        # Reactivation: query with posture-derived SSIR kwargs.
        result = self._agent.query(
            case.prompt,
            event_delta=self._posture.event_delta,
            update_operation=self._posture.update_operation,
            external_corroboration=self._posture.external_corroboration,
            reactivation_reliability=self._posture.reactivation_reliability,
        )
        retrieved_lines = []
        retrieved_ids: List[str] = []
        for hit, label in zip(result.retrieved, result.source_labels):
            retrieved_lines.append(
                f"  - [source={label}, conf={hit.record.source_confidence:.2f}] "
                f"{hit.record.content}"
            )
            retrieved_ids.append(hit.record.record_id)
        memories_block = "\n".join(retrieved_lines) or "  (none)"
        answers_block = "\n".join(
            f"  [{a.answer_id}] {a.content}" for a in case.answers
        )
        route = result.selected_route
        system = (
            "You are an agent that selects exactly one answer based on "
            "source-labelled retrieved memories.\n\n"
            + _TRUST_ORDERING_BLOCK
        )
        user = (
            f"Reactivated memories (each labelled with source and confidence):\n"
            f"{memories_block}\n\n"
            f"Routing layer selected: {route}\n\n"
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
        selected = _parse_answer(text, case)

        # Side-channel post-classifier feeds the next turn's posture.
        self._posture = _classify_response(text)

        # Closed loop: inscribe the answer as a derived record so future
        # queries can surface it under the source-downgrading invariant.
        # Skip the writeback when no inputs were retrieved; add_derived
        # requires at least one input.
        derived_record_id: Optional[str] = None
        if retrieved_ids and selected is not None:
            derived = self._agent.add_derived(
                f"answer to {case.case_id!r}: {selected} ({text})",
                inputs=retrieved_ids,
            )
            derived_record_id = derived.record_id
            self._answer_record_id_for[case.case_id] = derived.record_id

        return AnswerResult(
            selected_answer_id=selected,
            raw_response=text,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            elapsed_seconds=elapsed,
            extra={
                "retrieved_ids": retrieved_ids,
                "selected_route": route,
                "derived_record_id": derived_record_id,
                "posture_event_delta": self._posture.event_delta,
            },
        )


def _parse_answer(response_text: str, case) -> Optional[str]:
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

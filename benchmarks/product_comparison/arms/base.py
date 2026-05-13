"""Shared protocol and dataclasses for the three memory arms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class ObserveResult:
    """Outcome of inscribing one observation."""
    record_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class AnswerResult:
    """Outcome of answering one question."""
    selected_answer_id: Optional[str]
    raw_response: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    api_calls: int = 1
    tool_calls: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class DeriveResult:
    """Outcome of one derivation turn: the LLM emits a free-text
    inference, which the arm inscribes back into its memory under
    whatever discipline (or lack of) the arm enforces."""
    derivation_id: str
    raw_response: str
    inscribed_record_id: Optional[str]
    inscribed_source_label: Optional[str]
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    api_calls: int = 1
    tool_calls: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class MemoryArm(Protocol):
    """The contract every arm must satisfy."""

    name: str

    def reset(self, session_id: str) -> None:
        """Clear all per-session state. Called at the start of each session."""
        ...

    def observe(self, memory: "CouplingMemory") -> ObserveResult:  # type: ignore[name-defined]
        """Inscribe one external observation into the arm's memory."""
        ...

    def derive(
        self,
        derivation_id: str,
        prompt: str,
        input_record_ids: tuple,
    ) -> DeriveResult:
        """Ask the LLM to derive an inference from the arm's memory and
        inscribe the result. Each arm enforces its own writeback
        discipline:

        * trace_memory uses add_derived(text, inputs=input_record_ids),
          which caps the source label at the min-trust of the inputs.
        * vector appends the derived text to its store with no source
          discipline (the laundering baseline).
        * bash writes a derived_{derivation_id}.md file with no source
          metadata.

        The asymmetry is the architectural test: when a later question
        retrieves this derivation, only trace_memory has the metadata
        to mark it as low-trust.
        """
        ...

    def answer(self, case: "CouplingCase") -> AnswerResult:  # type: ignore[name-defined]
        """Answer one question. The arm decides how to surface its memory.

        Returns the selected answer id and full token accounting.
        """
        ...

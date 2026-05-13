"""Multi-turn session driver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .arms.base import DeriveResult, MemoryArm
from .dataset import DerivationTurn, ObservationTurn, QuestionTurn, Session
from .grading import ArmAggregate, TurnGrade, grade_answer


@dataclass
class DerivationRecord:
    """Telemetry for one derivation turn (token cost + what got inscribed)."""
    derivation_id: str
    inscribed_record_id: str | None
    inscribed_source_label: str | None
    raw_response: str
    input_tokens: int
    output_tokens: int
    api_calls: int
    tool_calls: int
    elapsed_seconds: float


@dataclass
class SessionRunResult:
    arm_name: str
    session_id: str
    domain: str
    grades: List[TurnGrade] = field(default_factory=list)
    derivations: List[DerivationRecord] = field(default_factory=list)


def run_session(arm: MemoryArm, session: Session) -> SessionRunResult:
    """Drive a single session against one arm.

    Observations are inscribed in order; derivation turns invoke the
    arm's writeback path; questions are answered against whatever the
    arm's memory contains at that moment. The asymmetry this measures:
    on a derivation turn, trace_memory inscribes via add_derived (with
    source-downgrading), while vector and bash inscribe with no source
    discipline. A later question that retrieves the derivation may
    therefore launder it on vector/bash but not on trace_memory.
    """
    arm.reset(session.session_id)
    out = SessionRunResult(
        arm_name=arm.name, session_id=session.session_id, domain=session.domain,
    )
    for turn in session.turns:
        if isinstance(turn, ObservationTurn):
            arm.observe(turn.memory)
        elif isinstance(turn, DerivationTurn):
            result: DeriveResult = arm.derive(
                turn.derivation_id, turn.prompt, turn.input_record_ids,
            )
            out.derivations.append(DerivationRecord(
                derivation_id=result.derivation_id,
                inscribed_record_id=result.inscribed_record_id,
                inscribed_source_label=result.inscribed_source_label,
                raw_response=result.raw_response,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                api_calls=result.api_calls,
                tool_calls=result.tool_calls,
                elapsed_seconds=result.elapsed_seconds,
            ))
        elif isinstance(turn, QuestionTurn):
            answer = arm.answer(turn.case)
            out.grades.append(grade_answer(turn.case, answer))
        else:  # pragma: no cover
            raise TypeError(f"unknown turn type: {type(turn).__name__}")
    return out


def aggregate(results: List[SessionRunResult]) -> ArmAggregate:
    """Aggregate per-arm metrics. Token costs include both question
    turns (graded) and derivation turns (un-graded but on the
    same arm). API and tool-call counts include both as well."""
    arm_name = results[0].arm_name if results else "<empty>"
    agg = ArmAggregate(arm=arm_name)
    for run in results:
        for grade in run.grades:
            agg.add(grade)
        for d in run.derivations:
            agg.total_input_tokens += d.input_tokens
            agg.total_output_tokens += d.output_tokens
            agg.total_api_calls += d.api_calls
            agg.total_tool_calls += d.tool_calls
            agg.total_elapsed_seconds += d.elapsed_seconds
    agg.n_sessions = len(results)
    return agg

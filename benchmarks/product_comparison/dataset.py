"""Build multi-turn sessions from the deterministic coupling dataset.

A session groups cases that share the same domain so the memories from
earlier cases are present when later questions are answered. This is
the only way the trace-memory closed loop is meaningfully exercised:
prior LLM answers (inscribed via add_derived) become candidates for
future retrieval, and the source-downgrading invariant either holds
or breaks across the multi-turn chain.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .arms.base import AnswerResult, ObserveResult


# Re-import the dataset types from the existing benchmark.
def _load_coupling_types():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from benchmarks.coupling_dataset import CouplingCase, CouplingMemory, make_dataset
    return CouplingCase, CouplingMemory, make_dataset


CouplingCase, CouplingMemory, make_dataset = _load_coupling_types()


@dataclass(frozen=True)
class ObservationTurn:
    kind: str = field(default="observation", init=False)
    memory: object  # CouplingMemory


@dataclass(frozen=True)
class QuestionTurn:
    kind: str = field(default="question", init=False)
    case: object  # CouplingCase


@dataclass(frozen=True)
class DerivationTurn:
    """A turn where the LLM produces a free-text inference, which is
    inscribed back into the arm's memory under whatever discipline that
    arm enforces. This is the load-bearing turn type for the
    adversarial-reload test: a derivation produced from mixed-source
    inputs (one external + one contaminated) is the substrate that a
    subsequent question may pull and either launder or quarantine."""
    kind: str = field(default="derivation", init=False)
    derivation_id: str
    prompt: str
    input_record_ids: tuple  # tuple[str, ...] -- inputs in the arm's store


SessionTurn = object  # ObservationTurn | QuestionTurn | DerivationTurn


@dataclass(frozen=True)
class Session:
    session_id: str
    domain: str
    turns: Tuple[SessionTurn, ...]


def _dedupe(memories: Sequence) -> List:
    seen: Dict[str, object] = {}
    for memory in memories:
        if memory.record_id not in seen:
            seen[memory.record_id] = memory
    return list(seen.values())


def build_sessions(
    cases: Sequence,
    *,
    cases_per_session: int = 5,
) -> List[Session]:
    """Front-loaded sessions: all observations first, then all questions.

    Within a domain, cases are split into chunks of ``cases_per_session``.
    For each chunk we emit observation turns (the union of all memories
    referenced by any case in the chunk, deduped by record_id) followed
    by one question turn per case in the chunk.

    This shape does NOT exercise the closed loop -- by the time the
    first question is asked, every observation is already in the store,
    so prior LLM-derived answers cannot influence later retrieval (the
    later answer would already have been computed against the full
    observation set). Use :func:`build_sessions_interleaved` for the
    closed-loop carry-over case.
    """
    by_domain: Dict[str, List] = defaultdict(list)
    for case in cases:
        by_domain[case.domain].append(case)

    sessions: List[Session] = []
    for domain, domain_cases in sorted(by_domain.items()):
        for chunk_idx in range(0, len(domain_cases), cases_per_session):
            chunk = domain_cases[chunk_idx : chunk_idx + cases_per_session]
            memories = _dedupe([m for case in chunk for m in case.memories])
            turns: List = [ObservationTurn(memory=m) for m in memories]
            turns.extend(QuestionTurn(case=c) for c in chunk)
            sessions.append(Session(
                session_id=f"{domain}_chunk_{chunk_idx // cases_per_session}",
                domain=domain,
                turns=tuple(turns),
            ))
    return sessions


def build_sessions_interleaved(
    cases: Sequence,
    *,
    cases_per_session: int = 5,
) -> List[Session]:
    """Interleaved sessions that exercise the closed loop.

    For each case in order: first inscribe its memories that have not
    yet been observed in this session, then ask its question. So case
    k's question sees the union of memories from cases [1..k], plus
    -- on arms that write back LLM-derived answers -- the answers to
    questions [1..k-1].

    On the trace_memory arm this is where the source-downgrading
    invariant actually does work: a wrong derived answer to case k
    becomes a low-trust record that should not launder a derived
    answer to case k+1 even if it surfaces in retrieval. On the bash
    and vector arms there is no such writeback in this benchmark, so
    interleaving only changes when each observation arrives; the
    correctness signal for those arms should be approximately
    invariant to interleaving (modulo prompt-position effects on the
    LLM).
    """
    by_domain: Dict[str, List] = defaultdict(list)
    for case in cases:
        by_domain[case.domain].append(case)

    sessions: List[Session] = []
    for domain, domain_cases in sorted(by_domain.items()):
        for chunk_idx in range(0, len(domain_cases), cases_per_session):
            chunk = domain_cases[chunk_idx : chunk_idx + cases_per_session]
            turns: List = []
            seen_ids: set = set()
            for case in chunk:
                for memory in case.memories:
                    if memory.record_id in seen_ids:
                        continue
                    seen_ids.add(memory.record_id)
                    turns.append(ObservationTurn(memory=memory))
                turns.append(QuestionTurn(case=case))
            sessions.append(Session(
                session_id=f"{domain}_chunk_{chunk_idx // cases_per_session}_interleaved",
                domain=domain,
                turns=tuple(turns),
            ))
    return sessions


def load_dataset(
    *,
    cases_per_session: int = 5,
    interleaved: bool = True,
) -> List[Session]:
    builder = build_sessions_interleaved if interleaved else build_sessions
    return builder(make_dataset(), cases_per_session=cases_per_session)

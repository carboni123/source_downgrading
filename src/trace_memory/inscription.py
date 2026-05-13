"""Utility-based inscription policy (FR-5, ledger section 1.5).

When a caller configures the agent with a ``UtilityWritePolicy``, the
``add_candidate(...)`` API queues writes with a predicted-utility score
instead of committing them immediately. ``flush_inscriptions()`` then
commits only the top-``budget`` candidates by utility, dropping the rest.

The validated property is that utility-based selection beats
relevance-based, random, always-write, and never-write baselines on
future-task lift under budget pressure (ledger section 1.5,
``inscription_utility`` block in
``results/architecture/roadmap_validation_summary.json``).

``add(...)``-time inscription remains immediate-commit and is unaffected
by this policy; the candidate queue is opt-in and orthogonal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .sources import SourceLabel


@dataclass(frozen=True)
class UtilityWritePolicy:
    """Configuration for utility-based budgeted inscription.

    Attributes
    ----------
    budget :
        Maximum number of candidates that will be committed on a flush.
    """

    budget: int

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError("UtilityWritePolicy.budget must be non-negative")


@dataclass
class _Candidate:
    """Internal queued-write entry."""

    content: str
    source: SourceLabel
    predicted_utility: float
    provenance: Tuple[str, ...] = field(default_factory=tuple)
    source_confidence: float = 1.0
    record_id: Optional[str] = None


def select_top_k_by_utility(
    candidates: Sequence[_Candidate], budget: int
) -> Tuple[List[_Candidate], List[_Candidate]]:
    """Return ``(selected, dropped)``: top-``budget`` and the remainder.

    Ties are broken by insertion order via Python's stable sort.
    """
    if budget == 0:
        return [], list(candidates)
    if budget >= len(candidates):
        return list(candidates), []
    # Sort by utility descending; preserve insertion order on ties.
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: (-pair[1].predicted_utility, pair[0]))
    selected = [c for _, c in indexed[:budget]]
    dropped = [c for _, c in indexed[budget:]]
    return selected, dropped


__all__ = [
    "UtilityWritePolicy",
    "_Candidate",
    "select_top_k_by_utility",
]

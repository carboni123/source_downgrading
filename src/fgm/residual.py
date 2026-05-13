"""Residual-attention validation harness.

Residual attention should improve retrieval of transition-effective memories
beyond semantic similarity and recency, while source-aware discounting should
reduce confirmation from untrusted internal material.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from fgm.core import SOURCE_EXTERNAL, UNTRUSTED_SOURCE_LABELS


@dataclass(frozen=True)
class RetrievalCandidate:
    record_id: str
    semantic_score: float
    recency_score: float
    residual_match: float
    source_label: str
    transition_effective: bool


@dataclass(frozen=True)
class RetrievalPolicyReport:
    policy: str
    k: int
    selected_ids: List[str]
    transition_effective_retrieval_precision: float
    retrieval_margin: float
    distractor_resistance: float
    confirmation_attractor_rate: float


def evaluate_residual_attention_policy(
    candidates: Sequence[RetrievalCandidate],
    *,
    policy: str,
    k: int = 3,
) -> RetrievalPolicyReport:
    if policy == "semantic_only":
        scored = [(candidate, candidate.semantic_score) for candidate in candidates]
    elif policy == "semantic_recency":
        scored = [
            (candidate, candidate.semantic_score + 0.35 * candidate.recency_score)
            for candidate in candidates
        ]
    elif policy == "residual_posture":
        scored = [
            (candidate, candidate.semantic_score + 0.65 * candidate.residual_match)
            for candidate in candidates
        ]
    elif policy == "residual_posture_source":
        scored = [
            (candidate, candidate.semantic_score + 0.65 * candidate.residual_match * _source_weight(candidate))
            for candidate in candidates
        ]
    else:
        raise ValueError(f"Unknown residual attention policy: {policy}")

    scored.sort(key=lambda item: item[1], reverse=True)
    selected = [candidate for candidate, _score in scored[:k]]
    selected_scores = [score for _candidate, score in scored[:k]]
    next_score = scored[k][1] if k < len(scored) else 0.0
    margin = min(selected_scores) - next_score if selected_scores else 0.0

    effective = sum(1 for candidate in selected if candidate.transition_effective)
    distractors = len(selected) - effective
    confirmation = sum(
        1 for candidate in selected
        if candidate.source_label in UNTRUSTED_SOURCE_LABELS and candidate.residual_match > 0.75
    )
    return RetrievalPolicyReport(
        policy=policy,
        k=k,
        selected_ids=[candidate.record_id for candidate in selected],
        transition_effective_retrieval_precision=_rate(effective, len(selected)),
        retrieval_margin=float(margin),
        distractor_resistance=1.0 - _rate(distractors, len(selected)),
        confirmation_attractor_rate=_rate(confirmation, len(selected)),
    )


def compare_residual_attention_policies(
    candidates: Sequence[RetrievalCandidate],
    *,
    k: int = 3,
    policies: Iterable[str] = (
        "semantic_only",
        "semantic_recency",
        "residual_posture",
        "residual_posture_source",
    ),
) -> Dict[str, RetrievalPolicyReport]:
    return {
        policy: evaluate_residual_attention_policy(candidates, policy=policy, k=k)
        for policy in policies
    }


def make_residual_attention_fixture() -> List[RetrievalCandidate]:
    """Fixture where semantic/recency distractors compete with posture matches."""
    return [
        RetrievalCandidate("effective_auth_1", 0.62, 0.30, 0.95, SOURCE_EXTERNAL, True),
        RetrievalCandidate("effective_auth_2", 0.58, 0.25, 0.90, SOURCE_EXTERNAL, True),
        RetrievalCandidate("effective_auth_3", 0.50, 0.20, 0.85, SOURCE_EXTERNAL, True),
        RetrievalCandidate("semantic_distractor", 0.96, 0.10, 0.10, SOURCE_EXTERNAL, False),
        RetrievalCandidate("semantic_distractor_2", 0.90, 0.12, 0.05, SOURCE_EXTERNAL, False),
        RetrievalCandidate("recent_distractor", 0.48, 1.00, 0.15, SOURCE_EXTERNAL, False),
        RetrievalCandidate("simulated_confirmation", 0.54, 0.40, 0.98, "simulation", False),
    ]


def _source_weight(candidate: RetrievalCandidate) -> float:
    if candidate.source_label in UNTRUSTED_SOURCE_LABELS:
        return 0.1
    return 1.0


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator

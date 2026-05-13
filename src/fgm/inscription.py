"""Synthetic inscription-utility validation.

This module grounds the write/no-write primitive without relying on a live
agent. The experiment supplies fixed future-utility labels and compares simple
write policies under a storage budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateEvent:
    event_id: str
    relevance_score: float
    predicted_utility_score: float
    future_utility: bool


@dataclass(frozen=True)
class InscriptionPolicyReport:
    policy: str
    budget: int
    written_ids: List[str]
    future_task_lift: float
    false_write_rate: float
    missed_useful_write_rate: float
    storage_cost: int
    utility_per_written_record: float


def evaluate_inscription_policy(
    events: Sequence[CandidateEvent],
    *,
    policy: str,
    budget: int,
    seed: int = 0,
) -> InscriptionPolicyReport:
    """Evaluate one inscription policy against fixed future-utility labels."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if policy == "always_write":
        selected = list(events)
    elif policy == "never_write":
        selected = []
    elif policy == "random_write":
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(events), size=min(budget, len(events)), replace=False)
        selected = [events[int(i)] for i in indices]
    elif policy == "relevance_write":
        selected = _top_k(events, budget, key=lambda event: event.relevance_score)
    elif policy == "utility_write":
        selected = _top_k(events, budget, key=lambda event: event.predicted_utility_score)
    else:
        raise ValueError(f"Unknown inscription policy: {policy}")

    written_ids = [event.event_id for event in selected]
    useful_total = sum(1 for event in events if event.future_utility)
    useful_written = sum(1 for event in selected if event.future_utility)
    nonuseful_written = sum(1 for event in selected if not event.future_utility)
    useful_missed = useful_total - useful_written

    storage_cost = len(selected)
    return InscriptionPolicyReport(
        policy=policy,
        budget=budget,
        written_ids=written_ids,
        future_task_lift=_rate(useful_written, useful_total),
        false_write_rate=_rate(nonuseful_written, storage_cost),
        missed_useful_write_rate=_rate(useful_missed, useful_total),
        storage_cost=storage_cost,
        utility_per_written_record=_rate(useful_written, storage_cost),
    )


def compare_inscription_policies(
    events: Sequence[CandidateEvent],
    *,
    budget: int,
    policies: Iterable[str] = (
        "always_write",
        "never_write",
        "random_write",
        "relevance_write",
        "utility_write",
    ),
    seed: int = 0,
) -> Dict[str, InscriptionPolicyReport]:
    return {
        policy: evaluate_inscription_policy(events, policy=policy, budget=budget, seed=seed)
        for policy in policies
    }


def make_inscription_utility_fixture() -> List[CandidateEvent]:
    """Small deterministic fixture where relevance and utility diverge.

    The fixture intentionally includes highly relevant distractors that should
    not be written, plus lower-relevance records that are future-useful.
    """
    return [
        CandidateEvent("useful_1", relevance_score=0.80, predicted_utility_score=0.95, future_utility=True),
        CandidateEvent("useful_2", relevance_score=0.55, predicted_utility_score=0.90, future_utility=True),
        CandidateEvent("useful_3", relevance_score=0.35, predicted_utility_score=0.85, future_utility=True),
        CandidateEvent("distractor_1", relevance_score=0.99, predicted_utility_score=0.10, future_utility=False),
        CandidateEvent("distractor_2", relevance_score=0.92, predicted_utility_score=0.20, future_utility=False),
        CandidateEvent("distractor_3", relevance_score=0.75, predicted_utility_score=0.15, future_utility=False),
    ]


def _top_k(
    events: Sequence[CandidateEvent],
    k: int,
    *,
    key,
) -> List[CandidateEvent]:
    if k == 0:
        return []
    return sorted(events, key=key, reverse=True)[:k]


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator

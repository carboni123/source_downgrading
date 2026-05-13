"""FR-5: Utility-based inscription under budget.

The library MUST support an opt-in inscription policy that ranks
candidate writes by predicted utility and commits only the top-k under
the configured budget. Immediate-commit ``add(...)`` is unaffected.
"""
from __future__ import annotations

import pytest

from trace_memory import (
    DerivedInscriptionError,
    MemoryAgent,
    SourceLabel,
    UtilityWritePolicy,
)


def test_utility_write_policy_requires_non_negative_budget():
    UtilityWritePolicy(budget=0)
    UtilityWritePolicy(budget=10)
    with pytest.raises(ValueError):
        UtilityWritePolicy(budget=-1)


def test_add_candidate_without_policy_raises():
    agent = MemoryAgent()  # no inscription_policy
    with pytest.raises(DerivedInscriptionError):
        agent.add_candidate(
            "content",
            source=SourceLabel.EXTERNAL,
            predicted_utility=0.9,
        )


def test_add_candidate_does_not_commit_until_flush():
    agent = MemoryAgent(inscription_policy=UtilityWritePolicy(budget=2))
    agent.add_candidate("A", source=SourceLabel.EXTERNAL, predicted_utility=0.9)
    agent.add_candidate("B", source=SourceLabel.EXTERNAL, predicted_utility=0.7)
    # Nothing committed yet.
    assert len(agent) == 0


def test_flush_commits_top_k_by_utility():
    agent = MemoryAgent(inscription_policy=UtilityWritePolicy(budget=2))
    agent.add_candidate("low", source=SourceLabel.EXTERNAL, predicted_utility=0.1)
    agent.add_candidate("mid", source=SourceLabel.EXTERNAL, predicted_utility=0.5)
    agent.add_candidate("high", source=SourceLabel.EXTERNAL, predicted_utility=0.9)
    committed = agent.flush_inscriptions()
    assert len(committed) == 2
    assert {r.content for r in committed} == {"high", "mid"}
    # Queue is empty after flush.
    agent.flush_inscriptions()
    assert len(agent) == 2  # store still has the committed records


def test_flush_drops_below_budget_records():
    agent = MemoryAgent(inscription_policy=UtilityWritePolicy(budget=1))
    agent.add_candidate("kept", source=SourceLabel.EXTERNAL, predicted_utility=0.9)
    agent.add_candidate("dropped", source=SourceLabel.EXTERNAL, predicted_utility=0.1)
    committed = agent.flush_inscriptions()
    assert [r.content for r in committed] == ["kept"]
    assert len(agent) == 1


def test_flush_with_zero_budget_drops_everything():
    agent = MemoryAgent(inscription_policy=UtilityWritePolicy(budget=0))
    agent.add_candidate("A", source=SourceLabel.EXTERNAL, predicted_utility=0.9)
    agent.add_candidate("B", source=SourceLabel.EXTERNAL, predicted_utility=0.5)
    committed = agent.flush_inscriptions()
    assert committed == []
    assert len(agent) == 0


def test_immediate_add_unaffected_by_inscription_policy():
    # add() is immediate-commit even when a policy is configured.
    agent = MemoryAgent(inscription_policy=UtilityWritePolicy(budget=1))
    record = agent.add("immediate", source=SourceLabel.EXTERNAL)
    assert len(agent) == 1
    assert record.content == "immediate"


def test_utility_validates_against_relevance_dominance_pattern():
    # Replicates the structural finding from ledger section 1.5:
    # utility-based selection separates high-relevance distractors from
    # low-relevance-but-useful records.
    agent = MemoryAgent(inscription_policy=UtilityWritePolicy(budget=3))

    # Three useful records with mid/low relevance but high utility.
    agent.add_candidate(
        "useful_1", source=SourceLabel.EXTERNAL, predicted_utility=0.95
    )
    agent.add_candidate(
        "useful_2", source=SourceLabel.EXTERNAL, predicted_utility=0.90
    )
    agent.add_candidate(
        "useful_3", source=SourceLabel.EXTERNAL, predicted_utility=0.85
    )
    # Three distractors with high relevance (irrelevant here) but low utility.
    agent.add_candidate(
        "distractor_1", source=SourceLabel.EXTERNAL, predicted_utility=0.10
    )
    agent.add_candidate(
        "distractor_2", source=SourceLabel.EXTERNAL, predicted_utility=0.20
    )
    agent.add_candidate(
        "distractor_3", source=SourceLabel.EXTERNAL, predicted_utility=0.15
    )

    committed = agent.flush_inscriptions()
    contents = {r.content for r in committed}
    # All three useful records committed; no distractors.
    assert contents == {"useful_1", "useful_2", "useful_3"}

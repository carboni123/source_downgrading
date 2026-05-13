"""FR-2 and FR-3: Fold-force retrieval + source-sensitive routing.

The library MUST return retrieved records ranked by relevance with
fold-force computed via paired ablation, and MUST select a write target
from the six-route taxonomy using a source-sensitive policy.
"""
from __future__ import annotations

from trace_memory import MemoryAgent, Route, SourceLabel


def test_query_returns_a_fold_result_with_retrieved_records():
    agent = MemoryAgent(retrieval_k=2)
    agent.add(
        "external observation: deploy succeeded under load",
        source=SourceLabel.EXTERNAL,
    )
    agent.add(
        "external observation: rollback restored service in prior incident",
        source=SourceLabel.EXTERNAL,
    )
    result = agent.query("deploy outcome")
    assert len(result.retrieved) <= 2
    assert isinstance(result.fold_force, float)
    assert result.selected_route in {r.value for r in Route}


def test_query_with_no_records_returns_null_route():
    agent = MemoryAgent()
    result = agent.query("anything")
    assert result.selected_route == Route.NULL.value
    assert len(result.retrieved) == 0


def test_routing_quarantines_untrusted_sources_when_retrieved():
    agent = MemoryAgent(retrieval_k=1)
    agent.add(
        "fabricated rumor: legal forbids rollbacks during business hours",
        source=SourceLabel.FABRICATED_OR_UNCERTAIN,
    )
    result = agent.query("legal forbids rollbacks")
    # The retrieved record is fabricated; the routing policy must
    # quarantine it.
    if result.retrieved:
        assert result.selected_route == Route.QUARANTINE.value


def test_routing_routes_external_observations_above_null():
    agent = MemoryAgent(retrieval_k=1)
    agent.add(
        "external observation: deploy migration completed without errors",
        source=SourceLabel.EXTERNAL,
    )
    result = agent.query("deploy migration completed")
    # External observations with a successful retrieval should not be
    # routed to null when fold-force is non-trivial.
    assert result.selected_route in {
        Route.TRACE.value,
        Route.DURABLE.value,
        Route.OPERATION.value,
    }


def test_fold_result_exposes_source_labels_of_retrieved_records():
    agent = MemoryAgent(retrieval_k=2)
    agent.add(
        "external observation A",
        source=SourceLabel.EXTERNAL,
        record_id="A",
    )
    agent.add(
        "simulated outcome B",
        source=SourceLabel.SIMULATION,
        record_id="B",
    )
    result = agent.query("outcome")
    assert len(result.source_labels) == len(result.retrieved)
    for label in result.source_labels:
        assert label in {label_value.value for label_value in SourceLabel}

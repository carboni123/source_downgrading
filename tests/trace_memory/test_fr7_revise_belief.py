"""FR-7: Correction-chain node schema.

The library MUST record belief revisions as full correction nodes with
the schema (prior_belief, evidence, update_operation, revised_belief,
delta, provenance, confidence). Nodes MUST be retrievable and the full
list inspectable.
"""
from __future__ import annotations

from trace_memory import CorrectionNode, MemoryAgent, SourceLabel


def _make_revision(agent: MemoryAgent) -> CorrectionNode:
    return agent.revise_belief(
        prior_belief="high CPU on web tier means add web servers",
        evidence="after adding servers CPU stayed high; profiler found O(n^2) handler",
        update_operation="replace capacity hypothesis with code-path hypothesis",
        revised_belief="fix handler complexity before scaling infrastructure",
        delta="root_cause:web_capacity->handler_complexity",
        provenance=("profiling_run_17",),
        confidence=0.93,
    )


def test_revise_belief_returns_correction_node_with_full_schema():
    agent = MemoryAgent()
    node = _make_revision(agent)
    assert isinstance(node, CorrectionNode)
    assert node.prior_belief.startswith("high CPU")
    assert node.evidence.startswith("after adding servers")
    assert node.update_operation.startswith("replace capacity")
    assert node.revised_belief.startswith("fix handler")
    assert node.delta == "root_cause:web_capacity->handler_complexity"
    assert node.provenance == ("profiling_run_17",)
    assert node.confidence == 0.93
    assert node.record_id is not None
    assert node.timestamp > 0


def test_correction_node_is_persisted_in_store():
    agent = MemoryAgent()
    node = _make_revision(agent)
    record = agent.store.get(node.record_id)
    assert record is not None
    assert record.source_label == SourceLabel.OPERATION_RECORD.value


def test_correction_nodes_returns_all_revisions():
    agent = MemoryAgent()
    n1 = _make_revision(agent)
    n2 = agent.revise_belief(
        prior_belief="cache is fine",
        evidence="cache hit rate dropped 40%",
        update_operation="invalidate cache hypothesis",
        revised_belief="cache regression after key rotation",
        delta="cache:fine->regressed",
        confidence=0.88,
    )
    nodes = agent.correction_nodes()
    assert len(nodes) == 2
    assert {n.node_id for n in nodes} == {n1.node_id, n2.node_id}


def test_revise_belief_does_not_default_confidence_to_external_strength():
    # Even if a caller omits confidence (defaults 1.0), the node's
    # record is stored as OPERATION_RECORD source, not EXTERNAL.
    agent = MemoryAgent()
    node = agent.revise_belief(
        prior_belief="X",
        evidence="Y",
        update_operation="Z",
        revised_belief="W",
        delta="d",
    )
    record = agent.store.get(node.record_id)
    assert record.source_label != SourceLabel.EXTERNAL.value
    assert record.source_label == SourceLabel.OPERATION_RECORD.value

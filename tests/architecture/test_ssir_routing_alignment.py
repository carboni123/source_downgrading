"""Source-Sensitive Inscription Routing alignment tests.

These tests lock in the SSIR §6 (correction-chain) and §7 Prop. 2
(echo amplification) routing semantics after the fixes that distinguish
current-event delta from retrieval-of-correction and that gate durable
promotion of reactivated content on corroboration or reliability.
"""
from __future__ import annotations

from fgm.core import (
    EVENT_DELTA_THRESHOLD,
    FGMAgent,
    REACTIVATION_RELIABILITY_THRESHOLD,
    ROUTE_CORRECTION_CHAIN,
    ROUTE_DURABLE_MEMORY,
    ROUTE_OPERATION_MEMORY,
    SOURCE_EXTERNAL,
)


# ---------------------------------------------------------------------------
# Issue 1: correction-chain routes on current event delta, not on retrieval
# class. SSIR §6: |Δ_t| ≥ θ_Δ AND u_t identifiable.
# ---------------------------------------------------------------------------


def test_retrieving_a_correction_record_without_delta_routes_to_operation():
    """A query that retrieves a prior correction node does not itself
    constitute a new revision event. Without an explicit event_delta the
    route stays at operation_memory."""
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.store.add(
        "correction: prior=X; evidence=Y; update=replace_X_with_Z; revised=Z",
        record_id="corr_1",
        record_type="correction",
        operation_type="correction",
        source_label=SOURCE_EXTERNAL,
        source_confidence=1.0,
    )
    result = agent.query("what is the revised belief after the prior X claim")
    assert result.retrieved
    # The correction-chain score must NOT dominate operation_memory in the
    # absence of a current revision event.
    assert result.selected_route == ROUTE_OPERATION_MEMORY
    assert (
        result.route_scores[ROUTE_OPERATION_MEMORY]
        > result.route_scores[ROUTE_CORRECTION_CHAIN]
    )


def test_query_with_event_delta_and_update_op_routes_to_correction_chain():
    """When the caller signals a real revision event (delta ≥ θ_Δ AND an
    update operation), the route is correction_chain."""
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "prior belief: high CPU means add web servers",
        record_id="prior_1",
        source_label=SOURCE_EXTERNAL,
    )
    result = agent.query(
        "profiler shows O(n^2) handler is the root cause",
        event_delta=0.4,
        update_operation="replace_capacity_with_code_path",
    )
    assert result.selected_route == ROUTE_CORRECTION_CHAIN


def test_event_delta_below_threshold_does_not_trigger_correction():
    """A small delta does not count as a revision — guards against
    noise inflating the correction-chain rate."""
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add("observation", record_id="o1", source_label=SOURCE_EXTERNAL)
    tiny_delta = EVENT_DELTA_THRESHOLD * 0.5
    result = agent.query(
        "observation revisited",
        event_delta=tiny_delta,
        update_operation="minor_tweak",
    )
    assert result.selected_route != ROUTE_CORRECTION_CHAIN


def test_event_delta_without_update_operation_does_not_trigger_correction():
    """The SSIR condition is delta ≥ θ_Δ AND u_t identifiable. Delta
    alone is insufficient."""
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add("observation", record_id="o1", source_label=SOURCE_EXTERNAL)
    result = agent.query(
        "observation revisited",
        event_delta=0.4,
        update_operation=None,
    )
    assert result.selected_route != ROUTE_CORRECTION_CHAIN


# ---------------------------------------------------------------------------
# Issue 2: SSIR §7 Prop. 2 echo guard. π_t^H(s=react) is suppressed unless
# external corroboration is present in the retrieval, the caller signals
# corroboration explicitly, or reactivation reliability ≥ θ_R.
# ---------------------------------------------------------------------------


def test_durable_score_zero_for_pure_reactivation_without_corroboration():
    """A query that retrieves only non-external (e.g. inference) content
    and without explicit corroboration must have a zero durable score."""
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "internally derived claim about deploy outcomes",
        record_id="d1",
        source_label="inference",
        source_confidence=1.0,
        provenance=("e1",),
    )
    result = agent.query("deploy outcome reasoning")
    assert result.retrieved
    # Source labels must NOT include external for this guard to apply.
    assert all(label != SOURCE_EXTERNAL for label in result.source_labels)
    assert result.route_scores[ROUTE_DURABLE_MEMORY] == 0.0


def test_durable_score_allowed_when_external_record_present_in_retrieval():
    """Corroboration via at least one externally-sourced retrieved
    record unlocks the durable score."""
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "external observation: migration succeeded",
        record_id="e1",
        source_label=SOURCE_EXTERNAL,
        source_confidence=1.0,
    )
    result = agent.query("migration succeeded")
    assert result.retrieved
    assert result.source_labels[0] == SOURCE_EXTERNAL
    assert result.route_scores[ROUTE_DURABLE_MEMORY] > 0.0


def test_durable_score_allowed_when_caller_signals_corroboration():
    """The caller can vouch for corroboration even when the retrieval
    set is reactivation-only."""
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "internally derived claim",
        record_id="d1",
        source_label="inference",
        source_confidence=1.0,
        provenance=("e1",),
    )
    result = agent.query("derived claim", external_corroboration=True)
    assert result.route_scores[ROUTE_DURABLE_MEMORY] > 0.0


def test_durable_score_allowed_when_reliability_meets_threshold():
    """High caller-supplied reactivation reliability also unlocks the
    durable score, matching SSIR Prop. 2's R_t ≥ θ_R branch."""
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "internally derived claim",
        record_id="d1",
        source_label="inference",
        source_confidence=1.0,
        provenance=("e1",),
    )
    result = agent.query(
        "derived claim",
        reactivation_reliability=REACTIVATION_RELIABILITY_THRESHOLD,
    )
    assert result.route_scores[ROUTE_DURABLE_MEMORY] > 0.0


def test_durable_score_zero_when_reliability_below_threshold():
    agent = FGMAgent(dim=32, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "internally derived claim",
        record_id="d1",
        source_label="inference",
        source_confidence=1.0,
        provenance=("e1",),
    )
    result = agent.query(
        "derived claim",
        reactivation_reliability=REACTIVATION_RELIABILITY_THRESHOLD * 0.5,
    )
    assert result.route_scores[ROUTE_DURABLE_MEMORY] == 0.0

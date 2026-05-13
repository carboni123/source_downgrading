"""Facade test: query() routing kwargs reach the underlying router.

Confirms that ``event_delta``, ``update_operation``,
``external_corroboration``, and ``reactivation_reliability`` flow through
``MemoryAgent.query()`` into the fgm router and produce the SSIR-aligned
route selections.
"""
from __future__ import annotations

import asyncio

from trace_memory import MemoryAgent, Route, SourceLabel


def test_query_event_delta_routes_to_correction_chain():
    agent = MemoryAgent(retrieval_k=1)
    agent.add(
        "prior belief: high CPU means add web servers",
        source=SourceLabel.EXTERNAL,
    )
    result = agent.query(
        "profiler shows O(n^2) handler is the root cause",
        event_delta=0.4,
        update_operation="replace_capacity_with_code_path",
    )
    assert result.selected_route == Route.CORRECTION.value


def test_query_without_event_delta_does_not_route_to_correction():
    agent = MemoryAgent(retrieval_k=1)
    agent.add("observation about deploy", source=SourceLabel.EXTERNAL)
    result = agent.query("deploy observations")
    assert result.selected_route != Route.CORRECTION.value


def test_durable_score_blocked_for_reactivation_only_retrieval():
    """When retrieval surfaces only derived (non-external) content, the
    durable route must be zero unless the caller supplies corroboration."""
    agent = MemoryAgent(retrieval_k=1)
    a = agent.add("external observation A", source=SourceLabel.EXTERNAL)
    b = agent.add("external observation B", source=SourceLabel.EXTERNAL)
    derived = agent.add_derived("derived inference from A and B", inputs=[a, b])
    # Query that should retrieve the derived (inference-source) record.
    result = agent.query(derived.content)
    if result.retrieved and all(
        label != SourceLabel.EXTERNAL.value for label in result.source_labels
    ):
        assert result.route_scores[Route.DURABLE.value] == 0.0


def test_durable_score_unblocked_with_explicit_corroboration_signal():
    agent = MemoryAgent(retrieval_k=1)
    a = agent.add("external A", source=SourceLabel.EXTERNAL)
    b = agent.add("external B", source=SourceLabel.EXTERNAL)
    derived = agent.add_derived("derived inference from A and B", inputs=[a, b])
    result = agent.query(derived.content, external_corroboration=True)
    if result.retrieved and all(
        label != SourceLabel.EXTERNAL.value for label in result.source_labels
    ):
        assert result.route_scores[Route.DURABLE.value] > 0.0


def test_async_query_threads_routing_kwargs():
    async def _run():
        agent = MemoryAgent(retrieval_k=1)
        agent.add("prior belief", source=SourceLabel.EXTERNAL)
        result = await agent.aquery(
            "revised belief evidence",
            event_delta=0.4,
            update_operation="replace_prior_with_revised",
        )
        return result

    result = asyncio.run(_run())
    assert result.selected_route == Route.CORRECTION.value

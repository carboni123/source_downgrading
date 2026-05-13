"""Async API tests.

Each async wrapper is a thin asyncio.to_thread offload of its sync
counterpart. We verify:

1. Every aX method works in an asyncio.run context and returns the
   right thing.
2. Async results equal sync results (modulo record_id generation and
   timestamps, which are inherently non-deterministic).
3. SQLite-backed agents work across the threadpool boundary (verifies
   check_same_thread=False).
4. Concurrent mutations serialize through the internal lock without
   data corruption.

No pytest-asyncio dependency: we use asyncio.run inline so the test
file remains compatible with stock pytest.
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from trace_memory import (
    MemoryAgent,
    SQLiteStorage,
    SelfIndex,
    SourceLabel,
    UtilityWritePolicy,
    ainfer_source,
)


# ---------------------------------------------------------------------------
# Each aX method works
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def test_aadd_returns_a_record():
    agent = MemoryAgent()

    async def go():
        return await agent.aadd("hello", source=SourceLabel.EXTERNAL)

    r = _run(go())
    assert r.content == "hello"
    assert r.source_label == "external"


def test_aadd_derived_caps_trust_at_min_input():
    agent = MemoryAgent()

    async def go():
        r1 = await agent.aadd("E", source=SourceLabel.EXTERNAL)
        r2 = await agent.aadd("S", source=SourceLabel.SIMULATION)
        return await agent.aadd_derived("derived", inputs=[r1, r2])

    derived = _run(go())
    assert derived.source_label == "simulation"


def test_aquery_returns_a_fold_result():
    agent = MemoryAgent(retrieval_k=2)

    async def go():
        await agent.aadd("server returned 500", source=SourceLabel.EXTERNAL)
        return await agent.aquery("server status")

    result = _run(go())
    assert hasattr(result, "fold_force")
    assert hasattr(result, "selected_route")


def test_aadd_candidate_and_aflush_inscriptions():
    agent = MemoryAgent(inscription_policy=UtilityWritePolicy(budget=2))

    async def go():
        await agent.aadd_candidate(
            "low", source=SourceLabel.EXTERNAL, predicted_utility=0.1
        )
        await agent.aadd_candidate(
            "high", source=SourceLabel.EXTERNAL, predicted_utility=0.9
        )
        await agent.aadd_candidate(
            "mid", source=SourceLabel.EXTERNAL, predicted_utility=0.5
        )
        return await agent.aflush_inscriptions()

    committed = _run(go())
    assert {r.content for r in committed} == {"high", "mid"}


def test_aadd_with_inferred_source_inscribes_a_record():
    agent = MemoryAgent()

    async def go():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return await agent.aadd_with_inferred_source(
                "fabricated rumor: something"
            )

    record = _run(go())
    assert record.source_label == SourceLabel.FABRICATED_OR_UNCERTAIN.value


def test_arevise_belief_returns_correction_node():
    agent = MemoryAgent()

    async def go():
        return await agent.arevise_belief(
            prior_belief="X",
            evidence="Y",
            update_operation="Z",
            revised_belief="W",
            delta="d",
            confidence=0.9,
        )

    node = _run(go())
    assert node.prior_belief == "X"
    assert node.confidence == 0.9


def test_aaudit_laundering_returns_audit_with_warning():
    agent = MemoryAgent()

    async def go():
        r1 = await agent.aadd("E", source=SourceLabel.EXTERNAL)
        r2 = await agent.aadd("S", source=SourceLabel.SIMULATION)
        await agent.aadd_derived("D", inputs=[r1, r2])
        return await agent.aaudit_laundering()

    audit = _run(go())
    assert audit.cascade_invisibility_warning is True
    assert audit.n_records == 1


def test_ainfer_source_classifies_marker_content():
    async def go():
        return await ainfer_source("hypothetical: if traffic doubled")

    label = _run(go())
    assert label == SourceLabel.SIMULATION


# ---------------------------------------------------------------------------
# Sync/async equivalence
# ---------------------------------------------------------------------------


def test_async_results_match_sync_when_inputs_match():
    sync_agent = MemoryAgent()
    async_agent = MemoryAgent()

    sync_record = sync_agent.add(
        "shared content",
        source=SourceLabel.EXTERNAL,
        record_id="shared",
    )

    async def go():
        return await async_agent.aadd(
            "shared content",
            source=SourceLabel.EXTERNAL,
            record_id="shared",
        )

    async_record = _run(go())

    assert sync_record.content == async_record.content
    assert sync_record.source_label == async_record.source_label
    assert sync_record.record_id == async_record.record_id


# ---------------------------------------------------------------------------
# SQLite cross-thread access
# ---------------------------------------------------------------------------


def test_async_writes_persist_through_sqlite_disk():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    async def session_one():
        agent = MemoryAgent(storage=SQLiteStorage(db_path))
        try:
            await agent.aadd("E1", source=SourceLabel.EXTERNAL, record_id="E1")
            await agent.aadd("E2", source=SourceLabel.TOOL_OUTPUT, record_id="E2")
        finally:
            agent.close()

    async def session_two():
        agent = MemoryAgent(storage=SQLiteStorage(db_path))
        try:
            await agent.aadd("E3", source=SourceLabel.EXTERNAL, record_id="E3")
            return len(agent)
        finally:
            agent.close()

    try:
        _run(session_one())
        record_count = _run(session_two())
        assert record_count == 3
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Concurrency: lock serializes mutations
# ---------------------------------------------------------------------------


def test_concurrent_aadd_does_not_corrupt_store():
    agent = MemoryAgent()
    n = 50

    async def writer(i: int):
        await agent.aadd(
            f"content #{i}",
            source=SourceLabel.EXTERNAL,
            record_id=f"r{i:03d}",
        )

    async def go():
        await asyncio.gather(*(writer(i) for i in range(n)))

    _run(go())
    assert len(agent) == n
    # All record ids must be present and distinct.
    ids = {rec.record_id for rec in agent.store.all_records()}
    assert ids == {f"r{i:03d}" for i in range(n)}


def test_concurrent_aquery_during_aadd_does_not_crash():
    agent = MemoryAgent()
    for i in range(20):
        agent.add(f"seed #{i}", source=SourceLabel.EXTERNAL, record_id=f"s{i:03d}")

    async def reader():
        for _ in range(20):
            await agent.aquery("seed")

    async def writer():
        for i in range(20, 40):
            await agent.aadd(
                f"writer #{i}",
                source=SourceLabel.EXTERNAL,
                record_id=f"w{i:03d}",
            )

    async def go():
        await asyncio.gather(reader(), writer())

    _run(go())
    # Note: agent.query() can write operation-memory records back to the
    # store when folds are gated, so len(agent) may exceed 40. We only
    # check that every record we explicitly added is present.
    ids = {rec.record_id for rec in agent.store.all_records()}
    expected = {f"s{i:03d}" for i in range(20)} | {f"w{i:03d}" for i in range(20, 40)}
    assert expected.issubset(ids)


def test_concurrent_async_derivations_preserve_provenance():
    agent = MemoryAgent()
    r1 = agent.add("E1", source=SourceLabel.EXTERNAL, record_id="E1")
    r2 = agent.add("E2", source=SourceLabel.EXTERNAL, record_id="E2")

    async def deriver(i: int):
        await agent.aadd_derived(
            f"derived #{i}",
            inputs=[r1, r2],
            record_id=f"d{i:03d}",
        )

    async def go():
        await asyncio.gather(*(deriver(i) for i in range(20)))

    _run(go())
    # Each derived record must have both inputs in its provenance.
    for i in range(20):
        derived = agent.store.get(f"d{i:03d}")
        assert derived is not None
        assert "E1" in derived.provenance
        assert "E2" in derived.provenance
        assert derived.source_label == SourceLabel.INFERENCE.value


# ---------------------------------------------------------------------------
# self-index round-trips through async path
# ---------------------------------------------------------------------------


def test_aadd_respects_active_self_index():
    agent = MemoryAgent(
        self_index=SelfIndex(user_id="alice", project_id="X"),
    )

    async def go():
        return await agent.aadd("alice content", source=SourceLabel.EXTERNAL)

    record = _run(go())
    from trace_memory.self_index import record_self_index
    idx = record_self_index(record)
    assert idx is not None
    assert idx.user_id == "alice"
    assert idx.project_id == "X"

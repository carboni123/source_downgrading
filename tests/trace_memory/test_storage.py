"""Storage backend tests.

Covers the Storage protocol, InMemoryStorage no-op shim, and
SQLiteStorage round-trip + multi-session persistence.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from fgm.core import MemoryRecord

from trace_memory import (
    InMemoryStorage,
    MemoryAgent,
    SelfIndex,
    SourceLabel,
    SQLiteStorage,
    Storage,
)


# ---------------------------------------------------------------------------
# InMemoryStorage
# ---------------------------------------------------------------------------


def test_in_memory_storage_satisfies_protocol():
    storage = InMemoryStorage()
    assert isinstance(storage, Storage)


def test_in_memory_storage_is_a_no_op():
    storage = InMemoryStorage()
    record = MemoryRecord(
        record_id="x",
        content="hello",
        vector=np.zeros(64),
        timestamp=0.0,
        record_type="content",
        source_label="external",
    )
    storage.save(record)
    assert list(storage.load_all()) == []
    assert not storage.contains("x")
    assert not storage.delete("x")


# ---------------------------------------------------------------------------
# SQLiteStorage round-trip
# ---------------------------------------------------------------------------


def _make_record(record_id: str = "r1") -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content="hello",
        vector=np.arange(8, dtype=np.float64) / 10.0,
        timestamp=123.456,
        record_type="content",
        operation_type=None,
        decision_content=None,
        source_label="external",
        source_confidence=0.9,
        provenance=("origin_a", "origin_b"),
        metadata={"key": "value", "flag": True},
    )


def test_sqlite_storage_round_trip_in_memory_db():
    storage = SQLiteStorage(":memory:")
    original = _make_record()
    storage.save(original)
    loaded = list(storage.load_all())
    assert len(loaded) == 1
    rec = loaded[0]
    assert rec.record_id == original.record_id
    assert rec.content == original.content
    assert np.array_equal(rec.vector, original.vector)
    assert rec.timestamp == original.timestamp
    assert rec.record_type == original.record_type
    assert rec.source_label == original.source_label
    assert rec.source_confidence == original.source_confidence
    assert rec.provenance == original.provenance
    assert rec.metadata == original.metadata
    storage.close()


def test_sqlite_storage_preserves_insertion_order():
    storage = SQLiteStorage(":memory:")
    for i in range(5):
        storage.save(_make_record(record_id=f"r{i}"))
    ids = [r.record_id for r in storage.load_all()]
    assert ids == [f"r{i}" for i in range(5)]
    storage.close()


def test_sqlite_storage_save_is_idempotent_on_record_id():
    storage = SQLiteStorage(":memory:")
    storage.save(_make_record(record_id="r"))
    # Save again with different content; insertion order MUST NOT change.
    storage.save(_make_record(record_id="r"))
    storage.save(_make_record(record_id="other"))
    ids = [r.record_id for r in storage.load_all()]
    assert ids == ["r", "other"]
    storage.close()


def test_sqlite_storage_contains_and_delete():
    storage = SQLiteStorage(":memory:")
    storage.save(_make_record(record_id="r"))
    assert storage.contains("r")
    assert not storage.contains("missing")
    assert storage.delete("r") is True
    assert storage.delete("r") is False
    assert not storage.contains("r")
    storage.close()


def test_sqlite_storage_context_manager_closes_connection():
    with SQLiteStorage(":memory:") as storage:
        storage.save(_make_record())
        assert len(list(storage.load_all())) == 1
    # After context exit, connection is closed.
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        storage.connection.execute("SELECT 1")


# ---------------------------------------------------------------------------
# Agent-level persistence across sessions
# ---------------------------------------------------------------------------


def _tmp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def test_agent_state_survives_a_close_and_reopen_cycle():
    db = _tmp_db_path()
    try:
        # Session 1.
        with MemoryAgent(storage=SQLiteStorage(db)) as agent:
            agent.add("E1", source=SourceLabel.EXTERNAL, record_id="E1")
            agent.add("E2", source=SourceLabel.TOOL_OUTPUT, record_id="E2")
            agent.add_derived(
                "derived",
                inputs=["E1", "E2"],
                record_id="D",
            )
            agent.revise_belief(
                prior_belief="X",
                evidence="Y",
                update_operation="Z",
                revised_belief="W",
                delta="d",
                confidence=0.9,
            )
            assert len(agent) == 4
            assert len(agent.correction_nodes()) == 1

        # Session 2: reopen on the same DB.
        with MemoryAgent(storage=SQLiteStorage(db)) as agent2:
            assert len(agent2) == 4
            assert len(agent2.correction_nodes()) == 1
            derived = agent2.store.get("D")
            assert derived is not None
            assert derived.source_label == SourceLabel.INFERENCE.value
            assert "E1" in derived.provenance
            assert "E2" in derived.provenance
            # The derived flag survived round-trip.
            assert "D" in agent2._derived_record_ids
    finally:
        os.unlink(db)


def test_agent_persistence_round_trips_self_index_metadata():
    db = _tmp_db_path()
    try:
        with MemoryAgent(
            storage=SQLiteStorage(db),
            self_index=SelfIndex(user_id="alice", project_id="X"),
        ) as agent:
            agent.add("alice record", source=SourceLabel.EXTERNAL, record_id="A")

        # Reopen WITHOUT setting the active index -- the record is scoped
        # and must NOT be visible until the right index is set.
        with MemoryAgent(storage=SQLiteStorage(db)) as agent:
            result = agent.query("record")
            assert "A" not in {h.record.record_id for h in result.retrieved}
            agent.active_self_index = SelfIndex(user_id="alice", project_id="X")
            result = agent.query("record")
            assert "A" in {h.record.record_id for h in result.retrieved}
    finally:
        os.unlink(db)


def test_agent_persistence_round_trips_audit_state():
    db = _tmp_db_path()
    try:
        with MemoryAgent(storage=SQLiteStorage(db)) as agent:
            r_ext = agent.add("E", source=SourceLabel.EXTERNAL)
            r_sim = agent.add("S", source=SourceLabel.SIMULATION)
            agent.add_derived("D", inputs=[r_ext, r_sim], record_id="D")

        with MemoryAgent(storage=SQLiteStorage(db)) as agent2:
            audit = agent2.audit_laundering(
                truth_ceilings={"D": SourceLabel.SIMULATION},
            )
            assert audit.n_records == 1
            assert audit.local_laundering_rate == 0.0
            assert audit.truth_grounded_rate == 0.0
            assert audit.is_clean is True
    finally:
        os.unlink(db)


def test_default_storage_is_in_memory():
    agent = MemoryAgent()
    assert isinstance(agent.storage, InMemoryStorage)


def test_operation_memory_records_persist_through_query():
    # Regression: FGMAgent.OperationMemory.record_fold writes operation
    # records via store.add(...) directly, bypassing trace-memory's
    # public add() wrapper. The agent must intercept these and persist
    # them, otherwise query-side state is lost on reload.
    db = _tmp_db_path()
    try:
        with MemoryAgent(storage=SQLiteStorage(db)) as agent:
            agent.add("E1", source=SourceLabel.EXTERNAL, record_id="E1")
            agent.add("E2", source=SourceLabel.EXTERNAL, record_id="E2")
            result = agent.query("E")
            # If the query gated a fold, an operation record was written.
            if result.operation_record_id is not None:
                expected_op_id = result.operation_record_id
                pre_close_count = len(agent)
            else:
                pytest.skip("query did not gate a fold; nothing to test")

        with MemoryAgent(storage=SQLiteStorage(db)) as agent2:
            assert len(agent2) == pre_close_count
            assert agent2.store.get(expected_op_id) is not None
    finally:
        os.unlink(db)

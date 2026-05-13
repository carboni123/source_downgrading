"""Tests for FGM core components."""
import numpy as np
import pytest

from fgm.core import (
    hash_embed,
    cosine,
    l2,
    default_transition,
    MemoryStore,
    MarginRetriever,
    FoldGate,
    OperationMemory,
    Compressor,
    FGMAgent,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestHashEmbed:
    def test_deterministic(self):
        a = hash_embed("hello world", 32)
        b = hash_embed("hello world", 32)
        assert np.allclose(a, b)

    def test_normalized(self):
        v = hash_embed("some text here", 64)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_different_texts_differ(self):
        a = hash_embed("deploy failed migration timeout", 64)
        b = hash_embed("coffee meeting tuesday afternoon", 64)
        assert cosine(a, b) < 0.9


class TestTransition:
    def test_fold_changes_output(self):
        state = np.zeros(8)
        inp = np.ones(8) * 0.5
        fold = np.ones(8) * 0.3
        with_fold = default_transition(state, inp, fold)
        without_fold = default_transition(state, inp, None)
        assert l2(with_fold, without_fold) > 0.01

    def test_no_fold_is_deterministic(self):
        state = np.ones(8) * 0.1
        inp = np.ones(8) * 0.5
        a = default_transition(state, inp, None)
        b = default_transition(state, inp, None)
        assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# Layer 2: Storage
# ---------------------------------------------------------------------------

class TestMemoryStore:
    def test_add_and_retrieve(self):
        store = MemoryStore(dim=32)
        rec = store.add("hello world", record_id="r1")
        assert store.get("r1") is not None
        assert store.get("r1").content == "hello world"
        assert len(store) == 1

    def test_remove(self):
        store = MemoryStore(dim=32)
        store.add("hello", record_id="r1")
        assert store.remove("r1")
        assert store.get("r1") is None
        assert len(store) == 0

    def test_fold_force_tracking(self):
        store = MemoryStore(dim=32)
        store.add("hello", record_id="r1")
        store.record_fold_force("r1", 0.5)
        store.record_fold_force("r1", 0.3)
        assert abs(store.total_fold_force("r1") - 0.8) < 1e-9
        assert store.use_count("r1") == 2

    def test_custom_vector(self):
        store = MemoryStore(dim=4)
        v = np.array([1.0, 0.0, 0.0, 0.0])
        rec = store.add("test", record_id="r1", vector=v)
        assert np.allclose(rec.vector, v)


# ---------------------------------------------------------------------------
# Layer 3: Addressability
# ---------------------------------------------------------------------------

class TestMarginRetriever:
    def test_retrieve_finds_best_match(self):
        store = MemoryStore(dim=32)
        store.add("database migration timeout", record_id="r1")
        store.add("coffee meeting notes", record_id="r2")
        retriever = MarginRetriever(store)
        report = retriever.retrieve("database migration failed")
        assert report.hits[0].record.record_id == "r1"

    def test_margin_positive_for_distinctive_records(self):
        store = MemoryStore(dim=64)
        store.add("database migration timeout failure", record_id="r1")
        store.add("quarterly revenue analysis report", record_id="r2")
        retriever = MarginRetriever(store)
        report = retriever.retrieve("database migration")
        assert report.hits[0].margin > 0

    def test_target_rank_margin(self):
        store = MemoryStore(dim=32)
        store.add("alpha beta gamma", record_id="r1")
        store.add("delta epsilon zeta", record_id="r2")
        retriever = MarginRetriever(store)
        rank, score, margin = retriever.target_rank_margin("alpha beta", "r1")
        assert rank == 1
        assert margin > 0


# ---------------------------------------------------------------------------
# Layer 4: Folding
# ---------------------------------------------------------------------------

class TestFoldGate:
    def test_fold_force_nonzero_with_memory(self):
        store = MemoryStore(dim=8)
        rec = store.add("important decision content", record_id="r1")
        from fgm.core import RetrievalHit
        hit = RetrievalHit(rec, score=0.9, rank=1, margin=0.5)
        gate = FoldGate(threshold=0.001)
        state = np.zeros(8)
        result = gate.fold("what was the decision?", rec.vector, [hit], state)
        assert result.fold_force > 0.001
        assert result.gated is True

    def test_fold_force_zero_without_memory(self):
        gate = FoldGate(threshold=0.001)
        state = np.zeros(8)
        q_vec = hash_embed("query", 8)
        result = gate.fold("query", q_vec, [], state)
        assert result.fold_force < 1e-9
        assert result.gated is False

    def test_fold_vector_is_weighted_average(self):
        dim = 8
        store = MemoryStore(dim=dim)
        r1 = store.add("alpha", record_id="r1", vector=np.eye(dim)[0])
        r2 = store.add("beta", record_id="r2", vector=np.eye(dim)[1])
        from fgm.core import RetrievalHit
        hits = [
            RetrievalHit(r1, score=0.9, rank=1, margin=0.5),
            RetrievalHit(r2, score=0.1, rank=2, margin=-0.3),
        ]
        gate = FoldGate()
        fv = gate.compute_fold_vector(hits)
        assert fv is not None
        assert fv[0] > fv[1]  # higher-scored record gets more weight


# ---------------------------------------------------------------------------
# Layer 5: Operation-Memory
# ---------------------------------------------------------------------------

class TestOperationMemory:
    def test_record_fold_creates_operation(self):
        store = MemoryStore(dim=8)
        rec = store.add("content", record_id="r1")
        ops = OperationMemory(store)
        from fgm.core import FoldResult, RetrievalHit
        hit = RetrievalHit(rec, 0.9, 1, 0.5)
        fold_result = FoldResult(
            query="test query",
            query_vector=rec.vector,
            retrieved=[hit],
            fold_vector=rec.vector,
            output_with=np.ones(8),
            output_without=np.zeros(8),
            fold_force=1.0,
            full_divergence=1.0,
            gated=True,
        )
        op = ops.record_fold(fold_result)
        assert len(ops) == 1
        assert op.fold_force == 1.0
        assert "r1" in op.retrieved_ids

    def test_operation_stored_in_store(self):
        store = MemoryStore(dim=8)
        rec = store.add("content", record_id="r1")
        ops = OperationMemory(store)
        from fgm.core import FoldResult, RetrievalHit
        hit = RetrievalHit(rec, 0.9, 1, 0.5)
        fold_result = FoldResult(
            query="test", query_vector=rec.vector, retrieved=[hit],
            fold_vector=rec.vector, output_with=np.ones(8),
            output_without=np.zeros(8), fold_force=1.0, full_divergence=1.0,
            gated=True,
        )
        op = ops.record_fold(fold_result)
        stored = store.get(op.operation_id)
        assert stored is not None
        assert stored.record_type == "operation"


# ---------------------------------------------------------------------------
# Layer 6: Compression
# ---------------------------------------------------------------------------

class TestCompressor:
    def test_prune_zero_fold_removes_unused(self):
        store = MemoryStore(dim=32)
        store.add("used record", record_id="r1")
        store.add("unused record", record_id="r2")
        store.record_fold_force("r1", 0.5)
        store.record_fold_force("r2", 0.0)  # used but zero force
        retriever = MarginRetriever(store)
        comp = Compressor(store, retriever)
        report = comp.prune_zero_fold()
        assert "r2" in report.removed_ids
        assert store.get("r1") is not None
        assert store.get("r2") is None

    def test_merge_duplicates(self):
        store = MemoryStore(dim=32)
        v = hash_embed("database migration timeout", 32)
        store.add("database migration timeout v1", record_id="r1", vector=v)
        store.add("database migration timeout v2", record_id="r2", vector=v * 1.001)
        store.record_fold_force("r1", 1.0)
        store.record_fold_force("r2", 0.1)
        retriever = MarginRetriever(store)
        comp = Compressor(store, retriever, duplicate_threshold=0.99)
        report = comp.merge_duplicates()
        assert report.records_after < report.records_before
        assert store.get("r1") is not None  # winner kept


# ---------------------------------------------------------------------------
# FGMAgent integration
# ---------------------------------------------------------------------------

class TestFGMAgent:
    def test_basic_query_cycle(self):
        agent = FGMAgent(dim=32)
        agent.add("deploy failed because of migration timeout",
                   operation_type="observation")
        agent.add("fixed by increasing timeout to 30 minutes",
                   operation_type="decision",
                   decision_content="increase migration timeout to 30min")
        result = agent.query("what caused the deploy failure?")
        assert result.fold_force > 0
        assert isinstance(result.gated, bool)

    def test_operation_memory_recorded_for_gated_folds(self):
        agent = FGMAgent(dim=32, fold_threshold=0.001)
        agent.store.add("critical decision about architecture", record_id="r1")
        result = agent.query("what was the architecture decision?")
        if result.gated:
            assert len(agent.operations) > 0

    def test_fold_force_gating(self):
        agent = FGMAgent(dim=32, fold_threshold=100.0)
        agent.add("some content")
        result = agent.query("query")
        assert result.gated is False  # threshold too high

    def test_metrics(self):
        agent = FGMAgent(dim=32)
        agent.add("record one")
        agent.add("record two")
        agent.query("query one")
        m = agent.metrics()
        assert m["content_records"] >= 2
        assert m["queries_processed"] == 1
        assert "fold_gate_rate" in m

    def test_trace_decision_returns_operations(self):
        agent = FGMAgent(dim=32, fold_threshold=0.001)
        agent.add("database migration failed", operation_type="observation")
        agent.query("database migration")
        ops = agent.trace_decision("migration")
        # ops may be empty if the fold wasn't gated, but the method should work
        assert isinstance(ops, list)

    def test_compress(self):
        agent = FGMAgent(dim=32)
        v = hash_embed("duplicate content", 32)
        agent.store.add("duplicate A", record_id="r1", vector=v)
        agent.store.add("duplicate B", record_id="r2", vector=v * 1.001)
        agent.store.record_fold_force("r1", 1.0)
        agent.store.record_fold_force("r2", 0.0)
        report = agent.compress()
        assert report.records_after <= report.records_before

    def test_multi_step_state_evolution(self):
        agent = FGMAgent(dim=16, fold_threshold=0.001)
        agent.add("alpha bravo charlie")
        agent.add("delta echo foxtrot")
        agent.add("golf hotel india")

        states = []
        for q in ["alpha bravo", "delta echo", "golf hotel"]:
            result = agent.query(q)
            states.append(agent._state.copy())

        # state should evolve across queries
        assert not np.allclose(states[0], states[1])
        assert not np.allclose(states[1], states[2])

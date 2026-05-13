"""Tests for h_cog projection in the fold gate.

Validates:
  1. When cog_dims is set, fold-force is computed only over those dimensions
  2. Bookkeeping dimensions are excluded from the gating decision
  3. full_divergence always tracks the full output divergence
  4. A fold that only changes bookkeeping dims is gated out
  5. A fold that changes cognitive dims passes the gate
  6. Default (cog_dims=None) matches full-output behavior
"""
import numpy as np
import pytest

from fgm.core import (
    FGMAgent,
    FoldGate,
    MemoryStore,
    RetrievalHit,
    hash_embed,
    l2,
    default_transition,
)


def _make_transition_with_bookkeeping(cog_size: int, book_size: int):
    """Create a transition where bookkeeping dims echo the fold vector
    but cognitive dims respond differently based on fold presence."""
    dim = cog_size + book_size

    def transition(state, input_vec, fold_vec=None):
        out = np.tanh(0.65 * state[:dim] + 0.75 * input_vec[:dim])
        if fold_vec is not None:
            out[:cog_size] = np.tanh(
                0.65 * state[:cog_size] + 0.75 * input_vec[:cog_size] + 0.85 * fold_vec[:cog_size]
            )
            out[cog_size:] = fold_vec[cog_size:]
        return out

    return transition, list(range(cog_size))


class TestHCogProjection:
    def test_fold_force_restricted_to_cog_dims(self):
        """Fold-force should only measure divergence over cog_dims."""
        dim = 16
        cog_dims = list(range(12))
        gate = FoldGate(threshold=0.001, cog_dims=cog_dims)
        store = MemoryStore(dim=dim)
        rec = store.add("test content", record_id="r1")
        hit = RetrievalHit(rec, score=0.9, rank=1, margin=0.5)

        state = np.zeros(dim)
        result = gate.fold("query", rec.vector, [hit], state)

        expected_cog_force = l2(
            result.output_with[cog_dims], result.output_without[cog_dims]
        )
        assert abs(result.fold_force - expected_cog_force) < 1e-9

    def test_full_divergence_includes_all_dims(self):
        """full_divergence should always measure over all dimensions."""
        dim = 16
        cog_dims = list(range(8))
        gate = FoldGate(threshold=0.001, cog_dims=cog_dims)
        store = MemoryStore(dim=dim)
        rec = store.add("content", record_id="r1")
        hit = RetrievalHit(rec, score=0.9, rank=1, margin=0.5)

        state = np.zeros(dim)
        result = gate.fold("q", rec.vector, [hit], state)

        expected_full = l2(result.output_with, result.output_without)
        assert abs(result.full_divergence - expected_full) < 1e-9
        assert result.full_divergence >= result.fold_force

    def test_bookkeeping_only_change_gated_out(self):
        """A fold that only changes bookkeeping dims should be gated out."""
        cog_size, book_size = 8, 8
        dim = cog_size + book_size
        transition, cog_dims = _make_transition_with_bookkeeping(cog_size, book_size)

        gate = FoldGate(transition_fn=transition, threshold=0.01, cog_dims=cog_dims)
        store = MemoryStore(dim=dim)

        vec = np.zeros(dim)
        vec[cog_size:] = np.ones(book_size) * 0.5
        rec = store.add("bookkeeping only", record_id="r1", vector=vec)
        hit = RetrievalHit(rec, score=0.9, rank=1, margin=0.5)

        state = np.zeros(dim)
        input_vec = np.zeros(dim)
        result = gate.fold("q", input_vec, [hit], state)

        assert result.full_divergence > 0.01, "Full divergence should be nonzero"
        assert result.fold_force < 0.01, "h_cog fold-force should be near zero"
        assert result.gated is False, "Should be gated out — only bookkeeping changed"

    def test_cognitive_change_passes_gate(self):
        """A fold that changes cognitive dims should pass the gate."""
        cog_size, book_size = 8, 8
        dim = cog_size + book_size
        transition, cog_dims = _make_transition_with_bookkeeping(cog_size, book_size)

        gate = FoldGate(transition_fn=transition, threshold=0.01, cog_dims=cog_dims)
        store = MemoryStore(dim=dim)

        vec = np.zeros(dim)
        vec[:cog_size] = np.ones(cog_size) * 0.5
        rec = store.add("cognitive content", record_id="r1", vector=vec)
        hit = RetrievalHit(rec, score=0.9, rank=1, margin=0.5)

        state = np.zeros(dim)
        input_vec = np.zeros(dim)
        result = gate.fold("q", input_vec, [hit], state)

        assert result.fold_force > 0.01, "h_cog fold-force should be nonzero"
        assert result.gated is True, "Should pass gate — cognitive dims changed"

    def test_default_cog_dims_matches_full(self):
        """With cog_dims=None, fold_force should equal full_divergence."""
        gate = FoldGate(threshold=0.001)
        store = MemoryStore(dim=8)
        rec = store.add("test", record_id="r1")
        hit = RetrievalHit(rec, score=0.9, rank=1, margin=0.5)

        result = gate.fold("q", rec.vector, [hit], np.zeros(8))
        assert abs(result.fold_force - result.full_divergence) < 1e-12

    def test_empty_cog_dims_always_gates_out(self):
        """With cog_dims=[] (no cognitive dims), fold-force is always zero."""
        gate = FoldGate(threshold=0.001, cog_dims=[])
        store = MemoryStore(dim=8)
        rec = store.add("content", record_id="r1")
        hit = RetrievalHit(rec, score=0.9, rank=1, margin=0.5)

        result = gate.fold("q", rec.vector, [hit], np.zeros(8))
        assert result.fold_force == 0.0
        assert result.gated is False
        assert result.full_divergence > 0


class TestHCogAgent:
    def test_agent_with_cog_dims(self):
        """FGMAgent should pass cog_dims through to FoldGate."""
        cog_dims = list(range(24))
        agent = FGMAgent(dim=32, cog_dims=cog_dims, fold_threshold=0.001)
        assert agent.fold_gate.cog_dims is not None
        assert len(agent.fold_gate.cog_dims) == 24

    def test_agent_cog_dims_affects_gating(self):
        """An agent with restricted cog_dims should gate differently than full."""
        cog_size, book_size = 8, 8
        dim = cog_size + book_size
        transition, cog_dims = _make_transition_with_bookkeeping(cog_size, book_size)

        agent_cog = FGMAgent(
            dim=dim, transition_fn=transition, cog_dims=cog_dims, fold_threshold=0.01,
        )
        agent_full = FGMAgent(
            dim=dim, transition_fn=transition, fold_threshold=0.01,
        )

        vec = np.zeros(dim)
        vec[cog_size:] = np.ones(book_size) * 0.5
        agent_cog.store.add("bookkeeping record", record_id="r1", vector=vec)
        agent_full.store.add("bookkeeping record", record_id="r1", vector=vec)

        result_cog = agent_cog.query("bookkeeping")
        result_full = agent_full.query("bookkeeping")

        assert result_full.full_divergence > 0.01
        assert result_cog.gated is False or result_cog.fold_force < result_full.fold_force

    def test_agent_metrics_with_cog_dims(self):
        """Agent metrics should work with cog_dims set."""
        agent = FGMAgent(dim=16, cog_dims=list(range(12)), fold_threshold=0.001)
        agent.add("some content here")
        agent.query("content")
        m = agent.metrics()
        assert "fold_gate_rate" in m
        assert m["queries_processed"] == 1

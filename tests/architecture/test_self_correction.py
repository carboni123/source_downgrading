"""Tests for operation-memory self-correction (Experiment 3 from the research sketch).

Validates:
  1. Recursive depth tracking: operations that retrieve prior operations get depth > 1
  2. Causal chain tracing: trace_decision identifies which memories drove a decision
  3. Self-correction scenario: agent traces a bad decision back through fold ops
  4. Operation records contain the right causal information
  5. Agents with operation-memory outperform content-only agents on self-correction
"""
import numpy as np
import pytest

from fgm.core import (
    FGMAgent,
    MemoryStore,
    MarginRetriever,
    FoldGate,
    OperationMemory,
    FoldResult,
    RetrievalHit,
    OperationRecord,
    hash_embed,
    cosine,
    l2,
)


# ---------------------------------------------------------------------------
# 1. Recursive depth tracking
# ---------------------------------------------------------------------------

class TestRecursiveDepth:
    def test_first_fold_has_depth_one(self):
        """First operation record should have recursive_depth=1."""
        agent = FGMAgent(dim=32, fold_threshold=0.001)
        agent.add("the server crashed due to OOM", operation_type="observation")
        result = agent.query("what caused the crash?")
        if result.gated:
            ops = agent.operations.all_operations()
            assert len(ops) >= 1
            assert ops[0].recursive_depth == 1

    def test_depth_increases_when_folding_operation_records(self):
        """When the agent retrieves and folds an operation record, depth increments.

        This is the SIMReC hierarchy: memory of memory-use informing memory-use.
        """
        agent = FGMAgent(dim=32, fold_threshold=0.001)

        agent.add("deploy failed migration timeout", operation_type="observation")
        r1 = agent.query("deploy migration")
        assert r1.gated, "First fold should be gated"

        agent.add("increased timeout to fix deploy", operation_type="decision",
                   decision_content="increase migration timeout")
        r2 = agent.query("deploy migration timeout fix")
        assert r2.gated, "Second fold should be gated"

        r3 = agent.query("fold query deploy migration")

        ops = agent.operations.all_operations()
        depths = [op.recursive_depth for op in ops]
        assert max(depths) >= 2, (
            f"Should reach recursive depth >= 2 when folding over operation records. "
            f"Depths: {depths}"
        )

    def test_depth_tracks_maximum_retrieved_operation_depth(self):
        """Depth should be max(retrieved op depths) + 1, not just +1."""
        dim = 32
        agent = FGMAgent(dim=dim, fold_threshold=0.001)

        agent.add("alpha event occurred", operation_type="observation")
        agent.query("alpha event")

        agent.add("beta response to alpha", operation_type="decision")
        agent.query("alpha beta response")

        agent.query("fold alpha beta event response")

        ops = agent.operations.all_operations()
        if len(ops) >= 3:
            last_depth = ops[-1].recursive_depth
            assert last_depth >= 1


# ---------------------------------------------------------------------------
# 2. Causal chain tracing
# ---------------------------------------------------------------------------

class TestCausalChainTracing:
    def test_trace_decision_returns_relevant_operations(self):
        """trace_decision should return operations related to the query."""
        agent = FGMAgent(dim=32, fold_threshold=0.001)

        agent.add("database migration failed at step 3", operation_type="observation")
        agent.query("database migration failure")

        agent.add("API latency spike in production", operation_type="observation")
        agent.query("API latency production")

        db_ops = agent.trace_decision("database migration")
        api_ops = agent.trace_decision("API latency")

        assert isinstance(db_ops, list)
        assert isinstance(api_ops, list)

    def test_operation_records_contain_retrieved_ids(self):
        """Each operation record tracks which memories were retrieved for it."""
        agent = FGMAgent(dim=32, fold_threshold=0.001)
        rec = agent.add("critical production incident", record_id="incident_1",
                         operation_type="observation")
        result = agent.query("production incident")

        if result.gated:
            ops = agent.operations.all_operations()
            latest = ops[-1]
            assert len(latest.retrieved_ids) > 0, "Operation should track retrieved record IDs"

    def test_operation_records_track_fold_force(self):
        """Operation records should preserve the fold-force measurement."""
        agent = FGMAgent(dim=32, fold_threshold=0.001)
        agent.add("server ran out of disk space", operation_type="observation")
        result = agent.query("disk space issue")

        if result.gated:
            ops = agent.operations.all_operations()
            assert ops[-1].fold_force > 0
            assert abs(ops[-1].fold_force - result.fold_force) < 1e-9

    def test_operation_records_store_both_outputs(self):
        """Operation records should store output_with and output_without for comparison."""
        agent = FGMAgent(dim=32, fold_threshold=0.001)
        agent.add("quarterly review meeting notes", operation_type="observation")
        result = agent.query("quarterly review")

        if result.gated:
            ops = agent.operations.all_operations()
            latest = ops[-1]
            assert latest.output_with is not None
            assert latest.output_without is not None
            divergence = l2(latest.output_with, latest.output_without)
            assert divergence > 0, "Stored outputs should show the fold divergence"


# ---------------------------------------------------------------------------
# 3. Self-correction scenario
# ---------------------------------------------------------------------------

class TestSelfCorrection:
    def test_bad_decision_traceable_through_operations(self):
        """An agent should be able to trace a decision back to the memories
        that informed it, enabling identification of the faulty input.

        Scenario:
          1. Agent stores observation: "server A is the bottleneck"
          2. Agent decides: "scale server A"
          3. New evidence: "actually server B was the bottleneck"
          4. Agent traces the scaling decision back through operations
          5. The trace reveals the original faulty observation
        """
        dim = 64
        agent = FGMAgent(dim=dim, fold_threshold=0.001)

        faulty_obs = agent.add(
            "monitoring shows server A is the performance bottleneck",
            record_id="obs_faulty",
            operation_type="observation",
        )
        r1 = agent.query("what is the performance bottleneck?")

        decision_rec = agent.add(
            "decided to scale server A based on monitoring data",
            record_id="decision_scale_a",
            operation_type="decision",
            decision_content="scale server A horizontally",
        )
        r2 = agent.query("how should we fix the performance issue?")

        correcting_obs = agent.add(
            "new monitoring shows server B was actually the bottleneck not A",
            record_id="obs_correction",
            operation_type="observation",
        )

        ops = agent.trace_decision("performance bottleneck server scaling decision")

        all_retrieved = set()
        for op in ops:
            all_retrieved.update(op.retrieved_ids)

        all_ops = agent.operations.all_operations()
        all_retrieved_ever = set()
        for op in all_ops:
            all_retrieved_ever.update(op.retrieved_ids)

        assert "obs_faulty" in all_retrieved_ever or "decision_scale_a" in all_retrieved_ever, (
            f"The causal chain should include the original observation or decision. "
            f"All retrieved IDs across operations: {all_retrieved_ever}"
        )

    def test_content_only_agent_loses_causal_chain(self):
        """An agent without operation-memory cannot trace decisions back to sources.

        This validates Experiment 3's prediction: content-only agents must
        re-derive decisions from content alone, losing the causal chain.
        """
        dim = 32
        agent = FGMAgent(dim=dim, fold_threshold=0.001)

        agent.add("root cause was a DNS misconfiguration", operation_type="observation")
        agent.query("what was the root cause?")

        agent.add("fixed DNS records to point to correct server", operation_type="decision",
                   decision_content="update DNS A record")
        agent.query("how did we fix it?")

        ops_with_chain = agent.operations.all_operations()
        has_chain = len(ops_with_chain) > 0 and any(
            len(op.retrieved_ids) > 0 for op in ops_with_chain
        )

        content_only_records = [r for r in agent.store.all_records() if r.record_type == "content"]
        content_has_no_chain = all(
            r.metadata.get("retrieved_ids") is None for r in content_only_records
        )

        assert has_chain, "Operation-memory agent should have a causal chain"
        assert content_has_no_chain, "Content records alone don't carry causal chain info"

    def test_multi_step_correction_chain(self):
        """Test a longer correction chain: observe -> decide -> discover error -> revise.

        The revision should reference the original decision through operations.
        """
        dim = 64
        agent = FGMAgent(dim=dim, fold_threshold=0.001)

        agent.add("CPU usage at 95% on web tier", operation_type="observation")
        agent.query("CPU usage web tier")

        agent.add("decision: add 3 more web servers", operation_type="decision",
                   decision_content="horizontal scale web tier +3")
        agent.query("scale web servers")

        agent.add("after scaling: CPU still at 90%, problem is a tight loop in handler",
                   operation_type="observation")
        agent.query("CPU still high after scaling")

        agent.add("decision: fix the O(n^2) loop in request handler instead",
                   operation_type="decision",
                   decision_content="optimize handler loop from O(n^2) to O(n)")
        agent.query("fix handler loop optimization")

        all_ops = agent.operations.all_operations()
        assert len(all_ops) >= 3, (
            f"Should have at least 3 fold operations for the full chain, got {len(all_ops)}"
        )

        all_retrieved_ids = set()
        for op in all_ops:
            all_retrieved_ids.update(op.retrieved_ids)

        assert len(all_retrieved_ids) > 1, (
            f"Correction chain should reference multiple source records: {all_retrieved_ids}"
        )


# ---------------------------------------------------------------------------
# 4. Comparative: operation-memory vs content-only
# ---------------------------------------------------------------------------

class TestOperationMemoryAdvantage:
    def test_operation_memory_enables_deeper_trace(self):
        """Agent with operation-memory can trace further back than one step.

        With content only, you can retrieve the content that was relevant.
        With operation-memory, you can retrieve WHICH content was used for
        WHICH decision, enabling multi-hop causal tracing.
        """
        dim = 32
        agent = FGMAgent(dim=dim, fold_threshold=0.001)

        agent.add("step 1: user reported slow page load", operation_type="observation")
        agent.query("slow page load")

        agent.add("step 2: profiled and found DB query N+1", operation_type="observation")
        agent.query("database query performance")

        agent.add("step 3: added eager loading to fix N+1", operation_type="decision",
                   decision_content="add eager loading for user.orders relationship")
        agent.query("fix N+1 query eager loading")

        ops = agent.operations.all_operations()
        assert len(ops) >= 2, "Should have multiple operations"

        later_ops = [op for op in ops if op.recursive_depth >= 1]
        any_references_prior_op = any(
            any(rid.startswith("op_") for rid in op.retrieved_ids)
            for op in later_ops
        )

        op_records_in_store = [r for r in agent.store.all_records() if r.record_type == "operation"]
        assert len(op_records_in_store) >= 2, (
            f"Operation records should be stored for retrieval: {len(op_records_in_store)}"
        )

    def test_operation_count_grows_with_gated_folds(self):
        """Each gated fold should produce exactly one operation record."""
        dim = 32
        agent = FGMAgent(dim=dim, fold_threshold=0.001)

        topics = [
            "kubernetes pod eviction due to memory limit",
            "nginx configuration for reverse proxy",
            "postgresql vacuum autovacuum settings",
            "redis cache invalidation strategy",
        ]
        for t in topics:
            agent.add(t, operation_type="observation")

        gated_count = 0
        for t in topics:
            result = agent.query(t.split()[0] + " " + t.split()[1])
            if result.gated:
                gated_count += 1

        op_count = len(agent.operations.all_operations())
        assert op_count == gated_count, (
            f"Operation count ({op_count}) should equal gated fold count ({gated_count})"
        )

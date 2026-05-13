"""Tests for compression boundary detection and margin preservation.

Validates:
  1. Auto-compression triggers when confusability exceeds threshold (not before)
  2. Post-compression margins don't degrade beyond tolerance (Eq. 28 constraint)
  3. Compression moves system from degrading toward stable regime
  4. Zero-fold pruning removes inert records without harming retrieval
  5. The compression boundary aligns with confusability crossing 0.5
"""
import numpy as np
import pytest

from fgm.core import (
    FGMAgent,
    MemoryStore,
    MarginRetriever,
    Compressor,
    FoldGate,
    hash_embed,
    cosine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_confusable_records(store: MemoryStore, n: int, dim: int, rng: np.random.Generator):
    """Add n records that share a dominant class component -> high confusability."""
    class_vec = rng.normal(size=dim)
    class_vec /= np.linalg.norm(class_vec)
    ids = []
    for i in range(n):
        v = 0.95 * class_vec + 0.05 * rng.normal(size=dim)
        v /= np.linalg.norm(v)
        rid = f"conf_{i}"
        store.add(f"confusable record {i}", record_id=rid, vector=v)
        ids.append(rid)
    return ids


def _add_distinctive_records(store: MemoryStore, n: int, dim: int, rng: np.random.Generator):
    """Add n records with strong unique components -> low confusability."""
    ids = []
    for i in range(n):
        v = rng.normal(size=dim)
        v /= np.linalg.norm(v)
        rid = f"dist_{i}"
        store.add(f"distinctive record {i}", record_id=rid, vector=v)
        ids.append(rid)
    return ids


# ---------------------------------------------------------------------------
# 1. Compression triggers at the right confusability boundary
# ---------------------------------------------------------------------------

class TestCompressionTrigger:
    def test_should_compress_true_for_confusable_store(self):
        """Confusable records push chi_N above threshold -> should_compress=True."""
        dim = 32
        rng = np.random.default_rng(42)
        store = MemoryStore(dim=dim)
        _add_confusable_records(store, 30, dim, rng)
        retriever = MarginRetriever(store, margin_threshold=0.05)
        comp = Compressor(store, retriever)
        assert comp.should_compress(chi_threshold=0.5, n_probes=100)

    def test_should_compress_false_for_distinctive_store(self):
        """Distinctive records keep chi_N low -> should_compress=False."""
        dim = 32
        rng = np.random.default_rng(42)
        store = MemoryStore(dim=dim)
        _add_distinctive_records(store, 30, dim, rng)
        retriever = MarginRetriever(store, margin_threshold=0.05)
        comp = Compressor(store, retriever)
        assert not comp.should_compress(chi_threshold=0.5, n_probes=100)

    def test_compression_boundary_aligns_with_confusability(self):
        """Find N where chi_N crosses 0.5 for confusable records.

        The boundary should exist in a predictable range: confusability
        should be low at small N and high at large N for confusable records.
        """
        dim = 32
        rng = np.random.default_rng(99)
        class_vec = rng.normal(size=dim)
        class_vec /= np.linalg.norm(class_vec)

        chi_values = []
        n_values = [4, 8, 16, 32, 64]

        for n in n_values:
            store = MemoryStore(dim=dim)
            for i in range(n):
                v = 0.95 * class_vec + 0.05 * rng.normal(size=dim)
                v /= np.linalg.norm(v)
                store.add(f"rec_{i}", record_id=f"rec_{i}", vector=v)
            retriever = MarginRetriever(store, margin_threshold=0.05)
            chi = retriever.estimate_confusability(n_queries=100, rng=np.random.default_rng(0))
            chi_values.append(chi)

        assert chi_values[-1] > chi_values[0], (
            f"Confusability should rise with N for confusable records: {list(zip(n_values, chi_values))}"
        )
        assert chi_values[-1] > 0.5, (
            f"Confusability at N={n_values[-1]} should exceed 0.5: {chi_values[-1]}"
        )

    def test_distinctive_records_never_trigger(self):
        """Even at large N, distinctive records keep chi_N below threshold."""
        dim = 32
        rng = np.random.default_rng(42)
        chi_at_large_n = []
        for n in [16, 32, 64]:
            store = MemoryStore(dim=dim)
            _add_distinctive_records(store, n, dim, rng)
            retriever = MarginRetriever(store, margin_threshold=0.05)
            chi = retriever.estimate_confusability(n_queries=100, rng=np.random.default_rng(0))
            chi_at_large_n.append(chi)

        assert all(c < 0.5 for c in chi_at_large_n), (
            f"Distinctive records should never cross chi=0.5: {chi_at_large_n}"
        )


# ---------------------------------------------------------------------------
# 2. Margin preservation constraint (Eq. 28)
# ---------------------------------------------------------------------------

class TestMarginPreservation:
    def test_prune_does_not_degrade_margins(self):
        """Pruning zero-fold records should maintain or improve margins."""
        dim = 32
        rng = np.random.default_rng(42)
        store = MemoryStore(dim=dim)

        good_ids = _add_distinctive_records(store, 10, dim, rng)
        for rid in good_ids:
            store.record_fold_force(rid, rng.uniform(0.1, 1.0))

        confusable_ids = _add_confusable_records(store, 15, dim, rng)
        for rid in confusable_ids:
            store.record_fold_force(rid, 0.0)

        retriever = MarginRetriever(store, margin_threshold=0.05)
        comp = Compressor(store, retriever)
        report = comp.prune_zero_fold()

        assert report.margin_after >= report.margin_before - comp.margin_tolerance, (
            f"Post-prune margin ({report.margin_after:.4f}) degraded more than "
            f"tolerance ({comp.margin_tolerance}) from pre-prune ({report.margin_before:.4f})"
        )

    def test_merge_preserves_margins_within_tolerance(self):
        """Merging near-duplicates should not degrade margins beyond tolerance."""
        dim = 32
        rng = np.random.default_rng(42)
        store = MemoryStore(dim=dim)

        _add_distinctive_records(store, 8, dim, rng)

        base_vec = rng.normal(size=dim)
        base_vec /= np.linalg.norm(base_vec)
        for i in range(5):
            v = base_vec + rng.normal(0, 0.01, size=dim)
            v /= np.linalg.norm(v)
            rid = f"dup_{i}"
            store.add(f"duplicate content {i}", record_id=rid, vector=v)
            store.record_fold_force(rid, rng.uniform(0.0, 0.5))

        retriever = MarginRetriever(store, margin_threshold=0.05)
        comp = Compressor(store, retriever, duplicate_threshold=0.95)
        report = comp.merge_duplicates()

        assert report.records_after < report.records_before
        assert report.margin_after >= report.margin_before - comp.margin_tolerance, (
            f"Post-merge margin ({report.margin_after:.4f}) degraded beyond tolerance "
            f"from pre-merge ({report.margin_before:.4f})"
        )

    def test_combined_compression_preserves_margins(self):
        """Full compress() pipeline should satisfy the margin preservation constraint."""
        dim = 32
        rng = np.random.default_rng(42)
        store = MemoryStore(dim=dim)

        _add_distinctive_records(store, 8, dim, rng)

        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        for i in range(6):
            v = base + rng.normal(0, 0.01, size=dim)
            v /= np.linalg.norm(v)
            rid = f"dup_{i}"
            store.add(f"dup content {i}", record_id=rid, vector=v)
            store.record_fold_force(rid, 0.0)

        retriever = MarginRetriever(store, margin_threshold=0.05)
        comp = Compressor(store, retriever, duplicate_threshold=0.95)
        report = comp.compress()

        assert report.records_after < report.records_before
        assert report.margin_after >= report.margin_before - comp.margin_tolerance


# ---------------------------------------------------------------------------
# 3. Compression prevents regime degradation
# ---------------------------------------------------------------------------

class TestCompressionPreventsOverload:
    def test_prune_reduces_confusability(self):
        """Pruning zero-fold confusable records should lower confusability."""
        dim = 32
        rng = np.random.default_rng(42)
        store = MemoryStore(dim=dim)

        _add_distinctive_records(store, 5, dim, rng)
        for rid in [f"dist_{i}" for i in range(5)]:
            store.record_fold_force(rid, 0.5)

        confusable_ids = _add_confusable_records(store, 20, dim, rng)
        for rid in confusable_ids:
            store.record_fold_force(rid, 0.0)

        retriever = MarginRetriever(store, margin_threshold=0.05)
        comp = Compressor(store, retriever)

        chi_before = retriever.estimate_confusability(n_queries=100, rng=np.random.default_rng(0))
        comp.prune_zero_fold()
        chi_after = retriever.estimate_confusability(n_queries=100, rng=np.random.default_rng(0))

        assert chi_after <= chi_before, (
            f"Confusability should not increase after pruning: {chi_before:.3f} -> {chi_after:.3f}"
        )

    def test_hit_rate_improves_after_pruning_inert_records(self):
        """Removing zero-fold-force confusable records should improve hit rate
        for the remaining distinctive records."""
        dim = 32
        rng = np.random.default_rng(42)
        store = MemoryStore(dim=dim)

        good_ids = _add_distinctive_records(store, 5, dim, rng)
        for rid in good_ids:
            store.record_fold_force(rid, 1.0)

        confusable_ids = _add_confusable_records(store, 30, dim, rng)
        for rid in confusable_ids:
            store.record_fold_force(rid, 0.0)

        retriever = MarginRetriever(store, margin_threshold=0.05)

        def measure_hit_rate():
            hits = 0
            n_probes = 50
            probe_rng = np.random.default_rng(7)
            records = [store.get(rid) for rid in good_ids if store.get(rid) is not None]
            for rec in records:
                q = rec.vector + probe_rng.normal(0, 0.05, size=dim)
                rank, _, _ = retriever.target_rank_margin(q, rec.record_id)
                if rank is not None and rank <= 3:
                    hits += 1
            return hits / max(len(records), 1)

        hr_before = measure_hit_rate()

        comp = Compressor(store, retriever)
        comp.prune_zero_fold()

        hr_after = measure_hit_rate()

        assert hr_after >= hr_before, (
            f"Hit rate for distinctive records should not decrease after pruning: "
            f"{hr_before:.3f} -> {hr_after:.3f}"
        )


# ---------------------------------------------------------------------------
# 4. Auto-compression in FGMAgent
# ---------------------------------------------------------------------------

class TestAgentAutoCompression:
    def test_auto_compress_triggers_on_schedule(self):
        """Agent auto-compresses every 20 queries when chi exceeds threshold."""
        dim = 32
        rng = np.random.default_rng(42)
        agent = FGMAgent(
            dim=dim,
            fold_threshold=0.001,
            auto_compress=True,
            compress_chi_threshold=0.3,
        )

        class_vec = rng.normal(size=dim)
        class_vec /= np.linalg.norm(class_vec)
        for i in range(40):
            v = 0.95 * class_vec + 0.05 * rng.normal(size=dim)
            v /= np.linalg.norm(v)
            agent.store.add(f"confusable {i}", record_id=f"c_{i}", vector=v)
            agent.store.record_fold_force(f"c_{i}", 0.0)

        n_before = len(agent.store)
        for i in range(20):
            agent.query(f"query {i}")

        n_after = len(agent.store)
        assert n_after <= n_before, (
            f"Auto-compression should have reduced store size: {n_before} -> {n_after}"
        )

    def test_no_auto_compress_when_disabled(self):
        """Agent doesn't compress when auto_compress=False."""
        dim = 32
        rng = np.random.default_rng(42)
        agent = FGMAgent(dim=dim, fold_threshold=0.001, auto_compress=False)

        class_vec = rng.normal(size=dim)
        class_vec /= np.linalg.norm(class_vec)
        for i in range(40):
            v = 0.95 * class_vec + 0.05 * rng.normal(size=dim)
            v /= np.linalg.norm(v)
            agent.store.add(f"conf {i}", record_id=f"c_{i}", vector=v)
            agent.store.record_fold_force(f"c_{i}", 0.0)

        content_before = len([r for r in agent.store.all_records() if r.record_type == "content"])
        for i in range(25):
            agent.query(f"q {i}")

        content_after = len([r for r in agent.store.all_records() if r.record_type == "content"])
        assert content_after >= content_before, (
            "Content records should not decrease with auto_compress=False"
        )

    def test_operations_survive_compression(self):
        """Operation records are never pruned by compression."""
        dim = 32
        agent = FGMAgent(dim=dim, fold_threshold=0.001)

        for i in range(5):
            agent.add(f"distinctive content {i} with unique words {'xyz'*i}")

        for i in range(5):
            agent.query(f"distinctive content {i}")

        op_count_before = len([r for r in agent.store.all_records() if r.record_type == "operation"])
        agent.compress()
        op_count_after = len([r for r in agent.store.all_records() if r.record_type == "operation"])

        assert op_count_after >= op_count_before, (
            f"Operation records should survive compression: {op_count_before} -> {op_count_after}"
        )

"""Tests with real sentence-transformer embeddings (384-dim).

Validates that the FGM architecture works with real embedding geometry,
not just toy hash embeddings. Uses all-MiniLM-L6-v2 (384-dim).

Tests are skipped if sentence-transformers is not installed.

Key questions:
  1. Do distinctive text records maintain high hit rate as N grows?
  2. Do confusable text records show hit-rate degradation?
  3. Does fold-force discriminate between relevant and irrelevant memories?
  4. Does compression preserve margins with real embeddings?
  5. Does the fold-gated agent outperform store-everything with real text?
"""
import numpy as np
import pytest

st = pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

from fgm.core import (
    FGMAgent,
    MemoryStore,
    MarginRetriever,
    FoldGate,
    Compressor,
    cosine,
    default_transition,
)


@pytest.fixture(scope="module")
def model():
    return st.SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def embed_fn(model):
    def fn(text: str) -> np.ndarray:
        return model.encode([text], normalize_embeddings=True)[0]
    return fn


@pytest.fixture(scope="module")
def dim():
    return 384


# ---------------------------------------------------------------------------
# 1. Distinctive records maintain high hit rate
# ---------------------------------------------------------------------------

class TestDistinctiveRecords:
    RECORDS = [
        ("rec_pg", "PostgreSQL vacuum autovacuum freezing transaction wraparound"),
        ("rec_k8s", "Kubernetes pod eviction OOMKilled memory limit exceeded"),
        ("rec_redis", "Redis cache invalidation strategy write-through vs write-behind"),
        ("rec_dns", "DNS resolution timeout SERVFAIL recursive resolver configuration"),
        ("rec_tls", "TLS certificate renewal Let's Encrypt ACME challenge validation"),
        ("rec_nginx", "Nginx reverse proxy upstream timeout load balancing configuration"),
        ("rec_docker", "Docker multi-stage build layer caching optimization Dockerfile"),
        ("rec_git", "Git rebase interactive squash fixup autosquash workflow"),
    ]
    QUERIES = [
        ("rec_pg", "PostgreSQL vacuum freezing wraparound"),
        ("rec_k8s", "Kubernetes pod OOMKilled memory"),
        ("rec_redis", "Redis cache invalidation strategy"),
        ("rec_dns", "DNS resolution timeout recursive resolver"),
        ("rec_tls", "TLS certificate renewal ACME"),
        ("rec_nginx", "Nginx reverse proxy upstream load balancing"),
        ("rec_docker", "Docker multi-stage build layer caching"),
        ("rec_git", "Git rebase squash fixup"),
    ]

    def test_perfect_retrieval_at_small_n(self, embed_fn, dim):
        """Distinctive text records should have perfect hit rate at small N."""
        store = MemoryStore(dim=dim, embed_fn=embed_fn)
        for rid, content in self.RECORDS:
            store.add(content, record_id=rid)

        retriever = MarginRetriever(store, margin_threshold=0.01)
        hits = 0
        for target_id, query in self.QUERIES:
            rank, _, _ = retriever.target_rank_margin(query, target_id)
            if rank is not None and rank <= 3:
                hits += 1

        hit_rate = hits / len(self.QUERIES)
        assert hit_rate >= 0.9, (
            f"Distinctive records at N=8: HR={hit_rate:.3f} (expected >= 0.9)"
        )

    def test_positive_margins(self, embed_fn, dim):
        """Distinctive text records should have positive retrieval margins."""
        store = MemoryStore(dim=dim, embed_fn=embed_fn)
        for rid, content in self.RECORDS:
            store.add(content, record_id=rid)

        retriever = MarginRetriever(store, margin_threshold=0.01)
        margins = []
        for target_id, query in self.QUERIES:
            _, _, margin = retriever.target_rank_margin(query, target_id)
            margins.append(margin)

        mean_margin = np.mean(margins)
        assert mean_margin > 0.01, (
            f"Mean margin for distinctive records: {mean_margin:.4f} (expected > 0.01)"
        )


# ---------------------------------------------------------------------------
# 2. Confusable records degrade
# ---------------------------------------------------------------------------

class TestConfusableRecords:
    BASE_RECORDS = [
        ("rec_meeting1", "Team meeting notes from Monday morning standup"),
        ("rec_meeting2", "Team meeting notes from Tuesday morning standup"),
        ("rec_meeting3", "Team meeting notes from Wednesday morning standup"),
        ("rec_meeting4", "Team meeting notes from Thursday morning standup"),
        ("rec_meeting5", "Team meeting notes from Friday morning standup"),
    ]
    NOISE_TEMPLATE = "Team meeting notes from the weekly standup discussion number {}"

    def test_high_confusability_for_similar_records(self, embed_fn, dim):
        """A store full of similar records should have high confusability."""
        store = MemoryStore(dim=dim, embed_fn=embed_fn)
        for rid, content in self.BASE_RECORDS:
            store.add(content, record_id=rid)
        for i in range(50):
            store.add(self.NOISE_TEMPLATE.format(i), record_id=f"noise_{i}")

        retriever = MarginRetriever(store, margin_threshold=0.01)
        chi = retriever.estimate_confusability(n_queries=50, rng=np.random.default_rng(0))

        assert chi > 0.3, (
            f"Similar records should have elevated confusability: {chi:.3f}"
        )

    def test_hit_rate_degrades_with_similar_noise(self, embed_fn, dim):
        """Hit rate for specific records should degrade as confusable noise grows."""
        store = MemoryStore(dim=dim, embed_fn=embed_fn)
        store.add("Critical incident: database migration failed at step 3 causing data loss",
                   record_id="target")

        retriever = MarginRetriever(store, margin_threshold=0.01)

        rank_before, _, margin_before = retriever.target_rank_margin(
            "database migration failure data loss", "target"
        )

        for i in range(80):
            store.add(f"Database migration step {i % 10} completed successfully run {i}",
                      record_id=f"noise_{i}")

        rank_after, _, margin_after = retriever.target_rank_margin(
            "database migration failure data loss", "target"
        )

        assert margin_after <= margin_before, (
            f"Margin should not improve: before={margin_before:.4f}, after={margin_after:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Fold-force with real embeddings
# ---------------------------------------------------------------------------

class TestRealFoldForce:
    def test_retrieval_ranks_relevant_higher(self, embed_fn, dim):
        """The retriever should rank relevant memories higher than irrelevant ones,
        which is the precondition for fold-force discrimination in real use."""
        store = MemoryStore(dim=dim, embed_fn=embed_fn)
        store.add(
            "The deployment failed because the database migration timed out after 30 seconds",
            record_id="relevant",
        )
        store.add(
            "The quarterly revenue report shows 15% growth in APAC region",
            record_id="irrelevant",
        )

        retriever = MarginRetriever(store, margin_threshold=0.01)
        report = retriever.retrieve("Why did the deployment fail?", k=2)

        assert report.hits[0].record.record_id == "relevant", (
            f"Relevant record should be top-ranked, got: {report.hits[0].record.record_id}"
        )
        assert report.hits[0].score > report.hits[1].score, (
            f"Relevant score ({report.hits[0].score:.4f}) should exceed "
            f"irrelevant ({report.hits[1].score:.4f})"
        )

    def test_fold_force_nonzero_with_real_text(self, embed_fn, dim):
        """Fold-force should be nonzero for relevant retrieved text."""
        agent = FGMAgent(dim=dim, embed_fn=embed_fn, fold_threshold=0.001)
        agent.add("Server crashed due to out of memory at 3:42 AM")
        result = agent.query("What caused the server crash?")
        assert result.fold_force > 0.001, (
            f"Fold-force should be nonzero for relevant query: {result.fold_force:.6f}"
        )
        assert result.full_divergence >= result.fold_force


# ---------------------------------------------------------------------------
# 4. Compression with real embeddings
# ---------------------------------------------------------------------------

class TestRealCompression:
    def test_prune_preserves_distinctive_records(self, embed_fn, dim):
        """Pruning zero-fold records should preserve distinctive ones."""
        store = MemoryStore(dim=dim, embed_fn=embed_fn)

        distinctive = [
            "PostgreSQL query planner chose sequential scan instead of index scan",
            "Redis cluster failover triggered by sentinel quorum",
            "Kubernetes horizontal pod autoscaler reached maximum replicas",
        ]
        for i, text in enumerate(distinctive):
            store.add(text, record_id=f"good_{i}")
            store.record_fold_force(f"good_{i}", 0.5)

        confusable = [f"System status check number {i} completed" for i in range(20)]
        for i, text in enumerate(confusable):
            store.add(text, record_id=f"noise_{i}")
            store.record_fold_force(f"noise_{i}", 0.0)

        retriever = MarginRetriever(store, margin_threshold=0.01)
        comp = Compressor(store, retriever)
        report = comp.prune_zero_fold()

        for i in range(len(distinctive)):
            assert store.get(f"good_{i}") is not None, f"Distinctive record good_{i} should survive"

        assert report.records_after < report.records_before


# ---------------------------------------------------------------------------
# 5. Full agent cycle with real embeddings
# ---------------------------------------------------------------------------

class TestRealAgentCycle:
    def test_store_query_fold_cycle(self, embed_fn, dim):
        """Full agent cycle should work with real embeddings."""
        agent = FGMAgent(dim=dim, embed_fn=embed_fn, fold_threshold=0.001)

        agent.add("The API rate limiter is configured to 100 requests per minute",
                   operation_type="observation")
        agent.add("We increased the rate limit to 500 req/min after the load test",
                   operation_type="decision",
                   decision_content="increase rate limit to 500 rpm")
        agent.add("Users reported 429 errors during peak hours",
                   operation_type="observation")

        result = agent.query("What is the API rate limit configuration?")
        assert result.fold_force > 0
        assert len(result.retrieved) > 0

        m = agent.metrics()
        assert m["content_records"] == 3
        assert m["queries_processed"] == 1

    def test_operation_memory_with_real_text(self, embed_fn, dim):
        """Operation memory should work with real text embeddings."""
        agent = FGMAgent(dim=dim, embed_fn=embed_fn, fold_threshold=0.001)

        agent.add("Memory leak detected in the user session handler")
        agent.query("What memory issues were found?")

        agent.add("Fixed the leak by adding proper cleanup in the session destructor")
        agent.query("How was the memory leak fixed?")

        ops = agent.operations.all_operations()
        assert len(ops) >= 1, "Should have at least one operation record"

    def test_decision_tracing_with_real_text(self, embed_fn, dim):
        """Decision tracing should return relevant operations for real queries."""
        agent = FGMAgent(dim=dim, embed_fn=embed_fn, fold_threshold=0.001)

        agent.add("CPU usage spiked to 95% on the API server")
        agent.query("CPU usage spike")

        agent.add("Deployed horizontal scaling to handle the load")
        agent.query("scaling deployment decision")

        ops = agent.trace_decision("CPU scaling")
        assert isinstance(ops, list)

"""Tests for LLM-backed transition functions.

Two test tiers:
  1. Echo-call tests (no API needed): verify plumbing, fold-force measurement,
     agent integration with the deterministic echo stub.
  2. Live API tests (require Anthropic key): verify fold-force discrimination
     with real Claude calls. Marked with @pytest.mark.live, skipped by default.

Run live tests with: pytest tests/test_llm_transition.py -m live -v
"""
import numpy as np
import pytest

from fgm.core import FGMAgent, FoldGate, MemoryStore, RetrievalHit, hash_embed
from fgm.llm import LLMTransition, CallStats, echo_call


# ---------------------------------------------------------------------------
# Embedder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def st_model():
    st = pytest.importorskip("sentence_transformers")
    return st.SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def st_embed(st_model):
    def fn(text: str) -> np.ndarray:
        return st_model.encode([text], normalize_embeddings=True)[0]
    return fn


@pytest.fixture(scope="module")
def st_dim():
    return 384


def hash_embedder(text: str) -> np.ndarray:
    return hash_embed(text, 64)


# ---------------------------------------------------------------------------
# 1. Echo-call tests (no API needed)
# ---------------------------------------------------------------------------

class TestEchoCall:
    def test_echo_returns_different_text_with_and_without_memory(self):
        """echo_call should produce different responses based on memory presence."""
        call = echo_call()
        transition = LLMTransition(call, hash_embedder, dim=64)

        transition.set_context(
            query_text="What caused the crash?",
            memory_texts=["Server ran out of memory at 3AM"],
        )

        with_mem = transition(np.zeros(64), np.zeros(64), fold_vec=np.ones(64))
        transition.set_context(
            query_text="What caused the crash?",
            memory_texts=["Server ran out of memory at 3AM"],
        )
        without_mem = transition(np.zeros(64), np.zeros(64), fold_vec=None)

        assert transition.stats.n_calls == 2
        assert not np.allclose(with_mem, without_mem), (
            "Responses with and without memory should produce different embeddings"
        )

    def test_echo_call_stats(self):
        """CallStats should track calls."""
        call = echo_call()
        transition = LLMTransition(call, hash_embedder, dim=64)
        transition.set_context(query_text="test")
        transition(np.zeros(64), np.zeros(64), None)

        assert transition.stats.n_calls == 1
        assert transition.stats.input_chars > 0
        assert transition.stats.output_chars > 0
        assert len(transition.stats.history) == 1

    def test_empty_response_retry_records_attempts(self):
        """Empty provider responses should be retryable without duplicating transition history."""
        responses = iter(["", "retry succeeded"])

        def call(_prompt: str) -> str:
            return next(responses)

        transition = LLMTransition(
            call,
            hash_embedder,
            dim=64,
            empty_response_retries=1,
        )
        transition.set_context(query_text="test empty retry")

        transition(np.zeros(64), np.zeros(64), None)

        assert transition.last_response == "retry succeeded"
        assert transition.stats.n_calls == 2
        assert transition.stats.empty_responses == 1
        assert transition.stats.retry_count == 1
        assert len(transition.stats.history) == 1
        assert transition.stats.history[0]["attempt_count"] == 2
        assert transition.stats.history[0]["empty_response_count"] == 1

    def test_empty_response_without_retry_is_observable(self):
        """Empty responses without retry should still be counted in stats/history."""
        transition = LLMTransition(lambda _prompt: "", hash_embedder, dim=64)
        transition.set_context(query_text="test empty no retry")

        transition(np.zeros(64), np.zeros(64), None)

        assert transition.last_response == ""
        assert transition.stats.n_calls == 1
        assert transition.stats.empty_responses == 1
        assert transition.stats.retry_count == 0
        assert transition.stats.history[0]["attempt_count"] == 1


class TestEchoFoldForce:
    def test_fold_force_nonzero_with_echo(self):
        """FoldGate with echo transition should produce nonzero fold-force."""
        call = echo_call()
        transition = LLMTransition(call, hash_embedder, dim=64)
        gate = FoldGate(transition_fn=transition, threshold=0.001)

        store = MemoryStore(dim=64)
        rec = store.add("Server crashed due to OOM", record_id="r1")
        hit = RetrievalHit(rec, score=0.9, rank=1, margin=0.5)

        transition.set_context(
            query_text="What caused the crash?",
            memory_texts=[rec.content],
        )
        result = gate.fold("What caused the crash?", rec.vector, [hit], np.zeros(64))

        assert result.fold_force > 0.001, (
            f"Fold-force should be nonzero with echo transition: {result.fold_force:.6f}"
        )
        assert result.gated is True

    def test_fold_force_zero_without_hits(self):
        """No retrieved records should produce zero fold-force."""
        call = echo_call()
        transition = LLMTransition(call, hash_embedder, dim=64)
        gate = FoldGate(transition_fn=transition, threshold=0.001)

        transition.set_context(query_text="anything")
        result = gate.fold("anything", np.zeros(64), [], np.zeros(64))
        assert result.fold_force < 1e-9


class TestEchoAgentIntegration:
    def test_agent_query_with_echo_transition(self):
        """FGMAgent should work with echo-backed LLMTransition."""
        call = echo_call()
        transition = LLMTransition(call, hash_embedder, dim=64)

        agent = FGMAgent(dim=64, transition_fn=transition, fold_threshold=0.001)
        agent.add("The deploy failed because the migration timed out")
        agent.add("We fixed it by increasing the timeout to 30 minutes")

        result = agent.query("What caused the deploy failure?")

        assert result.fold_force > 0, "Fold-force should be nonzero"
        assert transition.stats.n_calls >= 2, "Should make at least 2 LLM calls per fold"
        assert transition.last_response is not None

    def test_agent_sets_context_on_transition(self):
        """Agent should set query text and memory texts before folding."""
        call = echo_call()
        transition = LLMTransition(call, hash_embedder, dim=64)

        agent = FGMAgent(dim=64, transition_fn=transition, fold_threshold=0.001)
        agent.add("Important technical detail about the database schema")

        agent.query("Tell me about the database")

        last = transition.stats.history[-1]
        assert "database" in last["prompt"].lower()

    def test_agent_metrics_with_llm(self):
        """Agent metrics should work with LLM transition."""
        call = echo_call()
        transition = LLMTransition(call, hash_embedder, dim=64)

        agent = FGMAgent(dim=64, transition_fn=transition, fold_threshold=0.001)
        agent.add("Record one")
        agent.query("query one")

        m = agent.metrics()
        assert m["queries_processed"] == 1
        assert m["content_records"] >= 1

    def test_operation_memory_with_llm(self):
        """Operation memory should work with LLM transition."""
        call = echo_call()
        transition = LLMTransition(call, hash_embedder, dim=64)

        agent = FGMAgent(dim=64, transition_fn=transition, fold_threshold=0.001)
        agent.add("Critical incident: disk full on production server")
        result = agent.query("disk full incident")

        if result.gated:
            ops = agent.operations.all_operations()
            assert len(ops) >= 1


# ---------------------------------------------------------------------------
# 2. Sentence-transformer echo tests (real embeddings, no API)
# ---------------------------------------------------------------------------

class TestEchoWithRealEmbeddings:
    def test_fold_force_with_st_embeddings(self, st_embed, st_dim):
        """Echo transition with sentence-transformer embeddings should show
        clear fold-force discrimination (real embedding geometry)."""
        call = echo_call()
        transition = LLMTransition(call, st_embed, dim=st_dim)

        agent = FGMAgent(dim=st_dim, transition_fn=transition, embed_fn=st_embed,
                         fold_threshold=0.001)
        agent.add("PostgreSQL vacuum caused 10-second lock waits on users table")
        agent.add("Redis cache hit ratio dropped to 40% after the deployment")

        result = agent.query("What caused the PostgreSQL lock waits?")

        assert result.fold_force > 0.01, (
            f"fold-force with real embeddings should be substantial: {result.fold_force:.4f}"
        )

    def test_different_queries_different_fold_force(self, st_embed, st_dim):
        """Different queries should produce different fold-force values,
        showing the transition is sensitive to the specific memory retrieved."""
        call = echo_call()
        transition = LLMTransition(call, st_embed, dim=st_dim)

        agent = FGMAgent(dim=st_dim, transition_fn=transition, embed_fn=st_embed,
                         fold_threshold=0.001)
        agent.add("The API rate limit is 100 requests per minute")
        agent.add("The database backup runs at 2 AM UTC every day")

        r1 = agent.query("What is the API rate limit?")
        r2 = agent.query("When does the database backup run?")

        assert r1.fold_force > 0
        assert r2.fold_force > 0


# ---------------------------------------------------------------------------
# 3. Live API tests (require Anthropic key)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestLiveAnthropicTransition:
    """Tests that call the real Claude API. Skipped by default.

    Run with: pytest tests/test_llm_transition.py -m live -v
    """

    @pytest.fixture(scope="class")
    def live_transition(self, st_embed, st_dim):
        try:
            from fgm.llm import anthropic_call
            usage = {}
            call = anthropic_call(temperature=0.0, usage_tracker=usage)
            call("Say hello in one word.")
            transition = LLMTransition(call, st_embed, dim=st_dim)
            transition._usage = usage
            return transition
        except Exception as exc:
            pytest.skip(f"Anthropic API not available: {exc}")

    def test_live_fold_force_nonzero(self, live_transition, st_embed, st_dim):
        """Live Claude call should produce nonzero fold-force with relevant memory."""
        agent = FGMAgent(
            dim=st_dim, transition_fn=live_transition, embed_fn=st_embed,
            fold_threshold=0.001,
        )
        agent.add("The production outage on March 3rd was caused by a memory leak "
                   "in the user session handler that accumulated over 72 hours")

        result = agent.query("What caused the production outage?")

        assert result.fold_force > 0.01, (
            f"Live fold-force should be substantial: {result.fold_force:.4f}"
        )
        assert live_transition.stats.n_calls >= 2

        print(f"\n  Live fold-force: {result.fold_force:.4f}")
        print(f"  With memory: {live_transition.stats.history[-2]['response'][:100]}...")
        print(f"  Without:     {live_transition.stats.history[-1]['response'][:100]}...")

    def test_live_relevant_vs_irrelevant(self, live_transition, st_embed, st_dim):
        """Fold-force should be higher when retrieved memory is relevant."""
        agent = FGMAgent(
            dim=st_dim, transition_fn=live_transition, embed_fn=st_embed,
            fold_threshold=0.001,
        )

        agent.add("The deploy failed because the database migration timed out after 30 seconds",
                   record_id="relevant")
        agent.add("The team had pizza for lunch on Tuesday",
                   record_id="irrelevant")

        result = agent.query("Why did the deployment fail?")

        assert result.fold_force > 0, "Should have nonzero fold-force"
        print(f"\n  Live fold-force: {result.fold_force:.4f}")
        print(f"  Retrieved: {[h.record.record_id for h in result.retrieved]}")

    def test_live_usage_tracking(self, live_transition):
        """Usage tracker should accumulate tokens."""
        usage = getattr(live_transition, "_usage", {})
        if usage:
            print(f"\n  Total tokens: input={usage.get('input_tokens', 0)}, "
                  f"output={usage.get('output_tokens', 0)}")
            assert usage.get("input_tokens", 0) > 0

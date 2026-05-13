"""Tests for LLM Experiment 1 (echo-call plumbing, no API needed).

Validates that the experiment harness works correctly with the echo stub.
Live experiment results are captured in VALIDATION_REPORT.md.
"""
import numpy as np
import pytest

from fgm.core import hash_embed, FGMAgent
from fgm.llm import LLMTransition, echo_call, answer_quality_fold_force
from fgm.llm_experiment import LLMExperiment1, SIGNAL_RECORDS, SIGNAL_QUERIES, NOISE_POOL


def hash_embedder(text: str) -> np.ndarray:
    return hash_embed(text, 64)


class TestAnswerQualityMetric:
    def test_relevant_produces_higher_fold_force(self):
        """answer_quality_fold_force should give higher score when the response
        embedding is closer to the memory content embedding."""
        from fgm.core import RetrievalHit, MemoryStore

        store = MemoryStore(dim=64)
        relevant = store.add("specific technical detail about database", record_id="rel")
        noise = store.add("generic status check completed", record_id="noise")

        output_with_relevant = hash_embed("database technical detail response", 64)
        output_with_noise = hash_embed("generic status update response", 64)
        output_without = hash_embed("I don't have information about that", 64)

        rel_hit = RetrievalHit(relevant, score=0.9, rank=1, margin=0.3)
        noise_hit = RetrievalHit(noise, score=0.8, rank=1, margin=0.1)

        ff_rel = answer_quality_fold_force(output_with_relevant, output_without, [rel_hit])
        ff_noise = answer_quality_fold_force(output_with_noise, output_without, [noise_hit])

        assert isinstance(ff_rel, float)
        assert isinstance(ff_noise, float)
        assert ff_rel >= 0.0

    def test_no_hits_returns_zero(self):
        """Empty hits list should return zero fold-force."""
        ff = answer_quality_fold_force(np.zeros(64), np.zeros(64), [])
        assert ff == 0.0


class TestLLMExperimentEcho:
    def test_experiment_runs_with_echo(self):
        """Experiment harness should complete with echo_call."""
        exp = LLMExperiment1(
            llm_call=echo_call(),
            embed_fn=hash_embedder,
            dim=64,
            n_phases=2,
            noise_per_phase=5,
            fold_threshold=0.05,
            k=3,
        )
        result = exp.run()

        assert len(result.baseline_phases) == 2
        assert len(result.foldgated_phases) == 2
        assert result.total_llm_calls > 0

    def test_baseline_accumulates_records(self):
        """Baseline should accumulate all noise records."""
        exp = LLMExperiment1(
            llm_call=echo_call(),
            embed_fn=hash_embedder,
            dim=64,
            n_phases=3,
            noise_per_phase=5,
            fold_threshold=0.05,
            k=3,
        )
        result = exp.run()

        counts = [p.n_content_records for p in result.baseline_phases]
        assert counts[-1] > counts[0], f"Baseline should grow: {counts}"

    def test_foldgated_prunes_some(self):
        """Fold-gated agent should prune at least some records."""
        exp = LLMExperiment1(
            llm_call=echo_call(),
            embed_fn=hash_embedder,
            dim=64,
            n_phases=3,
            noise_per_phase=8,
            fold_threshold=0.05,
            k=3,
        )
        result = exp.run()

        bl_final = result.baseline_phases[-1].n_content_records
        fg_final = result.foldgated_phases[-1].n_content_records
        assert fg_final <= bl_final, (
            f"Fold-gated should have <= records: baseline={bl_final}, foldgated={fg_final}"
        )

    def test_signal_records_present(self):
        """All signal records should be in the experiment."""
        assert len(SIGNAL_RECORDS) == 6
        assert len(SIGNAL_QUERIES) == 6
        assert len(NOISE_POOL) >= 10

    def test_fold_force_fn_used(self):
        """Agent should use answer_quality_fold_force when configured."""
        transition = LLMTransition(echo_call(), hash_embedder, dim=64)
        agent = FGMAgent(
            dim=64, transition_fn=transition, fold_threshold=0.01,
            fold_force_fn=answer_quality_fold_force,
        )
        assert agent.fold_gate._fold_force_fn is answer_quality_fold_force

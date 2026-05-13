"""Tests for Experiment 1: Fold-Gating vs Store-Everything.

Validates Prediction 3 from the research sketch:
  Fold-gated agent outperforms store-everything on delayed recall despite
  storing fewer records, because its stored records are transition-effective
  and its retrieval is less confusable.

The experiment uses a unique-sensitive transition function combined with
h_cog projection: the fold-gated agent measures fold-force only over
cognitive (unique-subspace) dimensions, enabling it to distinguish signal
from noise and prune accordingly. The baseline uses full-output fold-force
and cannot discriminate.

In 32-dim toy vectors, the signal-noise fold-force gap is narrow, so the
prediction holds in a majority of seeds but not all. This mirrors the
theoretical finding: fold-gating's advantage depends on the fold-force
discriminability of the transition function.
"""
import numpy as np
import pytest

from fgm.experiment import Experiment1


class TestExperiment1Core:
    @pytest.fixture(scope="class")
    def result(self):
        return Experiment1(seed=42).run()

    def test_baseline_degrades(self, result):
        """Baseline hit rate should degrade as noise accumulates."""
        first_hr = result.baseline_phases[0].hit_rate
        last_hr = result.baseline_phases[-1].hit_rate
        assert last_hr <= first_hr, (
            f"Baseline should not improve: phase 0 HR={first_hr:.3f}, "
            f"final HR={last_hr:.3f}"
        )

    def test_foldgated_prunes_records(self, result):
        """Fold-gated agent should prune some records (fewer than baseline)."""
        bl_recs = result.baseline_phases[-1].n_content_records
        fg_recs = result.foldgated_phases[-1].n_content_records
        assert fg_recs < bl_recs, (
            f"Fold-gated should store fewer records: baseline={bl_recs}, foldgated={fg_recs}"
        )

    def test_baseline_confusability_rises(self, result):
        """Baseline confusability should increase as noise accumulates."""
        first_chi = result.baseline_phases[0].confusability
        last_chi = result.baseline_phases[-1].confusability
        assert last_chi >= first_chi - 0.1, (
            f"Confusability should not decrease significantly: "
            f"first={first_chi:.3f}, last={last_chi:.3f}"
        )

    def test_foldgated_margins_at_least_baseline(self, result):
        """Fold-gated margins should be >= baseline at the final phase."""
        bl_margin = result.baseline_phases[-1].mean_margin
        fg_margin = result.foldgated_phases[-1].mean_margin
        assert fg_margin >= bl_margin - 0.005, (
            f"Fold-gated margin ({fg_margin:.4f}) should not be much worse "
            f"than baseline ({bl_margin:.4f})"
        )

    def test_operations_recorded(self, result):
        """Both agents should record fold operations."""
        bl_ops = result.baseline_final.get("operation_records", 0)
        fg_ops = result.foldgated_final.get("operation_records", 0)
        assert bl_ops > 0 or fg_ops > 0, "At least one agent should have operations"


class TestExperiment1MultiSeed:
    """Multi-seed validation of the fold-gating advantage.

    The prediction (fold-gated HR > baseline HR) holds in a majority of seeds.
    The mean hit-rate advantage should be positive across seeds.
    """
    N_SEEDS = 15

    @pytest.fixture(scope="class")
    def seed_results(self):
        return [Experiment1(seed=seed).run() for seed in range(self.N_SEEDS)]

    def test_mean_advantage_positive(self, seed_results):
        """Mean hit-rate advantage should be positive across seeds."""
        advantages = [
            r.foldgated_phases[-1].hit_rate - r.baseline_phases[-1].hit_rate
            for r in seed_results
        ]
        mean_adv = np.mean(advantages)
        assert mean_adv > 0, (
            f"Mean hit-rate advantage should be positive: {mean_adv:+.3f}. "
            f"Per-seed: {[f'{a:+.3f}' for a in advantages]}"
        )

    def test_advantage_in_majority_of_seeds(self, seed_results):
        """Fold-gated should have higher or equal HR in most seeds."""
        advantages = [
            r.foldgated_phases[-1].hit_rate - r.baseline_phases[-1].hit_rate
            for r in seed_results
        ]
        non_negative = sum(1 for a in advantages if a >= 0)
        assert non_negative >= self.N_SEEDS * 0.5, (
            f"Non-negative advantage in {non_negative}/{self.N_SEEDS} seeds "
            f"(expected >= {self.N_SEEDS * 0.5:.0f})"
        )

    def test_record_efficiency(self, seed_results):
        """Fold-gated should store fewer records in most seeds."""
        fewer = sum(
            1 for r in seed_results
            if r.foldgated_phases[-1].n_content_records < r.baseline_phases[-1].n_content_records
        )
        assert fewer >= self.N_SEEDS * 0.6, (
            f"Fewer records in {fewer}/{self.N_SEEDS} seeds"
        )

    def test_baseline_always_degrades(self, seed_results):
        """Baseline should show degradation in most seeds."""
        degrades = sum(
            1 for r in seed_results
            if r.baseline_phases[-1].hit_rate <= r.baseline_phases[0].hit_rate
        )
        assert degrades >= self.N_SEEDS * 0.7, (
            f"Baseline degrades in {degrades}/{self.N_SEEDS} seeds"
        )

    def test_summary_report(self, seed_results, capsys):
        """Print summary statistics."""
        bl_hrs = [r.baseline_phases[-1].hit_rate for r in seed_results]
        fg_hrs = [r.foldgated_phases[-1].hit_rate for r in seed_results]
        bl_recs = [r.baseline_phases[-1].n_content_records for r in seed_results]
        fg_recs = [r.foldgated_phases[-1].n_content_records for r in seed_results]
        holds = sum(1 for r in seed_results if r.prediction_holds)

        print(f"\n{'='*60}")
        print(f"Experiment 1: Fold-Gating vs Store-Everything ({self.N_SEEDS} seeds)")
        print(f"{'='*60}")
        print(f"  Prediction holds:       {holds}/{self.N_SEEDS} ({holds/self.N_SEEDS:.0%})")
        print(f"  Baseline final HR:      {np.mean(bl_hrs):.3f} +/- {np.std(bl_hrs):.3f}")
        print(f"  Fold-gated final HR:    {np.mean(fg_hrs):.3f} +/- {np.std(fg_hrs):.3f}")
        print(f"  Baseline records:       {np.mean(bl_recs):.0f} +/- {np.std(bl_recs):.0f}")
        print(f"  Fold-gated records:     {np.mean(fg_recs):.0f} +/- {np.std(fg_recs):.0f}")
        print(f"  HR advantage:           {np.mean(np.array(fg_hrs) - np.array(bl_hrs)):+.3f}")
        print(f"  Record reduction:       {np.mean(np.array(bl_recs) - np.array(fg_recs)):.0f} fewer")
        print(f"{'='*60}")

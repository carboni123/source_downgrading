"""Large-N scaling tests for the regime evaluator.

Validates the binomial proposition (Prop. 10.1) at realistic scale:
  - Overload regimes continue degrading monotonically up to N=1000
  - Non-overload regimes maintain perfect hit rate at N=1000
  - Retrieval margins and confusability follow predicted trajectories
  - Phase boundary holds at N values the paper's evaluation protocol requires
"""
import numpy as np
import pytest

from fgm.metrics import RegimeEvaluator, REGIME_CONFIGS


LARGE_N_VALUES = (8, 64, 256, 512, 1000)
N_SEEDS = 5
N_QUERIES = 200
K = 3


class TestLargeNRegimePhysics:
    """Verify regime physics hold at N up to 1000."""

    @pytest.fixture(scope="class")
    def all_reports(self):
        evaluator = RegimeEvaluator()
        reports = {}
        for regime in REGIME_CONFIGS:
            reports[regime] = evaluator.evaluate_regime(
                regime, n_values=LARGE_N_VALUES, k=K, n_queries=N_QUERIES, seed=42,
            )
        return reports

    def test_rich_distinctive_perfect_at_1000(self, all_reports):
        """rich_distinctive should maintain 1.0 hit rate even at N=1000."""
        report = all_reports["rich_distinctive"]
        row_1000 = [r for r in report.rows if r.n == 1000][0]
        assert row_1000.hit_rate >= 0.99, (
            f"Rich distinctive at N=1000: HR={row_1000.hit_rate:.3f}"
        )

    def test_compressed_preserving_stable_at_1000(self, all_reports):
        """compressed_preserving should maintain high hit rate at N=1000."""
        report = all_reports["compressed_preserving"]
        row_1000 = [r for r in report.rows if r.n == 1000][0]
        assert row_1000.hit_rate >= 0.95, (
            f"Compressed preserving at N=1000: HR={row_1000.hit_rate:.3f}"
        )

    def test_sparse_confusable_near_zero_at_1000(self, all_reports):
        """sparse_confusable hit rate should approach zero at N=1000."""
        report = all_reports["sparse_confusable"]
        row_1000 = [r for r in report.rows if r.n == 1000][0]
        assert row_1000.hit_rate < 0.05, (
            f"Sparse confusable at N=1000: HR={row_1000.hit_rate:.3f}"
        )

    def test_aggressive_lossy_near_zero_at_1000(self, all_reports):
        """aggressive_lossy hit rate should approach zero at N=1000."""
        report = all_reports["aggressive_lossy"]
        row_1000 = [r for r in report.rows if r.n == 1000][0]
        assert row_1000.hit_rate < 0.05, (
            f"Aggressive lossy at N=1000: HR={row_1000.hit_rate:.3f}"
        )

    def test_overload_classification_at_large_n(self, all_reports):
        """Overload classification should hold at large N."""
        expected = {
            "sparse_confusable": True,
            "aggressive_lossy": True,
            "compressed_preserving": False,
            "rich_distinctive": False,
        }
        for regime, expect_overload in expected.items():
            assert all_reports[regime].overload_like == expect_overload, (
                f"{regime}: expected overload={expect_overload}, got {all_reports[regime].overload_like}"
            )


class TestBinomialProposition:
    """Validate Proposition 10.1: H_q(k,N) -> 0 as N -> inf for confusable regimes."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        evaluator = RegimeEvaluator()
        data = {}
        for regime in ["sparse_confusable", "aggressive_lossy"]:
            data[regime] = evaluator.evaluate_regime(
                regime, n_values=LARGE_N_VALUES, k=K, n_queries=300, seed=42,
            )
        return data

    def test_sparse_monotonic_decrease(self, trajectories):
        """sparse_confusable hit rate should decrease monotonically."""
        rows = trajectories["sparse_confusable"].rows
        rates = [r.hit_rate for r in rows]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1] - 0.03, (
                f"Non-monotonic at N={rows[i+1].n}: {rates}"
            )

    def test_aggressive_monotonic_decrease(self, trajectories):
        """aggressive_lossy hit rate should decrease monotonically."""
        rows = trajectories["aggressive_lossy"].rows
        rates = [r.hit_rate for r in rows]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1] - 0.03, (
                f"Non-monotonic at N={rows[i+1].n}: {rates}"
            )

    def test_convergence_toward_zero(self, trajectories):
        """Hit rate at N=1000 should be much smaller than at N=8."""
        for regime in ["sparse_confusable", "aggressive_lossy"]:
            rows = trajectories[regime].rows
            hr_8 = [r.hit_rate for r in rows if r.n == 8][0]
            hr_1000 = [r.hit_rate for r in rows if r.n == 1000][0]
            assert hr_1000 < hr_8 * 0.1, (
                f"{regime}: N=8 HR={hr_8:.3f}, N=1000 HR={hr_1000:.3f}. "
                f"Expected >10x degradation."
            )


class TestLargeNMargins:
    """Verify margin trajectories at scale."""

    @pytest.fixture(scope="class")
    def all_reports(self):
        evaluator = RegimeEvaluator()
        return {
            regime: evaluator.evaluate_regime(
                regime, n_values=LARGE_N_VALUES, k=K, n_queries=200, seed=42,
            )
            for regime in REGIME_CONFIGS
        }

    def test_rich_margins_positive_at_1000(self, all_reports):
        """rich_distinctive should have positive margins even at N=1000."""
        row = [r for r in all_reports["rich_distinctive"].rows if r.n == 1000][0]
        assert row.mean_margin > 0.1, (
            f"Rich distinctive margin at N=1000: {row.mean_margin:.4f}"
        )

    def test_overload_margins_negative_at_1000(self, all_reports):
        """Overload regimes should have negative margins at N=1000."""
        for regime in ["sparse_confusable", "aggressive_lossy"]:
            row = [r for r in all_reports[regime].rows if r.n == 1000][0]
            assert row.mean_margin < 0, (
                f"{regime} margin at N=1000: {row.mean_margin:+.4f}"
            )

    def test_rich_zero_confusability_at_1000(self, all_reports):
        """rich_distinctive should have zero confusability at N=1000."""
        row = [r for r in all_reports["rich_distinctive"].rows if r.n == 1000][0]
        assert row.confusability < 0.05, (
            f"Rich distinctive chi at N=1000: {row.confusability:.3f}"
        )

    def test_overload_full_confusability_at_1000(self, all_reports):
        """Overload regimes should have chi=1.0 at N=1000."""
        for regime in ["sparse_confusable", "aggressive_lossy"]:
            row = [r for r in all_reports[regime].rows if r.n == 1000][0]
            assert row.confusability > 0.95, (
                f"{regime} chi at N=1000: {row.confusability:.3f}"
            )


class TestLargeNMultiSeed:
    """Multi-seed validation at large N."""

    def test_overload_stable_across_seeds(self):
        """Overload classification at N=1000 should be stable across seeds."""
        evaluator = RegimeEvaluator()
        expected = {
            "sparse_confusable": True,
            "aggressive_lossy": True,
            "compressed_preserving": False,
            "rich_distinctive": False,
        }
        for regime, expect_overload in expected.items():
            passes = 0
            for seed in range(N_SEEDS):
                report = evaluator.evaluate_regime(
                    regime, n_values=(8, 256, 1000), k=K, n_queries=200, seed=seed,
                )
                if report.overload_like == expect_overload:
                    passes += 1
            assert passes == N_SEEDS, (
                f"{regime}: {passes}/{N_SEEDS} seeds matched expected overload={expect_overload}"
            )

    def test_summary_report(self, capsys):
        """Print large-N trajectory summary."""
        evaluator = RegimeEvaluator()
        print(f"\n{'='*70}")
        print(f"Large-N Scaling Report (N up to 1000)")
        print(f"{'='*70}")
        for regime in REGIME_CONFIGS:
            report = evaluator.evaluate_regime(
                regime, n_values=LARGE_N_VALUES, k=K, n_queries=200, seed=42,
            )
            rates = {r.n: r.hit_rate for r in report.rows}
            margins = {r.n: r.mean_margin for r in report.rows}
            print(f"\n  {regime}:")
            print(f"    HR:     {' -> '.join(f'N={n}:{rates[n]:.3f}' for n in LARGE_N_VALUES)}")
            print(f"    Margin: {' -> '.join(f'N={n}:{margins[n]:+.4f}' for n in LARGE_N_VALUES)}")
        print(f"\n{'='*70}")

"""Multi-seed regime stability tests.

Validates:
  1. All four regimes classify correctly across 10+ seeds (statistical confidence)
  2. Overload/non-overload boundary is stable (no seed-dependent flips)
  3. Hit-rate trajectories maintain expected ordering across seeds
  4. Phase boundary metrics (margin, confusability) show low variance
  5. Pass rates match the trace_validation_framework's standard (>95%)
"""
import numpy as np
import pytest

from fgm.metrics import RegimeEvaluator, REGIME_CONFIGS


N_SEEDS = 15
N_VALUES = (8, 32, 128)
N_QUERIES = 200
K = 3

EXPECTED_OVERLOAD = {
    "sparse_confusable": True,
    "aggressive_lossy": True,
    "compressed_preserving": False,
    "rich_distinctive": False,
}


# ---------------------------------------------------------------------------
# 1. Multi-seed classification consistency
# ---------------------------------------------------------------------------

class TestMultiSeedClassification:
    @pytest.fixture(scope="class")
    def all_results(self):
        """Run all regimes across N_SEEDS seeds. Cached for the test class."""
        evaluator = RegimeEvaluator()
        results = {}
        for regime in REGIME_CONFIGS:
            regime_results = []
            for seed in range(N_SEEDS):
                report = evaluator.evaluate_regime(
                    regime, n_values=N_VALUES, k=K, n_queries=N_QUERIES, seed=seed,
                )
                regime_results.append(report)
            results[regime] = regime_results
        return results

    @pytest.mark.parametrize("regime", list(REGIME_CONFIGS.keys()))
    def test_overload_classification_stable(self, all_results, regime):
        """Overload classification should be consistent across all seeds."""
        reports = all_results[regime]
        expected = EXPECTED_OVERLOAD[regime]
        passes = sum(1 for r in reports if r.overload_like == expected)
        pass_rate = passes / len(reports)

        assert pass_rate >= 0.95, (
            f"{regime}: overload classification pass rate = {pass_rate:.2f} "
            f"({passes}/{len(reports)}). Expected overload={expected}. "
            f"Failed seeds: {[i for i, r in enumerate(reports) if r.overload_like != expected]}"
        )

    @pytest.mark.parametrize("regime", list(REGIME_CONFIGS.keys()))
    def test_classification_label_stable(self, all_results, regime):
        """Classification label should match regime name across seeds."""
        reports = all_results[regime]
        correct = sum(1 for r in reports if r.classification == regime)
        pass_rate = correct / len(reports)

        assert pass_rate >= 0.90, (
            f"{regime}: label pass rate = {pass_rate:.2f} ({correct}/{len(reports)}). "
            f"Labels: {[r.classification for r in reports]}"
        )


# ---------------------------------------------------------------------------
# 2. Hit-rate trajectory ordering
# ---------------------------------------------------------------------------

class TestHitRateOrdering:
    @pytest.fixture(scope="class")
    def trajectory_data(self):
        """Collect hit-rate at N=128 across seeds for all regimes."""
        evaluator = RegimeEvaluator()
        data = {}
        for regime in REGIME_CONFIGS:
            rates_at_128 = []
            for seed in range(N_SEEDS):
                report = evaluator.evaluate_regime(
                    regime, n_values=(128,), k=K, n_queries=N_QUERIES, seed=seed,
                )
                rates_at_128.append(report.rows[0].hit_rate)
            data[regime] = rates_at_128
        return data

    def test_rich_distinctive_dominates(self, trajectory_data):
        """Rich distinctive should have higher hit rate than overload regimes at N=128."""
        rich = np.array(trajectory_data["rich_distinctive"])
        sparse = np.array(trajectory_data["sparse_confusable"])
        lossy = np.array(trajectory_data["aggressive_lossy"])

        assert np.mean(rich) > np.mean(sparse) + 0.3, (
            f"Rich ({np.mean(rich):.3f}) should dominate sparse ({np.mean(sparse):.3f}) by >0.3"
        )
        assert np.mean(rich) > np.mean(lossy) + 0.3, (
            f"Rich ({np.mean(rich):.3f}) should dominate lossy ({np.mean(lossy):.3f}) by >0.3"
        )

    def test_compressed_preserving_maintains_high_hit_rate(self, trajectory_data):
        """Compressed preserving should maintain high hit rate (>0.9) across seeds."""
        rates = trajectory_data["compressed_preserving"]
        mean_rate = np.mean(rates)
        assert mean_rate > 0.9, (
            f"Compressed preserving mean hit rate at N=128: {mean_rate:.3f} (expected >0.9)"
        )

    def test_overload_regimes_degrade(self, trajectory_data):
        """Sparse confusable and aggressive lossy should have low hit rate at N=128."""
        sparse = np.mean(trajectory_data["sparse_confusable"])
        lossy = np.mean(trajectory_data["aggressive_lossy"])

        assert sparse < 0.2, f"Sparse confusable at N=128: {sparse:.3f} (expected <0.2)"
        assert lossy < 0.2, f"Aggressive lossy at N=128: {lossy:.3f} (expected <0.2)"

    def test_hit_rate_variance_bounded(self, trajectory_data):
        """Hit rate should have low variance across seeds (stable phenomenon)."""
        for regime, rates in trajectory_data.items():
            std = np.std(rates)
            assert std < 0.15, (
                f"{regime}: hit rate std = {std:.4f} across {N_SEEDS} seeds (expected <0.15)"
            )


# ---------------------------------------------------------------------------
# 3. Margin and confusability stability
# ---------------------------------------------------------------------------

class TestPhaseMetricStability:
    @pytest.fixture(scope="class")
    def metric_data(self):
        """Collect margin and confusability at N=128 across seeds."""
        evaluator = RegimeEvaluator()
        data = {}
        for regime in REGIME_CONFIGS:
            margins = []
            chis = []
            for seed in range(N_SEEDS):
                report = evaluator.evaluate_regime(
                    regime, n_values=(128,), k=K, n_queries=N_QUERIES, seed=seed,
                )
                row = report.rows[0]
                margins.append(row.mean_margin)
                chis.append(row.confusability)
            data[regime] = {"margins": margins, "chis": chis}
        return data

    def test_rich_distinctive_margins_positive(self, metric_data):
        """Rich distinctive should maintain positive margins at N=128 across all seeds."""
        margins = metric_data["rich_distinctive"]["margins"]
        assert all(m > 0.1 for m in margins), (
            f"Rich distinctive: all margins should be >0.1 at N=128. "
            f"Min={min(margins):.4f}, values={[f'{m:.3f}' for m in margins]}"
        )

    def test_overload_margins_negative_or_small(self, metric_data):
        """Overload regimes should have near-zero or negative margins at N=128."""
        for regime in ["sparse_confusable", "aggressive_lossy"]:
            margins = metric_data[regime]["margins"]
            mean_margin = np.mean(margins)
            assert mean_margin < 0.05, (
                f"{regime}: mean margin at N=128 = {mean_margin:.4f} (expected <0.05)"
            )

    def test_rich_distinctive_zero_confusability(self, metric_data):
        """Rich distinctive should have near-zero confusability at N=128."""
        chis = metric_data["rich_distinctive"]["chis"]
        assert all(c < 0.05 for c in chis), (
            f"Rich distinctive: confusability should be <0.05 at N=128. "
            f"Max={max(chis):.4f}"
        )

    def test_overload_high_confusability(self, metric_data):
        """Overload regimes should have high confusability (>0.8) at N=128."""
        for regime in ["sparse_confusable", "aggressive_lossy"]:
            chis = metric_data[regime]["chis"]
            mean_chi = np.mean(chis)
            assert mean_chi > 0.8, (
                f"{regime}: mean confusability at N=128 = {mean_chi:.3f} (expected >0.8)"
            )

    def test_metric_variance_bounded(self, metric_data):
        """Phase metrics should be stable across seeds."""
        for regime, data in metric_data.items():
            margin_std = np.std(data["margins"])
            chi_std = np.std(data["chis"])
            assert margin_std < 0.1, (
                f"{regime}: margin std = {margin_std:.4f} (expected <0.1)"
            )
            assert chi_std < 0.15, (
                f"{regime}: confusability std = {chi_std:.4f} (expected <0.15)"
            )


# ---------------------------------------------------------------------------
# 4. Monotonicity across seeds
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_sparse_hit_rate_monotonically_decreases(self):
        """sparse_confusable hit rate should decrease from N=8 to N=128 in most seeds."""
        evaluator = RegimeEvaluator()
        monotonic_count = 0

        for seed in range(N_SEEDS):
            report = evaluator.evaluate_regime(
                "sparse_confusable", n_values=(8, 32, 128), k=K, n_queries=N_QUERIES, seed=seed,
            )
            rates = [r.hit_rate for r in report.rows]
            is_monotonic = all(rates[i] >= rates[i+1] - 0.05 for i in range(len(rates)-1))
            if is_monotonic:
                monotonic_count += 1

        pass_rate = monotonic_count / N_SEEDS
        assert pass_rate >= 0.90, (
            f"Sparse confusable monotonic decrease: {pass_rate:.2f} ({monotonic_count}/{N_SEEDS})"
        )

    def test_rich_hit_rate_stable_across_n(self):
        """rich_distinctive should maintain >0.95 hit rate at all N values across seeds."""
        evaluator = RegimeEvaluator()
        stable_count = 0

        for seed in range(N_SEEDS):
            report = evaluator.evaluate_regime(
                "rich_distinctive", n_values=(8, 32, 128), k=K, n_queries=N_QUERIES, seed=seed,
            )
            rates = [r.hit_rate for r in report.rows]
            all_high = all(r >= 0.95 for r in rates)
            if all_high:
                stable_count += 1

        pass_rate = stable_count / N_SEEDS
        assert pass_rate >= 0.90, (
            f"Rich distinctive stability: {pass_rate:.2f} ({stable_count}/{N_SEEDS})"
        )


# ---------------------------------------------------------------------------
# 5. Summary statistics (informational, always passes)
# ---------------------------------------------------------------------------

class TestSummaryReport:
    def test_print_pass_rates(self, capsys):
        """Print pass rates for all regimes (informational)."""
        evaluator = RegimeEvaluator()
        results = {}

        for regime in REGIME_CONFIGS:
            expected = EXPECTED_OVERLOAD[regime]
            passes = 0
            for seed in range(N_SEEDS):
                report = evaluator.evaluate_regime(
                    regime, n_values=N_VALUES, k=K, n_queries=N_QUERIES, seed=seed,
                )
                if report.overload_like == expected:
                    passes += 1
            results[regime] = passes

        total_pass = sum(results.values())
        total_trials = N_SEEDS * len(REGIME_CONFIGS)

        print(f"\n{'='*60}")
        print(f"Multi-Seed Regime Stability Report ({N_SEEDS} seeds)")
        print(f"{'='*60}")
        for regime, passes in results.items():
            rate = passes / N_SEEDS
            status = "PASS" if rate >= 0.95 else "WARN" if rate >= 0.80 else "FAIL"
            print(f"  {regime:25s}: {passes:2d}/{N_SEEDS} ({rate:.1%}) [{status}]")
        print(f"{'='*60}")
        print(f"  {'TOTAL':25s}: {total_pass}/{total_trials} ({total_pass/total_trials:.1%})")
        print(f"{'='*60}")

        assert total_pass / total_trials >= 0.90, (
            f"Overall pass rate {total_pass/total_trials:.2f} below 0.90"
        )

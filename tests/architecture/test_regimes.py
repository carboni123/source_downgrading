"""Tests for the four-regime classification.

Validates the central prediction: sparse_confusable and aggressive_lossy
show overload; rich_distinctive and compressed_preserving do not.
"""
import pytest
from fgm.metrics import RegimeEvaluator


EXPECTED = {
    "sparse_confusable": True,
    "compressed_preserving": False,
    "rich_distinctive": False,
    "aggressive_lossy": True,
}


class TestRegimeClassification:
    @pytest.fixture(scope="class")
    def evaluator(self):
        return RegimeEvaluator()

    @pytest.mark.parametrize("regime,expect_overload", list(EXPECTED.items()))
    def test_regime(self, evaluator, regime, expect_overload):
        report = evaluator.evaluate_regime(
            regime, n_values=(8, 32, 128), k=3, n_queries=200, seed=42,
        )
        assert report.overload_like == expect_overload, (
            f"{regime}: expected overload_like={expect_overload}, "
            f"got {report.overload_like}. "
            f"Hit rates: {[r.hit_rate for r in report.rows]}"
        )

    def test_rich_maintains_perfect_hit_rate(self, evaluator):
        report = evaluator.evaluate_regime(
            "rich_distinctive", n_values=(8, 64, 128), k=3, n_queries=300, seed=7,
        )
        for row in report.rows:
            assert row.hit_rate >= 0.95, (
                f"Rich distinctive at N={row.n}: hit_rate={row.hit_rate}"
            )

    def test_sparse_degrades_monotonically(self, evaluator):
        report = evaluator.evaluate_regime(
            "sparse_confusable", n_values=(8, 16, 32, 64), k=3, n_queries=300, seed=7,
        )
        hit_rates = [r.hit_rate for r in report.rows]
        for i in range(len(hit_rates) - 1):
            assert hit_rates[i] >= hit_rates[i + 1] - 0.05, (
                f"Expected monotonic decrease: {hit_rates}"
            )

    def test_classification_labels(self, evaluator):
        for regime in EXPECTED:
            report = evaluator.evaluate_regime(
                regime, n_values=(8, 32, 128), k=3, n_queries=200, seed=42,
            )
            assert report.classification != "unknown", (
                f"{regime} classified as unknown"
            )

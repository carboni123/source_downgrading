"""Tests for the Source(.) inference validation harness."""
from __future__ import annotations

import math

import pytest

from fgm import (
    SOURCE_CLASSES,
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_INFERENCE,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_SIMULATION,
    SOURCE_TOOL_OUTPUT,
    compare_source_inference_policies,
    evaluate_source_inference,
    make_source_inference_fixture,
)


def test_fixture_covers_all_six_source_classes():
    cases = make_source_inference_fixture()
    classes = {case.true_source for case in cases}
    assert classes == set(SOURCE_CLASSES)


def test_fixture_includes_ambiguous_cases():
    cases = make_source_inference_fixture()
    assert sum(1 for c in cases if c.is_ambiguous) >= 5


def test_uniform_external_is_the_laundering_baseline():
    cases = make_source_inference_fixture()
    report = evaluate_source_inference(cases, policy="uniform_external")
    # Predicting "external" for everything: accuracy equals the share of true
    # external cases, false_externalization_rate is 1.0 for every non-external
    # case, ambiguous accuracy is whatever fraction of ambiguous cases happen
    # to be true-external.
    assert report.false_externalization_rate == 1.0


def test_combined_beats_uniform_external_by_at_least_0_3():
    cases = make_source_inference_fixture()
    reports = compare_source_inference_policies(cases)
    delta = reports["combined"].overall_accuracy - reports["uniform_external"].overall_accuracy
    assert delta >= 0.3, f"combined improvement over uniform_external is {delta:.3f}"


def test_combined_drives_false_externalization_below_threshold():
    cases = make_source_inference_fixture()
    report = evaluate_source_inference(cases, policy="combined")
    assert report.false_externalization_rate < 0.3, (
        f"combined false_externalization_rate {report.false_externalization_rate:.3f} "
        f"should be < 0.3"
    )


def test_combined_recovers_at_least_four_classes_above_half():
    cases = make_source_inference_fixture()
    report = evaluate_source_inference(cases, policy="combined")
    classes_above_half = sum(
        1 for cls, acc in report.per_class_accuracy.items() if acc >= 0.5
    )
    assert classes_above_half >= 4, (
        f"combined recovered only {classes_above_half}/6 classes above 0.5"
    )


def test_lexical_rules_dominates_marker_classes():
    cases = make_source_inference_fixture()
    report = evaluate_source_inference(cases, policy="lexical_rules")
    # The marker classes are designed to have lexically clear cases; lexical
    # rules should hit them strongly. Tolerate one ambiguous miss per class.
    assert report.per_class_accuracy[SOURCE_FABRICATED] >= 0.8
    assert report.per_class_accuracy[SOURCE_SIMULATION] >= 0.8
    assert report.per_class_accuracy[SOURCE_INFERENCE] >= 0.8
    assert report.per_class_accuracy[SOURCE_TOOL_OUTPUT] >= 0.8


def test_confusion_matrix_is_complete():
    cases = make_source_inference_fixture()
    report = evaluate_source_inference(cases, policy="combined")
    for true_cls in SOURCE_CLASSES:
        assert true_cls in report.confusion_matrix
        row_total = sum(report.confusion_matrix[true_cls].values())
        true_count = sum(1 for c in cases if c.true_source == true_cls)
        assert row_total == true_count


@pytest.mark.parametrize("policy", ["uniform_external", "lexical_rules", "feature_threshold", "combined"])
def test_metrics_are_finite_or_nan(policy):
    cases = make_source_inference_fixture()
    report = evaluate_source_inference(cases, policy=policy)
    values = [report.overall_accuracy, report.false_externalization_rate, report.ambiguous_accuracy]
    values.extend(report.per_class_accuracy.values())
    for value in values:
        assert math.isfinite(value) or math.isnan(value)

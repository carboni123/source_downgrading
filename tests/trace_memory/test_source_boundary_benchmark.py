"""Source-boundary benchmark tests.

These tests guard the dataset shape and the high-level empirical signal. The
benchmark is intentionally not perfect: decoy cases should expose current
Source(.) inference limits instead of being tuned away.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.run_source_boundary_benchmark import run_benchmark
from benchmarks.source_boundary_dataset import (
    BOUNDARY_TYPES,
    CONTENT_SOURCE_LABELS,
    DIFFICULTIES,
    DOMAINS,
    dataset_summary,
    make_dataset,
    read_jsonl,
    validate_cases,
    write_jsonl,
)


def test_source_boundary_dataset_shape_and_balance():
    cases = make_dataset()
    validate_cases(cases)
    assert len(cases) == 126

    summary = dataset_summary(cases)
    assert set(summary["by_source"]) == set(CONTENT_SOURCE_LABELS)
    assert set(summary["by_boundary_type"]) == set(BOUNDARY_TYPES)
    assert set(summary["by_difficulty"]) == set(DIFFICULTIES)
    assert set(summary["by_domain"]) == {domain.name for domain in DOMAINS}
    assert set(summary["by_source"].values()) == {21}
    assert set(summary["by_boundary_type"].values()) == {42}
    assert set(summary["by_difficulty"].values()) == {42}
    assert set(summary["by_domain"].values()) == {18}


def test_source_boundary_dataset_jsonl_round_trips():
    cases = make_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "source_boundary_dataset.jsonl"
        write_jsonl(cases, path)
        loaded = read_jsonl(path)
    assert loaded == cases


def test_source_boundary_benchmark_exposes_inference_boundary_limits():
    results = run_benchmark(make_dataset())

    uniform = results["uniform_external"]["aggregate"]
    combined = results["combined"]["aggregate"]

    assert uniform["overall_accuracy"] == 1 / 6
    assert uniform["false_externalization_rate"] == 1.0
    assert combined["overall_accuracy"] > uniform["overall_accuracy"]
    assert combined["false_externalization_rate"] < uniform["false_externalization_rate"]

    assert combined["overall_accuracy"] >= 0.95
    assert combined["false_externalization_rate"] <= 0.05

    # Source(.) is improved but still not solved: some retrieved-memory decoys
    # remain indistinguishable from fresh external claims without app-owned
    # boundary metadata.
    assert combined["per_boundary_accuracy"]["source_decoy"] < 1.0
    assert combined["trust_upgrade_rate"] > 0.0

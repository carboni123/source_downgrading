"""Mnestic-attentional coupling benchmark tests.

These tests guard the fixture truth and the high-level signal: source-blind
memory attention is unsafe, labels help selection, and trace-memory writeback
is needed for zero trust-ceiling violations with full provenance.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.coupling_dataset import (
    COUPLING_MODES,
    DOMAINS,
    dataset_summary,
    make_dataset,
    read_jsonl,
    validate_cases,
    write_jsonl,
)
from benchmarks.run_coupling_benchmark import run_benchmark


def test_coupling_dataset_shape_and_balance():
    cases = make_dataset()
    validate_cases(cases)
    assert len(cases) == 70

    summary = dataset_summary(cases)
    assert set(summary["by_domain"]) == {domain.name for domain in DOMAINS}
    assert set(summary["by_mode"]) == set(COUPLING_MODES)
    assert set(summary["by_domain"].values()) == {10}
    assert set(summary["by_mode"].values()) == {14}
    assert summary["by_expected_answer"] == {"safe": 42, "quarantine": 28}
    assert summary["by_expected_source"] == {
        "fabricated_or_uncertain": 14,
        "inference": 42,
        "simulation": 14,
    }


def test_coupling_dataset_jsonl_round_trips():
    cases = make_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "coupling_dataset.jsonl"
        write_jsonl(cases, path)
        loaded = read_jsonl(path)
    assert loaded == cases


def test_coupling_benchmark_exposes_attention_memory_boundary():
    results = run_benchmark(make_dataset())

    no_memory = results["no_memory"]["aggregate"]
    raw_memory = results["raw_memory"]["aggregate"]
    labels_only = results["labels_only"]["aggregate"]
    trace_memory = results["trace_memory"]["aggregate"]

    assert no_memory["decision_accuracy"] == 0.0

    assert raw_memory["decision_accuracy"] == 0.2
    assert raw_memory["unsafe_contamination_rate"] == 0.8
    assert raw_memory["trust_ceiling_violation_rate"] == 1.0

    assert labels_only["decision_accuracy"] == 1.0
    assert labels_only["unsafe_contamination_rate"] == 0.0
    assert labels_only["trust_ceiling_violation_rate"] == 0.4
    assert labels_only["provenance_recall"] == 0.5

    assert trace_memory["decision_accuracy"] == 1.0
    assert trace_memory["unsafe_contamination_rate"] == 0.0
    assert trace_memory["trust_ceiling_violation_rate"] == 0.0
    assert trace_memory["provenance_recall"] == 1.0
    assert trace_memory["source_match_rate"] == 1.0

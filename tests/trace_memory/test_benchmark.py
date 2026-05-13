"""Benchmark smoke tests.

Confirms that each baseline behaves as documented and the dataset has
the expected shape. The benchmark itself is the source of truth for
performance comparisons; these tests just guard the moving parts that
support the comparison.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.laundering_baselines import (
    BASELINES,
    run_no_source,
    run_provenance_only,
    run_trace_memory,
)
from benchmarks.laundering_dataset import (
    DOMAINS,
    FAILURE_MODES,
    TEMPLATED_VARIANTS,
    expected_chain_sources,
    expected_final_origins,
    expected_final_source,
    make_dataset,
    make_hand_crafted_scenarios,
    make_templated_scenarios,
    read_jsonl,
    validate_scenarios,
    write_jsonl,
)
from trace_memory import SourceLabel


# ---------------------------------------------------------------------------
# Dataset shape
# ---------------------------------------------------------------------------


def test_make_dataset_has_163_scenarios():
    scenarios = make_dataset()
    assert len(scenarios) == 163


def test_templated_dataset_covers_every_domain_failure_mode_pair():
    scenarios = make_templated_scenarios()
    pairs = {(s.domain, s.failure_mode) for s in scenarios}
    expected_pairs = {(d.name, m) for d in DOMAINS for m in FAILURE_MODES}
    assert pairs == expected_pairs
    assert len(scenarios) == len(DOMAINS) * len(FAILURE_MODES) * TEMPLATED_VARIANTS


def test_hand_crafted_scenarios_have_distinct_ids():
    scenarios = make_hand_crafted_scenarios()
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))
    assert len(scenarios) == 16


def test_dataset_truth_fields_are_computed_from_graph():
    scenarios = make_dataset()
    validate_scenarios(scenarios)
    for scenario in scenarios:
        assert scenario.expected_max_trust == expected_final_source(scenario)
        assert scenario.expected_provenance_origins == expected_final_origins(scenario)
        assert scenario.chain[-1].derived_id in expected_chain_sources(scenario)


def test_scenario_seeds_and_chain_consistency():
    """Every chain step's input ids must resolve in either earlier chain steps or seeds."""
    for scenario in make_dataset():
        known_ids = {seed.record_id for seed in scenario.seeds}
        for step in scenario.chain:
            for input_id in step.input_ids:
                assert input_id in known_ids, (
                    f"scenario {scenario.scenario_id} step {step.derived_id} "
                    f"references unknown input {input_id}"
                )
            known_ids.add(step.derived_id)
        # The later_target_id must resolve too.
        assert scenario.later_target_id in known_ids


def test_dataset_jsonl_round_trips():
    scenarios = make_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dataset.jsonl"
        write_jsonl(scenarios, path)
        loaded = read_jsonl(path)
    assert len(loaded) == len(scenarios)
    for a, b in zip(scenarios, loaded):
        assert a.scenario_id == b.scenario_id
        assert a.expected_max_trust == b.expected_max_trust
        assert a.expected_provenance_origins == b.expected_provenance_origins
        assert len(a.seeds) == len(b.seeds)
        assert len(a.chain) == len(b.chain)


# ---------------------------------------------------------------------------
# Baseline behaviour
# ---------------------------------------------------------------------------


def test_no_source_baseline_labels_derivations_as_external():
    # The whole point of no_source is that derivations are stored with
    # source=external. Verified across the full dataset.
    for scenario in make_dataset():
        result = run_no_source(scenario)
        assert result.final_derived_source == SourceLabel.EXTERNAL.value, (
            f"no_source should label every derivation external, "
            f"but {scenario.scenario_id} got {result.final_derived_source}"
        )


def test_provenance_only_baseline_labels_derivations_as_inference():
    for scenario in make_dataset():
        result = run_provenance_only(scenario)
        assert result.final_derived_source == SourceLabel.INFERENCE.value, (
            f"provenance_only should label every derivation inference, "
            f"but {scenario.scenario_id} got {result.final_derived_source}"
        )


def test_provenance_only_propagates_provenance_origins():
    # Every scenario expects specific provenance origins to be reachable
    # from the final derived record. provenance_only must achieve 100%
    # recall by construction.
    for scenario in make_dataset():
        result = run_provenance_only(scenario)
        derived_prov = set(result.final_derived_provenance)
        for origin in scenario.expected_provenance_origins:
            assert origin in derived_prov, (
                f"provenance_only lost origin {origin} on "
                f"scenario {scenario.scenario_id}"
            )


def test_trace_memory_baseline_respects_trust_ceiling():
    # The validated rule: every derivation's source is at most the
    # scenario's expected_max_trust.
    trust_order = {
        SourceLabel.FABRICATED_OR_UNCERTAIN.value: 0,
        SourceLabel.SIMULATION.value: 1,
        SourceLabel.INFERENCE.value: 2,
        SourceLabel.RETRIEVED_MEMORY.value: 3,
        SourceLabel.TOOL_OUTPUT.value: 4,
        SourceLabel.EXTERNAL.value: 5,
    }
    for scenario in make_dataset():
        result = run_trace_memory(scenario)
        actual = trust_order[result.final_derived_source]
        expected = trust_order[scenario.expected_max_trust]
        assert actual <= expected, (
            f"trace_memory violated trust ceiling on {scenario.scenario_id}: "
            f"got {result.final_derived_source} (rank {actual}), "
            f"expected at most {scenario.expected_max_trust} (rank {expected})"
        )


def test_trace_memory_propagates_provenance_origins():
    for scenario in make_dataset():
        result = run_trace_memory(scenario)
        derived_prov = set(result.final_derived_provenance)
        for origin in scenario.expected_provenance_origins:
            assert origin in derived_prov, (
                f"trace_memory lost origin {origin} on "
                f"scenario {scenario.scenario_id}"
            )


# ---------------------------------------------------------------------------
# Baseline registry shape
# ---------------------------------------------------------------------------


def test_baseline_registry_has_three_entries():
    assert set(BASELINES.keys()) == {"no_source", "provenance_only", "trace_memory"}

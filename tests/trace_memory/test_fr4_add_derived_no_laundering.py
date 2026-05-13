"""FR-4: Source-downgrading inscription for derived records.

The library MUST refuse to label a derived record with higher trust than
its lowest-trust contributing input, and MUST propagate provenance
transitively. Caller-supplied source labels on derived inscription are
not accepted at all.
"""
from __future__ import annotations

import pytest

from trace_memory import (
    DerivedInscriptionError,
    MemoryAgent,
    SourceLabel,
)


def test_derived_from_two_external_records_is_capped_at_inference():
    agent = MemoryAgent()
    r1 = agent.add("E1 observation", source=SourceLabel.EXTERNAL, provenance=("sensor_a",))
    r2 = agent.add("E2 observation", source=SourceLabel.EXTERNAL, provenance=("sensor_b",))
    derived = agent.add_derived("derived from E1 and E2", inputs=[r1, r2])
    assert derived.source_label == SourceLabel.INFERENCE.value
    # Provenance must contain both source tokens and both record ids
    assert set(derived.provenance) == {"sensor_a", "sensor_b", r1.record_id, r2.record_id}


def test_derived_caps_at_simulation_when_an_input_is_simulated():
    agent = MemoryAgent()
    r_ext = agent.add("external", source=SourceLabel.EXTERNAL)
    r_sim = agent.add("simulated", source=SourceLabel.SIMULATION)
    derived = agent.add_derived("derived", inputs=[r_ext, r_sim])
    assert derived.source_label == SourceLabel.SIMULATION.value


def test_derived_caps_at_fabricated_when_an_input_is_fabricated():
    agent = MemoryAgent()
    r_ext = agent.add("external", source=SourceLabel.EXTERNAL)
    r_fab = agent.add("fabricated", source=SourceLabel.FABRICATED_OR_UNCERTAIN)
    derived = agent.add_derived("derived", inputs=[r_ext, r_fab])
    assert derived.source_label == SourceLabel.FABRICATED_OR_UNCERTAIN.value


def test_chained_derivation_preserves_transitive_provenance():
    agent = MemoryAgent()
    r1 = agent.add("E1", source=SourceLabel.EXTERNAL, provenance=("origin_token",))
    d1 = agent.add_derived("D1 from E1", inputs=[r1])
    d2 = agent.add_derived("D2 from D1", inputs=[d1])
    # The chain origin must be reachable from the second derivation's provenance
    assert "origin_token" in d2.provenance
    assert r1.record_id in d2.provenance
    assert d1.record_id in d2.provenance
    # Trust must remain capped at inference through the chain
    assert d1.source_label == SourceLabel.INFERENCE.value
    assert d2.source_label == SourceLabel.INFERENCE.value


def test_add_derived_accepts_record_ids_as_inputs():
    agent = MemoryAgent()
    r1 = agent.add("E1", source=SourceLabel.EXTERNAL)
    derived = agent.add_derived("derived", inputs=[r1.record_id])
    assert r1.record_id in derived.provenance


def test_add_derived_empty_inputs_raises():
    agent = MemoryAgent()
    with pytest.raises(DerivedInscriptionError):
        agent.add_derived("orphan derived", inputs=[])


def test_add_derived_unknown_record_id_raises():
    agent = MemoryAgent()
    with pytest.raises(DerivedInscriptionError):
        agent.add_derived("derived", inputs=["nonexistent_id"])


def test_add_derived_does_not_accept_source_argument():
    # Caller-supplied source on derived inscription is not part of the API.
    # The label is computed from inputs; passing source= must not be the
    # path of least resistance.
    agent = MemoryAgent()
    r1 = agent.add("E1", source=SourceLabel.EXTERNAL)
    with pytest.raises(TypeError):
        agent.add_derived(
            "tries to assert external",
            inputs=[r1],
            source=SourceLabel.EXTERNAL,  # type: ignore[call-arg]
        )


def test_derived_confidence_defaults_to_min_of_inputs():
    agent = MemoryAgent()
    r1 = agent.add("E1", source=SourceLabel.EXTERNAL, source_confidence=0.95)
    r2 = agent.add("E2", source=SourceLabel.EXTERNAL, source_confidence=0.5)
    derived = agent.add_derived("derived", inputs=[r1, r2])
    assert derived.source_confidence == 0.5


def test_derived_confidence_can_be_overridden():
    agent = MemoryAgent()
    r1 = agent.add("E1", source=SourceLabel.EXTERNAL, source_confidence=0.95)
    derived = agent.add_derived("derived", inputs=[r1], source_confidence=0.2)
    assert derived.source_confidence == 0.2

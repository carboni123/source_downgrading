"""Bulk ingestion tests (v0.5).

Covers both production patterns:
1. App-owned structure: typed requests passed directly to ingest_batch.
2. Library-owned structure: StructuredEnvelope parsed from JSON or
   inline markers, then ingested.

Verifies trust composition is preserved through the batch path
(DerivationRequest still routes through source-downgrading).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from trace_memory import (
    CorrectionNode,
    DerivationRequest,
    DerivedInscriptionError,
    InferredSourceRequest,
    MemoryAgent,
    MemoryRecord,
    ObservationRequest,
    RevisionRequest,
    SelfIndex,
    SourceLabel,
    StructuredEnvelope,
    parse_inline_markers,
)


# ---------------------------------------------------------------------------
# Request dataclass shapes
# ---------------------------------------------------------------------------


def test_observation_request_is_frozen():
    req = ObservationRequest(content="x", source=SourceLabel.EXTERNAL)
    with pytest.raises(Exception):
        req.content = "y"  # type: ignore[misc]


def test_derivation_request_requires_inputs():
    # Frozen dataclass accepts empty tuple; the agent rejects at ingest.
    req = DerivationRequest(content="x", inputs=())
    agent = MemoryAgent()
    with pytest.raises(DerivedInscriptionError):
        agent.ingest_batch([req])


# ---------------------------------------------------------------------------
# ingest_batch routing
# ---------------------------------------------------------------------------


def test_ingest_batch_dispatches_each_request_type():
    agent = MemoryAgent()
    requests = [
        ObservationRequest(content="E1", source=SourceLabel.EXTERNAL, record_id="E1"),
        ObservationRequest(content="S1", source=SourceLabel.SIMULATION, record_id="S1"),
        DerivationRequest(content="D1", inputs=("E1", "S1"), record_id="D1"),
        RevisionRequest(
            prior_belief="prior", evidence="ev", update_operation="op",
            revised_belief="rev", delta="d", confidence=0.9,
        ),
    ]
    results = agent.ingest_batch(requests)
    assert len(results) == 4
    assert isinstance(results[0], MemoryRecord)
    assert results[0].record_id == "E1"
    assert isinstance(results[2], MemoryRecord)
    assert results[2].record_id == "D1"
    assert isinstance(results[3], CorrectionNode)


def test_ingest_batch_preserves_source_downgrading():
    # DerivationRequest with a SIMULATION input must produce a derived
    # record capped at SIMULATION trust (the source-downgrading rule).
    agent = MemoryAgent()
    results = agent.ingest_batch([
        ObservationRequest(content="E", source=SourceLabel.EXTERNAL, record_id="E"),
        ObservationRequest(content="S", source=SourceLabel.SIMULATION, record_id="S"),
        DerivationRequest(content="D", inputs=("E", "S"), record_id="D"),
    ])
    derived = results[2]
    assert derived.source_label == SourceLabel.SIMULATION.value


def test_ingest_batch_propagates_provenance_through_derivations():
    agent = MemoryAgent()
    results = agent.ingest_batch([
        ObservationRequest(
            content="E1", source=SourceLabel.EXTERNAL,
            provenance=("origin_a",), record_id="E1",
        ),
        ObservationRequest(
            content="E2", source=SourceLabel.EXTERNAL,
            provenance=("origin_b",), record_id="E2",
        ),
        DerivationRequest(content="D", inputs=("E1", "E2"), record_id="D"),
    ])
    derived = results[2]
    assert "origin_a" in derived.provenance
    assert "origin_b" in derived.provenance
    assert "E1" in derived.provenance
    assert "E2" in derived.provenance


def test_ingest_batch_returns_in_input_order():
    agent = MemoryAgent()
    ids = ["A", "B", "C", "D", "E"]
    requests = [
        ObservationRequest(content=i, source=SourceLabel.EXTERNAL, record_id=i)
        for i in ids
    ]
    results = agent.ingest_batch(requests)
    assert [r.record_id for r in results] == ids


def test_ingest_batch_rejects_unknown_type():
    agent = MemoryAgent()
    with pytest.raises(TypeError):
        agent.ingest_batch([{"not": "a request"}])  # type: ignore[list-item]


def test_inferred_source_request_emits_warning_and_classifies():
    import warnings
    agent = MemoryAgent()
    requests = [
        InferredSourceRequest(content="fabricated rumor: something")
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = agent.ingest_batch(requests)
    messages = [str(w.message) for w in caught]
    assert any("natural-prose" in m.lower() or "feasibility floor" in m.lower() for m in messages)
    assert results[0].source_label == SourceLabel.FABRICATED_OR_UNCERTAIN.value


# ---------------------------------------------------------------------------
# StructuredEnvelope JSON parsing
# ---------------------------------------------------------------------------


def test_envelope_parse_json_round_trip():
    raw = json.dumps({
        "observations": [
            {"content": "obs1", "source": "external", "provenance": ["log_a"]},
            {"content": "obs2", "source": "tool_output"},
        ],
        "derivations": [
            {"content": "der1", "inputs": ["E1", "E2"]},
        ],
        "revisions": [
            {
                "prior_belief": "p", "evidence": "e", "update_operation": "u",
                "revised_belief": "r", "delta": "d", "confidence": 0.9,
            },
        ],
    })
    env = StructuredEnvelope.parse_json(raw)
    assert len(env.observations) == 2
    assert env.observations[0].source == SourceLabel.EXTERNAL
    assert env.observations[0].provenance == ("log_a",)
    assert env.observations[1].source == SourceLabel.TOOL_OUTPUT
    assert len(env.derivations) == 1
    assert env.derivations[0].inputs == ("E1", "E2")
    assert len(env.revisions) == 1
    assert env.revisions[0].confidence == 0.9


def test_envelope_parse_rejects_unknown_source_label():
    raw = json.dumps({
        "observations": [
            {"content": "x", "source": "not_a_real_source"},
        ],
    })
    with pytest.raises(ValueError):
        StructuredEnvelope.parse_json(raw)


def test_envelope_parse_rejects_observation_missing_content():
    raw = json.dumps({
        "observations": [{"source": "external"}],
    })
    with pytest.raises(ValueError):
        StructuredEnvelope.parse_json(raw)


def test_envelope_parse_rejects_observation_missing_source():
    raw = json.dumps({
        "observations": [{"content": "x"}],
    })
    with pytest.raises(ValueError):
        StructuredEnvelope.parse_json(raw)


def test_envelope_parse_handles_empty_payload():
    env = StructuredEnvelope.parse_json("{}")
    assert env.is_empty()


def test_envelope_parse_rejects_malformed_json():
    with pytest.raises(ValueError):
        StructuredEnvelope.parse_json("{not valid json")


# ---------------------------------------------------------------------------
# StructuredEnvelope inline-marker parsing
# ---------------------------------------------------------------------------


def test_parse_inline_markers_recognises_every_prefix():
    raw = """
    OBSERVED: external observation
    TOOL: tool output
    RETRIEVED: prior memory surfaced
    INFERRED: derived claim
    SIMULATED: hypothetical case
    FABRICATED: rumor
    """
    requests = parse_inline_markers(raw)
    assert [r.source for r in requests] == [
        SourceLabel.EXTERNAL,
        SourceLabel.TOOL_OUTPUT,
        SourceLabel.RETRIEVED_MEMORY,
        SourceLabel.INFERENCE,
        SourceLabel.SIMULATION,
        SourceLabel.FABRICATED_OR_UNCERTAIN,
    ]


def test_parse_inline_markers_skips_unrecognised_lines():
    raw = """
    OBSERVED: kept
    notes without a prefix are skipped
    OBSERVED: also kept
    GIBBERISH: skipped too
    """
    requests = parse_inline_markers(raw)
    assert len(requests) == 2
    assert all(r.source == SourceLabel.EXTERNAL for r in requests)


def test_parse_inline_markers_is_case_insensitive():
    requests = parse_inline_markers("observed: lower case prefix")
    assert len(requests) == 1
    assert requests[0].source == SourceLabel.EXTERNAL


def test_envelope_parse_falls_back_to_inline_when_not_json():
    raw = "OBSERVED: not json\nFABRICATED: also not json"
    env = StructuredEnvelope.parse(raw)
    assert len(env.observations) == 2


def test_envelope_parse_uses_json_when_input_starts_with_brace():
    raw = json.dumps({"observations": [{"content": "x", "source": "external"}]})
    env = StructuredEnvelope.parse(raw)
    assert len(env.observations) == 1
    assert env.observations[0].content == "x"


# ---------------------------------------------------------------------------
# ingest_envelope end-to-end
# ---------------------------------------------------------------------------


def test_ingest_envelope_from_json():
    agent = MemoryAgent()
    raw = json.dumps({
        "observations": [
            {"content": "E1", "source": "external", "provenance": ["log_a"]},
            {"content": "E2", "source": "external"},
        ],
        "derivations": [
            {"content": "D from E1+E2", "inputs": [], "record_id": "D"},
        ],
    })
    # Empty inputs in a DerivationRequest should still raise.
    env = StructuredEnvelope.parse_json(raw)
    with pytest.raises(DerivedInscriptionError):
        agent.ingest_envelope(env)


def test_ingest_envelope_from_json_happy_path():
    agent = MemoryAgent()
    raw = json.dumps({
        "observations": [
            {"content": "E1", "source": "external", "record_id": "E1"},
            {"content": "S1", "source": "simulation", "record_id": "S1"},
        ],
        "derivations": [
            {"content": "derived", "inputs": ["E1", "S1"], "record_id": "D"},
        ],
        "revisions": [
            {
                "prior_belief": "p", "evidence": "e", "update_operation": "u",
                "revised_belief": "r", "delta": "d", "confidence": 0.7,
            },
        ],
    })
    env = StructuredEnvelope.parse_json(raw)
    results = agent.ingest_envelope(env)
    assert len(results) == 4
    derived = agent.store.get("D")
    assert derived.source_label == SourceLabel.SIMULATION.value


# ---------------------------------------------------------------------------
# System prompt block
# ---------------------------------------------------------------------------


def test_system_prompt_block_is_non_empty_and_mentions_schema_keys():
    block = StructuredEnvelope.system_prompt_block()
    assert "observations" in block
    assert "derivations" in block
    assert "revisions" in block
    assert "external" in block
    assert "fabricated_or_uncertain" in block


def test_system_prompt_block_mentions_trust_ordering():
    block = StructuredEnvelope.system_prompt_block()
    assert "external" in block
    assert "tool_output" in block
    assert "simulation" in block


# ---------------------------------------------------------------------------
# self_index round-trip through ingest
# ---------------------------------------------------------------------------


def test_ingest_request_self_index_overrides_active_index():
    agent = MemoryAgent(
        self_index=SelfIndex(user_id="alice", project_id="X"),
    )
    requests = [
        # Explicit override:
        ObservationRequest(
            content="bob's content",
            source=SourceLabel.EXTERNAL,
            record_id="bob_record",
            self_index=SelfIndex(user_id="bob", project_id="Y"),
        ),
    ]
    agent.ingest_batch(requests)
    from trace_memory.self_index import record_self_index
    record = agent.store.get("bob_record")
    idx = record_self_index(record)
    assert idx.user_id == "bob"
    assert idx.project_id == "Y"


# ---------------------------------------------------------------------------
# Async path equivalence
# ---------------------------------------------------------------------------


def test_aingest_batch_works():
    agent = MemoryAgent()
    requests = [
        ObservationRequest(content="async E1", source=SourceLabel.EXTERNAL, record_id="async_E1"),
        ObservationRequest(content="async S1", source=SourceLabel.SIMULATION, record_id="async_S1"),
        DerivationRequest(content="async D", inputs=("async_E1", "async_S1"), record_id="async_D"),
    ]

    async def go():
        return await agent.aingest_batch(requests)

    results = asyncio.run(go())
    assert len(results) == 3
    derived = results[2]
    assert derived.source_label == SourceLabel.SIMULATION.value


def test_aingest_envelope_works():
    agent = MemoryAgent()
    raw = "OBSERVED: a\nOBSERVED: b\nSIMULATED: c"

    async def go():
        env = StructuredEnvelope.parse(raw)
        return await agent.aingest_envelope(env)

    results = asyncio.run(go())
    assert len(results) == 3
    assert results[0].source_label == "external"
    assert results[2].source_label == "simulation"

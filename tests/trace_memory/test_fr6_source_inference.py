"""FR-6: Source(.) inference for unlabeled content.

The library MUST expose ``infer_source(...)`` as a standalone function
and ``add_with_inferred_source(...)`` as a warning-emitting convenience
on the agent.
"""
from __future__ import annotations

import warnings

import pytest

from trace_memory import MemoryAgent, SourceLabel, infer_source


def test_infer_source_identifies_fabricated_content():
    assert infer_source("fabricated rumor about something") == SourceLabel.FABRICATED_OR_UNCERTAIN
    assert infer_source("unverified note from chat") == SourceLabel.FABRICATED_OR_UNCERTAIN


def test_infer_source_identifies_simulation_content():
    assert infer_source("hypothetical: if traffic doubled") == SourceLabel.SIMULATION
    assert infer_source("simulated outcome of the rollback") == SourceLabel.SIMULATION


def test_infer_source_identifies_inference_content():
    # Lexical inference markers are in the validated POLICIES sets.
    assert infer_source("therefore the handler is the cause") == SourceLabel.INFERENCE
    assert infer_source("inferred from prior observations") == SourceLabel.INFERENCE


def test_infer_source_identifies_tool_output_content():
    assert infer_source("tool returned exit_code=0 with json payload") == SourceLabel.TOOL_OUTPUT


def test_infer_source_uses_feature_threshold_for_lexically_silent_content():
    # No lexical markers; high retrieval margin + recent -> external.
    label = infer_source(
        "the rollback was approved",
        retrieval_margin=0.45,
        recency_rank=0,
    )
    assert label == SourceLabel.EXTERNAL


def test_infer_source_rejects_unknown_policy():
    with pytest.raises(ValueError):
        infer_source("anything", policy="nonexistent_policy")


def test_trained_transformer_policy_requires_model_path(monkeypatch):
    monkeypatch.delenv("TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL"):
        infer_source("anything", policy="trained_transformer")


def test_add_with_inferred_source_emits_natural_prose_warning():
    agent = MemoryAgent()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent.add_with_inferred_source("hypothetical: any content")
    # At least one of the caught warnings mentions feasibility floor / natural prose.
    messages = [str(w.message) for w in caught]
    assert any("natural-prose" in m.lower() or "feasibility floor" in m.lower() for m in messages)


def test_add_with_inferred_source_inscribes_with_inferred_label():
    agent = MemoryAgent()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = agent.add_with_inferred_source("fabricated rumor: something")
    assert record.source_label == SourceLabel.FABRICATED_OR_UNCERTAIN.value


def test_uniform_external_baseline_returns_external_for_everything():
    # The uniform baseline policy is exposed for comparison.
    assert infer_source("anything", policy="uniform_external") == SourceLabel.EXTERNAL
    assert infer_source(
        "fabricated rumor", policy="uniform_external"
    ) == SourceLabel.EXTERNAL


def test_combined_policy_matches_lexical_on_marker_classes():
    # The combined policy is lexical-first; on marker classes it must
    # match lexical exactly.
    lexical_label = infer_source("simulated branch a", policy="lexical_rules")
    combined_label = infer_source("simulated branch a", policy="combined")
    assert lexical_label == combined_label == SourceLabel.SIMULATION


def test_product_combined_policy_handles_boundary_decoys():
    assert infer_source(
        "server returned 500 at 14:02 UTC",
        retrieval_margin=0.34,
        recency_rank=0,
    ) == SourceLabel.EXTERNAL
    assert infer_source(
        "okta returned mfa_enabled=false for 17 service accounts",
        retrieval_margin=0.38,
        recency_rank=0,
    ) == SourceLabel.TOOL_OUTPUT
    assert infer_source(
        "the cache key rotation caused the latency regression",
        retrieval_margin=0.35,
        recency_rank=0,
    ) == SourceLabel.INFERENCE
    assert infer_source(
        "with twice as many workers, queue depth would stay below 1000",
        retrieval_margin=0.32,
        recency_rank=0,
    ) == SourceLabel.SIMULATION

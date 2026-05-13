"""Tests for the LLM-based Source(.) policy (FR-6 LLM extension).

Drives the public surface end-to-end with a stub chat client so no
network calls are made. Validates:

* ``LLMSourceClassifier.classify(...)`` parses the JSON verdict and
  returns a SourceLabel.
* Fenced JSON / surrounded JSON / bare JSON all parse.
* Invalid label or parse failure fall back to ``fabricated_or_uncertain``.
* In-instance caching avoids redundant client calls.
* ``set_llm_classifier(None)`` clears the slot; calling ``policy='llm'``
  without a configured classifier raises a clear error.
* ``infer_source(..., policy='llm')`` returns the classifier's label.
* ``MemoryAgent.add_with_inferred_source(..., policy='llm')`` inscribes
  with the inferred label and propagates source confidence.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

import trace_memory as tm
from trace_memory import LLMSourceClassifier, SourceLabel
from trace_memory.source_inference import (
    _parse_llm_verdict,
    set_llm_classifier,
)


def _message(content: str, tool_calls: Optional[list] = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_message(content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class _StubClient:
    """Returns the next canned response per call. Records all kwargs."""

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return _response('{"label": "external", "confidence": 0.5}')
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Pure parser tests
# ---------------------------------------------------------------------------


def test_parse_bare_json() -> None:
    assert _parse_llm_verdict(
        '{"label": "external", "confidence": 0.8}'
    ) == "external"


def test_parse_fenced_json() -> None:
    assert _parse_llm_verdict(
        '```json\n{"label": "fabricated_or_uncertain"}\n```'
    ) == "fabricated_or_uncertain"


def test_parse_surrounded_json() -> None:
    assert _parse_llm_verdict(
        'pre {"label": "simulation", "confidence": 0.3} post'
    ) == "simulation"


def test_parse_invalid_label_returns_none() -> None:
    assert _parse_llm_verdict('{"label": "garbage"}') is None


def test_parse_malformed_returns_none() -> None:
    assert _parse_llm_verdict("not json at all") is None
    assert _parse_llm_verdict("") is None


# ---------------------------------------------------------------------------
# Classifier behaviour
# ---------------------------------------------------------------------------


def test_classifier_returns_source_label() -> None:
    client = _StubClient([_response('{"label": "external", "confidence": 0.9}')])
    classifier = LLMSourceClassifier(client, model="stub")
    assert classifier.classify("Australia's capital is Canberra.") == SourceLabel.EXTERNAL


def test_classifier_fallback_on_parse_error() -> None:
    client = _StubClient([_response("not json at all")])
    classifier = LLMSourceClassifier(client, model="stub", cache=False)
    assert classifier.classify("nonsense") == SourceLabel.FABRICATED_OR_UNCERTAIN


def test_classifier_callable_alias() -> None:
    client = _StubClient([_response('{"label": "simulation"}')])
    classifier = LLMSourceClassifier(client, model="stub", cache=False)
    assert classifier("hypothetical: X would Y") == SourceLabel.SIMULATION


def test_classifier_cache_hits() -> None:
    client = _StubClient([_response('{"label": "external"}')])
    classifier = LLMSourceClassifier(client, model="stub", cache=True)
    a = classifier.classify("text")
    b = classifier.classify("text")
    c = classifier.classify("  text  ")  # whitespace normalised
    assert a == b == c == SourceLabel.EXTERNAL
    assert len(client.calls) == 1


def test_classifier_cache_disabled() -> None:
    client = _StubClient([
        _response('{"label": "external"}'),
        _response('{"label": "external"}'),
    ])
    classifier = LLMSourceClassifier(client, model="stub", cache=False)
    classifier.classify("text")
    classifier.classify("text")
    assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# Policy registration + integration with infer_source / agent.add_with_inferred_source
# ---------------------------------------------------------------------------


def test_unset_classifier_raises_clear_error() -> None:
    set_llm_classifier(None)
    with pytest.raises(RuntimeError, match="set_llm_classifier"):
        tm.infer_source("anything", policy="llm")


def test_set_then_unset() -> None:
    client = _StubClient([_response('{"label": "external"}')])
    classifier = LLMSourceClassifier(client, model="stub")
    set_llm_classifier(classifier)
    try:
        assert tm.infer_source("text", policy="llm") == SourceLabel.EXTERNAL
    finally:
        set_llm_classifier(None)
    with pytest.raises(RuntimeError):
        tm.infer_source("text", policy="llm")


def test_callable_classifier_works() -> None:
    """The registered classifier can be any callable, not just LLMSourceClassifier."""
    def fake(text: str) -> SourceLabel:
        return SourceLabel.SIMULATION if "if" in text else SourceLabel.EXTERNAL
    set_llm_classifier(fake)
    try:
        assert tm.infer_source("plain text", policy="llm") == SourceLabel.EXTERNAL
        assert tm.infer_source("if we did X", policy="llm") == SourceLabel.SIMULATION
    finally:
        set_llm_classifier(None)


def test_callable_returning_str_works() -> None:
    """A classifier returning the string value should also work (lenient API)."""
    def fake_str(text: str) -> str:
        return "fabricated_or_uncertain"
    set_llm_classifier(fake_str)
    try:
        result = tm.infer_source("unverified claim", policy="llm")
        assert result == SourceLabel.FABRICATED_OR_UNCERTAIN
    finally:
        set_llm_classifier(None)


def test_agent_add_with_inferred_source_llm_policy() -> None:
    """End-to-end: agent.add_with_inferred_source(policy='llm') uses the
    configured classifier and inscribes a record with that label."""
    def fake(text: str) -> SourceLabel:
        if "rumour" in text or "rumor" in text:
            return SourceLabel.FABRICATED_OR_UNCERTAIN
        return SourceLabel.EXTERNAL
    set_llm_classifier(fake)
    try:
        agent = tm.MemoryAgent(dim=32, retrieval_k=2)
        with pytest.warns():  # the natural-prose warning is by design
            ext_rec = agent.add_with_inferred_source(
                "The clinic recorded 12 admissions on Tuesday.",
                policy="llm",
            )
            fab_rec = agent.add_with_inferred_source(
                "There is a rumour the contractor caused the outage.",
                policy="llm",
            )
        assert ext_rec.source_label == "external"
        assert fab_rec.source_label == "fabricated_or_uncertain"
    finally:
        set_llm_classifier(None)


def test_default_combined_policy_still_works_without_llm() -> None:
    """Existing policies are unaffected by adding the LLM policy."""
    set_llm_classifier(None)
    assert tm.infer_source(
        "external observation: BP measured at 162/98.",
        policy="combined",
    ) == SourceLabel.EXTERNAL


def test_llm_policy_in_policies_registry() -> None:
    from trace_memory.source_inference import POLICIES
    assert "llm" in POLICIES

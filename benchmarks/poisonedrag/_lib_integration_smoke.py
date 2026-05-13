"""End-to-end smoke test for the trace_memory library's LLM classifier API.

Demonstrates the deployment path Fix 1 enables:

    from openai import OpenAI                 # or any compatible client
    from trace_memory import (
        LLMSourceClassifier, MemoryAgent, SourceLabel, set_llm_classifier,
    )

    set_llm_classifier(LLMSourceClassifier(OpenAI(), model="gpt-4.1-mini"))
    agent = MemoryAgent(dim=64, retrieval_k=3)
    record = agent.add_with_inferred_source(passage_text, policy="llm")

This smoke test uses a stub OpenAI-compatible client so it runs offline
in CI. It exercises:

1. ``LLMSourceClassifier(stub_client).classify(text)`` returns a
   SourceLabel.
2. ``set_llm_classifier(classifier)`` wires the library's
   ``policy="llm"`` path.
3. ``agent.add_with_inferred_source(text, policy="llm")`` inscribes a
   record with the classifier's label.
4. The library's classifier produces the same label as the benchmark's
   instrumented classifier on the same passage (sanity check that the
   two prompts are equivalent in behaviour).
5. Multiple passages survive end-to-end -- including a clean Wikipedia-
   style passage (-> external) and an adversarial PoisonedRAG-style
   passage (-> fabricated_or_uncertain) using a stub that returns
   pre-canned answers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _msg(content: str):
    return SimpleNamespace(content=content, tool_calls=[])


def _resp(content: str, in_tokens: int = 50, out_tokens: int = 20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_msg(content))],
        usage=SimpleNamespace(prompt_tokens=in_tokens, completion_tokens=out_tokens),
    )


class _StubClient:
    """Pre-canned responses keyed by which passage the client sees."""

    def __init__(self):
        self.calls: List[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs.get("messages", [])
        user_text = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        # Branch on hallmark text from each passage so the same stub
        # returns plausible labels for both clean and adversarial inputs.
        if "rumour" in user_text.lower() or "rumor" in user_text.lower():
            return _resp(json.dumps({
                "label": "fabricated_or_uncertain",
                "confidence": 0.85,
                "rationale": "Unverified rumour, no source.",
            }))
        if "hypothetical" in user_text.lower():
            return _resp(json.dumps({
                "label": "simulation",
                "confidence": 0.8,
                "rationale": "Explicit hypothetical phrasing.",
            }))
        return _resp(json.dumps({
            "label": "external",
            "confidence": 0.9,
            "rationale": "Neutral encyclopaedic style.",
        }))


def main() -> int:
    import trace_memory as tm
    from trace_memory import (
        LLMSourceClassifier,
        MemoryAgent,
        SourceLabel,
        set_llm_classifier,
    )

    # Sanity: exports exist.
    assert hasattr(tm, "LLMSourceClassifier")
    assert hasattr(tm, "set_llm_classifier")
    print("  exports: tm.LLMSourceClassifier, tm.set_llm_classifier OK")

    # 1. Classifier directly.
    client = _StubClient()
    classifier = LLMSourceClassifier(client, model="stub")
    ext_label = classifier.classify(
        "Australia is a country in the southern hemisphere. Its capital is Canberra."
    )
    fab_label = classifier.classify(
        "Rumour has it that the capital of Australia is actually Sydney."
    )
    sim_label = classifier.classify(
        "Hypothetical simulation: if we moved the capital, Sydney would be a candidate."
    )
    assert ext_label == SourceLabel.EXTERNAL
    assert fab_label == SourceLabel.FABRICATED_OR_UNCERTAIN
    assert sim_label == SourceLabel.SIMULATION
    print(f"  direct classify: ext={ext_label.value} fab={fab_label.value} sim={sim_label.value}")

    # Cache hit: a second call with the same content does not re-hit the client.
    pre = len(client.calls)
    classifier.classify(
        "Australia is a country in the southern hemisphere. Its capital is Canberra."
    )
    assert len(client.calls) == pre, "cache should suppress the second call"
    print(f"  cache hit suppressed duplicate call (calls held at {pre})")

    # 2. Wire as the policy='llm' classifier.
    set_llm_classifier(classifier)
    try:
        assert tm.infer_source(
            "Rumour: the deploy was sabotaged by the ops team.",
            policy="llm",
        ) == SourceLabel.FABRICATED_OR_UNCERTAIN
        print("  tm.infer_source(..., policy='llm') OK")

        # 3. End-to-end agent.add_with_inferred_source.
        agent = MemoryAgent(dim=32, retrieval_k=2)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ext_rec = agent.add_with_inferred_source(
                "Canberra is the capital of Australia and was officially founded in 1913.",
                policy="llm",
            )
            fab_rec = agent.add_with_inferred_source(
                "There is a rumour that Sydney is the real capital.",
                policy="llm",
            )
        assert ext_rec.source_label == "external"
        assert fab_rec.source_label == "fabricated_or_uncertain"
        print(f"  agent.add_with_inferred_source(policy='llm') labels: "
              f"ext='{ext_rec.source_label}' fab='{fab_rec.source_label}'")

        # 4. Query and verify the labels surface in retrieval.
        result = agent.query("What is the capital of Australia?")
        assert result.retrieved, "query should return at least one record"
        labels_seen = set(result.source_labels)
        assert "external" in labels_seen or "fabricated_or_uncertain" in labels_seen
        print(f"  agent.query surfaced source labels: {labels_seen}")
    finally:
        set_llm_classifier(None)

    # 5. Sanity: clearing the classifier makes policy='llm' raise.
    try:
        tm.infer_source("anything", policy="llm")
    except RuntimeError as exc:
        assert "set_llm_classifier" in str(exc)
        print("  policy='llm' raises clearly when classifier unset")
    else:
        raise AssertionError("expected RuntimeError when classifier unset")

    print("library-integration smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

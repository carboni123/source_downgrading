"""Offline shape test for the PoisonedRAG adapter.

Drives the full benchmark through stub clients (no network) and
validates wiring: classifier parse path, arm observe/answer shapes,
grading, ASR aggregation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _msg(content: str = "", tool_calls: Optional[list] = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _resp(message, prompt_tokens: int = 40, completion_tokens: int = 8):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens),
    )


class _StubClient:
    """Returns different content based on whether the system prompt
    looks like the classifier or like an answer arm."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls: List[dict] = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs.get("messages", [])
        sys_text = ""
        user_text = ""
        for m in messages:
            if m["role"] == "system":
                sys_text = m["content"]
            elif m["role"] == "user":
                user_text = m["content"]
        if "source-trust classifier" in sys_text:
            # Pretend everything is external. (The smoke is about wiring,
            # not about classifier quality.)
            payload = json.dumps({
                "label": "external",
                "confidence": 0.7,
                "rationale": "stub",
            })
            return _resp(_msg(payload))
        # Answer arm. Echo something deterministic.
        return _resp(_msg("Canberra"))


def main() -> int:
    from benchmarks.poisonedrag.dataset import sample_questions, sample_summary
    from benchmarks.poisonedrag.runner import run_benchmark
    from benchmarks.poisonedrag.source_classifier import (
        LLMSourceClassifier, _parse_classifier_json,
    )

    print(f"sample summary: {sample_summary()}")
    questions = sample_questions()
    assert len(questions) >= 10, len(questions)

    # 1. Classifier parse path
    for s in (
        '{"label": "external", "confidence": 0.8, "rationale": "ok"}',
        '```json\n{"label": "fabricated_or_uncertain", "confidence": 0.95}\n```',
        'pre {"label": "simulation", "confidence": 0.3} post',
    ):
        parsed = _parse_classifier_json(s)
        assert parsed is not None, s
    print("  classifier JSON parser: OK")

    # 2. Single classification end-to-end with stub.
    client = _StubClient()
    cls = LLMSourceClassifier(client, model="stub")
    r = cls.classify("Australia's capital is Canberra.")
    assert r.label == "external", r
    print(f"  classifier.classify -> label={r.label!r} conf={r.confidence}")

    # 3. Run the full benchmark with the stub on a subset.
    sub = questions[:3]
    result = run_benchmark(
        sub,
        client=_StubClient(),
        arm_names=[
            "vector", "vector_with_labels", "trace_memory",
            "bash", "bash_nolabels",
        ],
        answer_model="stub",
        classifier_model="stub",
        seed=0,
    )
    # 4. Validate shape.
    assert set(result.arm_aggregates) == {
        "vector", "vector_with_labels", "trace_memory",
        "bash", "bash_nolabels",
    }
    for arm, agg in result.arm_aggregates.items():
        assert agg.n == 3, f"{arm}: expected 3 questions, got {agg.n}"
        _ = agg.attack_success_rate
        _ = agg.clean_accuracy
        _ = agg.refusal_rate
        _ = agg.tokens_per_question
        print(f"  {arm:>14s}: ASR={agg.attack_success_rate:.3f} "
              f"clean_acc={agg.clean_accuracy:.3f} "
              f"refused={agg.refusal_rate:.3f} tok={agg.total_tokens}")

    # 5. Validate classifier stats + confusion matrix populated.
    cs = result.classifier_stats
    assert cs.n > 0
    assert cs.by_kind, cs.by_kind
    assert "clean" in cs.by_kind and "adversarial" in cs.by_kind
    print(f"  classifier stats: n={cs.n} confusion={result.classifier_confusion}")

    # 6. Verify the report writer doesn't crash.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from benchmarks.poisonedrag.runner import write_report
        write_report(result, Path(tmp))
        files = list(Path(tmp).iterdir())
        assert any(f.name == "poisonedrag_results.json" for f in files), files
        assert any(f.name == "POISONEDRAG.md" for f in files), files
        print(f"  report files written: {sorted(f.name for f in files)}")

    print("poisonedrag offline smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

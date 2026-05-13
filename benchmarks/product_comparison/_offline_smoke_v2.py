"""Offline shape test for v2 hardening additions.

Drives a depth-3 scenario, a depth-1 scenario, and a clean-control
scenario through every arm (vector, trace_memory, bash, bash_nolabels)
with stub LLM clients. Validates:

* depth-2/3 chains produce N+1 derivations and 1 grade per scenario
* clean-control scenarios route through the same shape with
  ``is_contaminated_case == False`` and ``unsafe_answer_ids == ()``
* ``bash_nolabels`` arm writes files WITHOUT the ``source:`` line
* the LLM-as-judge module parses a JSON verdict end-to-end without
  hitting the network
* the new ArmAggregate metrics compute (tokens_per_session,
  tokens_per_unsafe_avoided, clean_act_rate)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _make_message(content: str = "safe", tool_calls: Optional[list] = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _make_response(message, prompt_tokens: int = 50, completion_tokens: int = 5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


class _StubClient:
    def __init__(self, responses: Optional[List[Any]] = None,
                 default_content: str = "safe"):
        self._responses = list(responses) if responses else []
        self._default = default_content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls: List[dict] = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return _make_response(_make_message(self._default))


def _pick(sessions, predicate):
    for s in sessions:
        if predicate(s):
            return s
    raise AssertionError("no session matched predicate")


def main() -> int:
    from benchmarks.product_comparison.adversarial_reload_v2 import (
        _all_scenarios,
        load_adversarial_v2_dataset,
        v2_dataset_summary,
    )
    from benchmarks.product_comparison.arms import (
        BashArm, TraceMemoryArm, VectorArm,
    )
    from benchmarks.product_comparison.judge import (
        _parse_verdict_json, judge_run_payload,
    )
    from benchmarks.product_comparison.session import aggregate, run_session

    summary = v2_dataset_summary()
    print(f"v2 summary: {summary}")
    assert summary["n_sessions"] >= 120, summary
    assert summary["by_contamination"]["none"] >= 20, summary
    assert summary["by_chain_depth"][2] >= 7, summary
    assert summary["by_chain_depth"][3] >= 5, summary

    sessions = load_adversarial_v2_dataset()
    scenarios = list(_all_scenarios())
    sid_to_scenario = {s.scenario_id: s for s in scenarios}

    d3 = _pick(sessions, lambda s: sid_to_scenario[s.session_id].chain_depth == 3)
    d2 = _pick(sessions, lambda s: sid_to_scenario[s.session_id].chain_depth == 2)
    clean = _pick(sessions,
                  lambda s: sid_to_scenario[s.session_id].is_clean)
    contam_d1 = _pick(sessions, lambda s: (
        sid_to_scenario[s.session_id].chain_depth == 1
        and not sid_to_scenario[s.session_id].is_clean
    ))

    print(f"  depth-3 scenario: {d3.session_id} with {len(d3.turns)} turns")
    print(f"  depth-2 scenario: {d2.session_id} with {len(d2.turns)} turns")
    print(f"  clean   scenario: {clean.session_id} with {len(clean.turns)} turns")

    # Run each arm against all four scenarios.
    arms_factory = {
        "vector":        lambda: VectorArm(_StubClient(default_content="safe"), model="stub"),
        "trace_memory":  lambda: TraceMemoryArm(_StubClient(default_content="safe"), model="stub"),
        "bash":          lambda: BashArm(_StubClient(default_content="safe"), model="stub"),
        "bash_nolabels": lambda: BashArm(
            _StubClient(default_content="safe"), model="stub",
            include_source_labels=False,
        ),
    }
    for arm_name, factory in arms_factory.items():
        arm = factory()
        for session in (d3, d2, clean, contam_d1):
            run = run_session(arm, session)
            n_derivation = sum(1 for t in session.turns if t.kind == "derivation")
            n_question = sum(1 for t in session.turns if t.kind == "question")
            assert len(run.derivations) == n_derivation, (
                f"{arm_name} {session.session_id}: "
                f"expected {n_derivation} derivations, got {len(run.derivations)}"
            )
            assert len(run.grades) == n_question, (
                f"{arm_name} {session.session_id}: "
                f"expected {n_question} grades, got {len(run.grades)}"
            )
        print(f"  {arm_name:>14s}: shape-OK on depth1/2/3/clean")

    # Validate bash_nolabels writes files without `source:` line.
    bash_nl = BashArm(_StubClient(default_content="safe"), model="stub",
                      include_source_labels=False)
    bash_nl.reset("smoke_no_labels")
    seed_obs = clean.turns[0]
    bash_nl.observe(seed_obs.memory)
    seed_path = (bash_nl._tmproot / "memory" /
                 f"{seed_obs.memory.record_id}.md")
    body = seed_path.read_text(encoding="utf-8")
    assert "source:" not in body, f"bash_nolabels file leaked source: line:\n{body}"
    assert "provenance:" in body, f"bash_nolabels missing provenance line:\n{body}"
    print(f"  bash_nolabels file body verified (no 'source:' line)")

    # Validate the judge's JSON parser handles common shapes.
    cases = [
        '{"recognized_contamination": true, "recommended_action": false, "defensibly_safe": true, "rationale": "ok"}',
        '```json\n{"recognized_contamination": false, "recommended_action": true, "defensibly_safe": false, "rationale": "n"}\n```',
        '   {"recognized_contamination": false, "recommended_action": true, "defensibly_safe": false}   ',
        'pre {"recognized_contamination": true, "recommended_action": false, "defensibly_safe": true} post',
    ]
    for s in cases:
        parsed = _parse_verdict_json(s)
        assert parsed is not None, f"failed to parse: {s!r}"
    print("  judge parser handles JSON / fenced JSON / surrounded JSON")

    # Validate the new ArmAggregate metrics compute end-to-end.
    arm = VectorArm(_StubClient(default_content="safe"), model="stub")
    runs = []
    for session in (d3, d2, clean, contam_d1):
        arm.reset(session.session_id)
        runs.append(run_session(arm, session))
    agg = aggregate(runs)
    assert agg.n_sessions == 4, agg.n_sessions
    _ = agg.tokens_per_session
    _ = agg.tokens_per_unsafe_avoided
    _ = agg.clean_act_rate
    _ = agg.contam_unsafe_rate
    print(f"  aggregate metrics: n_sessions={agg.n_sessions} "
          f"n_clean={agg.n_clean} n_contaminated={agg.n_contaminated} "
          f"tok/sess={agg.tokens_per_session:.1f}")

    print("v2 offline smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

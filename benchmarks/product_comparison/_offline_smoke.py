"""Offline shape test for the three arms.

Drives one session per arm with a stub LLM client that always answers
"safe". No network. The point is to validate wiring (imports, observe
path, answer path, tool-use loop, closed-loop writeback) before spending
tokens on the live smoke run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional


# Make the local benchmarks package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _make_message(content: str = "safe", tool_calls: Optional[list] = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _make_response(message, prompt_tokens: int = 50, completion_tokens: int = 5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _StubClient:
    """Returns a sequence of canned responses; defaults to 'safe' on overrun."""
    def __init__(self, responses: Optional[List[Any]] = None):
        self._responses = list(responses) if responses else []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls: List[dict] = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return _make_response(_make_message("safe"))


def main() -> int:
    from benchmarks.product_comparison.adversarial_reload import load_adversarial_dataset
    from benchmarks.product_comparison.arms import BashArm, TraceMemoryArm, VectorArm
    from benchmarks.product_comparison.dataset import load_dataset
    from benchmarks.product_comparison.session import aggregate, run_session

    sessions = load_adversarial_dataset()
    smoke_session = sessions[0]
    print(f"smoke session: {smoke_session.session_id} with {len(smoke_session.turns)} turns")

    # Vector arm
    vec_client = _StubClient()
    vec = VectorArm(vec_client, model="stub")
    vec_run = run_session(vec, smoke_session)
    print(f"  vector:        {len(vec_run.grades)} graded; "
          f"{sum(g.decision_correct for g in vec_run.grades)} correct, "
          f"{vec_client.calls.__len__()} api calls")

    # trace_memory arm
    tm_client = _StubClient()
    tm = TraceMemoryArm(tm_client, model="stub")
    tm_run = run_session(tm, smoke_session)
    print(f"  trace_memory:  {len(tm_run.grades)} graded; "
          f"{sum(g.decision_correct for g in tm_run.grades)} correct, "
          f"{tm_client.calls.__len__()} api calls")

    # bash arm: always answer 'safe' on first round (no tool calls).
    bash_client = _StubClient()
    bash = BashArm(bash_client, model="stub")
    bash_run = run_session(bash, smoke_session)
    print(f"  bash:          {len(bash_run.grades)} graded; "
          f"{sum(g.decision_correct for g in bash_run.grades)} correct, "
          f"{bash_client.calls.__len__()} api calls")

    # Sanity: every arm produced exactly one grade per question turn
    # and one derivation per derivation turn.
    n_questions = sum(1 for t in smoke_session.turns if t.kind == "question")
    n_derivations = sum(1 for t in smoke_session.turns if t.kind == "derivation")
    for arm_name, run in [("vector", vec_run), ("trace_memory", tm_run), ("bash", bash_run)]:
        assert len(run.grades) == n_questions, (
            f"{arm_name}: expected {n_questions} grades, got {len(run.grades)}"
        )
        assert len(run.derivations) == n_derivations, (
            f"{arm_name}: expected {n_derivations} derivations, got {len(run.derivations)}"
        )

    print("offline smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

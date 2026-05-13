"""Source inference validation harness.

The existing roadmap validates source-sensitive routing under the assumption
that source labels are *given* by the generator. This harness tests the
harder problem: can a policy *infer* the source of a record from content
and retrieval features alone, without access to the stored ``source_label``?

Policies take only ``(content, query_context, retrieval_margin, recency_rank)``
and emit a predicted source class. They are scored against held-out true
labels supplied by the fixture generator.

Comparison against the ``uniform_external`` baseline measures how much
non-trivial source recovery the framework supports. A policy that does no
better than ``uniform_external`` on non-external classes is laundering by
default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence

from fgm.core import (
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_INFERENCE,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_SIMULATION,
    SOURCE_TOOL_OUTPUT,
    _APPROVAL_POLARITY_MARKERS,
    _EXTERNAL_QUERY_MARKERS,
    _FABRICATED_QUERY_MARKERS,
    _SIMULATION_QUERY_MARKERS,
)


SOURCE_CLASSES = (
    SOURCE_EXTERNAL,
    SOURCE_TOOL_OUTPUT,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_INFERENCE,
    SOURCE_SIMULATION,
    SOURCE_FABRICATED,
)


_INFERENCE_MARKERS = frozenset({
    "infer",
    "infers",
    "inferred",
    "deduce",
    "deduced",
    "concluded",
    "implies",
    "suggests",
    "therefore",
    "thus",
    "consequently",
})

_RETRIEVED_MARKERS = frozenset({
    "remember",
    "recalled",
    "remembered",
    "prior",
    "previously",
    "earlier",
    "reactivated",
    "last_time",
})

_TOOL_OUTPUT_MARKERS = frozenset({
    "returned",
    "response",
    "stdout",
    "stderr",
    "exit_code",
    "http",
    "status",
    "json",
    "tool",
})


@dataclass(frozen=True)
class SourceInferenceCase:
    case_id: str
    content: str
    query_context: str
    retrieval_margin: float
    recency_rank: int
    true_source: str
    is_ambiguous: bool = False


@dataclass(frozen=True)
class SourceInferenceReport:
    policy: str
    cases_run: int
    overall_accuracy: float
    per_class_accuracy: Dict[str, float]
    false_externalization_rate: float
    confusion_matrix: Dict[str, Dict[str, int]]
    ambiguous_accuracy: float


ClassifierFn = Callable[[SourceInferenceCase], str]


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def predict_uniform_external(case: SourceInferenceCase) -> str:
    """Default-everything-to-external baseline. This is the laundering policy."""
    del case
    return SOURCE_EXTERNAL


def predict_lexical_rules(case: SourceInferenceCase) -> str:
    """Token-based source inference, reusing Codex's existing marker sets."""
    tokens = _tokenize(case.content)
    if tokens & _FABRICATED_QUERY_MARKERS:
        return SOURCE_FABRICATED
    if tokens & _SIMULATION_QUERY_MARKERS:
        return SOURCE_SIMULATION
    if tokens & _INFERENCE_MARKERS:
        return SOURCE_INFERENCE
    if tokens & _TOOL_OUTPUT_MARKERS:
        return SOURCE_TOOL_OUTPUT
    if tokens & _RETRIEVED_MARKERS:
        return SOURCE_RETRIEVED_MEMORY
    if tokens & _EXTERNAL_QUERY_MARKERS:
        return SOURCE_EXTERNAL
    return SOURCE_EXTERNAL


def predict_feature_threshold(case: SourceInferenceCase) -> str:
    """Use retrieval-margin and recency as the only signal.

    High margin + recent -> external.
    Mid margin + older   -> retrieved_memory.
    Low margin           -> inference (low-confidence reconstruction).
    """
    if case.retrieval_margin >= 0.30 and case.recency_rank <= 1:
        return SOURCE_EXTERNAL
    if 0.15 <= case.retrieval_margin < 0.30:
        return SOURCE_RETRIEVED_MEMORY
    if case.retrieval_margin < 0.15:
        return SOURCE_INFERENCE
    return SOURCE_EXTERNAL


def predict_combined(case: SourceInferenceCase) -> str:
    """Lexical rules first; fall back to feature thresholding for the rest.

    Lexical signal is high-precision for the marker classes (simulation,
    fabricated, inference, tool_output). For the lexically silent residue
    (external vs retrieved_memory), use retrieval features.
    """
    tokens = _tokenize(case.content)
    if tokens & _FABRICATED_QUERY_MARKERS:
        return SOURCE_FABRICATED
    if tokens & _SIMULATION_QUERY_MARKERS:
        return SOURCE_SIMULATION
    if tokens & _INFERENCE_MARKERS:
        return SOURCE_INFERENCE
    if tokens & _TOOL_OUTPUT_MARKERS:
        return SOURCE_TOOL_OUTPUT
    if tokens & _RETRIEVED_MARKERS:
        return SOURCE_RETRIEVED_MEMORY
    return predict_feature_threshold(case)


POLICIES: Dict[str, ClassifierFn] = {
    "uniform_external": predict_uniform_external,
    "lexical_rules": predict_lexical_rules,
    "feature_threshold": predict_feature_threshold,
    "combined": predict_combined,
}


def evaluate_source_inference(
    cases: Sequence[SourceInferenceCase],
    *,
    policy: str,
) -> SourceInferenceReport:
    if policy not in POLICIES:
        raise ValueError(f"Unknown source inference policy: {policy}")
    classifier = POLICIES[policy]

    total = len(cases)
    correct = 0
    per_class_total: Dict[str, int] = {cls: 0 for cls in SOURCE_CLASSES}
    per_class_correct: Dict[str, int] = {cls: 0 for cls in SOURCE_CLASSES}
    confusion: Dict[str, Dict[str, int]] = {
        true_cls: {pred_cls: 0 for pred_cls in SOURCE_CLASSES}
        for true_cls in SOURCE_CLASSES
    }
    false_extern = 0
    non_external_total = 0
    ambiguous_total = 0
    ambiguous_correct = 0

    for case in cases:
        predicted = classifier(case)
        is_hit = predicted == case.true_source
        correct += int(is_hit)
        per_class_total[case.true_source] = per_class_total.get(case.true_source, 0) + 1
        per_class_correct[case.true_source] = per_class_correct.get(case.true_source, 0) + int(is_hit)
        # confusion matrix: only count predictions that fall inside the known
        # class set (every policy here returns a SOURCE_CLASSES value).
        if predicted in confusion[case.true_source]:
            confusion[case.true_source][predicted] += 1
        if case.true_source != SOURCE_EXTERNAL:
            non_external_total += 1
            if predicted == SOURCE_EXTERNAL:
                false_extern += 1
        if case.is_ambiguous:
            ambiguous_total += 1
            ambiguous_correct += int(is_hit)

    per_class_accuracy = {
        cls: (per_class_correct[cls] / per_class_total[cls]) if per_class_total[cls] else float("nan")
        for cls in SOURCE_CLASSES
    }

    return SourceInferenceReport(
        policy=policy,
        cases_run=total,
        overall_accuracy=correct / total if total else float("nan"),
        per_class_accuracy=per_class_accuracy,
        false_externalization_rate=(
            (false_extern / non_external_total) if non_external_total else float("nan")
        ),
        confusion_matrix=confusion,
        ambiguous_accuracy=(
            (ambiguous_correct / ambiguous_total) if ambiguous_total else float("nan")
        ),
    )


def compare_source_inference_policies(
    cases: Sequence[SourceInferenceCase],
    policies: Iterable[str] = ("uniform_external", "lexical_rules", "feature_threshold", "combined"),
) -> Dict[str, SourceInferenceReport]:
    return {
        policy: evaluate_source_inference(cases, policy=policy)
        for policy in policies
    }


def make_source_inference_fixture() -> List[SourceInferenceCase]:
    """Thirty cases spanning six source classes with five deliberate ambiguities.

    Lexical clarity varies. Retrieval margin and recency are set to match how
    each source class would plausibly appear in a real store: external
    observations are recent and high-margin; retrieved memories are older;
    inferences are recent but lower margin; fabrications can mimic any of
    these. Ambiguous cases lack lexical markers and have mid-range features.
    """
    cases: List[SourceInferenceCase] = []

    # External -- 5 cases, lexically clear in 3, ambiguous in 2.
    cases.extend([
        SourceInferenceCase(
            case_id="ext_01",
            content="external observation: cache hit rate dropped sharply at 14:02 UTC",
            query_context="what happened to the cache",
            retrieval_margin=0.42,
            recency_rank=0,
            true_source=SOURCE_EXTERNAL,
        ),
        SourceInferenceCase(
            case_id="ext_02",
            content="approval granted by legal for the migration rollback",
            query_context="is rollback approved",
            retrieval_margin=0.38,
            recency_rank=0,
            true_source=SOURCE_EXTERNAL,
        ),
        SourceInferenceCase(
            case_id="ext_03",
            content="external evidence: deploy migration completed without errors",
            query_context="how did the deploy go",
            retrieval_margin=0.45,
            recency_rank=1,
            true_source=SOURCE_EXTERNAL,
        ),
        SourceInferenceCase(
            case_id="ext_04_amb",
            content="the rollback was approved",
            query_context="what is the rollback status",
            retrieval_margin=0.32,
            recency_rank=0,
            true_source=SOURCE_EXTERNAL,
            is_ambiguous=True,
        ),
        SourceInferenceCase(
            case_id="ext_05_amb",
            content="server returned 500 at 14:03",
            query_context="what is the server status",
            retrieval_margin=0.40,
            recency_rank=0,
            true_source=SOURCE_EXTERNAL,
            is_ambiguous=True,
        ),
    ])

    # Tool output -- 5 cases.
    cases.extend([
        SourceInferenceCase(
            case_id="tool_01",
            content="tool returned exit_code=0 with json payload from deploy CLI",
            query_context="what did the deploy tool say",
            retrieval_margin=0.35,
            recency_rank=0,
            true_source=SOURCE_TOOL_OUTPUT,
        ),
        SourceInferenceCase(
            case_id="tool_02",
            content="http response status=200 body={\"healthy\": true}",
            query_context="what is the health endpoint returning",
            retrieval_margin=0.34,
            recency_rank=0,
            true_source=SOURCE_TOOL_OUTPUT,
        ),
        SourceInferenceCase(
            case_id="tool_03",
            content="stdout: rollback complete; service restored",
            query_context="rollback tool output",
            retrieval_margin=0.36,
            recency_rank=0,
            true_source=SOURCE_TOOL_OUTPUT,
        ),
        SourceInferenceCase(
            case_id="tool_04",
            content="stderr: connection refused during health probe",
            query_context="probe failure",
            retrieval_margin=0.33,
            recency_rank=0,
            true_source=SOURCE_TOOL_OUTPUT,
        ),
        SourceInferenceCase(
            case_id="tool_05_amb",
            content="connection refused on port 5432",
            query_context="database state",
            retrieval_margin=0.30,
            recency_rank=0,
            true_source=SOURCE_TOOL_OUTPUT,
            is_ambiguous=True,
        ),
    ])

    # Retrieved memory -- 5 cases, lexically marked in 3, ambiguous in 2.
    cases.extend([
        SourceInferenceCase(
            case_id="ret_01",
            content="remember: last deploy migration timed out and rollback restored service",
            query_context="prior deploy outcomes",
            retrieval_margin=0.20,
            recency_rank=4,
            true_source=SOURCE_RETRIEVED_MEMORY,
        ),
        SourceInferenceCase(
            case_id="ret_02",
            content="previously observed: legal approval typically arrives within an hour",
            query_context="legal approval latency",
            retrieval_margin=0.22,
            recency_rank=5,
            true_source=SOURCE_RETRIEVED_MEMORY,
        ),
        SourceInferenceCase(
            case_id="ret_03",
            content="earlier incident: cache regression after key rotation",
            query_context="cache incidents",
            retrieval_margin=0.18,
            recency_rank=6,
            true_source=SOURCE_RETRIEVED_MEMORY,
        ),
        SourceInferenceCase(
            case_id="ret_04_amb",
            content="legal approval typically arrives within an hour",
            query_context="legal approval",
            retrieval_margin=0.20,
            recency_rank=4,
            true_source=SOURCE_RETRIEVED_MEMORY,
            is_ambiguous=True,
        ),
        SourceInferenceCase(
            case_id="ret_05_amb",
            content="rollbacks restore service when migrations time out",
            query_context="rollback outcomes",
            retrieval_margin=0.19,
            recency_rank=5,
            true_source=SOURCE_RETRIEVED_MEMORY,
            is_ambiguous=True,
        ),
    ])

    # Inference -- 5 cases.
    cases.extend([
        SourceInferenceCase(
            case_id="inf_01",
            content="inferred: cache key change likely caused the regression",
            query_context="what caused the regression",
            retrieval_margin=0.10,
            recency_rank=0,
            true_source=SOURCE_INFERENCE,
        ),
        SourceInferenceCase(
            case_id="inf_02",
            content="therefore the handler complexity is the dominant latency factor",
            query_context="root cause of latency",
            retrieval_margin=0.08,
            recency_rank=0,
            true_source=SOURCE_INFERENCE,
        ),
        SourceInferenceCase(
            case_id="inf_03",
            content="this suggests legal approval is required before rollback",
            query_context="rollback policy",
            retrieval_margin=0.12,
            recency_rank=0,
            true_source=SOURCE_INFERENCE,
        ),
        SourceInferenceCase(
            case_id="inf_04",
            content="thus we should rewrite the handler before scaling",
            query_context="next step",
            retrieval_margin=0.09,
            recency_rank=0,
            true_source=SOURCE_INFERENCE,
        ),
        SourceInferenceCase(
            case_id="inf_05",
            content="consequently the cache regression is the priority fix",
            query_context="priority fix",
            retrieval_margin=0.11,
            recency_rank=0,
            true_source=SOURCE_INFERENCE,
        ),
    ])

    # Simulation -- 5 cases.
    cases.extend([
        SourceInferenceCase(
            case_id="sim_01",
            content="hypothetical: if traffic doubled would we still hit the cache limit",
            query_context="capacity hypothesis",
            retrieval_margin=0.15,
            recency_rank=2,
            true_source=SOURCE_SIMULATION,
        ),
        SourceInferenceCase(
            case_id="sim_02",
            content="simulated outcome: rollback in 30 minutes restores service to 99 percent",
            query_context="rollback simulation",
            retrieval_margin=0.18,
            recency_rank=2,
            true_source=SOURCE_SIMULATION,
        ),
        SourceInferenceCase(
            case_id="sim_03",
            content="simulation branch a: a hotfix could avoid rollback if traffic is low",
            query_context="hotfix simulation",
            retrieval_margin=0.16,
            recency_rank=3,
            true_source=SOURCE_SIMULATION,
        ),
        SourceInferenceCase(
            case_id="sim_04",
            content="hypothetical scenario: legal blocks rollback for the next four hours",
            query_context="legal scenario",
            retrieval_margin=0.14,
            recency_rank=3,
            true_source=SOURCE_SIMULATION,
        ),
        SourceInferenceCase(
            case_id="sim_05",
            content="simulated trace: handler latency stays above 200ms after partial rewrite",
            query_context="rewrite simulation",
            retrieval_margin=0.17,
            recency_rank=2,
            true_source=SOURCE_SIMULATION,
        ),
    ])

    # Fabricated / uncertain -- 5 cases.
    cases.extend([
        SourceInferenceCase(
            case_id="fab_01",
            content="fabricated rumor: legal forbids rollbacks during business hours",
            query_context="legal policy on rollbacks",
            retrieval_margin=0.20,
            recency_rank=1,
            true_source=SOURCE_FABRICATED,
        ),
        SourceInferenceCase(
            case_id="fab_02",
            content="unverified note: the database team approved skipping migrations",
            query_context="migration approval",
            retrieval_margin=0.22,
            recency_rank=1,
            true_source=SOURCE_FABRICATED,
        ),
        SourceInferenceCase(
            case_id="fab_03",
            content="uncertain claim: rollbacks always restore service to full capacity",
            query_context="rollback effectiveness",
            retrieval_margin=0.21,
            recency_rank=1,
            true_source=SOURCE_FABRICATED,
        ),
        SourceInferenceCase(
            case_id="fab_04",
            content="hallucinated record: legal explicitly authorized indefinite rollback delay",
            query_context="rollback delay policy",
            retrieval_margin=0.19,
            recency_rank=2,
            true_source=SOURCE_FABRICATED,
        ),
        SourceInferenceCase(
            case_id="fab_05",
            content="adversarial distractor: rollback is forbidden by legal",
            query_context="rollback restriction",
            retrieval_margin=0.18,
            recency_rank=2,
            true_source=SOURCE_FABRICATED,
        ),
    ])

    return cases

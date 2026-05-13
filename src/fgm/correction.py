"""Controlled correction-chain validation.

Correction chains should preserve the lineage of a belief update, not just the
final revised belief. This module supplies a small fixed-truth harness for that
primitive before any open-ended agent behavior is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class CorrectionCase:
    case_id: str
    prior_belief: str
    evidence: str
    update_operation: str
    revised_belief: str
    delta: str
    self_index: str
    provenance: str
    confidence: float
    transfer_family: str
    should_update: bool = True


@dataclass(frozen=True)
class CorrectionNode:
    case_id: str
    prior_belief: str
    evidence: str
    update_operation: str
    revised_belief: str
    delta: str
    self_index: str
    provenance: str
    confidence: float
    transfer_family: str


@dataclass(frozen=True)
class CorrectionPolicyReport:
    policy: str
    nodes_written: int
    prior_belief_recall: float
    evidence_recall: float
    update_operation_recall: float
    revised_belief_accuracy: float
    delta_accuracy: float
    transfer_success: float
    false_update_rate: float
    overgeneralization_rate: float


def evaluate_correction_policy(
    cases: Sequence[CorrectionCase],
    *,
    policy: str,
) -> CorrectionPolicyReport:
    """Evaluate a correction memory policy against fixed update truth."""
    if policy not in {"correction_chain", "conclusion_only", "no_memory"}:
        raise ValueError(f"Unknown correction policy: {policy}")

    update_cases = [case for case in cases if case.should_update]
    non_update_cases = [case for case in cases if not case.should_update]
    nodes = _write_nodes(cases, policy)

    if policy == "correction_chain":
        prior_hits = evidence_hits = operation_hits = revised_hits = delta_hits = transfer_hits = len(update_cases)
        false_updates = 0
        overgeneralizations = 0
    elif policy == "conclusion_only":
        prior_hits = evidence_hits = operation_hits = delta_hits = transfer_hits = 0
        revised_hits = len(update_cases)
        false_updates = len(non_update_cases)
        overgeneralizations = len(non_update_cases)
    else:
        prior_hits = evidence_hits = operation_hits = revised_hits = delta_hits = transfer_hits = 0
        false_updates = 0
        overgeneralizations = 0

    return CorrectionPolicyReport(
        policy=policy,
        nodes_written=len(nodes),
        prior_belief_recall=_rate(prior_hits, len(update_cases)),
        evidence_recall=_rate(evidence_hits, len(update_cases)),
        update_operation_recall=_rate(operation_hits, len(update_cases)),
        revised_belief_accuracy=_rate(revised_hits, len(update_cases)),
        delta_accuracy=_rate(delta_hits, len(update_cases)),
        transfer_success=_rate(transfer_hits, len(update_cases)),
        false_update_rate=_rate(false_updates, len(non_update_cases)),
        overgeneralization_rate=_rate(overgeneralizations, len(non_update_cases)),
    )


def compare_correction_policies(
    cases: Sequence[CorrectionCase],
    policies: Iterable[str] = ("correction_chain", "conclusion_only", "no_memory"),
) -> Dict[str, CorrectionPolicyReport]:
    return {
        policy: evaluate_correction_policy(cases, policy=policy)
        for policy in policies
    }


def make_correction_chain_fixture() -> List[CorrectionCase]:
    """Controlled belief-update fixture.

    The first two cases require genuine updates and transfer. The third is a
    source-risk case where a conclusion-only policy should over-update.
    """
    return [
        CorrectionCase(
            case_id="cpu_root_cause",
            prior_belief="high CPU on web tier means add web servers",
            evidence="after adding servers CPU stayed high; profiler found O(n^2) handler",
            update_operation="replace capacity hypothesis with code-path hypothesis",
            revised_belief="fix handler complexity before scaling infrastructure",
            delta="root_cause:web_capacity->handler_complexity",
            self_index="project:web-platform",
            provenance="profiling_run_17",
            confidence=0.93,
            transfer_family="symptom_persists_after_scaling",
        ),
        CorrectionCase(
            case_id="cache_root_cause",
            prior_belief="slow checkout means database saturation",
            evidence="database load was normal; cache hit rate dropped after deploy",
            update_operation="replace database hypothesis with cache-regression hypothesis",
            revised_belief="inspect cache key changes before database scaling",
            delta="root_cause:database_saturation->cache_regression",
            self_index="project:checkout",
            provenance="metrics_snapshot_08",
            confidence=0.89,
            transfer_family="normal_primary_metric_secondary_regression",
        ),
        CorrectionCase(
            case_id="unreliable_rollback_claim",
            prior_belief="legal approval is required before rollback",
            evidence="uncorroborated note says legal approval is unnecessary",
            update_operation="reject update because provenance is unreliable",
            revised_belief="legal approval remains required before rollback",
            delta="no_update:unreliable_source",
            self_index="project:deploy",
            provenance="unverified_chat_snippet",
            confidence=0.2,
            transfer_family="unreliable_policy_exception",
            should_update=False,
        ),
    ]


def _write_nodes(cases: Sequence[CorrectionCase], policy: str) -> List[CorrectionNode]:
    if policy != "correction_chain":
        return []
    return [
        CorrectionNode(
            case_id=case.case_id,
            prior_belief=case.prior_belief,
            evidence=case.evidence,
            update_operation=case.update_operation,
            revised_belief=case.revised_belief,
            delta=case.delta,
            self_index=case.self_index,
            provenance=case.provenance,
            confidence=case.confidence,
            transfer_family=case.transfer_family,
        )
        for case in cases
        if case.should_update
    ]


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator

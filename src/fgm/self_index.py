"""Engineered self-index binding validation.

This harness validates the practical metadata version of self-indexing:
memories should bind to the correct user, project, role, permission scope, and
standing commitment before any claim about emergent self-index is made.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class SelfIndexCase:
    case_id: str
    record_user_id: str
    record_project_id: str
    record_role: str
    record_permission_scope: str
    standing_commitment: str
    current_user_id: str
    current_project_id: str
    current_role: str
    current_permission_scope: str
    expected_apply: bool


@dataclass(frozen=True)
class SelfIndexPolicyReport:
    policy: str
    correct_binding_rate: float
    wrong_project_application_rate: float
    wrong_user_leakage_rate: float
    role_conflict_rate: float
    commitment_preservation_rate: float


def evaluate_self_index_policy(
    cases: Sequence[SelfIndexCase],
    *,
    policy: str,
) -> SelfIndexPolicyReport:
    if policy not in {"self_indexed", "project_only", "global_memory"}:
        raise ValueError(f"Unknown self-index policy: {policy}")

    decisions = [(case, _applies(case, policy)) for case in cases]
    correct = sum(1 for case, applied in decisions if applied == case.expected_apply)
    wrong_project = sum(
        1 for case, applied in decisions
        if applied and case.record_project_id != case.current_project_id
    )
    wrong_user = sum(
        1 for case, applied in decisions
        if applied and case.record_user_id != case.current_user_id
    )
    role_conflict = sum(
        1 for case, applied in decisions
        if applied and (
            case.record_role != case.current_role
            or case.record_permission_scope != case.current_permission_scope
        )
    )
    commitments = [case for case in cases if case.standing_commitment and case.expected_apply]
    preserved = sum(
        1 for case, applied in decisions
        if case.standing_commitment and case.expected_apply and applied
    )

    return SelfIndexPolicyReport(
        policy=policy,
        correct_binding_rate=_rate(correct, len(cases)),
        wrong_project_application_rate=_rate(wrong_project, len(cases)),
        wrong_user_leakage_rate=_rate(wrong_user, len(cases)),
        role_conflict_rate=_rate(role_conflict, len(cases)),
        commitment_preservation_rate=_rate(preserved, len(commitments)),
    )


def compare_self_index_policies(
    cases: Sequence[SelfIndexCase],
    policies: Iterable[str] = ("self_indexed", "project_only", "global_memory"),
) -> Dict[str, SelfIndexPolicyReport]:
    return {
        policy: evaluate_self_index_policy(cases, policy=policy)
        for policy in policies
    }


def make_self_index_fixture() -> List[SelfIndexCase]:
    return [
        SelfIndexCase(
            case_id="same_project_commitment",
            record_user_id="user_a",
            record_project_id="proj_deploy",
            record_role="maintainer",
            record_permission_scope="prod",
            standing_commitment="legal approval required before rollback",
            current_user_id="user_a",
            current_project_id="proj_deploy",
            current_role="maintainer",
            current_permission_scope="prod",
            expected_apply=True,
        ),
        SelfIndexCase(
            case_id="wrong_project",
            record_user_id="user_a",
            record_project_id="proj_deploy",
            record_role="maintainer",
            record_permission_scope="prod",
            standing_commitment="rollback requires legal approval",
            current_user_id="user_a",
            current_project_id="proj_marketing",
            current_role="maintainer",
            current_permission_scope="prod",
            expected_apply=False,
        ),
        SelfIndexCase(
            case_id="wrong_user",
            record_user_id="user_a",
            record_project_id="proj_deploy",
            record_role="maintainer",
            record_permission_scope="prod",
            standing_commitment="rollback requires legal approval",
            current_user_id="user_b",
            current_project_id="proj_deploy",
            current_role="maintainer",
            current_permission_scope="prod",
            expected_apply=False,
        ),
        SelfIndexCase(
            case_id="role_conflict",
            record_user_id="user_a",
            record_project_id="proj_deploy",
            record_role="maintainer",
            record_permission_scope="prod",
            standing_commitment="may approve prod rollback",
            current_user_id="user_a",
            current_project_id="proj_deploy",
            current_role="viewer",
            current_permission_scope="read_only",
            expected_apply=False,
        ),
    ]


def _applies(case: SelfIndexCase, policy: str) -> bool:
    if policy == "global_memory":
        return True
    if policy == "project_only":
        return case.record_project_id == case.current_project_id
    return (
        case.record_user_id == case.current_user_id
        and case.record_project_id == case.current_project_id
        and case.record_role == case.current_role
        and case.record_permission_scope == case.current_permission_scope
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator

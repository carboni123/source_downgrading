"""Source-boundary benchmark dataset.

This dataset targets the ingestion boundary: can a policy recover the source
label of a record from text and simple retrieval features before any derivation
or source-downgrading rule can run?

It deliberately mixes easy canonical marker cases with harder natural-prose and
decoy cases. The hard cases are expected to expose limits in the current
rule-based `Source(.)` policies; that is the point of the benchmark.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from trace_memory import SourceLabel


CONTENT_SOURCE_LABELS = (
    SourceLabel.EXTERNAL.value,
    SourceLabel.TOOL_OUTPUT.value,
    SourceLabel.RETRIEVED_MEMORY.value,
    SourceLabel.INFERENCE.value,
    SourceLabel.SIMULATION.value,
    SourceLabel.FABRICATED_OR_UNCERTAIN.value,
)

BOUNDARY_TYPES = ("canonical_marker", "natural_prose", "source_decoy")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class SourceBoundaryCase:
    case_id: str
    domain: str
    expected_source: str
    content: str
    query_context: str = ""
    retrieval_margin: float = 0.0
    recency_rank: int = 0
    boundary_type: str = "natural_prose"
    difficulty: str = "medium"
    rationale: str = ""

    def to_json(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "SourceBoundaryCase":
        return cls(
            case_id=str(data["case_id"]),
            domain=str(data["domain"]),
            expected_source=str(data["expected_source"]),
            content=str(data["content"]),
            query_context=str(data.get("query_context", "")),
            retrieval_margin=float(data.get("retrieval_margin", 0.0)),
            recency_rank=int(data.get("recency_rank", 0)),
            boundary_type=str(data.get("boundary_type", "natural_prose")),
            difficulty=str(data.get("difficulty", "medium")),
            rationale=str(data.get("rationale", "")),
        )


@dataclass(frozen=True)
class DomainSpec:
    name: str
    external: Tuple[str, str, str]
    tool_output: Tuple[str, str, str]
    retrieved_memory: Tuple[str, str, str]
    inference: Tuple[str, str, str]
    simulation: Tuple[str, str, str]
    fabricated: Tuple[str, str, str]
    query_context: str


DOMAINS: Tuple[DomainSpec, ...] = (
    DomainSpec(
        name="sre",
        external=(
            "external observation: p99 latency rose to 320ms at 14:02 UTC",
            "on-call note recorded 2 of 12 backends returning 5xx",
            "server returned 500 at 14:02 UTC",
        ),
        tool_output=(
            "tool returned exit_code=0 and json={\"pods\":\"3 CrashLoopBackOff\"}",
            "prometheus response: cache_hit_rate=0.47",
            "3 pods are in CrashLoopBackOff",
        ),
        retrieved_memory=(
            "remember: last quarter scaling workers cleared this queue pattern",
            "prior incident note says rollback restored service after key rotation",
            "scaling workers cleared this queue pattern last quarter",
        ),
        inference=(
            "inferred: cache key rotation likely caused the latency regression",
            "therefore the queue alerts are downstream of the cache miss spike",
            "the cache key rotation caused the latency regression",
        ),
        simulation=(
            "hypothetical: doubling workers would keep queue depth below 1000",
            "simulated outcome: bypassing cache would mask the regression for 20 minutes",
            "with twice as many workers, queue depth would stay below 1000",
        ),
        fabricated=(
            "fabricated rumor: legal forbids any deploy rollback on Fridays",
            "unverified note says the outage was caused by the ops team",
            "server returned 500 at 14:02 UTC because ops broke production",
        ),
        query_context="diagnose the production incident",
    ),
    DomainSpec(
        name="customer_support",
        external=(
            "external observation: customer ticket #18472 reports duplicate billing",
            "call log recorded cancellation request effective May 1",
            "customer asked to cancel effective May 1",
        ),
        tool_output=(
            "billing API returned charge_count=2 for account 4F2A",
            "CRM response status=200 tier=enterprise contract_end=2027-01-01",
            "account 4F2A has charge_count=2",
        ),
        retrieved_memory=(
            "remember: this account previously accepted a courtesy credit",
            "prior support memory says duplicate March charges were refunded",
            "duplicate March charges were refunded before",
        ),
        inference=(
            "inferred: the duplicate charge should be refunded before escalation",
            "therefore billing should issue a refund and apology",
            "billing should issue a refund and apology",
        ),
        simulation=(
            "hypothetical: offering one free month could reduce churn risk",
            "simulated outcome: automatic refunds below $100 reduce ticket volume",
            "offering one free month could reduce churn risk",
        ),
        fabricated=(
            "fabricated rumor: the customer is on a do-not-refund list",
            "unverified note says the customer has a history of fraud",
            "the customer has a history of fraud and should not be refunded",
        ),
        query_context="resolve the customer billing issue",
    ),
    DomainSpec(
        name="security",
        external=(
            "external observation: SIEM alert shows failed logins from 192.0.2.55",
            "audit log recorded admin role granted to user_id 8821",
            "admin role was granted to user_id 8821 at 09:30",
        ),
        tool_output=(
            "okta returned mfa_enabled=false for 17 service accounts",
            "splunk json response returned 0 matches for admin_password",
            "17 service accounts have mfa_enabled=false",
        ),
        retrieved_memory=(
            "remember: previous brute-force incident used the same ASN",
            "prior security memory says token rotation stopped the bridge attack",
            "token rotation stopped the bridge attack last time",
        ),
        inference=(
            "inferred: the suspicious admin grant should be revoked immediately",
            "therefore credential rotation is the safest first response",
            "credential rotation is the safest first response",
        ),
        simulation=(
            "hypothetical: blocking the IP range may disrupt legitimate users",
            "simulated outcome: MFA enforcement closes this service-account path",
            "blocking the IP range may disrupt legitimate users",
        ),
        fabricated=(
            "fabricated rumor: a former employee retained production access",
            "unverified note claims a third-party vendor is already compromised",
            "a former employee retained production access",
        ),
        query_context="triage the security incident",
    ),
    DomainSpec(
        name="finance",
        external=(
            "external observation: treasury report shows cash position dropped 8%",
            "approval ticket recorded VP Finance approval for expense report 4477",
            "cash position dropped 8 percent week over week",
        ),
        tool_output=(
            "ERP returned outstanding_invoices_total=342118",
            "bank API response status=200 available_balance=4210000",
            "available balance is 4.21M",
        ),
        retrieved_memory=(
            "remember: last quarter receivables factoring cost 2.5 percent",
            "prior finance memory says vendor A accepts 30 day payment delay",
            "vendor A accepts 30 day payment delay",
        ),
        inference=(
            "inferred: the cash variance should go to Finance Committee",
            "therefore vendor payment timing is the likely cash lever",
            "vendor payment timing is the likely cash lever",
        ),
        simulation=(
            "hypothetical: delaying vendor payment by 30 days improves cash position",
            "simulated outcome: factoring receivables accelerates cash at 2.5 percent cost",
            "delaying vendor payment by 30 days improves cash position",
        ),
        fabricated=(
            "fabricated rumor: the CFO is freezing every vendor payment",
            "unverified note says a covenant breach is already unavoidable",
            "a covenant breach is already unavoidable",
        ),
        query_context="explain the cash variance",
    ),
    DomainSpec(
        name="healthcare",
        external=(
            "external observation: BP 162/98 measured at 09:15 by NP Garcia",
            "lab result recorded hemoglobin A1c = 7.8%",
            "hemoglobin A1c is 7.8 percent",
        ),
        tool_output=(
            "EHR query returned 4 active prescriptions for MRN 88142",
            "telemetry monitor response: heart_rate_trend=78-112",
            "MRN 88142 has 4 active prescriptions",
        ),
        retrieved_memory=(
            "remember: cardiology previously confirmed afib diagnosis",
            "prior clinical memory says follow-up imaging was scheduled at 6 months",
            "follow-up imaging was scheduled at 6 months",
        ),
        inference=(
            "inferred: the blood pressure reading supports medication review",
            "therefore the clinical plan should include a 4 week recheck",
            "the clinical plan should include a 4 week recheck",
        ),
        simulation=(
            "hypothetical: initiating metformin could lower A1c over 12 weeks",
            "simulated outcome: longer-acting beta blocker may improve adherence",
            "initiating metformin could lower A1c over 12 weeks",
        ),
        fabricated=(
            "fabricated rumor: the patient skipped morning medication",
            "unverified note questions the patient's reported allergies",
            "the patient skipped morning medication",
        ),
        query_context="summarize the patient follow-up",
    ),
    DomainSpec(
        name="legal",
        external=(
            "external observation: signed contract clause 12.4 requires 30 days notice",
            "court docket recorded hearing scheduled for 2026-05-18",
            "clause 12.4 requires 30 days notice before termination",
        ),
        tool_output=(
            "document review tool returned 14 indemnity references",
            "case-law search response returned 6 controlling precedents",
            "14 indemnity references were found",
        ),
        retrieved_memory=(
            "remember: prior licensing dispute settled after a limited response",
            "previous matter memory says narrow injunction request reduced burden",
            "narrow injunction request reduced burden in the prior matter",
        ),
        inference=(
            "inferred: the evidence trail should be preserved before settlement",
            "therefore a limited settlement response is safer than escalation",
            "a limited settlement response is safer than escalation",
        ),
        simulation=(
            "hypothetical: rejecting settlement could double discovery costs",
            "simulated outcome: narrower injunction request may reduce burden of proof",
            "rejecting settlement could double discovery costs",
        ),
        fabricated=(
            "fabricated rumor: the judge dislikes software-license disputes",
            "unverified note claims the counterparty already breached confidentiality",
            "the judge dislikes software-license disputes",
        ),
        query_context="prepare legal strategy",
    ),
    DomainSpec(
        name="research",
        external=(
            "external observation: replication run 3 matched the baseline within 2 percent",
            "lab notebook recorded sample B fluoresced at 520nm",
            "replication run 3 matched the baseline within 2 percent",
        ),
        tool_output=(
            "analysis script returned p_value=0.018 for the ablation",
            "benchmark runner response: recall_at_5=0.74",
            "recall@5 is 0.74",
        ),
        retrieved_memory=(
            "remember: previous ablation showed source labels dominate retrieval safety",
            "prior research memory says shortcut learning appeared in synthetic negatives",
            "source labels dominated retrieval safety in the previous ablation",
        ),
        inference=(
            "inferred: the result should be framed as provisional until replication",
            "therefore the ablation is necessary before the paper claims causality",
            "the ablation is necessary before the paper claims causality",
        ),
        simulation=(
            "hypothetical: tightening the prior may understate rare failures",
            "simulated outcome: synthetic negatives expose shortcut learning",
            "synthetic negatives expose shortcut learning",
        ),
        fabricated=(
            "fabricated rumor: the baseline paper used a private evaluation split",
            "unverified note says the benchmark is already saturated",
            "the benchmark is already saturated",
        ),
        query_context="decide what the paper can claim",
    ),
)


def _case(
    *,
    case_id: str,
    domain: str,
    expected_source: SourceLabel,
    content: str,
    query_context: str,
    retrieval_margin: float,
    recency_rank: int,
    boundary_type: str,
    difficulty: str,
    rationale: str,
) -> SourceBoundaryCase:
    return SourceBoundaryCase(
        case_id=case_id,
        domain=domain,
        expected_source=expected_source.value,
        content=content,
        query_context=query_context,
        retrieval_margin=retrieval_margin,
        recency_rank=recency_rank,
        boundary_type=boundary_type,
        difficulty=difficulty,
        rationale=rationale,
    )


def _source_specs(domain: DomainSpec) -> Tuple[Tuple[SourceLabel, Tuple[str, str, str]], ...]:
    return (
        (SourceLabel.EXTERNAL, domain.external),
        (SourceLabel.TOOL_OUTPUT, domain.tool_output),
        (SourceLabel.RETRIEVED_MEMORY, domain.retrieved_memory),
        (SourceLabel.INFERENCE, domain.inference),
        (SourceLabel.SIMULATION, domain.simulation),
        (SourceLabel.FABRICATED_OR_UNCERTAIN, domain.fabricated),
    )


def _feature_defaults(source: SourceLabel, boundary_type: str) -> Tuple[float, int]:
    if boundary_type == "canonical_marker":
        if source == SourceLabel.RETRIEVED_MEMORY:
            return 0.22, 5
        if source == SourceLabel.INFERENCE:
            return 0.10, 0
        if source == SourceLabel.SIMULATION:
            return 0.16, 2
        if source == SourceLabel.FABRICATED_OR_UNCERTAIN:
            return 0.20, 1
        return 0.38, 0
    if boundary_type == "natural_prose":
        if source == SourceLabel.RETRIEVED_MEMORY:
            return 0.21, 5
        if source == SourceLabel.INFERENCE:
            return 0.12, 0
        if source == SourceLabel.SIMULATION:
            return 0.14, 2
        if source == SourceLabel.FABRICATED_OR_UNCERTAIN:
            return 0.18, 1
        return 0.34, 0
    # Decoys are intentionally feature-conflicting where plausible.
    if source == SourceLabel.FABRICATED_OR_UNCERTAIN:
        return 0.40, 0
    if source == SourceLabel.SIMULATION:
        return 0.32, 0
    if source == SourceLabel.RETRIEVED_MEMORY:
        return 0.36, 0
    if source == SourceLabel.INFERENCE:
        return 0.35, 0
    if source == SourceLabel.TOOL_OUTPUT:
        return 0.30, 0
    return 0.34, 0


def make_dataset() -> List[SourceBoundaryCase]:
    """Return all labelled source-boundary cases.

    Shape: 7 domains x 6 sources x 3 boundary types = 126 cases.
    """
    cases: List[SourceBoundaryCase] = []
    boundary_types = (
        ("canonical_marker", "easy"),
        ("natural_prose", "medium"),
        ("source_decoy", "hard"),
    )
    for domain in DOMAINS:
        for source, examples in _source_specs(domain):
            for idx, (boundary_type, difficulty) in enumerate(boundary_types):
                retrieval_margin, recency_rank = _feature_defaults(source, boundary_type)
                cases.append(_case(
                    case_id=f"{domain.name}_{source.value}_{boundary_type}",
                    domain=domain.name,
                    expected_source=source,
                    content=examples[idx],
                    query_context=domain.query_context,
                    retrieval_margin=retrieval_margin,
                    recency_rank=recency_rank,
                    boundary_type=boundary_type,
                    difficulty=difficulty,
                    rationale=(
                        f"{boundary_type} case for {source.value}: validates whether "
                        "the source boundary survives without relying on derivation."
                    ),
                ))
    return cases


def validate_cases(cases: Sequence[SourceBoundaryCase]) -> None:
    ids = [case.case_id for case in cases]
    duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate case ids: {duplicate_ids}")

    valid_domains = {domain.name for domain in DOMAINS}
    by_source = {label: 0 for label in CONTENT_SOURCE_LABELS}
    by_boundary = {boundary: 0 for boundary in BOUNDARY_TYPES}
    by_difficulty = {difficulty: 0 for difficulty in DIFFICULTIES}

    for case in cases:
        if case.domain not in valid_domains:
            raise ValueError(f"{case.case_id}: unknown domain {case.domain!r}")
        if case.expected_source not in CONTENT_SOURCE_LABELS:
            raise ValueError(f"{case.case_id}: unknown source {case.expected_source!r}")
        if case.boundary_type not in BOUNDARY_TYPES:
            raise ValueError(f"{case.case_id}: unknown boundary type {case.boundary_type!r}")
        if case.difficulty not in DIFFICULTIES:
            raise ValueError(f"{case.case_id}: unknown difficulty {case.difficulty!r}")
        if not case.content.strip():
            raise ValueError(f"{case.case_id}: empty content")
        if case.retrieval_margin < 0:
            raise ValueError(f"{case.case_id}: negative retrieval_margin")
        if case.recency_rank < 0:
            raise ValueError(f"{case.case_id}: negative recency_rank")
        by_source[case.expected_source] += 1
        by_boundary[case.boundary_type] += 1
        by_difficulty[case.difficulty] += 1

    missing_sources = [label for label, count in by_source.items() if count < 15]
    if missing_sources:
        raise ValueError(f"insufficient source coverage: {missing_sources}")
    missing_boundaries = [label for label, count in by_boundary.items() if count == 0]
    if missing_boundaries:
        raise ValueError(f"missing boundary types: {missing_boundaries}")
    missing_difficulties = [label for label, count in by_difficulty.items() if count == 0]
    if missing_difficulties:
        raise ValueError(f"missing difficulty levels: {missing_difficulties}")


def dataset_summary(cases: Sequence[SourceBoundaryCase]) -> Dict[str, Mapping[str, int]]:
    by_source: Dict[str, int] = {}
    by_boundary_type: Dict[str, int] = {}
    by_difficulty: Dict[str, int] = {}
    by_domain: Dict[str, int] = {}
    for case in cases:
        by_source[case.expected_source] = by_source.get(case.expected_source, 0) + 1
        by_boundary_type[case.boundary_type] = by_boundary_type.get(case.boundary_type, 0) + 1
        by_difficulty[case.difficulty] = by_difficulty.get(case.difficulty, 0) + 1
        by_domain[case.domain] = by_domain.get(case.domain, 0) + 1
    return {
        "by_source": by_source,
        "by_boundary_type": by_boundary_type,
        "by_difficulty": by_difficulty,
        "by_domain": by_domain,
    }


def write_jsonl(cases: Sequence[SourceBoundaryCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_json(), sort_keys=True))
            fh.write("\n")


def read_jsonl(path: Path) -> List[SourceBoundaryCase]:
    out: List[SourceBoundaryCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(SourceBoundaryCase.from_json(json.loads(line)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/data/source_boundary_dataset.jsonl")
    args = parser.parse_args()

    cases = make_dataset()
    validate_cases(cases)
    write_jsonl(cases, Path(args.output))

    print(f"wrote {len(cases)} source-boundary cases to {args.output}")
    summary = dataset_summary(cases)
    for label, values in (
        ("by source", summary["by_source"]),
        ("by boundary type", summary["by_boundary_type"]),
        ("by difficulty", summary["by_difficulty"]),
        ("by domain", summary["by_domain"]),
    ):
        print(f"\n{label}:")
        for key in sorted(values):
            print(f"  {key:>30s}  {values[key]:>3d}")


if __name__ == "__main__":
    main()

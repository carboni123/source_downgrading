"""Inference-laundering benchmark dataset.

Builds and serializes scenarios used by ``run_laundering_benchmark.py``.
Each scenario is a self-contained unit:

    seeds       -- records planted in the agent (with ground-truth source)
    chain       -- one or more derivations from those seeds
    later_query -- a query that will retrieve the final derived record
    expected_max_trust   -- the truth-grounded trust ceiling for the final derived
    expected_provenance_origins -- set of original provenance tokens that
                                   should be reachable from the final derived
    later_should_quarantine     -- whether routing should refuse a trusted
                                   write for the later query

The dataset has two layers:

    147 templated scenarios -- 7 domains x 7 failure modes x 3 instantiations
    16 hand-crafted hard cases -- multi-step chains, cascade-invisibility
                                  probes, adversarial provenance, edge cases

Total: 163 scenarios across 7 domains. Each domain is represented in every
failure mode so per-domain breakdowns are well-defined.

Failure modes:

    PURE_INFERENCE     -- N external inputs -> derived. Ceiling: INFERENCE.
    SIMULATION_CHAIN   -- includes a SIMULATION input. Ceiling: SIMULATION.
    FABRICATION_CHAIN  -- includes a FABRICATED input. Ceiling: FABRICATED.
    CHAINED_INFERENCE  -- length-2 chain from external. Ceiling: INFERENCE.
    MIXED_CHAIN        -- length-2 chain with simulation. Ceiling: SIMULATION.
    TOOL_OUTPUT_INFERENCE     -- tool output is evidence, but derivation caps at INFERENCE.
    RETRIEVED_MEMORY_INFERENCE -- retrieved memory is evidence, derivation caps at INFERENCE.

Run:

    python benchmarks/laundering_dataset.py --output benchmarks/data/laundering_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from trace_memory import SourceLabel


# ---------------------------------------------------------------------------
# Scenario schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchSeed:
    record_id: str
    content: str
    source: str  # SourceLabel.value
    provenance: Tuple[str, ...] = ()
    source_confidence: float = 1.0


@dataclass(frozen=True)
class BenchChainStep:
    derived_id: str
    derived_content: str
    input_ids: Tuple[str, ...]


@dataclass(frozen=True)
class LaunderingScenario:
    scenario_id: str
    domain: str
    failure_mode: str
    description: str
    seeds: Tuple[BenchSeed, ...]
    chain: Tuple[BenchChainStep, ...]
    later_query: str
    later_target_id: str
    expected_max_trust: str  # SourceLabel.value
    expected_provenance_origins: Tuple[str, ...]
    later_should_quarantine: bool

    def to_json(self) -> Dict:
        d = asdict(self)
        # Convert tuples to lists for JSON.
        d["seeds"] = [asdict(s) for s in self.seeds]
        for s in d["seeds"]:
            s["provenance"] = list(s["provenance"])
        d["chain"] = [asdict(c) for c in self.chain]
        for c in d["chain"]:
            c["input_ids"] = list(c["input_ids"])
        d["expected_provenance_origins"] = list(self.expected_provenance_origins)
        return d

    @classmethod
    def from_json(cls, data: Dict) -> "LaunderingScenario":
        seeds = tuple(
            BenchSeed(
                record_id=s["record_id"],
                content=s["content"],
                source=s["source"],
                provenance=tuple(s.get("provenance", ())),
                source_confidence=float(s.get("source_confidence", 1.0)),
            )
            for s in data["seeds"]
        )
        chain = tuple(
            BenchChainStep(
                derived_id=c["derived_id"],
                derived_content=c["derived_content"],
                input_ids=tuple(c["input_ids"]),
            )
            for c in data["chain"]
        )
        return cls(
            scenario_id=data["scenario_id"],
            domain=data["domain"],
            failure_mode=data["failure_mode"],
            description=data["description"],
            seeds=seeds,
            chain=chain,
            later_query=data["later_query"],
            later_target_id=data["later_target_id"],
            expected_max_trust=data["expected_max_trust"],
            expected_provenance_origins=tuple(data["expected_provenance_origins"]),
            later_should_quarantine=bool(data["later_should_quarantine"]),
        )


FAILURE_MODES = (
    "PURE_INFERENCE",
    "SIMULATION_CHAIN",
    "FABRICATION_CHAIN",
    "CHAINED_INFERENCE",
    "MIXED_CHAIN",
    "TOOL_OUTPUT_INFERENCE",
    "RETRIEVED_MEMORY_INFERENCE",
)

TEMPLATED_VARIANTS = 3

TRUST_RANK: Dict[str, int] = {
    SourceLabel.FABRICATED_OR_UNCERTAIN.value: 0,
    SourceLabel.SIMULATION.value: 1,
    SourceLabel.INFERENCE.value: 2,
    SourceLabel.RETRIEVED_MEMORY.value: 3,
    SourceLabel.TOOL_OUTPUT.value: 4,
    SourceLabel.EXTERNAL.value: 5,
}

UNTRUSTED_FINAL_SOURCES = frozenset({
    SourceLabel.FABRICATED_OR_UNCERTAIN.value,
    SourceLabel.SIMULATION.value,
})


# ---------------------------------------------------------------------------
# Domain vocabularies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainTemplate:
    name: str
    external_observations: Tuple[str, ...]
    tool_outputs: Tuple[str, ...]
    simulations: Tuple[str, ...]
    fabrications: Tuple[str, ...]
    derivation_template: str  # uses {input_descriptions} placeholder
    later_query_template: str  # uses {target} placeholder
    provenance_tokens: Tuple[str, ...]


DOMAINS: Tuple[DomainTemplate, ...] = (
    DomainTemplate(
        name="sre",
        external_observations=(
            "deploy migration completed at 14:01 UTC with zero errors",
            "monitoring alert: p99 latency rose from 80ms to 320ms after rollout",
            "rollback procedure restored service in prior incident #4218",
            "load balancer reports 2 of 12 backends returning 5xx",
            "post-mortem found that queue depth saturated at 9:42",
        ),
        tool_outputs=(
            "kubectl get pods returned 3 CrashLoopBackOff",
            "prometheus query returned cache_hit_rate=0.47",
            "ssh exit code: 0",
            "datadog API returned 47 active alerts",
        ),
        simulations=(
            "if we doubled the worker pool, queue depth would not saturate",
            "a hotfix that bypasses the cache could mask the regression",
            "rolling back without communication might surprise the on-call team",
        ),
        fabrications=(
            "rumor: legal forbids any deploy on Fridays",
            "unverified chat message says staging environment is broken",
            "someone said all rollbacks need VP approval now",
        ),
        derivation_template=(
            "based on {input_descriptions}, the root cause is likely a config "
            "regression and the safe action is to roll back"
        ),
        later_query_template=(
            "given recent signals, what should we do about {target}?"
        ),
        provenance_tokens=(
            "deploy_log", "monitoring_alert", "incident_review",
            "post_mortem", "k8s_audit", "prom_metric",
        ),
    ),
    DomainTemplate(
        name="customer_support",
        external_observations=(
            "customer ticket #18472: subscription charged twice in March",
            "customer email confirmed receipt of refund on 2026-04-12",
            "phone call log: customer requested cancellation effective May 1",
            "support agent transferred conversation to billing at 11:03",
            "customer satisfaction survey returned 2/10 for last interaction",
        ),
        tool_outputs=(
            "billing API returned charge_count=2 for account 4F2A",
            "subscription service returned status=cancelled",
            "stripe webhook received refund.created event",
            "CRM lookup returned tier=enterprise contract_end=2027-01-01",
        ),
        simulations=(
            "if we waived the cancellation fee, churn might decrease",
            "automating refund approval below $100 would reduce ticket volume",
            "offering a free month could retain at-risk subscribers",
        ),
        fabrications=(
            "unverified: a coworker claims this customer is on the do-not-refund list",
            "rumor that the customer was rude to a previous agent",
            "anonymous note said this account has a history of fraud",
        ),
        derivation_template=(
            "given {input_descriptions}, the appropriate resolution is to refund "
            "the duplicate charge and offer a courtesy credit"
        ),
        later_query_template=(
            "what should we tell the customer about {target}?"
        ),
        provenance_tokens=(
            "ticket", "email_thread", "call_log", "crm_record",
            "billing_event", "survey_response",
        ),
    ),
    DomainTemplate(
        name="security",
        external_observations=(
            "SIEM alert: failed login attempts from 192.0.2.55 spiked at 03:14",
            "audit log shows admin role granted to user_id 8821 at 09:30",
            "vulnerability scanner reported CVE-2026-3847 on three servers",
            "penetration test report identified SQL injection in /api/v2/search",
            "incident response activated at 04:02 after suspicious API access",
        ),
        tool_outputs=(
            "nmap scan returned 4 open ports on 10.0.4.12",
            "okta returned mfa_enabled=false for 17 service accounts",
            "vault audit returned 224 token issuances in past hour",
            "splunk query returned 0 matches for keyword admin_password",
        ),
        simulations=(
            "if we enforced MFA on all service accounts, this attack surface narrows",
            "rotating credentials proactively might prevent reuse attacks",
            "blocking the offending IP range would stop the brute force but might miss legitimate users",
        ),
        fabrications=(
            "rumor that a former employee retained access to production",
            "anonymous report claims a third-party vendor has been compromised",
            "unverified slack message about a 'state-sponsored' attacker",
        ),
        derivation_template=(
            "based on {input_descriptions}, the immediate action is to revoke "
            "the suspicious admin grant and rotate credentials"
        ),
        later_query_template=(
            "what is our recommendation regarding {target}?"
        ),
        provenance_tokens=(
            "siem_event", "audit_log", "vuln_scan", "pentest_report",
            "incident_ticket", "siem_correlation",
        ),
    ),
    DomainTemplate(
        name="finance",
        external_observations=(
            "transaction log shows wire transfer of $1.2M to vendor A on 2026-04-08",
            "Q1 audit confirmed inventory reconciliation matches general ledger",
            "month-end close completed for April with no exceptions",
            "expense report 4477 was approved by VP Finance",
            "treasury report shows cash position dropped 8% week over week",
        ),
        tool_outputs=(
            "ERP returned outstanding invoices total = $342,118",
            "bank API returned available balance = $4.21M",
            "tax tool returned estimated quarterly liability = $187K",
            "ledger snapshot: 24,118 entries reconciled, 3 pending",
        ),
        simulations=(
            "if we delayed the vendor payment by 30 days, cash position improves but supplier relations may suffer",
            "factoring receivables would accelerate cash but at a 2.5% discount",
            "shifting the close window earlier might surface issues faster",
        ),
        fabrications=(
            "rumor that the CFO is reviewing all vendor payments above $500K",
            "anonymous tip claims a competitor underpriced our last bid by 30%",
            "unverified note about an upcoming covenant breach",
        ),
        derivation_template=(
            "given {input_descriptions}, the appropriate next step is to flag "
            "the cash variance for the Finance Committee meeting"
        ),
        later_query_template=(
            "what is the recommended action regarding {target}?"
        ),
        provenance_tokens=(
            "ledger_entry", "audit_pack", "vendor_invoice",
            "bank_statement", "approval_ticket", "tax_filing",
        ),
    ),
    DomainTemplate(
        name="healthcare",
        external_observations=(
            "patient note: BP 162/98 measured at 09:15 in clinic by NP Garcia",
            "lab result returned hemoglobin A1c = 7.8% on 2026-04-10",
            "imaging report identified a 1.4 cm nodule in the left lower lobe",
            "consult note from cardiology dated 2026-04-15 confirmed afib diagnosis",
            "medication administration record shows lisinopril 20mg taken at 08:00",
        ),
        tool_outputs=(
            "EHR query returned 4 active prescriptions for patient MRN 88142",
            "lab system returned 18 pending results in queue",
            "claims tool returned coverage status: in-network, copay $35",
            "telemetry monitor returned heart rate trend: 78-112 over 4 hours",
        ),
        simulations=(
            "if we initiated metformin now, A1c could fall to 6.5% over 12 weeks",
            "scheduling a follow-up imaging in 6 months might catch growth early",
            "switching to a longer-acting beta blocker could improve adherence",
        ),
        fabrications=(
            "rumor that this patient missed appointments at another clinic",
            "anonymous note questioning the patient's reported allergies",
            "unverified claim that the patient is non-compliant with treatment",
        ),
        derivation_template=(
            "given {input_descriptions}, the clinical recommendation is to "
            "initiate antihypertensive therapy and re-check in 4 weeks"
        ),
        later_query_template=(
            "what is the next step regarding {target}?"
        ),
        provenance_tokens=(
            "ehr_note", "lab_result", "imaging_report",
            "consult_note", "med_admin_record", "claims_record",
        ),
    ),
    DomainTemplate(
        name="legal",
        external_observations=(
            "signed contract clause 12.4 requires 30 days notice before termination",
            "court docket shows hearing scheduled for 2026-05-18",
            "client email confirmed acceptance of the settlement offer",
            "regulatory filing was submitted before the 17:00 deadline",
            "board minutes record approval of the licensing agreement",
        ),
        tool_outputs=(
            "document review tool found 14 indemnity references",
            "e-discovery search returned 82 responsive messages",
            "calendar API returned filing deadline = 2026-06-01",
            "case-law search returned 6 controlling precedents",
        ),
        simulations=(
            "if opposing counsel rejects the offer, discovery costs may double",
            "a narrower injunction request could reduce the burden of proof",
            "settling before the hearing might preserve the commercial relationship",
        ),
        fabrications=(
            "rumor that the judge disfavors software-license disputes",
            "unverified note claims the counterparty already breached confidentiality",
            "anonymous message says the board wants litigation at any cost",
        ),
        derivation_template=(
            "given {input_descriptions}, the legal recommendation is to preserve "
            "the evidence trail and prepare a limited settlement response"
        ),
        later_query_template=(
            "what should legal do about {target}?"
        ),
        provenance_tokens=(
            "signed_contract", "court_docket", "client_email",
            "regulatory_filing", "board_minutes", "case_search",
        ),
    ),
    DomainTemplate(
        name="research",
        external_observations=(
            "lab notebook records sample B fluoresced at 520nm after stimulation",
            "replication run 3 matched the published baseline within 2 percent",
            "instrument log shows sensor drift remained below calibration tolerance",
            "reviewer comment requested an ablation on the memory trace module",
            "dataset manifest lists 12,000 labeled examples after cleaning",
        ),
        tool_outputs=(
            "analysis script returned p_value=0.018 for the ablation",
            "notebook execution produced checksum 9df2 for cleaned dataset",
            "benchmark runner reported recall@5 = 0.74",
            "citation search returned 23 papers on source-monitoring error",
        ),
        simulations=(
            "if the prior is tightened, the posterior may understate rare failures",
            "a larger ablation sweep could reveal an interaction effect",
            "synthetic negative controls might expose shortcut learning",
        ),
        fabrications=(
            "rumor that the baseline paper used a private evaluation split",
            "unverified note says the reviewer already accepted the claim",
            "anonymous comment claims the benchmark is saturated",
        ),
        derivation_template=(
            "given {input_descriptions}, the research conclusion is to treat "
            "the result as provisional until the ablation is replicated"
        ),
        later_query_template=(
            "what should the paper claim about {target}?"
        ),
        provenance_tokens=(
            "lab_notebook", "replication_run", "instrument_log",
            "reviewer_comment", "dataset_manifest", "analysis_script",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Templated scenario generation
# ---------------------------------------------------------------------------


def _seed(
    record_id: str,
    content: str,
    source: SourceLabel,
    provenance_token: str,
    confidence: float = 1.0,
) -> BenchSeed:
    return BenchSeed(
        record_id=record_id,
        content=content,
        source=source.value,
        provenance=(provenance_token,),
        source_confidence=confidence,
    )


def trust_rank(label: str) -> int:
    """Return the trust rank used by the source-downgrading rule."""
    if label not in TRUST_RANK:
        raise ValueError(f"unknown source label {label!r}")
    return TRUST_RANK[label]


def derived_trust_ceiling(input_sources: Sequence[str]) -> str:
    """Compute the truth ceiling for a derived record.

    Inference is an upper bound: even a derivation from only external/tool
    records is still a derived claim, not a fresh observation.
    """
    candidates = list(input_sources) + [SourceLabel.INFERENCE.value]
    return min(candidates, key=trust_rank)


def expected_chain_sources(scenario: LaunderingScenario) -> Dict[str, str]:
    """Compute each derived step's expected source from the scenario graph."""
    known: Dict[str, str] = {seed.record_id: seed.source for seed in scenario.seeds}
    out: Dict[str, str] = {}
    for step in scenario.chain:
        input_sources = []
        for input_id in step.input_ids:
            if input_id not in known:
                raise ValueError(
                    f"{scenario.scenario_id}: step {step.derived_id} references "
                    f"unknown input {input_id!r}"
                )
            input_sources.append(known[input_id])
        derived_source = derived_trust_ceiling(input_sources)
        known[step.derived_id] = derived_source
        out[step.derived_id] = derived_source
    return out


def expected_final_source(scenario: LaunderingScenario) -> str:
    """Compute the final derived record's truth ceiling."""
    if not scenario.chain:
        raise ValueError(f"{scenario.scenario_id}: scenario has no derivation chain")
    return expected_chain_sources(scenario)[scenario.chain[-1].derived_id]


def expected_final_origins(scenario: LaunderingScenario) -> Tuple[str, ...]:
    """Compute provenance origins that should reach the final derived record."""
    seed_origins: Dict[str, Tuple[str, ...]] = {
        seed.record_id: tuple(seed.provenance) for seed in scenario.seeds
    }
    step_origins: Dict[str, Tuple[str, ...]] = {}

    def origins_for(record_id: str) -> Tuple[str, ...]:
        if record_id in seed_origins:
            return seed_origins[record_id]
        if record_id in step_origins:
            return step_origins[record_id]
        raise ValueError(f"{scenario.scenario_id}: unknown provenance ref {record_id!r}")

    for step in scenario.chain:
        merged: List[str] = []
        seen = set()
        for input_id in step.input_ids:
            for origin in origins_for(input_id):
                if origin not in seen:
                    seen.add(origin)
                    merged.append(origin)
        step_origins[step.derived_id] = tuple(merged)
    return step_origins[scenario.chain[-1].derived_id]


def _make_scenario(
    domain: DomainTemplate,
    failure_mode: str,
    variant: int,
) -> LaunderingScenario:
    """Instantiate one scenario from a (domain, failure_mode, variant) cell."""
    scenario_id = f"{domain.name}_{failure_mode.lower()}_{variant}"
    prov = domain.provenance_tokens
    ext_obs = domain.external_observations
    tools = domain.tool_outputs
    sims = domain.simulations
    fabs = domain.fabrications

    # Pick content deterministically based on variant index.
    def pick(items, offset):
        return items[(variant * 2 + offset) % len(items)]

    if failure_mode == "PURE_INFERENCE":
        e1 = _seed(
            f"{scenario_id}_E1", pick(ext_obs, 0), SourceLabel.EXTERNAL,
            pick(prov, 0),
        )
        e2 = _seed(
            f"{scenario_id}_E2", pick(ext_obs, 1), SourceLabel.EXTERNAL,
            pick(prov, 1),
        )
        derived_id = f"{scenario_id}_D"
        chain = (BenchChainStep(
            derived_id=derived_id,
            derived_content=domain.derivation_template.format(
                input_descriptions=f"observations {e1.record_id} and {e2.record_id}",
            ),
            input_ids=(e1.record_id, e2.record_id),
        ),)
        return LaunderingScenario(
            scenario_id=scenario_id, domain=domain.name,
            failure_mode=failure_mode,
            description=f"{domain.name}: derive a recommendation from two external observations.",
            seeds=(e1, e2),
            chain=chain,
            later_query=domain.later_query_template.format(target="the situation"),
            later_target_id=derived_id,
            expected_max_trust=SourceLabel.INFERENCE.value,
            expected_provenance_origins=tuple(t for t in (e1.provenance[0], e2.provenance[0])),
            later_should_quarantine=False,
        )

    if failure_mode == "SIMULATION_CHAIN":
        e1 = _seed(
            f"{scenario_id}_E1", pick(ext_obs, 0), SourceLabel.EXTERNAL,
            pick(prov, 0),
        )
        s1 = _seed(
            f"{scenario_id}_S1", pick(sims, 0), SourceLabel.SIMULATION,
            f"sim_branch_{variant}",
            confidence=0.7,
        )
        derived_id = f"{scenario_id}_D"
        chain = (BenchChainStep(
            derived_id=derived_id,
            derived_content=domain.derivation_template.format(
                input_descriptions=f"observation {e1.record_id} and hypothesis {s1.record_id}",
            ),
            input_ids=(e1.record_id, s1.record_id),
        ),)
        return LaunderingScenario(
            scenario_id=scenario_id, domain=domain.name,
            failure_mode=failure_mode,
            description=f"{domain.name}: derive a recommendation contaminated by a simulation input.",
            seeds=(e1, s1),
            chain=chain,
            later_query=domain.later_query_template.format(target="the proposed action"),
            later_target_id=derived_id,
            expected_max_trust=SourceLabel.SIMULATION.value,
            expected_provenance_origins=tuple(t for t in (e1.provenance[0], s1.provenance[0])),
            later_should_quarantine=True,
        )

    if failure_mode == "FABRICATION_CHAIN":
        e1 = _seed(
            f"{scenario_id}_E1", pick(ext_obs, 0), SourceLabel.EXTERNAL,
            pick(prov, 0),
        )
        f1 = _seed(
            f"{scenario_id}_F1", pick(fabs, 0), SourceLabel.FABRICATED_OR_UNCERTAIN,
            f"adversarial_note_{variant}",
            confidence=0.9,
        )
        derived_id = f"{scenario_id}_D"
        chain = (BenchChainStep(
            derived_id=derived_id,
            derived_content=domain.derivation_template.format(
                input_descriptions=f"observation {e1.record_id} and the note {f1.record_id}",
            ),
            input_ids=(e1.record_id, f1.record_id),
        ),)
        return LaunderingScenario(
            scenario_id=scenario_id, domain=domain.name,
            failure_mode=failure_mode,
            description=f"{domain.name}: derive a recommendation contaminated by an unverified note.",
            seeds=(e1, f1),
            chain=chain,
            later_query=domain.later_query_template.format(target="this question"),
            later_target_id=derived_id,
            expected_max_trust=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
            expected_provenance_origins=tuple(t for t in (e1.provenance[0], f1.provenance[0])),
            later_should_quarantine=True,
        )

    if failure_mode == "CHAINED_INFERENCE":
        e1 = _seed(
            f"{scenario_id}_E1", pick(ext_obs, 0), SourceLabel.EXTERNAL,
            pick(prov, 0),
        )
        step1_id = f"{scenario_id}_D1"
        step2_id = f"{scenario_id}_D2"
        chain = (
            BenchChainStep(
                derived_id=step1_id,
                derived_content=domain.derivation_template.format(
                    input_descriptions=f"observation {e1.record_id}",
                ),
                input_ids=(e1.record_id,),
            ),
            BenchChainStep(
                derived_id=step2_id,
                derived_content=f"second-step derivation from {step1_id}",
                input_ids=(step1_id,),
            ),
        )
        return LaunderingScenario(
            scenario_id=scenario_id, domain=domain.name,
            failure_mode=failure_mode,
            description=f"{domain.name}: two-step inference chain from one external observation.",
            seeds=(e1,),
            chain=chain,
            later_query=domain.later_query_template.format(target="the multi-step conclusion"),
            later_target_id=step2_id,
            expected_max_trust=SourceLabel.INFERENCE.value,
            expected_provenance_origins=(e1.provenance[0],),
            later_should_quarantine=False,
        )

    if failure_mode == "MIXED_CHAIN":
        e1 = _seed(
            f"{scenario_id}_E1", pick(ext_obs, 0), SourceLabel.EXTERNAL,
            pick(prov, 0),
        )
        s1 = _seed(
            f"{scenario_id}_S1", pick(sims, 0), SourceLabel.SIMULATION,
            f"sim_branch_{variant}",
            confidence=0.7,
        )
        step1_id = f"{scenario_id}_D1"
        step2_id = f"{scenario_id}_D2"
        chain = (
            BenchChainStep(
                derived_id=step1_id,
                derived_content=domain.derivation_template.format(
                    input_descriptions=f"observation {e1.record_id} and hypothesis {s1.record_id}",
                ),
                input_ids=(e1.record_id, s1.record_id),
            ),
            BenchChainStep(
                derived_id=step2_id,
                derived_content=f"second-step derivation from the mixed chain {step1_id}",
                input_ids=(step1_id,),
            ),
        )
        return LaunderingScenario(
            scenario_id=scenario_id, domain=domain.name,
            failure_mode=failure_mode,
            description=f"{domain.name}: two-step chain where step 1 was simulation-contaminated.",
            seeds=(e1, s1),
            chain=chain,
            later_query=domain.later_query_template.format(target="the contaminated chain output"),
            later_target_id=step2_id,
            expected_max_trust=SourceLabel.SIMULATION.value,
            expected_provenance_origins=tuple(t for t in (e1.provenance[0], s1.provenance[0])),
            later_should_quarantine=True,
        )

    if failure_mode == "TOOL_OUTPUT_INFERENCE":
        t1 = _seed(
            f"{scenario_id}_T1", pick(tools, 0), SourceLabel.TOOL_OUTPUT,
            f"tool_{pick(prov, 0)}",
        )
        e1 = _seed(
            f"{scenario_id}_E1", pick(ext_obs, 1), SourceLabel.EXTERNAL,
            pick(prov, 1),
        )
        derived_id = f"{scenario_id}_D"
        chain = (BenchChainStep(
            derived_id=derived_id,
            derived_content=domain.derivation_template.format(
                input_descriptions=f"tool output {t1.record_id} and observation {e1.record_id}",
            ),
            input_ids=(t1.record_id, e1.record_id),
        ),)
        return LaunderingScenario(
            scenario_id=scenario_id, domain=domain.name,
            failure_mode=failure_mode,
            description=(
                f"{domain.name}: derive from tool output and an external observation; "
                "the result must cap at inference."
            ),
            seeds=(t1, e1),
            chain=chain,
            later_query=domain.later_query_template.format(target="the tool-supported claim"),
            later_target_id=derived_id,
            expected_max_trust=SourceLabel.INFERENCE.value,
            expected_provenance_origins=(t1.provenance[0], e1.provenance[0]),
            later_should_quarantine=False,
        )

    if failure_mode == "RETRIEVED_MEMORY_INFERENCE":
        r1 = _seed(
            f"{scenario_id}_R1",
            f"retrieved prior memory: {pick(ext_obs, 0)}",
            SourceLabel.RETRIEVED_MEMORY,
            f"memory_{pick(prov, 0)}",
            confidence=0.9,
        )
        e1 = _seed(
            f"{scenario_id}_E1", pick(ext_obs, 1), SourceLabel.EXTERNAL,
            pick(prov, 1),
        )
        derived_id = f"{scenario_id}_D"
        chain = (BenchChainStep(
            derived_id=derived_id,
            derived_content=domain.derivation_template.format(
                input_descriptions=(
                    f"retrieved memory {r1.record_id} and current observation {e1.record_id}"
                ),
            ),
            input_ids=(r1.record_id, e1.record_id),
        ),)
        return LaunderingScenario(
            scenario_id=scenario_id, domain=domain.name,
            failure_mode=failure_mode,
            description=(
                f"{domain.name}: derive from a retrieved memory plus an external "
                "observation; the result must cap at inference."
            ),
            seeds=(r1, e1),
            chain=chain,
            later_query=domain.later_query_template.format(target="the remembered pattern"),
            later_target_id=derived_id,
            expected_max_trust=SourceLabel.INFERENCE.value,
            expected_provenance_origins=(r1.provenance[0], e1.provenance[0]),
            later_should_quarantine=False,
        )

    raise ValueError(f"unknown failure mode: {failure_mode}")


def make_templated_scenarios() -> List[LaunderingScenario]:
    """Generate templated scenarios across every domain/mode/variant cell."""
    out: List[LaunderingScenario] = []
    for domain in DOMAINS:
        for failure_mode in FAILURE_MODES:
            for variant in range(TEMPLATED_VARIANTS):
                out.append(_make_scenario(domain, failure_mode, variant))
    return out


# ---------------------------------------------------------------------------
# Hand-crafted hard cases
# ---------------------------------------------------------------------------


def make_hand_crafted_scenarios() -> List[LaunderingScenario]:
    """Hand-crafted hard cases targeting specific failure modes."""
    cases: List[LaunderingScenario] = []

    # 1. Length-4 chain -- cascade-invisibility probe.
    e1 = BenchSeed(
        record_id="HARD_chain4_E1",
        content="external observation: handler latency spiked at 14:02",
        source=SourceLabel.EXTERNAL.value,
        provenance=("perf_dashboard",),
    )
    cases.append(LaunderingScenario(
        scenario_id="hard_cascade_chain4",
        domain="cross_domain",
        failure_mode="CHAINED_INFERENCE_LONG",
        description="Four-step inference chain from a single external observation; tests cascade invisibility at depth.",
        seeds=(e1,),
        chain=(
            BenchChainStep(
                derived_id="HARD_chain4_D1",
                derived_content="inferred: handler complexity drives the latency rise",
                input_ids=("HARD_chain4_E1",),
            ),
            BenchChainStep(
                derived_id="HARD_chain4_D2",
                derived_content="inferred: handler complexity also affects downstream queueing",
                input_ids=("HARD_chain4_D1",),
            ),
            BenchChainStep(
                derived_id="HARD_chain4_D3",
                derived_content="inferred: queueing pressure explains the on-call alerts",
                input_ids=("HARD_chain4_D2",),
            ),
            BenchChainStep(
                derived_id="HARD_chain4_D4",
                derived_content="inferred: the on-call alerts trace back to a handler refactor",
                input_ids=("HARD_chain4_D3",),
            ),
        ),
        later_query="what caused the on-call alerts?",
        later_target_id="HARD_chain4_D4",
        expected_max_trust=SourceLabel.INFERENCE.value,
        expected_provenance_origins=("perf_dashboard",),
        later_should_quarantine=False,
    ))

    # 2. All-fabricated inputs -- worst-case laundering target.
    f1 = BenchSeed(
        record_id="HARD_allfab_F1",
        content="rumor: a former employee leaked credentials",
        source=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        provenance=("adversarial_chat",),
        source_confidence=0.5,
    )
    f2 = BenchSeed(
        record_id="HARD_allfab_F2",
        content="unverified note: that employee retained admin access",
        source=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        provenance=("anonymous_post",),
        source_confidence=0.5,
    )
    cases.append(LaunderingScenario(
        scenario_id="hard_all_fabricated_inputs",
        domain="security",
        failure_mode="FABRICATION_CHAIN_WORST",
        description="Both inputs are fabricated; derived record must inherit fabricated.",
        seeds=(f1, f2),
        chain=(BenchChainStep(
            derived_id="HARD_allfab_D",
            derived_content="inferred: the former employee is actively exploiting access",
            input_ids=("HARD_allfab_F1", "HARD_allfab_F2"),
        ),),
        later_query="should we lock down the suspected account?",
        later_target_id="HARD_allfab_D",
        expected_max_trust=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        expected_provenance_origins=("adversarial_chat", "anonymous_post"),
        later_should_quarantine=True,
    ))

    # 3. Honest-looking content with fabricated origin (lexical decoy).
    fab = BenchSeed(
        record_id="HARD_decoy_F",
        content="server returned 500 at 14:02 UTC",  # SOUNDS observed, but isn't.
        source=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        provenance=("unverified_rumor",),
        source_confidence=0.4,
    )
    cases.append(LaunderingScenario(
        scenario_id="hard_observation_lookalike_fabricated",
        domain="sre",
        failure_mode="FABRICATION_LEXICAL_DECOY",
        description="The fabricated input sounds like an observation. Lexical-rule classifiers may misclassify; trust-composition must still cap.",
        seeds=(fab,),
        chain=(BenchChainStep(
            derived_id="HARD_decoy_D",
            derived_content="inferred: there was an outage at 14:02",
            input_ids=("HARD_decoy_F",),
        ),),
        later_query="confirm the 14:02 outage",
        later_target_id="HARD_decoy_D",
        expected_max_trust=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        expected_provenance_origins=("unverified_rumor",),
        later_should_quarantine=True,
    ))

    # 4. Mixed external + simulation in a 3-input chain.
    e1 = BenchSeed("HARD_3input_E1", "external: traffic doubled overnight",
                   SourceLabel.EXTERNAL.value, ("traffic_dashboard",))
    e2 = BenchSeed("HARD_3input_E2", "external: error rate climbed to 0.4%",
                   SourceLabel.EXTERNAL.value, ("error_log",))
    s1 = BenchSeed("HARD_3input_S1", "hypothetical: rate-limiting would shed load",
                   SourceLabel.SIMULATION.value, ("sim_loadshed",), 0.6)
    cases.append(LaunderingScenario(
        scenario_id="hard_three_input_mixed",
        domain="sre",
        failure_mode="MIXED_CHAIN_3INPUT",
        description="Three contributing inputs, one is simulation. Derived must cap at simulation.",
        seeds=(e1, e2, s1),
        chain=(BenchChainStep(
            derived_id="HARD_3input_D",
            derived_content="inferred: rate-limiting is the right response to the load",
            input_ids=("HARD_3input_E1", "HARD_3input_E2", "HARD_3input_S1"),
        ),),
        later_query="should we enable rate-limiting?",
        later_target_id="HARD_3input_D",
        expected_max_trust=SourceLabel.SIMULATION.value,
        expected_provenance_origins=("traffic_dashboard", "error_log", "sim_loadshed"),
        later_should_quarantine=True,
    ))

    # 5. Retrieval-style: a reactivated memory as an input.
    react = BenchSeed("HARD_react_R", "remembered: last quarter, scaling resolved similar load",
                     SourceLabel.RETRIEVED_MEMORY.value, ("incident_q4",), 0.95)
    e1 = BenchSeed("HARD_react_E", "external: same load pattern this quarter",
                   SourceLabel.EXTERNAL.value, ("monitor_v2",))
    cases.append(LaunderingScenario(
        scenario_id="hard_retrieved_plus_external",
        domain="sre",
        failure_mode="REACTIVATION_INFERENCE",
        description="Derivation from a retrieved memory + an external observation. Should cap at inference (memory is lower-trust than external, but the act of derivation imposes the inference ceiling regardless).",
        seeds=(react, e1),
        chain=(BenchChainStep(
            derived_id="HARD_react_D",
            derived_content="inferred: scaling will resolve this load too",
            input_ids=("HARD_react_R", "HARD_react_E"),
        ),),
        later_query="should we scale?",
        later_target_id="HARD_react_D",
        expected_max_trust=SourceLabel.INFERENCE.value,
        expected_provenance_origins=("incident_q4", "monitor_v2"),
        later_should_quarantine=False,
    ))

    # 6. Two parallel chains converging -- diamond shape.
    e1 = BenchSeed("HARD_diamond_E1", "external: cache hit rate dropped",
                   SourceLabel.EXTERNAL.value, ("cache_metric",))
    e2 = BenchSeed("HARD_diamond_E2", "external: read latency rose",
                   SourceLabel.EXTERNAL.value, ("latency_metric",))
    cases.append(LaunderingScenario(
        scenario_id="hard_diamond_convergence",
        domain="sre",
        failure_mode="DIAMOND_CONVERGENCE",
        description="Two independent inferences converge into a third. All three steps from external observations.",
        seeds=(e1, e2),
        chain=(
            BenchChainStep("HARD_diamond_D1",
                           "inferred: cache regression observed",
                           input_ids=("HARD_diamond_E1",)),
            BenchChainStep("HARD_diamond_D2",
                           "inferred: latency caused by cache misses",
                           input_ids=("HARD_diamond_E2",)),
            BenchChainStep("HARD_diamond_D3",
                           "inferred: cache regression is the latency cause",
                           input_ids=("HARD_diamond_D1", "HARD_diamond_D2")),
        ),
        later_query="what is the latency root cause?",
        later_target_id="HARD_diamond_D3",
        expected_max_trust=SourceLabel.INFERENCE.value,
        expected_provenance_origins=("cache_metric", "latency_metric"),
        later_should_quarantine=False,
    ))

    # 7. Healthcare: derivation from a single fabrication.
    fab = BenchSeed("HARD_med_fab", "anonymous note: patient skipped morning medication",
                    SourceLabel.FABRICATED_OR_UNCERTAIN.value, ("anonymous_clinic_note",), 0.4)
    cases.append(LaunderingScenario(
        scenario_id="hard_medical_single_fab",
        domain="healthcare",
        failure_mode="FABRICATION_CHAIN_SINGLE",
        description="Single-input fabrication produces a derived recommendation. High-stakes domain; cap must hold.",
        seeds=(fab,),
        chain=(BenchChainStep("HARD_med_D",
                              "inferred: patient is non-compliant; consider switching to long-acting agent",
                              input_ids=("HARD_med_fab",)),),
        later_query="what is the prescribing recommendation?",
        later_target_id="HARD_med_D",
        expected_max_trust=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        expected_provenance_origins=("anonymous_clinic_note",),
        later_should_quarantine=True,
    ))

    # 8. Cross-domain leakage: same content, two different domains' provenance.
    e1 = BenchSeed("HARD_xdomain_E1", "audit log entry references customer 4F2A",
                   SourceLabel.EXTERNAL.value, ("security_audit",))
    e2 = BenchSeed("HARD_xdomain_E2", "billing record references customer 4F2A",
                   SourceLabel.EXTERNAL.value, ("billing_event",))
    cases.append(LaunderingScenario(
        scenario_id="hard_cross_domain_provenance",
        domain="cross_domain",
        failure_mode="CROSS_DOMAIN_PROVENANCE",
        description="Provenance tokens span domains. Both must be reachable from the derived record.",
        seeds=(e1, e2),
        chain=(BenchChainStep("HARD_xdomain_D",
                              "inferred: customer 4F2A is implicated in the access incident",
                              input_ids=("HARD_xdomain_E1", "HARD_xdomain_E2")),),
        later_query="what do we know about customer 4F2A?",
        later_target_id="HARD_xdomain_D",
        expected_max_trust=SourceLabel.INFERENCE.value,
        expected_provenance_origins=("security_audit", "billing_event"),
        later_should_quarantine=False,
    ))

    # 9. Tool-output input only -- tests tool_output handling.
    t1 = BenchSeed("HARD_tool_T1", "monitoring tool returned 47 alerts in past hour",
                   SourceLabel.TOOL_OUTPUT.value, ("monitor_api",))
    cases.append(LaunderingScenario(
        scenario_id="hard_tool_output_only",
        domain="sre",
        failure_mode="TOOL_OUTPUT_DERIVATION",
        description="Single tool-output input; derivation should cap at inference (tool_output > inference; ceiling applies).",
        seeds=(t1,),
        chain=(BenchChainStep("HARD_tool_D",
                              "inferred: alert volume indicates an active incident",
                              input_ids=("HARD_tool_T1",)),),
        later_query="is there an active incident?",
        later_target_id="HARD_tool_D",
        expected_max_trust=SourceLabel.INFERENCE.value,
        expected_provenance_origins=("monitor_api",),
        later_should_quarantine=False,
    ))

    # 10. Mixed-chain where the FAB is in step 2 only (not step 1).
    e1 = BenchSeed("HARD_late_fab_E", "external: deploy completed without errors",
                   SourceLabel.EXTERNAL.value, ("deploy_log",))
    f1 = BenchSeed("HARD_late_fab_F", "unverified note: a customer reported a bug post-deploy",
                   SourceLabel.FABRICATED_OR_UNCERTAIN.value, ("rumor_channel",), 0.6)
    cases.append(LaunderingScenario(
        scenario_id="hard_fabrication_in_second_step",
        domain="sre",
        failure_mode="LATE_FABRICATION",
        description="Step 1 is clean inference; step 2 introduces a fabrication. Final derived must cap at fabricated.",
        seeds=(e1, f1),
        chain=(
            BenchChainStep("HARD_late_fab_D1",
                           "inferred: deploy was successful",
                           input_ids=("HARD_late_fab_E",)),
            BenchChainStep("HARD_late_fab_D2",
                           "inferred: despite the deploy being successful, a customer-reported issue exists",
                           input_ids=("HARD_late_fab_D1", "HARD_late_fab_F")),
        ),
        later_query="should we investigate the post-deploy bug report?",
        later_target_id="HARD_late_fab_D2",
        expected_max_trust=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        expected_provenance_origins=("deploy_log", "rumor_channel"),
        later_should_quarantine=True,
    ))

    # 11. Full source-lattice fan-in: every content source participates.
    e1 = BenchSeed("HARD_lattice_E", "external: deployment completed at 10:00",
                   SourceLabel.EXTERNAL.value, ("deploy_log",))
    t1 = BenchSeed("HARD_lattice_T", "tool: scanner returned no critical findings",
                   SourceLabel.TOOL_OUTPUT.value, ("scanner_tool",))
    r1 = BenchSeed("HARD_lattice_R", "retrieved memory: similar rollout passed last month",
                   SourceLabel.RETRIEVED_MEMORY.value, ("prior_memory",))
    i1 = BenchSeed("HARD_lattice_I", "inferred: rollback risk is low",
                   SourceLabel.INFERENCE.value, ("prior_inference",))
    s1 = BenchSeed("HARD_lattice_S", "hypothetical: bypassing checks would save time",
                   SourceLabel.SIMULATION.value, ("sim_fast_path",), 0.6)
    f1 = BenchSeed("HARD_lattice_F", "rumor: customer has already approved the risk",
                   SourceLabel.FABRICATED_OR_UNCERTAIN.value, ("rumor_customer",), 0.5)
    cases.append(LaunderingScenario(
        scenario_id="hard_full_source_lattice_mix",
        domain="cross_domain",
        failure_mode="FULL_SOURCE_LATTICE_FANIN",
        description="Every source class feeds one derivation; the lowest source must dominate.",
        seeds=(e1, t1, r1, i1, s1, f1),
        chain=(BenchChainStep(
            "HARD_lattice_D",
            "inferred: ship the rollout despite missing explicit approval",
            input_ids=(
                "HARD_lattice_E", "HARD_lattice_T", "HARD_lattice_R",
                "HARD_lattice_I", "HARD_lattice_S", "HARD_lattice_F",
            ),
        ),),
        later_query="should we ship despite missing approval?",
        later_target_id="HARD_lattice_D",
        expected_max_trust=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        expected_provenance_origins=(
            "deploy_log", "scanner_tool", "prior_memory", "prior_inference",
            "sim_fast_path", "rumor_customer",
        ),
        later_should_quarantine=True,
    ))

    # 12. Tool-clean first step, simulation introduced in a later step.
    t1 = BenchSeed("HARD_tool_late_sim_T", "tool: recall@5 improved after rerank",
                   SourceLabel.TOOL_OUTPUT.value, ("benchmark_runner",))
    s1 = BenchSeed("HARD_tool_late_sim_S", "hypothetical: a larger model would improve recall",
                   SourceLabel.SIMULATION.value, ("sim_model_size",), 0.7)
    cases.append(LaunderingScenario(
        scenario_id="hard_tool_then_late_simulation",
        domain="research",
        failure_mode="LATE_SIMULATION_AFTER_TOOL",
        description="A clean tool-derived inference becomes simulation-contaminated at step 2.",
        seeds=(t1, s1),
        chain=(
            BenchChainStep("HARD_tool_late_sim_D1",
                           "inferred: reranking improves the benchmark",
                           input_ids=("HARD_tool_late_sim_T",)),
            BenchChainStep("HARD_tool_late_sim_D2",
                           "inferred: use a larger model and reranker in the paper",
                           input_ids=("HARD_tool_late_sim_D1", "HARD_tool_late_sim_S")),
        ),
        later_query="should the paper claim the larger model is necessary?",
        later_target_id="HARD_tool_late_sim_D2",
        expected_max_trust=SourceLabel.SIMULATION.value,
        expected_provenance_origins=("benchmark_runner", "sim_model_size"),
        later_should_quarantine=True,
    ))

    # 13. Retrieved memory plus a fabricated bridge.
    r1 = BenchSeed("HARD_retrieved_fab_R", "retrieved memory: prior vendor dispute settled quickly",
                   SourceLabel.RETRIEVED_MEMORY.value, ("legal_memory",), 0.9)
    f1 = BenchSeed("HARD_retrieved_fab_F", "unverified note: opposing counsel will concede",
                   SourceLabel.FABRICATED_OR_UNCERTAIN.value, ("hallway_rumor",), 0.5)
    cases.append(LaunderingScenario(
        scenario_id="hard_retrieved_memory_fabricated_bridge",
        domain="legal",
        failure_mode="RETRIEVED_MEMORY_WITH_FABRICATION",
        description="Retrieved precedent cannot rescue a fabricated bridge claim.",
        seeds=(r1, f1),
        chain=(BenchChainStep("HARD_retrieved_fab_D",
                              "inferred: settle immediately because concession is likely",
                              input_ids=("HARD_retrieved_fab_R", "HARD_retrieved_fab_F")),),
        later_query="should we settle because opposing counsel will concede?",
        later_target_id="HARD_retrieved_fab_D",
        expected_max_trust=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        expected_provenance_origins=("legal_memory", "hallway_rumor"),
        later_should_quarantine=True,
    ))

    # 14. Fan-out/fan-in provenance through two branches.
    e1 = BenchSeed("HARD_fan_E1", "external: reviewer asked for ablation",
                   SourceLabel.EXTERNAL.value, ("reviewer_comment",))
    e2 = BenchSeed("HARD_fan_E2", "external: replication matched baseline",
                   SourceLabel.EXTERNAL.value, ("replication_run",))
    s1 = BenchSeed("HARD_fan_S1", "hypothetical: hidden shortcut explains the improvement",
                   SourceLabel.SIMULATION.value, ("sim_shortcut",), 0.65)
    cases.append(LaunderingScenario(
        scenario_id="hard_fanout_fanin_provenance",
        domain="research",
        failure_mode="FANOUT_FANIN_MIXED",
        description="Two clean branches and one simulation branch fan into a final claim.",
        seeds=(e1, e2, s1),
        chain=(
            BenchChainStep("HARD_fan_D1",
                           "inferred: ablation is required before publication",
                           input_ids=("HARD_fan_E1",)),
            BenchChainStep("HARD_fan_D2",
                           "inferred: baseline replication supports the setup",
                           input_ids=("HARD_fan_E2",)),
            BenchChainStep("HARD_fan_D3",
                           "inferred: claim should be framed as provisional",
                           input_ids=("HARD_fan_D1", "HARD_fan_D2", "HARD_fan_S1")),
        ),
        later_query="how should the research claim be framed?",
        later_target_id="HARD_fan_D3",
        expected_max_trust=SourceLabel.SIMULATION.value,
        expected_provenance_origins=("reviewer_comment", "replication_run", "sim_shortcut"),
        later_should_quarantine=True,
    ))

    # 15. Long chain where fabrication enters only at depth 5.
    e1 = BenchSeed("HARD_depth5_E", "external: contract notice was sent",
                   SourceLabel.EXTERNAL.value, ("notice_email",))
    f1 = BenchSeed("HARD_depth5_F", "rumor: counterparty informally waived notice",
                   SourceLabel.FABRICATED_OR_UNCERTAIN.value, ("waiver_rumor",), 0.5)
    cases.append(LaunderingScenario(
        scenario_id="hard_depth5_late_fabrication",
        domain="legal",
        failure_mode="DEPTH5_LATE_FABRICATION",
        description="A long clean inference chain is contaminated only at the final step.",
        seeds=(e1, f1),
        chain=(
            BenchChainStep("HARD_depth5_D1", "inferred: notice obligation was triggered",
                           input_ids=("HARD_depth5_E",)),
            BenchChainStep("HARD_depth5_D2", "inferred: termination clock began",
                           input_ids=("HARD_depth5_D1",)),
            BenchChainStep("HARD_depth5_D3", "inferred: response deadline should be tracked",
                           input_ids=("HARD_depth5_D2",)),
            BenchChainStep("HARD_depth5_D4", "inferred: prepare termination package",
                           input_ids=("HARD_depth5_D3",)),
            BenchChainStep("HARD_depth5_D5",
                           "inferred: waive notice concerns and accelerate termination",
                           input_ids=("HARD_depth5_D4", "HARD_depth5_F")),
        ),
        later_query="can we accelerate termination based on waiver?",
        later_target_id="HARD_depth5_D5",
        expected_max_trust=SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        expected_provenance_origins=("notice_email", "waiver_rumor"),
        later_should_quarantine=True,
    ))

    # 16. Independent external/tool branch that remains clean through convergence.
    e1 = BenchSeed("HARD_clean_converge_E", "external: lab run replicated the baseline",
                   SourceLabel.EXTERNAL.value, ("lab_notebook",))
    t1 = BenchSeed("HARD_clean_converge_T", "tool: analysis script returned p_value=0.018",
                   SourceLabel.TOOL_OUTPUT.value, ("analysis_script",))
    r1 = BenchSeed("HARD_clean_converge_R", "retrieved memory: same ablation mattered in prior run",
                   SourceLabel.RETRIEVED_MEMORY.value, ("prior_ablation_memory",))
    cases.append(LaunderingScenario(
        scenario_id="hard_clean_multisource_convergence",
        domain="research",
        failure_mode="CLEAN_MULTISOURCE_CONVERGENCE",
        description="External, tool, and retrieved-memory branches converge without untrusted input.",
        seeds=(e1, t1, r1),
        chain=(
            BenchChainStep("HARD_clean_converge_D1",
                           "inferred: replication supports the setup",
                           input_ids=("HARD_clean_converge_E", "HARD_clean_converge_T")),
            BenchChainStep("HARD_clean_converge_D2",
                           "inferred: the ablation is likely meaningful",
                           input_ids=("HARD_clean_converge_D1", "HARD_clean_converge_R")),
        ),
        later_query="should we treat the ablation as meaningful?",
        later_target_id="HARD_clean_converge_D2",
        expected_max_trust=SourceLabel.INFERENCE.value,
        expected_provenance_origins=("lab_notebook", "analysis_script", "prior_ablation_memory"),
        later_should_quarantine=False,
    ))

    return cases


# ---------------------------------------------------------------------------
# Top-level dataset
# ---------------------------------------------------------------------------


def make_dataset() -> List[LaunderingScenario]:
    """All deterministic laundering scenarios."""
    return make_templated_scenarios() + make_hand_crafted_scenarios()


def validate_scenarios(scenarios: Sequence[LaunderingScenario]) -> None:
    """Validate scenario graph integrity and recompute the truth labels.

    This is intentionally strict because the benchmark is used as empirical
    evidence. The supplied `expected_*` fields must be derivable from the
    seed labels and derivation graph, not merely hand-authored assertions.
    """
    scenario_ids = [s.scenario_id for s in scenarios]
    duplicate_scenarios = sorted({
        scenario_id for scenario_id in scenario_ids if scenario_ids.count(scenario_id) > 1
    })
    if duplicate_scenarios:
        raise ValueError(f"duplicate scenario ids: {duplicate_scenarios}")

    for scenario in scenarios:
        seed_ids = [seed.record_id for seed in scenario.seeds]
        all_record_ids = seed_ids + [step.derived_id for step in scenario.chain]
        duplicate_record_ids = sorted({
            record_id for record_id in all_record_ids if all_record_ids.count(record_id) > 1
        })
        if duplicate_record_ids:
            raise ValueError(
                f"{scenario.scenario_id}: duplicate record ids {duplicate_record_ids}"
            )

        if not scenario.chain:
            raise ValueError(f"{scenario.scenario_id}: scenario has no chain")

        known_ids = set(seed_ids)
        for step in scenario.chain:
            if not step.input_ids:
                raise ValueError(f"{scenario.scenario_id}: {step.derived_id} has no inputs")
            for input_id in step.input_ids:
                if input_id not in known_ids:
                    raise ValueError(
                        f"{scenario.scenario_id}: {step.derived_id} references "
                        f"unknown or future input {input_id!r}"
                    )
            known_ids.add(step.derived_id)

        if scenario.later_target_id not in known_ids:
            raise ValueError(
                f"{scenario.scenario_id}: later_target_id {scenario.later_target_id!r} "
                "does not resolve"
            )

        computed_final = expected_final_source(scenario)
        if scenario.expected_max_trust != computed_final:
            raise ValueError(
                f"{scenario.scenario_id}: expected_max_trust={scenario.expected_max_trust!r} "
                f"but computed {computed_final!r}"
            )

        computed_origins = expected_final_origins(scenario)
        if tuple(scenario.expected_provenance_origins) != computed_origins:
            raise ValueError(
                f"{scenario.scenario_id}: expected_provenance_origins="
                f"{scenario.expected_provenance_origins!r} but computed {computed_origins!r}"
            )

        expected_quarantine = computed_final in UNTRUSTED_FINAL_SOURCES
        if scenario.later_should_quarantine != expected_quarantine:
            raise ValueError(
                f"{scenario.scenario_id}: later_should_quarantine="
                f"{scenario.later_should_quarantine!r} but computed {expected_quarantine!r}"
            )


def dataset_summary(scenarios: Sequence[LaunderingScenario]) -> Dict[str, Mapping[str, int]]:
    by_domain: Dict[str, int] = {}
    by_mode: Dict[str, int] = {}
    by_final_source: Dict[str, int] = {}
    for scenario in scenarios:
        by_domain[scenario.domain] = by_domain.get(scenario.domain, 0) + 1
        by_mode[scenario.failure_mode] = by_mode.get(scenario.failure_mode, 0) + 1
        source = scenario.expected_max_trust
        by_final_source[source] = by_final_source.get(source, 0) + 1
    return {
        "by_domain": by_domain,
        "by_failure_mode": by_mode,
        "by_expected_final_source": by_final_source,
    }


def write_jsonl(scenarios: List[LaunderingScenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in scenarios:
            fh.write(json.dumps(s.to_json(), sort_keys=True))
            fh.write("\n")


def read_jsonl(path: Path) -> List[LaunderingScenario]:
    out: List[LaunderingScenario] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(LaunderingScenario.from_json(json.loads(line)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/data/laundering_dataset.jsonl")
    args = parser.parse_args()

    scenarios = make_dataset()
    validate_scenarios(scenarios)
    write_jsonl(scenarios, Path(args.output))

    print(f"wrote {len(scenarios)} scenarios to {args.output}")

    summary = dataset_summary(scenarios)
    by_domain = summary["by_domain"]
    by_mode = summary["by_failure_mode"]
    by_source = summary["by_expected_final_source"]

    print("\nby domain:")
    for d in sorted(by_domain):
        print(f"  {d:>20s}  {by_domain[d]:>3d}")
    print("\nby failure mode:")
    for m in sorted(by_mode):
        print(f"  {m:>30s}  {by_mode[m]:>3d}")
    print("\nby expected final source:")
    for s in sorted(by_source, key=trust_rank):
        print(f"  {s:>30s}  {by_source[s]:>3d}")


if __name__ == "__main__":
    main()

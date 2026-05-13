"""Three inscription-policy baselines for the laundering benchmark.

All three baselines share the same underlying agent infrastructure (the
same retrieval, the same fold-gate, the same routing layer). They differ
only in their *inscription policy* for derived records:

    no_source         -- write every record with source=external. The
                         laundering baseline; represents the default
                         behaviour of memory libraries that have no
                         source semantics.

    provenance_only   -- preserve provenance metadata on derived records
                         (union of contributing provenance + ids), but
                         always label derived records as inference.
                         Represents systems that track lineage but do
                         not enforce trust composition.

    trace_memory      -- the validated source-downgrading rule: derived
                         records have source = min-trust of contributing
                         inputs (with inference as the upper bound),
                         and provenance is propagated transitively.

The three baselines instantiate seeds the same way (with the supplied
source label from the scenario) so the only difference is how
derivations are inscribed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from trace_memory import MemoryAgent, SourceLabel
from trace_memory.types import MemoryRecord

from .laundering_dataset import BenchChainStep, BenchSeed, LaunderingScenario


@dataclass(frozen=True)
class BaselineResult:
    """One baseline's outcome on one scenario.

    Captures everything the benchmark report needs to compute aggregate
    metrics without rerunning the agent. JSON-serializable.
    """

    baseline: str
    scenario_id: str
    domain: str
    failure_mode: str
    # Final derived record fields (the laundering target):
    final_derived_source: str
    final_derived_provenance: Tuple[str, ...]
    final_derived_record_id: str
    # Truth from the scenario:
    expected_max_trust: str
    expected_provenance_origins: Tuple[str, ...]
    later_should_quarantine: bool
    # Per-chain-step labels:
    chain_step_sources: Tuple[Tuple[str, str], ...]  # (derived_id, source)
    # Later-query outcome:
    later_route: str
    later_retrieved_target: bool


def _trust_rank(label: str) -> int:
    ordering = {
        SourceLabel.FABRICATED_OR_UNCERTAIN.value: 0,
        SourceLabel.SIMULATION.value: 1,
        SourceLabel.INFERENCE.value: 2,
        SourceLabel.RETRIEVED_MEMORY.value: 3,
        SourceLabel.TOOL_OUTPUT.value: 4,
        SourceLabel.EXTERNAL.value: 5,
    }
    return ordering.get(label, 2)


def _min_trust(labels: Sequence[str]) -> str:
    """Min-trust with inference as the upper bound."""
    if not labels:
        return SourceLabel.INFERENCE.value
    candidates = list(labels) + [SourceLabel.INFERENCE.value]
    return min(candidates, key=_trust_rank)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _plant_seeds(agent: MemoryAgent, scenario: LaunderingScenario) -> None:
    for seed in scenario.seeds:
        agent.add(
            seed.content,
            source=SourceLabel(seed.source),
            provenance=tuple(seed.provenance),
            source_confidence=seed.source_confidence,
            record_id=seed.record_id,
        )


def _resolve_contributing(agent: MemoryAgent, ids: Sequence[str]) -> List[MemoryRecord]:
    out: List[MemoryRecord] = []
    for rid in ids:
        rec = agent.store.get(rid)
        if rec is not None:
            out.append(rec)
    return out


def _collect_chain_sources(
    agent: MemoryAgent,
    scenario: LaunderingScenario,
) -> Tuple[Tuple[str, str], ...]:
    pairs: List[Tuple[str, str]] = []
    for step in scenario.chain:
        rec = agent.store.get(step.derived_id)
        if rec is not None:
            pairs.append((step.derived_id, rec.source_label))
    return tuple(pairs)


def _run_later_query(
    agent: MemoryAgent,
    scenario: LaunderingScenario,
) -> Tuple[str, bool]:
    result = agent.query(scenario.later_query)
    retrieved_ids = {hit.record.record_id for hit in result.retrieved}
    return result.selected_route, scenario.later_target_id in retrieved_ids


# ---------------------------------------------------------------------------
# Baseline 1: no_source
# ---------------------------------------------------------------------------


def run_no_source(scenario: LaunderingScenario) -> BaselineResult:
    """Every derived record stored with source=external. Provenance ignored."""
    agent = MemoryAgent()
    _plant_seeds(agent, scenario)
    for step in scenario.chain:
        # Use the contributing records as a pretext for the derivation,
        # but the policy itself does no propagation. Source defaults to
        # external (the laundering baseline).
        agent.add(
            step.derived_content,
            source=SourceLabel.EXTERNAL,
            record_id=step.derived_id,
        )
    final = agent.store.get(scenario.chain[-1].derived_id)
    route, retrieved = _run_later_query(agent, scenario)
    return BaselineResult(
        baseline="no_source",
        scenario_id=scenario.scenario_id,
        domain=scenario.domain,
        failure_mode=scenario.failure_mode,
        final_derived_source=final.source_label,
        final_derived_provenance=tuple(final.provenance),
        final_derived_record_id=final.record_id,
        expected_max_trust=scenario.expected_max_trust,
        expected_provenance_origins=scenario.expected_provenance_origins,
        later_should_quarantine=scenario.later_should_quarantine,
        chain_step_sources=_collect_chain_sources(agent, scenario),
        later_route=route,
        later_retrieved_target=retrieved,
    )


# ---------------------------------------------------------------------------
# Baseline 2: provenance_only
# ---------------------------------------------------------------------------


def run_provenance_only(scenario: LaunderingScenario) -> BaselineResult:
    """Provenance propagated; derived source is always inference."""
    agent = MemoryAgent()
    _plant_seeds(agent, scenario)
    for step in scenario.chain:
        contributing = _resolve_contributing(agent, step.input_ids)
        prov: List[str] = []
        for rec in contributing:
            prov.extend(rec.provenance)
            prov.append(rec.record_id)
        agent.add(
            step.derived_content,
            source=SourceLabel.INFERENCE,
            provenance=tuple(prov),
            record_id=step.derived_id,
            source_confidence=(
                min(r.source_confidence for r in contributing)
                if contributing
                else 0.5
            ),
        )
    final = agent.store.get(scenario.chain[-1].derived_id)
    route, retrieved = _run_later_query(agent, scenario)
    return BaselineResult(
        baseline="provenance_only",
        scenario_id=scenario.scenario_id,
        domain=scenario.domain,
        failure_mode=scenario.failure_mode,
        final_derived_source=final.source_label,
        final_derived_provenance=tuple(final.provenance),
        final_derived_record_id=final.record_id,
        expected_max_trust=scenario.expected_max_trust,
        expected_provenance_origins=scenario.expected_provenance_origins,
        later_should_quarantine=scenario.later_should_quarantine,
        chain_step_sources=_collect_chain_sources(agent, scenario),
        later_route=route,
        later_retrieved_target=retrieved,
    )


# ---------------------------------------------------------------------------
# Baseline 3: trace_memory (the validated source-downgrading rule)
# ---------------------------------------------------------------------------


def run_trace_memory(scenario: LaunderingScenario) -> BaselineResult:
    """Source-downgrading inscription via agent.add_derived()."""
    agent = MemoryAgent()
    _plant_seeds(agent, scenario)
    for step in scenario.chain:
        agent.add_derived(
            step.derived_content,
            inputs=tuple(step.input_ids),
            record_id=step.derived_id,
        )
    final = agent.store.get(scenario.chain[-1].derived_id)
    route, retrieved = _run_later_query(agent, scenario)
    return BaselineResult(
        baseline="trace_memory",
        scenario_id=scenario.scenario_id,
        domain=scenario.domain,
        failure_mode=scenario.failure_mode,
        final_derived_source=final.source_label,
        final_derived_provenance=tuple(final.provenance),
        final_derived_record_id=final.record_id,
        expected_max_trust=scenario.expected_max_trust,
        expected_provenance_origins=scenario.expected_provenance_origins,
        later_should_quarantine=scenario.later_should_quarantine,
        chain_step_sources=_collect_chain_sources(agent, scenario),
        later_route=route,
        later_retrieved_target=retrieved,
    )


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------


BASELINES: Dict[str, "callable"] = {
    "no_source": run_no_source,
    "provenance_only": run_provenance_only,
    "trace_memory": run_trace_memory,
}


__all__ = [
    "BaselineResult",
    "BASELINES",
    "run_no_source",
    "run_provenance_only",
    "run_trace_memory",
]

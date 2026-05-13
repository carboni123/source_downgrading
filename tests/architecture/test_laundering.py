"""Tests for the inference-laundering validation harness."""
from __future__ import annotations

import math

import pytest

from fgm import (
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_INFERENCE,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_SIMULATION,
    SOURCE_TOOL_OUTPUT,
    compare_laundering_policies,
    compare_laundering_policies_multiseed,
    evaluate_laundering_policy,
    make_extended_adversarial_fixture,
    make_laundering_fixture,
    min_trust_source,
    run_laundering_multiseed,
)
from fgm.laundering import (
    _make_agent,
    _plant_seeds,
    _resolve_contributing,
    write_inference_downgrading,
)


def test_min_trust_picks_lowest_input_under_inference_ceiling():
    # Only external inputs -> inference (because deriving introduces uncertainty)
    assert min_trust_source([SOURCE_EXTERNAL, SOURCE_EXTERNAL]) == SOURCE_INFERENCE
    # Add retrieved memory -> still capped at inference (retrieved > inference in trust)
    assert min_trust_source([SOURCE_EXTERNAL, SOURCE_RETRIEVED_MEMORY]) == SOURCE_INFERENCE
    # Add simulation -> simulation dominates
    assert min_trust_source([SOURCE_EXTERNAL, SOURCE_SIMULATION]) == SOURCE_SIMULATION
    # Add fabricated -> fabricated dominates
    assert min_trust_source([SOURCE_EXTERNAL, SOURCE_FABRICATED]) == SOURCE_FABRICATED
    # Empty -> inference fallback
    assert min_trust_source([]) == SOURCE_INFERENCE


def test_fixture_shape():
    cases = make_laundering_fixture()
    assert len(cases) >= 5
    case_ids = {case.case_id for case in cases}
    # Coverage: pure inference, simulation laundering, fabrication laundering,
    # chained, and mixed.
    assert {
        "pure_inference_from_observations",
        "laundering_from_simulation",
        "laundering_from_fabrication",
        "chained_inference_from_observation",
        "mixed_chain_simulation_then_inference",
    }.issubset(case_ids)


def test_naive_policy_violates_trust_ceiling_for_every_derived_record():
    cases = make_laundering_fixture()
    report = evaluate_laundering_policy(cases, policy="naive_inscribe", seed=0)
    # naive_inscribe uses agent.add() default source=external. Every derived
    # record's expected_max_trust is at most inference (because deriving
    # introduces uncertainty), so labeling derived as external is always a
    # trust-ceiling violation against ground truth.
    assert report.derived_trust_ceiling_violation_rate == 1.0
    # And provenance is never propagated by naive_inscribe.
    assert report.provenance_chain_recall == 0.0


def test_naive_laundering_is_locally_underdetected_due_to_cascade():
    # The local-view inference_laundering_rate inspects whether the inputs to
    # each step had non-external source labels AT THE TIME OF INSCRIPTION. When
    # naive_inscribe relabels a derived record as external, subsequent steps
    # that consume it see "external" inputs and the framework cannot detect
    # the cascading laundering from its own perspective. The truth-grounded
    # derived_trust_ceiling_violation_rate catches all 7 violations; the local
    # view sees fewer. The two metrics together reveal cascade-invisibility.
    cases = make_laundering_fixture()
    report = evaluate_laundering_policy(cases, policy="naive_inscribe", seed=0)
    assert report.inference_laundering_rate > 0.0
    assert report.inference_laundering_rate < report.derived_trust_ceiling_violation_rate


def test_provenance_propagating_eliminates_laundering():
    cases = make_laundering_fixture()
    report = evaluate_laundering_policy(cases, policy="provenance_propagating", seed=0)
    # Source is always set to inference, so no derived record is labeled external.
    assert report.inference_laundering_rate == 0.0
    # Provenance must be fully recoverable for every derived record.
    assert report.provenance_chain_recall == 1.0
    # Provenance chain depth must be at least 1 (we always push at least one id).
    assert report.transitive_provenance_depth_mean >= 1.0


def test_source_downgrading_respects_trust_ceiling():
    cases = make_laundering_fixture()
    report = evaluate_laundering_policy(cases, policy="source_downgrading", seed=0)
    # Source is capped at min-trust of inputs, so no ceiling violations.
    assert report.derived_trust_ceiling_violation_rate == 0.0
    assert report.inference_laundering_rate == 0.0
    assert report.provenance_chain_recall == 1.0


def test_propagating_beats_naive_on_provenance_and_laundering():
    cases = make_laundering_fixture()
    reports = compare_laundering_policies(cases, seed=0)
    naive = reports["naive_inscribe"]
    prop = reports["provenance_propagating"]
    assert prop.inference_laundering_rate < naive.inference_laundering_rate
    assert prop.provenance_chain_recall > naive.provenance_chain_recall


def test_downgrading_beats_naive_on_ceiling_and_externalization():
    cases = make_laundering_fixture()
    reports = compare_laundering_policies(cases, seed=0)
    naive = reports["naive_inscribe"]
    down = reports["source_downgrading"]
    assert down.derived_trust_ceiling_violation_rate <= naive.derived_trust_ceiling_violation_rate
    assert down.inference_laundering_rate < naive.inference_laundering_rate


def test_deterministic_under_fixed_seed():
    cases = make_laundering_fixture()
    a = evaluate_laundering_policy(cases, policy="provenance_propagating", seed=0)
    b = evaluate_laundering_policy(cases, policy="provenance_propagating", seed=0)
    assert a == b


def test_multiseed_source_downgrading_is_stable_across_seeds():
    cases = make_laundering_fixture()
    report = run_laundering_multiseed(cases, policy="source_downgrading", seeds=tuple(range(20)))
    assert report.n_seeds == 20
    # source_downgrading must keep zero across all four failure metrics
    # under embedding-noise perturbation. Provenance recall must stay at 1.0.
    assert report.max["inference_laundering_rate"] == 0.0
    assert report.max["derived_trust_ceiling_violation_rate"] == 0.0
    assert report.max["false_externalization_after_inference"] == 0.0
    assert report.min["provenance_chain_recall"] == 1.0


def test_multiseed_naive_ceiling_violations_persist_across_seeds():
    cases = make_laundering_fixture()
    report = run_laundering_multiseed(cases, policy="naive_inscribe", seeds=tuple(range(20)))
    # naive_inscribe must violate the trust ceiling on every derived record
    # under every seed (label defaults to external; expected_max_trust is
    # always inference or lower). The property must hold robustly.
    assert report.min["derived_trust_ceiling_violation_rate"] == 1.0
    assert report.max["provenance_chain_recall"] == 0.0


def test_multiseed_propagating_retains_residual_ceiling_violations():
    cases = make_laundering_fixture()
    report = run_laundering_multiseed(cases, policy="provenance_propagating", seeds=tuple(range(20)))
    # provenance_propagating eliminates direct laundering but cannot cap trust
    # at min-trust of inputs (always labels derived as inference). The
    # ceiling-violation rate must stay strictly above zero across seeds.
    assert report.min["inference_laundering_rate"] == 0.0
    assert report.min["provenance_chain_recall"] == 1.0
    assert report.min["derived_trust_ceiling_violation_rate"] > 0.0


def test_multiseed_compare_returns_all_three_policies():
    cases = make_laundering_fixture()
    seeds = tuple(range(5))
    reports = compare_laundering_policies_multiseed(cases, seeds=seeds)
    assert set(reports.keys()) == {"naive_inscribe", "provenance_propagating", "source_downgrading"}
    for report in reports.values():
        assert report.n_seeds == 5


@pytest.mark.parametrize("policy", ["naive_inscribe", "provenance_propagating", "source_downgrading"])
def test_metrics_are_finite_or_nan(policy):
    cases = make_laundering_fixture()
    report = evaluate_laundering_policy(cases, policy=policy, seed=0)
    for value in (
        report.inference_laundering_rate,
        report.provenance_chain_recall,
        report.false_externalization_after_inference,
        report.derived_trust_ceiling_violation_rate,
        report.transitive_provenance_depth_mean,
    ):
        assert math.isfinite(value) or math.isnan(value)


# ---------------------------------------------------------------------------
# Extended adversarial fixture (paper Section 5.1 edge cases).
#
# Each test below makes a per-record assertion against source_downgrading,
# rather than relying on the case-level aggregate metrics. Aggregate metrics
# bind one expected_max_trust to the whole case, which is the right shape
# for a linear chain but too coarse for DAG cases where intermediate
# records legitimately carry a higher trust than the final record.
# ---------------------------------------------------------------------------


def _run_source_downgrading(case):
    """Replay a single case under source_downgrading and return derived records by id."""
    agent = _make_agent(0)
    _plant_seeds(agent, case)
    derived_by_id = {}
    for step in case.chain:
        contributing = _resolve_contributing(agent, step.input_record_ids)
        if contributing:
            agent.query(step.derived_content)
        derived = write_inference_downgrading(agent, step, contributing)
        derived_by_id[step.derived_id] = derived
    return agent, derived_by_id


def test_extended_fixture_shape():
    cases = make_extended_adversarial_fixture()
    case_ids = {c.case_id for c in cases}
    assert case_ids == {
        "branching_dag",
        "reconvergent_dag",
        "same_rank_contradiction",
        "irrelevant_declared_contributor",
        "cross_agent_label_stripped",
    }


def test_branching_dag_all_branches_capped_at_contaminated_input():
    """Branching DAG: one contaminated seed fans out to three derivations.

    Each derived record must inherit the sim ceiling. The rule is pointwise,
    so a fan-out structure should not weaken it.
    """
    cases = {c.case_id: c for c in make_extended_adversarial_fixture()}
    _, derived = _run_source_downgrading(cases["branching_dag"])
    for rid in ("D_branch_a", "D_branch_b", "D_branch_c"):
        assert derived[rid].source_label == SOURCE_SIMULATION
        # Provenance must include both contributing seeds.
        assert "E1" in derived[rid].provenance
        assert "S1" in derived[rid].provenance


def test_reconvergent_dag_final_capped_at_dirty_branch():
    """Reconvergent DAG: clean and contaminated branches merge.

    The intermediate D_clean_branch legitimately gets ``inference`` because
    its sole contributor is external. The intermediate D_dirty_branch gets
    ``simulation`` because one contributor was simulated. The final
    D_reconverge takes the meet and must be ``simulation``. Provenance on
    the final record must reach all three seed origins.
    """
    cases = {c.case_id: c for c in make_extended_adversarial_fixture()}
    _, derived = _run_source_downgrading(cases["reconvergent_dag"])

    assert derived["D_clean_branch"].source_label == SOURCE_INFERENCE
    assert derived["D_dirty_branch"].source_label == SOURCE_SIMULATION
    assert derived["D_reconverge"].source_label == SOURCE_SIMULATION

    final_prov = set(derived["D_reconverge"].provenance)
    assert {"D_clean_branch", "D_dirty_branch", "E1", "S1", "E2"}.issubset(final_prov)


def test_same_rank_contradiction_falls_back_to_inference_ceiling():
    """Same-rank contradiction: two tool outputs disagree.

    Both inputs are ``tool_output`` (rank 4), above ``inference`` (rank 2),
    so the inference ceiling triggers and the derived record is labeled
    ``inference``. The rule cannot detect the contradiction itself
    (paper Section 5.1: "Same-rank contradictions are not resolved by
    source downgrading and should raise a separate contradiction flag for
    routing or belief revision"). This test asserts that boundary
    explicitly.
    """
    cases = {c.case_id: c for c in make_extended_adversarial_fixture()}
    _, derived = _run_source_downgrading(cases["same_rank_contradiction"])
    rec = derived["D_contradiction"]
    assert rec.source_label == SOURCE_INFERENCE
    # Both contributors must appear in provenance even though they disagree;
    # the contradiction is preserved for downstream belief-revision logic.
    assert "T1" in rec.provenance
    assert "T2" in rec.provenance


def test_irrelevant_declared_contributor_is_not_filtered():
    """Irrelevant declared contributor: a semantically unrelated input is declared.

    Both inputs are external, so the inference ceiling triggers and the
    derived label is ``inference``. The irrelevant contributor remains in
    provenance and contributes to the trust computation. The rule cannot
    detect semantic irrelevance (paper Section 5.1: "the rule assumes
    that the declared contributors are semantically relevant to the
    derived claim; validating that support relation requires a derivation
    witness, entailment check, or audit hook outside the source-label
    rule itself"). This test asserts that boundary explicitly.
    """
    cases = {c.case_id: c for c in make_extended_adversarial_fixture()}
    _, derived = _run_source_downgrading(cases["irrelevant_declared_contributor"])
    rec = derived["D_unsupported"]
    assert rec.source_label == SOURCE_INFERENCE
    assert "E_rel" in rec.provenance
    assert "E_irr" in rec.provenance


def test_cross_agent_label_stripping_breaks_truth_ceiling():
    """Cross-agent label stripping: an input loses its source label in transit.

    The simulation input was relabeled as external before reaching this
    agent. Source-downgrading honours the (corrupted) input labels: both
    inputs are external, so the inference ceiling triggers and the derived
    record gets ``inference``. The truth-grounded expectation is
    ``simulation``, so the rule fails this case by design. This is paper
    Section 2 adversary 4 (cross-agent label transport): the rule is no
    stronger than the weakest input label. Defence requires signed
    provenance envelopes or verified import policies, which are out of
    scope for source downgrading itself.
    """
    cases = {c.case_id: c for c in make_extended_adversarial_fixture()}
    case = cases["cross_agent_label_stripped"]
    _, derived = _run_source_downgrading(case)
    rec = derived["D_post_strip"]

    # Given the labels as planted, the rule produces inference (correct under
    # the rule's preconditions).
    assert rec.source_label == SOURCE_INFERENCE

    # But the case's truth-grounded ceiling is simulation, so the per-record
    # check against truth fails. The test asserts this gap explicitly.
    assert case.expected_max_trust == SOURCE_SIMULATION
    truth_rank = {
        SOURCE_FABRICATED: 0,
        SOURCE_SIMULATION: 1,
        SOURCE_INFERENCE: 2,
        SOURCE_RETRIEVED_MEMORY: 3,
        SOURCE_TOOL_OUTPUT: 4,
        SOURCE_EXTERNAL: 5,
    }
    assert truth_rank[rec.source_label] > truth_rank[case.expected_max_trust], (
        "expected the label-stripped case to over-trust the derivation against "
        "the truth-grounded ceiling"
    )


def test_extended_fixture_naive_and_propagating_remain_unsafe():
    """Sanity: naive launders, provenance-propagating still over-trusts.

    On the extended fixture, naive inscription should still violate the
    ceiling on every derived record, and pure provenance propagation
    should still over-trust on at least the cases where the expected
    ceiling is below ``inference`` (branching_dag and reconvergent_dag).
    """
    cases = make_extended_adversarial_fixture()
    naive = evaluate_laundering_policy(cases, policy="naive_inscribe", seed=0)
    prop = evaluate_laundering_policy(cases, policy="provenance_propagating", seed=0)

    assert naive.derived_trust_ceiling_violation_rate == 1.0
    assert naive.provenance_chain_recall == 0.0

    # Provenance propagation labels every derived as inference. The
    # extended fixture contains branching/reconvergent cases whose
    # expected ceiling is sim (below inference), so propagation must
    # over-trust on those records.
    assert prop.inference_laundering_rate == 0.0
    assert prop.derived_trust_ceiling_violation_rate > 0.0

import json
import os

import numpy as np
import pytest

from fgm import (
    FGMAgent,
    ROUTE_OPERATION_MEMORY,
    ROUTE_QUARANTINE,
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_OPERATION_RECORD,
    SOURCE_RETRIEVED_MEMORY,
    ValidationRecord,
    analyze_live_replication,
    apply_route_baseline,
    compare_correction_policies,
    compare_inscription_policies,
    compare_residual_attention_policies,
    compare_self_index_policies,
    coupled_field_probe,
    make_correction_chain_fixture,
    make_inscription_utility_fixture,
    make_residual_attention_fixture,
    make_self_index_fixture,
    read_validation_jsonl,
    run_controlled_replication,
    run_real_embedding_replication,
    run_real_embedding_validation,
    run_controlled_roadmap_validations,
    score_validation_records,
    write_controlled_replication_outputs,
    write_real_embedding_replication_output,
    write_real_component_validation_output,
    write_roadmap_validation_outputs,
    write_validation_jsonl,
    echo_call,
    hash_embed,
    run_live_llm_validation,
    run_live_llm_replication,
    run_rerank_boundary_regression,
    write_live_provider_model_comparison_outputs,
    write_live_llm_replication_outputs,
    write_live_llm_validation_output,
    write_live_replication_diagnostics,
    write_rerank_boundary_regression_output,
)


def test_source_labels_survive_add_retrieve_fold_and_operation_write():
    agent = FGMAgent(dim=32, fold_threshold=0.001, auto_compress=False)
    agent.add(
        "database migration timed out during deploy",
        record_id="obs_deploy_timeout",
        source_label=SOURCE_EXTERNAL,
        source_confidence=0.97,
        provenance=["incident_report_1"],
    )

    result = agent.query("database migration timed out")

    assert result.retrieved
    assert result.source_labels[0] == SOURCE_EXTERNAL
    assert result.active_source_labels[0] == SOURCE_RETRIEVED_MEMORY
    assert result.source_confidence[0] == 0.97
    assert result.selected_route == ROUTE_OPERATION_MEMORY
    assert result.operation_record_id is not None

    op = agent.operations.all_operations()[-1]
    assert op.source_labels == [SOURCE_EXTERNAL]
    assert op.active_source_labels == [SOURCE_RETRIEVED_MEMORY]
    assert op.operation_record_id == result.operation_record_id

    op_record = agent.store.get(result.operation_record_id)
    assert op_record is not None
    assert op_record.source_label == SOURCE_OPERATION_RECORD
    assert op_record.provenance == ("obs_deploy_timeout",)


def test_source_sensitive_route_quarantines_fabricated_content():
    agent = FGMAgent(dim=32, fold_threshold=0.001, auto_compress=False)
    agent.add(
        "fabricated claim: deploy was safe despite rollback evidence",
        record_id="fab_rollback",
        source_label=SOURCE_FABRICATED,
        source_confidence=0.99,
    )

    result = agent.query("deploy rollback evidence")

    assert result.retrieved
    assert result.source_labels == [SOURCE_FABRICATED]
    assert result.selected_route == ROUTE_QUARANTINE
    assert result.route_scores[ROUTE_QUARANTINE] > result.route_scores[ROUTE_OPERATION_MEMORY]


def test_source_aware_rerank_fixes_legal_rollback_polarity_boundary():
    def query_embed(_text):
        return np.array([1.0, 0.0])

    e1_vec = np.array([1.0, 0.0])
    f1_vec = np.array([0.98, 0.2])
    f1_vec = f1_vec / np.linalg.norm(f1_vec)
    query = "which uncertain claim says legal forbids rollback?"

    baseline = FGMAgent(
        dim=2,
        embed_fn=query_embed,
        fold_threshold=0.001,
        retrieval_k=1,
        auto_compress=False,
        source_aware_rerank=False,
    )
    baseline.add(
        "E1 external observation legal approved rollback for deploy migration",
        record_id="E1",
        vector=e1_vec,
        source_label=SOURCE_EXTERNAL,
    )
    baseline.add(
        "F1 fabricated distractor rollback is forbidden by legal",
        record_id="F1",
        vector=f1_vec,
        source_label=SOURCE_FABRICATED,
    )

    assert baseline.query(query).retrieved[0].record.record_id == "E1"

    agent = FGMAgent(
        dim=2,
        embed_fn=query_embed,
        fold_threshold=0.001,
        retrieval_k=1,
        auto_compress=False,
        source_aware_rerank=True,
        source_rerank_k=2,
    )
    agent.add(
        "E1 external observation legal approved rollback for deploy migration",
        record_id="E1",
        vector=e1_vec,
        source_label=SOURCE_EXTERNAL,
    )
    agent.add(
        "F1 fabricated distractor rollback is forbidden by legal",
        record_id="F1",
        vector=f1_vec,
        source_label=SOURCE_FABRICATED,
    )

    result = agent.query(query)

    assert result.retrieved[0].record.record_id == "F1"
    assert result.source_labels == [SOURCE_FABRICATED]
    assert result.selected_route == ROUTE_QUARANTINE


def test_low_confidence_source_routes_to_quarantine():
    agent = FGMAgent(dim=32, fold_threshold=0.001, auto_compress=False)
    agent.add(
        "uncertain note about production database ownership",
        record_id="uncertain_db_owner",
        source_label=SOURCE_EXTERNAL,
        source_confidence=0.2,
    )

    result = agent.query("production database ownership")

    assert result.selected_route == ROUTE_QUARANTINE


def test_validation_record_jsonl_round_trip(tmp_path):
    agent = FGMAgent(dim=16, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "API rate limit is 100 requests per minute",
        record_id="rate_limit",
        source_label=SOURCE_EXTERNAL,
    )
    result = agent.query("API rate limit")

    record = ValidationRecord.from_fold_result(
        run_id="source-routing-smoke",
        seed=7,
        turn_id=1,
        fold_result=result,
        external_input_ids=["rate_limit"],
        eligibility_score=0.8,
        inscription_score=0.7,
        predicted_fold_force=result.fold_force,
        expected_retrieved_ids=["rate_limit"],
        expected_source_labels={"rate_limit": SOURCE_EXTERNAL},
        expected_active_source_labels={"rate_limit": SOURCE_RETRIEVED_MEMORY},
        expected_route=ROUTE_OPERATION_MEMORY,
        future_utility_label=True,
    )

    path = tmp_path / "validation.jsonl"
    write_validation_jsonl(path, [record])
    rows = read_validation_jsonl(path)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "source-routing-smoke"
    assert row["retrieved_ids"] == ["rate_limit"]
    assert row["source_labels"] == {"rate_limit": SOURCE_EXTERNAL}
    assert row["active_source_labels"] == {"rate_limit": SOURCE_RETRIEVED_MEMORY}
    assert row["selected_route"] == ROUTE_OPERATION_MEMORY
    assert row["expected_route"] == ROUTE_OPERATION_MEMORY
    assert row["operation_record_id"] == result.operation_record_id
    assert isinstance(row["output_with_memory"], list)
    assert isinstance(row["transition_delta"], float)


def test_empty_retrieval_validation_record_has_no_sources():
    agent = FGMAgent(dim=8, fold_threshold=0.001, auto_compress=False)
    result = agent.query("nothing stored yet")

    record = ValidationRecord.from_fold_result(
        run_id="empty",
        seed=0,
        turn_id=1,
        fold_result=result,
    )

    assert result.retrieved == []
    assert record.retrieved_ids == []
    assert record.source_labels == {}
    assert record.active_source_labels == {}
    assert np.isclose(record.realized_fold_force, 0.0)


def test_validation_scoring_reports_source_route_and_utility_metrics():
    agent = FGMAgent(dim=16, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add("external deploy rollback approval", record_id="E1", source_label=SOURCE_EXTERNAL)
    agent.add("fabricated rollback forbidden claim", record_id="F1", source_label=SOURCE_FABRICATED)

    useful = ValidationRecord.from_fold_result(
        run_id="score",
        seed=0,
        turn_id=1,
        fold_result=agent.query("deploy rollback approval"),
        expected_retrieved_ids=["E1"],
        expected_source_labels={"E1": SOURCE_EXTERNAL},
        expected_active_source_labels={"E1": SOURCE_RETRIEVED_MEMORY},
        expected_route=ROUTE_OPERATION_MEMORY,
        future_utility_label=True,
    )
    unsafe = ValidationRecord.from_fold_result(
        run_id="score",
        seed=0,
        turn_id=2,
        fold_result=agent.query("fabricated rollback forbidden claim"),
        expected_retrieved_ids=["F1"],
        expected_source_labels={"F1": SOURCE_FABRICATED},
        expected_active_source_labels={"F1": SOURCE_RETRIEVED_MEMORY},
        expected_route=ROUTE_QUARANTINE,
        future_utility_label=False,
    )

    source_sensitive = score_validation_records([useful, unsafe])
    assert source_sensitive["retrieval_hit_rate"] == 1.0
    assert source_sensitive["source_label_accuracy"] == 1.0
    assert source_sensitive["active_source_accuracy"] == 1.0
    assert source_sensitive["route_accuracy"] == 1.0
    assert source_sensitive["quarantine_recall"] == 1.0
    assert source_sensitive["false_write_rate"] == 0.0
    assert source_sensitive["missed_useful_write_rate"] == 0.0

    always_write = score_validation_records(apply_route_baseline([useful, unsafe], "always_write"))
    never_write = score_validation_records(apply_route_baseline([useful, unsafe], "never_write"))
    source_blind = score_validation_records(apply_route_baseline([useful, unsafe], "source_blind"))

    assert always_write["false_write_rate"] == 1.0
    assert never_write["missed_useful_write_rate"] == 1.0
    assert source_blind["echo_promotion_rate"] == 1.0


def test_inscription_utility_policy_beats_relevance_and_degenerate_baselines():
    events = make_inscription_utility_fixture()
    reports = compare_inscription_policies(events, budget=3, seed=0)

    utility = reports["utility_write"]
    relevance = reports["relevance_write"]
    always = reports["always_write"]
    never = reports["never_write"]

    assert utility.future_task_lift == 1.0
    assert utility.false_write_rate == 0.0
    assert utility.missed_useful_write_rate == 0.0
    assert utility.utility_per_written_record == 1.0

    assert relevance.future_task_lift < utility.future_task_lift
    assert relevance.false_write_rate > utility.false_write_rate
    assert always.false_write_rate > utility.false_write_rate
    assert never.missed_useful_write_rate == 1.0


def test_correction_chain_preserves_update_lineage_vs_conclusion_only():
    cases = make_correction_chain_fixture()
    reports = compare_correction_policies(cases)

    chain = reports["correction_chain"]
    conclusion = reports["conclusion_only"]
    none = reports["no_memory"]

    assert chain.nodes_written == 2
    assert chain.prior_belief_recall == 1.0
    assert chain.evidence_recall == 1.0
    assert chain.update_operation_recall == 1.0
    assert chain.revised_belief_accuracy == 1.0
    assert chain.delta_accuracy == 1.0
    assert chain.transfer_success == 1.0
    assert chain.false_update_rate == 0.0
    assert chain.overgeneralization_rate == 0.0

    assert conclusion.revised_belief_accuracy == 1.0
    assert conclusion.prior_belief_recall == 0.0
    assert conclusion.evidence_recall == 0.0
    assert conclusion.update_operation_recall == 0.0
    assert conclusion.delta_accuracy == 0.0
    assert conclusion.transfer_success == 0.0
    assert conclusion.false_update_rate == 1.0

    assert none.revised_belief_accuracy == 0.0
    assert none.false_update_rate == 0.0


def test_residual_attention_improves_retrieval_and_source_discounting():
    candidates = make_residual_attention_fixture()
    reports = compare_residual_attention_policies(candidates, k=3)

    semantic = reports["semantic_only"]
    recency = reports["semantic_recency"]
    residual = reports["residual_posture"]
    source_aware = reports["residual_posture_source"]

    assert residual.transition_effective_retrieval_precision > semantic.transition_effective_retrieval_precision
    assert residual.transition_effective_retrieval_precision >= recency.transition_effective_retrieval_precision
    assert residual.confirmation_attractor_rate > source_aware.confirmation_attractor_rate
    assert source_aware.transition_effective_retrieval_precision == 1.0
    assert source_aware.distractor_resistance == 1.0


def test_self_index_binding_prevents_cross_project_user_and_role_leakage():
    cases = make_self_index_fixture()
    reports = compare_self_index_policies(cases)

    self_indexed = reports["self_indexed"]
    project_only = reports["project_only"]
    global_memory = reports["global_memory"]

    assert self_indexed.correct_binding_rate == 1.0
    assert self_indexed.wrong_project_application_rate == 0.0
    assert self_indexed.wrong_user_leakage_rate == 0.0
    assert self_indexed.role_conflict_rate == 0.0
    assert self_indexed.commitment_preservation_rate == 1.0

    assert project_only.correct_binding_rate < self_indexed.correct_binding_rate
    assert project_only.wrong_user_leakage_rate > self_indexed.wrong_user_leakage_rate
    assert project_only.role_conflict_rate > self_indexed.role_conflict_rate

    assert global_memory.correct_binding_rate < self_indexed.correct_binding_rate
    assert global_memory.wrong_project_application_rate > self_indexed.wrong_project_application_rate
    assert global_memory.wrong_user_leakage_rate > self_indexed.wrong_user_leakage_rate


def test_coupled_field_probe_has_nonzero_cross_effects_and_source_discounting():
    source_aware = coupled_field_probe(source_aware=True)
    source_blind = coupled_field_probe(source_aware=False)

    assert source_aware.attention_shift_after_memory_ablation > 0
    assert source_aware.write_shift_after_attention_ablation > 0
    assert source_aware.novelty_breakthrough_threshold < float("inf")

    assert source_blind.echo_amplification_rate > source_aware.echo_amplification_rate
    assert source_blind.novelty_breakthrough_threshold > source_aware.novelty_breakthrough_threshold


def test_controlled_roadmap_runner_summarizes_all_primitives(tmp_path):
    summary = run_controlled_roadmap_validations(seed=0)

    assert summary["source_routing"]["source_sensitive"]["route_accuracy"] == 1.0
    assert summary["source_routing"]["source_blind"]["echo_promotion_rate"] == 1.0
    assert summary["inscription_utility"]["utility_write"]["future_task_lift"] == 1.0
    assert summary["correction_chains"]["correction_chain"]["transfer_success"] == 1.0
    assert summary["residual_attention"]["residual_posture_source"]["transition_effective_retrieval_precision"] == 1.0
    assert summary["self_index_binding"]["self_indexed"]["correct_binding_rate"] == 1.0
    assert summary["coupled_field"]["source_aware"]["attention_shift_after_memory_ablation"] > 0

    paths = write_roadmap_validation_outputs(tmp_path, seed=0)
    assert paths["summary"].exists()
    assert paths["source_routing"].exists()
    rows = read_validation_jsonl(paths["source_routing"])
    assert len(rows) == 4


def test_controlled_replication_reports_effect_stability(tmp_path):
    summary = run_controlled_replication(seed_count=5, start_seed=0)

    assert summary["seed_count"] == 5
    assert summary["acceptance"]["toy_seed_count_met"] is False
    assert summary["acceptance"]["effect_direction_hold_rate_met"] is True
    assert summary["acceptance"]["minimum_effect_hold_rate"] == 1.0
    assert summary["metrics"]["source_sensitive.route_accuracy"]["mean"] == 1.0
    assert summary["metrics"]["source_blind.echo_promotion_rate"]["mean"] == 1.0
    assert summary["effect_directions"]["source_sensitive_route_beats_source_blind"]["hold_rate"] == 1.0

    paths = write_controlled_replication_outputs(tmp_path, seed_count=5, start_seed=10)
    assert paths["summary"].exists()
    assert paths["runs"].exists()
    assert len(paths["runs"].read_text(encoding="utf-8").strip().splitlines()) == 5


def test_real_component_report_structure_with_deterministic_embedder(tmp_path):
    report = run_real_embedding_validation(
        model_name="hash-embed-test",
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )

    assert report.embedding_available is True
    assert report.dim == 64
    assert report.source_route_accuracy == 1.0
    assert report.retrieval_hit_rate == 1.0
    assert report.fold_force is not None and report.fold_force > 0
    assert report.cost_ledger == {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}

    path = write_real_component_validation_output(
        tmp_path,
        model_name="hash-embed-test",
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )
    assert path.exists()


def test_real_embedding_replication_with_deterministic_embedder(tmp_path):
    summary = run_real_embedding_replication(
        model_name="hash-embed-test",
        seed_count=4,
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )

    assert summary["embedding_available"] is True
    assert summary["seed_count"] == 4
    assert summary["acceptance"]["real_seed_count_met"] is False
    assert summary["metrics"]["retrieval_hit_rate"]["n"] == 4
    assert summary["metrics"]["mean_fold_force"]["n"] == 4
    assert len(summary["runs"]) == 4

    paths = write_real_embedding_replication_output(
        tmp_path,
        model_name="hash-embed-test",
        seed_count=4,
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )
    assert paths["summary"].exists()
    assert paths["runs"].exists()


def test_live_llm_validation_skips_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    report = run_live_llm_validation()

    assert report.status == "skipped"
    assert report.provider == "openai"
    assert report.n_cases == 0
    assert report.route_accuracy is None
    assert report.reason == "OPENAI_API_KEY not set"
    assert report.cost_ledger == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }


def test_live_llm_validation_skips_without_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    report = run_live_llm_validation(provider="anthropic")

    assert report.status == "skipped"
    assert report.provider == "anthropic"
    assert report.reason == "ANTHROPIC_API_KEY not set"


def test_live_llm_validation_runs_with_stub_call():
    report = run_live_llm_validation(
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        llm_call=echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )

    assert report.status == "passed"
    assert report.provider == "openai"
    assert report.n_cases == 4
    assert report.route_accuracy == 1.0
    assert report.retrieval_hit_rate == 1.0
    assert report.echo_promotion_rate == 0.0
    assert report.operation_records == 4
    assert report.cost_ledger["api_calls"] == 8
    assert report.cost_ledger["total_tokens"] == 0


def test_live_llm_validation_writer_records_stub_artifact(tmp_path):
    path = write_live_llm_validation_output(
        tmp_path,
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        llm_call=echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )

    assert path.exists()
    rows = path.read_text(encoding="utf-8")
    assert '"status": "passed"' in rows
    assert '"api_calls": 8' in rows


def test_live_llm_replication_runs_with_stub_call_and_audit_log(tmp_path):
    report = run_live_llm_replication(
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        seed_count=1,
        llm_call=echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )

    assert report["status"] == "failed"
    assert report["seed_count"] == 1
    assert report["cost_ledger"]["api_calls"] == 8
    assert report["acceptance"]["live_seed_count_met"] is False
    assert report["acceptance"]["audit_event_count_met"] is True
    assert len(report["audit_events"]) == 4
    assert "with_memory_prompt" in report["audit_events"][0]
    assert "without_memory_response" in report["audit_events"][0]

    paths = write_live_llm_replication_outputs(
        tmp_path,
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        seed_count=1,
        llm_call=echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )
    assert paths["summary"].exists()
    assert paths["audit"].exists()
    assert len(paths["audit"].read_text(encoding="utf-8").strip().splitlines()) == 4

    custom_paths = write_live_llm_replication_outputs(
        tmp_path,
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        seed_count=1,
        llm_call=echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        summary_filename="custom_live_summary.json",
        audit_filename="custom_live_audit.jsonl",
    )
    assert custom_paths["summary"].name == "custom_live_summary.json"
    assert custom_paths["audit"].name == "custom_live_audit.jsonl"


def test_live_llm_replication_supports_billing_refund_case_family():
    report = run_live_llm_replication(
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        seed_count=1,
        llm_call=echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        case_family="billing_refund",
        include_audit_text=False,
    )

    assert report["case_family"] == "billing_refund"
    assert report["metrics"]["route_accuracy"]["mean"] == 1.0
    assert report["metrics"]["retrieval_hit_rate"]["mean"] == 1.0
    assert report["metrics"]["source_label_accuracy"]["mean"] == 1.0
    assert len(report["audit_events"]) == 4
    assert {event["case_family"] for event in report["audit_events"]} == {"billing_refund"}
    assert report["audit_events"][0]["expected_retrieved_ids"] == ["E1"]


def test_live_llm_replication_supports_security_rotation_case_family():
    report = run_live_llm_replication(
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        seed_count=1,
        llm_call=echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        case_family="security_rotation",
        include_audit_text=False,
    )

    assert report["case_family"] == "security_rotation"
    assert report["metrics"]["route_accuracy"]["mean"] == 1.0
    assert report["metrics"]["retrieval_hit_rate"]["mean"] == 1.0
    assert report["metrics"]["source_label_accuracy"]["mean"] == 1.0
    assert len(report["audit_events"]) == 4
    assert {event["case_family"] for event in report["audit_events"]} == {"security_rotation"}
    assert report["audit_events"][3]["expected_route"] == ROUTE_QUARANTINE


def test_live_llm_replication_rejects_unknown_case_family():
    with pytest.raises(ValueError, match="unsupported live case family"):
        run_live_llm_replication(
            provider="openai",
            model="echo-test",
            embedding_model="hash-embed-test",
            seed_count=1,
            llm_call=echo_call(),
            embed_fn=lambda text: hash_embed(text, 64),
            dim=64,
            case_family="unknown_family",
        )


def test_live_llm_replication_retries_empty_transition_response():
    echo = echo_call()
    calls = {"n": 0}

    def flaky_call(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""
        return echo(prompt)

    report = run_live_llm_replication(
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        seed_count=1,
        llm_call=flaky_call,
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        include_audit_text=False,
        empty_response_retries=1,
    )

    first_event = report["audit_events"][0]
    assert report["cost_ledger"]["api_calls"] == 9
    assert report["acceptance"]["paired_baseline_calls_met"] is True
    assert first_event["with_memory_attempt_count"] == 2
    assert first_event["with_memory_empty_response_count"] == 1
    assert first_event["with_memory_response_chars"] > 0


def test_live_provider_model_comparison_writer_runs_stub_targets(tmp_path):
    paths = write_live_provider_model_comparison_outputs(
        tmp_path,
        targets=[
            {"provider": "openai", "model": "echo-a"},
            {"provider": "anthropic", "model": "echo-b"},
        ],
        seed_count=5,
        min_passed_targets=2,
        max_output_tokens=777,
        artifact_prefix="custom_matrix",
        llm_call_factory=lambda _provider, _model: echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        include_audit_text=False,
    )

    report = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["target_count"] == 2
    assert report["attempted_count"] == 2
    assert report["passed_count"] == 2
    assert report["max_output_tokens"] == 777
    assert report["artifact_prefix"] == "custom_matrix"
    assert report["acceptance"]["provider_model_comparison_gate_met"] is True
    assert report["total_cost_ledger"]["api_calls"] == 80
    assert {target["provider"] for target in report["targets"]} == {"openai", "anthropic"}
    assert all(target["provider_valid_route_accuracy"] == 1.0 for target in report["targets"])
    assert all(target["primitive_failure_count"] == 0 for target in report["targets"])
    assert paths["openai_echo-a_summary"].exists()
    assert paths["openai_echo-a_summary"].name == "custom_matrix_openai_echo-a_summary.json"
    assert paths["anthropic_echo-b_diagnostics"].exists()

    audit = paths["openai_echo-a_audit"].read_text(encoding="utf-8")
    assert "with_memory_prompt" not in audit


def test_live_provider_model_comparison_rejects_invalid_max_output_tokens(tmp_path):
    with pytest.raises(ValueError, match="max_output_tokens"):
        write_live_provider_model_comparison_outputs(
            tmp_path,
            targets=["openai:echo-a"],
            max_output_tokens=0,
            llm_call_factory=lambda _provider, _model: echo_call(),
            embed_fn=lambda text: hash_embed(text, 64),
            dim=64,
        )


def test_live_provider_model_comparison_skips_missing_provider_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    paths = write_live_provider_model_comparison_outputs(
        tmp_path,
        targets=["openai:skip-model"],
        seed_count=5,
        min_passed_targets=1,
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
    )
    report = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert report["status"] == "skipped"
    assert report["attempted_count"] == 0
    assert report["skipped_count"] == 1
    assert report["targets"][0]["reason"] == "OPENAI_API_KEY not set"
    assert report["total_cost_ledger"]["api_calls"] == 0


def test_live_provider_model_comparison_keeps_provider_output_failure_as_nonprimitive(tmp_path):
    def factory(_provider, model):
        return (lambda _prompt: "") if model == "empty-output" else echo_call()

    paths = write_live_provider_model_comparison_outputs(
        tmp_path,
        targets=["openai:echo-a", "openai:empty-output"],
        seed_count=5,
        min_passed_targets=1,
        llm_call_factory=factory,
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        include_audit_text=False,
        empty_response_retries=1,
    )
    report = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["passed_count"] == 1
    assert report["failed_count"] == 1
    assert report["primitive_failure_count"] == 0
    assert report["provider_output_boundary_failure_count"] == 10
    assert report["acceptance"]["all_attempted_targets_passed"] is False
    assert report["acceptance"]["no_primitive_failures_met"] is True
    assert report["acceptance"]["provider_model_comparison_gate_met"] is True


def test_live_provider_model_comparison_reuses_existing_target_artifacts(tmp_path):
    first_paths = write_live_provider_model_comparison_outputs(
        tmp_path,
        targets=["openai:echo-a"],
        seed_count=5,
        llm_call_factory=lambda _provider, _model: echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        include_audit_text=False,
    )
    calls = {"n": 0}

    def fail_if_called(_provider, _model):
        calls["n"] += 1
        raise AssertionError("existing target should be reused")

    second_paths = write_live_provider_model_comparison_outputs(
        tmp_path,
        targets=["openai:echo-a"],
        seed_count=5,
        llm_call_factory=fail_if_called,
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        include_audit_text=False,
        reuse_existing=True,
    )
    report = json.loads(second_paths["summary"].read_text(encoding="utf-8"))

    assert calls["n"] == 0
    assert first_paths["openai_echo-a_summary"] == second_paths["openai_echo-a_summary"]
    assert report["status"] == "passed"
    assert report["targets"][0]["reused_existing"] is True
    assert report["targets"][0]["provider_valid_route_accuracy"] == 1.0


def test_live_llm_replication_stub_covers_seed4_boundary_case():
    report = run_rerank_boundary_regression()
    seed4_turn4 = report["seed4_turn4"]

    assert report["status"] == "passed"
    assert report["route_accuracy_mean"] == 1.0
    assert report["retrieval_hit_rate_mean"] == 1.0
    assert seed4_turn4["query"] == "which uncertain claim says legal forbids rollback?"
    assert seed4_turn4["retrieved_ids"] == ["F1"]
    assert seed4_turn4["selected_route"] == ROUTE_QUARANTINE


def test_rerank_boundary_regression_writer_records_artifact(tmp_path):
    path = write_rerank_boundary_regression_output(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["acceptance"]["boundary_retrieval_fixed"] is True
    assert report["acceptance"]["boundary_route_fixed"] is True


def test_live_replication_diagnostics_classifies_retrieval_boundary(tmp_path):
    summary = {
        "status": "passed",
        "provider": "openai",
        "model": "test-model",
        "seed_count": 1,
        "metrics": {"route_accuracy": {"mean": 0.75}},
        "cost_ledger": {"api_calls": 8, "total_tokens": 100},
    }
    audit_rows = [
        {
            "seed": 4,
            "turn_id": 4,
            "query": "which uncertain claim says legal forbids rollback?",
            "expected_retrieved_ids": ["F1"],
            "retrieved_ids": ["E1"],
            "expected_route": ROUTE_QUARANTINE,
            "selected_route": ROUTE_OPERATION_MEMORY,
            "realized_fold_force": 1.2,
            "transition_delta": 1.2,
            "with_memory_response_chars": 100,
            "without_memory_response_chars": 0,
        }
    ]
    summary_path = tmp_path / "summary.json"
    audit_path = tmp_path / "audit.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    audit_path.write_text("\n".join(json.dumps(row) for row in audit_rows) + "\n", encoding="utf-8")

    report = analyze_live_replication(summary_path, audit_path)

    assert report["failure_count"] == 1
    assert report["failure_counts"]["retrieval_miss+route_miss"] == 1
    assert report["failures"][0]["likely_boundary"].startswith("embedding_confusability")
    assert report["empty_response_count"] == 1
    assert report["gate_semantics"]["primitive_failure_count"] == 1
    assert report["gate_semantics"]["provider_output_boundary_failure_count"] == 0
    assert report["gate_semantics"]["route_failure_count_excluding_provider_output"] == 1
    assert report["gate_semantics"]["route_accuracy_if_provider_output_valid"] == 0.0
    assert report["gate_semantics"]["without_memory_final_empty_count"] == 1

    path = write_live_replication_diagnostics(
        tmp_path,
        summary_filename="summary.json",
        audit_filename="audit.jsonl",
        diagnostics_filename="diagnostics.json",
    )
    assert path.exists()
    assert path.name == "diagnostics.json"


def test_live_replication_diagnostics_classifies_empty_transition_route_boundary(tmp_path):
    summary = {
        "status": "passed",
        "provider": "openai",
        "model": "test-model",
        "seed_count": 1,
        "metrics": {"route_accuracy": {"mean": 0.75}},
        "cost_ledger": {"api_calls": 8, "total_tokens": 100},
    }
    audit_rows = [
        {
            "seed": 2,
            "turn_id": 2,
            "query": "what prior memory explains service restoration after deploy migration timeout?",
            "expected_retrieved_ids": ["R1"],
            "retrieved_ids": ["R1"],
            "expected_route": ROUTE_OPERATION_MEMORY,
            "selected_route": "null",
            "realized_fold_force": 0.0,
            "transition_delta": 0.0,
            "with_memory_response_chars": 0,
            "without_memory_response_chars": 0,
        }
    ]
    summary_path = tmp_path / "summary.json"
    audit_path = tmp_path / "audit.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    audit_path.write_text("\n".join(json.dumps(row) for row in audit_rows) + "\n", encoding="utf-8")

    report = analyze_live_replication(summary_path, audit_path)

    assert report["failure_count"] == 1
    assert report["failure_counts"]["route_miss"] == 1
    assert report["failures"][0]["likely_boundary"] == "provider_empty_with_memory_zero_fold_force"
    assert report["failures"][0]["failure_plane"] == "provider_output_validity"
    assert report["gate_semantics"]["primitive_failure_count"] == 0
    assert report["gate_semantics"]["provider_output_boundary_failure_count"] == 1
    assert report["gate_semantics"]["route_failure_count_excluding_provider_output"] == 0
    assert report["gate_semantics"]["route_accuracy_if_provider_output_valid"] is None
    assert report["gate_semantics"]["with_memory_final_empty_count"] == 1
    assert any(
        "Track provider-empty with-memory transitions separately" in recommendation
        for recommendation in report["recommendations"]
    )
    assert any(
        "provider-valid" in recommendation
        for recommendation in report["recommendations"]
    )


@pytest.mark.live
def test_live_openai_validation_gate_runs_with_configured_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not configured")

    report = run_live_llm_validation(
        provider="openai",
        model=os.environ.get("OPENAI_TEST_MODEL"),
    )

    assert report.status == "passed", report.reason
    assert report.provider == "openai"
    assert report.n_cases == 4
    assert report.route_accuracy == 1.0
    assert report.retrieval_hit_rate == 1.0
    assert report.echo_promotion_rate == 0.0
    assert report.cost_ledger["api_calls"] == 8
    assert report.cost_ledger["total_tokens"] > 0

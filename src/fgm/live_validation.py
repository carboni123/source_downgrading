"""Live LLM validation gate for the roadmap.

This module records a durable live-replication status. Without an API key it
returns a skipped report; with a key it runs the same source-routing primitive
through an LLM-backed transition and records a cost ledger.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from fgm.core import (
    FGMAgent,
    ROUTE_OPERATION_MEMORY,
    ROUTE_QUARANTINE,
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_SIMULATION,
    Array,
)
from fgm.llm import (
    LLMCall,
    LLMTransition,
    anthropic_call,
    openai_call,
)
from fgm.validation import ValidationRecord, score_validation_records


@dataclass(frozen=True)
class LiveValidationReport:
    status: str
    provider: str
    model: str
    embedding_model: str
    n_cases: int
    route_accuracy: Optional[float]
    retrieval_hit_rate: Optional[float]
    echo_promotion_rate: Optional[float]
    mean_fold_force: Optional[float]
    operation_records: int
    cost_ledger: Dict[str, int]
    reason: Optional[str]


def run_live_llm_validation(
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    api_key: Optional[str] = None,
    llm_call: Optional[LLMCall] = None,
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
    empty_response_retries: int = 0,
    max_output_tokens: Optional[int] = None,
    case_family: str = "deploy_rollback",
) -> LiveValidationReport:
    """Run or skip the live LLM validation gate."""
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive when provided")
    _live_case_family(case_family)
    provider = provider.lower()
    model = model or _default_model(provider)
    key_env = _provider_key_env(provider)
    key = api_key or os.environ.get(key_env)
    if llm_call is None and not key:
        return LiveValidationReport(
            status="skipped",
            provider=provider,
            model=model,
            embedding_model=embedding_model,
            n_cases=0,
            route_accuracy=None,
            retrieval_hit_rate=None,
            echo_promotion_rate=None,
            mean_fold_force=None,
            operation_records=0,
            cost_ledger={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
            reason=f"{key_env} not set",
        )

    if embed_fn is None:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(embedding_model)

            def embed_fn(text: str) -> Array:
                return st_model.encode([text], normalize_embeddings=True)[0]

            dim = int(embed_fn("dimension probe").shape[0])
        except Exception as exc:  # pragma: no cover - environment-dependent
            return LiveValidationReport(
                status="skipped",
                provider=provider,
                model=model,
                embedding_model=embedding_model,
                n_cases=0,
                route_accuracy=None,
                retrieval_hit_rate=None,
                echo_promotion_rate=None,
                mean_fold_force=None,
                operation_records=0,
                cost_ledger={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
                reason=f"embedding unavailable: {exc}",
            )
    if embed_fn is None or dim is None:
        raise ValueError("embed_fn and dim must be provided together")

    usage: Dict[str, int] = {}
    if llm_call is None:
        try:
            if provider == "openai":
                llm_call = openai_call(
                    api_key=key,
                    model=model,
                    max_output_tokens=max_output_tokens or 300,
                    temperature=None,
                    usage_tracker=usage,
                )
            elif provider == "anthropic":
                llm_call = anthropic_call(
                    api_key=key,
                    model=model,
                    max_tokens=max_output_tokens or 150,
                    temperature=0.0,
                    usage_tracker=usage,
                )
            else:
                raise ValueError(f"unsupported live LLM provider: {provider}")
        except ImportError as exc:  # pragma: no cover - environment-dependent
            return LiveValidationReport(
                status="skipped",
                provider=provider,
                model=model,
                embedding_model=embedding_model,
                n_cases=0,
                route_accuracy=None,
                retrieval_hit_rate=None,
                echo_promotion_rate=None,
                mean_fold_force=None,
                operation_records=0,
                cost_ledger={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
                reason=f"{provider} SDK unavailable: {exc}",
            )

    transition = LLMTransition(
        llm_call,
        embed_fn,
        dim=dim,
        empty_response_retries=empty_response_retries,
    )
    agent = FGMAgent(
        dim=dim,
        transition_fn=transition,
        embed_fn=embed_fn,
        fold_threshold=0.001,
        retrieval_k=1,
        auto_compress=False,
    )
    try:
        records = _run_live_source_cases(agent, model, case_family=case_family)
    except Exception as exc:  # pragma: no cover - provider/network-dependent
        return LiveValidationReport(
            status="failed",
            provider=provider,
            model=model,
            embedding_model=embedding_model,
            n_cases=0,
            route_accuracy=None,
            retrieval_hit_rate=None,
            echo_promotion_rate=None,
            mean_fold_force=None,
            operation_records=len(agent.operations.all_operations()),
            cost_ledger={
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": _total_tokens(usage),
                "api_calls": int(transition.stats.n_calls),
            },
            reason=f"live validation failed: {exc}",
        )
    metrics = score_validation_records(records)
    cost = {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "total_tokens": _total_tokens(usage),
        "api_calls": int(transition.stats.n_calls),
    }
    passed = (
        metrics["route_accuracy"] == 1.0
        and metrics["retrieval_hit_rate"] == 1.0
        and metrics["echo_promotion_rate"] == 0.0
    )
    return LiveValidationReport(
        status="passed" if passed else "failed",
        provider=provider,
        model=model,
        embedding_model=embedding_model,
        n_cases=len(records),
        route_accuracy=metrics["route_accuracy"],
        retrieval_hit_rate=metrics["retrieval_hit_rate"],
        echo_promotion_rate=metrics["echo_promotion_rate"],
        mean_fold_force=float(np.mean([record.realized_fold_force or 0.0 for record in records])),
        operation_records=len(agent.operations.all_operations()),
        cost_ledger=cost,
        reason=None if passed else "fixed-truth live metrics did not pass",
    )


def write_live_llm_validation_output(
    output_dir: str | Path = "results",
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    api_key: Optional[str] = None,
    llm_call: Optional[LLMCall] = None,
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
    empty_response_retries: int = 0,
    max_output_tokens: Optional[int] = None,
    case_family: str = "deploy_rollback",
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = run_live_llm_validation(
        provider=provider,
        model=model,
        embedding_model=embedding_model,
        api_key=api_key,
        llm_call=llm_call,
        embed_fn=embed_fn,
        dim=dim,
        empty_response_retries=empty_response_retries,
        max_output_tokens=max_output_tokens,
        case_family=case_family,
    )
    path = output / "live_llm_validation_summary.json"
    path.write_text(json.dumps(_json_safe(asdict(report)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_live_llm_replication(
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    seed_count: int = 5,
    start_seed: int = 0,
    api_key: Optional[str] = None,
    llm_call: Optional[LLMCall] = None,
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
    include_audit_text: bool = True,
    empty_response_retries: int = 0,
    max_output_tokens: Optional[int] = None,
    case_family: str = "deploy_rollback",
) -> Dict[str, Any]:
    """Run live source-routing replication across seeded prompt variants."""
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive when provided")
    _live_case_family(case_family)

    provider = provider.lower()
    model = model or _default_model(provider)
    key_env = _provider_key_env(provider)
    key = api_key or os.environ.get(key_env)
    external_provider = llm_call is None
    if llm_call is None and not key:
        return _json_safe({
            "status": "skipped",
            "provider": provider,
            "model": model,
            "embedding_model": embedding_model,
            "seed_count": seed_count,
            "start_seed": start_seed,
            "max_output_tokens": max_output_tokens,
            "case_family": case_family,
            "runs": [],
            "audit_events": [],
            "metrics": {},
            "cost_ledger": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
            "acceptance": {"live_seed_count_met": seed_count >= 5, "all_live_replication_gates_met": False},
            "reason": f"{key_env} not set",
        })

    if embed_fn is None:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(embedding_model)

            def embed_fn(text: str) -> Array:
                return st_model.encode([text], normalize_embeddings=True)[0]

            dim = int(embed_fn("dimension probe").shape[0])
        except Exception as exc:  # pragma: no cover - environment-dependent
            return _json_safe({
                "status": "skipped",
                "provider": provider,
                "model": model,
                "embedding_model": embedding_model,
                "seed_count": seed_count,
                "start_seed": start_seed,
                "max_output_tokens": max_output_tokens,
                "case_family": case_family,
                "runs": [],
                "audit_events": [],
                "metrics": {},
                "cost_ledger": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
                "acceptance": {"live_seed_count_met": seed_count >= 5, "all_live_replication_gates_met": False},
                "reason": f"embedding unavailable: {exc}",
            })
    if embed_fn is None or dim is None:
        raise ValueError("embed_fn and dim must be provided together")

    usage: Dict[str, int] = {}
    if llm_call is None:
        try:
            if provider == "openai":
                llm_call = openai_call(
                    api_key=key,
                    model=model,
                    max_output_tokens=max_output_tokens or 300,
                    temperature=None,
                    usage_tracker=usage,
                )
            elif provider == "anthropic":
                llm_call = anthropic_call(
                    api_key=key,
                    model=model,
                    max_tokens=max_output_tokens or 150,
                    temperature=0.0,
                    usage_tracker=usage,
                )
            else:
                raise ValueError(f"unsupported live LLM provider: {provider}")
        except ImportError as exc:  # pragma: no cover - environment-dependent
            return _json_safe({
                "status": "skipped",
                "provider": provider,
                "model": model,
                "embedding_model": embedding_model,
                "seed_count": seed_count,
                "start_seed": start_seed,
                "max_output_tokens": max_output_tokens,
                "case_family": case_family,
                "runs": [],
                "audit_events": [],
                "metrics": {},
                "cost_ledger": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
                "acceptance": {"live_seed_count_met": seed_count >= 5, "all_live_replication_gates_met": False},
                "reason": f"{provider} SDK unavailable: {exc}",
            })

    runs = []
    audit_events = []
    total_api_calls = 0
    for seed in range(start_seed, start_seed + seed_count):
        transition = LLMTransition(
            llm_call,
            embed_fn,
            dim=dim,
            empty_response_retries=empty_response_retries,
        )
        agent = FGMAgent(
            dim=dim,
            transition_fn=transition,
            embed_fn=embed_fn,
            fold_threshold=0.001,
            retrieval_k=1,
            auto_compress=False,
        )
        try:
            records = _run_live_source_cases(agent, model, seed=seed, case_family=case_family)
        except Exception as exc:  # pragma: no cover - provider/network-dependent
            cost = {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": _total_tokens(usage),
                "api_calls": total_api_calls + int(transition.stats.n_calls),
            }
            return _json_safe({
                "status": "failed",
                "provider": provider,
                "model": model,
                "embedding_model": embedding_model,
                "seed_count": seed_count,
                "start_seed": start_seed,
                "max_output_tokens": max_output_tokens,
                "case_family": case_family,
                "runs": runs,
                "audit_events": audit_events,
                "metrics": {},
                "cost_ledger": cost,
                "acceptance": {"live_seed_count_met": seed_count >= 5, "all_live_replication_gates_met": False},
                "reason": f"live replication failed at seed {seed}: {exc}",
            })
        total_api_calls += int(transition.stats.n_calls)
        metrics = score_validation_records(records)
        run = {
            "seed": seed,
            "route_accuracy": metrics["route_accuracy"],
            "retrieval_hit_rate": metrics["retrieval_hit_rate"],
            "source_label_accuracy": metrics["source_label_accuracy"],
            "echo_promotion_rate": metrics["echo_promotion_rate"],
            "false_write_rate": metrics["false_write_rate"],
            "quarantine_recall": metrics["quarantine_recall"],
            "mean_fold_force": float(np.mean([record.realized_fold_force or 0.0 for record in records])),
            "operation_records": len(agent.operations.all_operations()),
            "api_calls": int(transition.stats.n_calls),
        }
        runs.append(run)
        audit_events.extend(_make_live_audit_events(
            provider=provider,
            model=model,
            embedding_model=embedding_model,
            case_family=case_family,
            seed=seed,
            records=records,
            history=transition.stats.history,
            include_audit_text=include_audit_text,
        ))

    cost = {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "total_tokens": _total_tokens(usage),
        "api_calls": total_api_calls,
    }
    metric_summary = {
        "route_accuracy": _summarize_numeric(run["route_accuracy"] for run in runs),
        "retrieval_hit_rate": _summarize_numeric(run["retrieval_hit_rate"] for run in runs),
        "source_label_accuracy": _summarize_numeric(run["source_label_accuracy"] for run in runs),
        "echo_promotion_rate": _summarize_numeric(run["echo_promotion_rate"] for run in runs),
        "false_write_rate": _summarize_numeric(run["false_write_rate"] for run in runs),
        "quarantine_recall": _summarize_numeric(run["quarantine_recall"] for run in runs),
        "mean_fold_force": _summarize_numeric(run["mean_fold_force"] for run in runs),
        "operation_records": _summarize_numeric(run["operation_records"] for run in runs),
        "api_calls_per_seed": _summarize_numeric(run["api_calls"] for run in runs),
    }
    acceptance = {
        "live_seed_count_met": seed_count >= 5,
        "route_accuracy_mean_met": metric_summary["route_accuracy"]["mean"] >= 0.9,
        "retrieval_hit_rate_mean_met": metric_summary["retrieval_hit_rate"]["mean"] >= 0.9,
        "source_label_accuracy_mean_met": metric_summary["source_label_accuracy"]["mean"] >= 0.9,
        "echo_promotion_control_met": metric_summary["echo_promotion_rate"]["mean"] == 0.0,
        "fold_force_positive_met": metric_summary["mean_fold_force"]["min"] > 0.0,
        "operation_records_mean_met": metric_summary["operation_records"]["mean"] >= 2.0,
        "paired_baseline_calls_met": cost["api_calls"] >= seed_count * 8,
        "cost_ledger_nonzero_met": (not external_provider) or cost["total_tokens"] > 0,
        "audit_event_count_met": len(audit_events) == seed_count * 4,
    }
    acceptance["all_live_replication_gates_met"] = all(acceptance.values())

    return _json_safe({
        "status": "passed" if acceptance["all_live_replication_gates_met"] else "failed",
        "provider": provider,
        "model": model,
        "embedding_model": embedding_model,
        "seed_count": seed_count,
        "start_seed": start_seed,
        "max_output_tokens": max_output_tokens,
        "case_family": case_family,
        "runs": runs,
        "audit_events": audit_events,
        "metrics": metric_summary,
        "cost_ledger": cost,
        "acceptance": acceptance,
        "reason": None if acceptance["all_live_replication_gates_met"] else "live replication gates did not pass",
    })


def write_live_llm_replication_outputs(
    output_dir: str | Path = "results",
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    seed_count: int = 5,
    start_seed: int = 0,
    api_key: Optional[str] = None,
    llm_call: Optional[LLMCall] = None,
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
    include_audit_text: bool = True,
    summary_filename: str = "live_llm_replication_summary.json",
    audit_filename: str = "live_llm_replication_audit.jsonl",
    empty_response_retries: int = 0,
    max_output_tokens: Optional[int] = None,
    case_family: str = "deploy_rollback",
) -> Dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = run_live_llm_replication(
        provider=provider,
        model=model,
        embedding_model=embedding_model,
        seed_count=seed_count,
        start_seed=start_seed,
        api_key=api_key,
        llm_call=llm_call,
        embed_fn=embed_fn,
        dim=dim,
        include_audit_text=include_audit_text,
        empty_response_retries=empty_response_retries,
        max_output_tokens=max_output_tokens,
        case_family=case_family,
    )
    audit_events = report.pop("audit_events", [])

    summary_path = output / summary_filename
    summary_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")

    audit_path = output / audit_filename
    with audit_path.open("w", encoding="utf-8") as handle:
        for event in audit_events:
            handle.write(json.dumps(_json_safe(event), sort_keys=True))
            handle.write("\n")
    return {"summary": summary_path, "audit": audit_path}


def _default_model(provider: str) -> str:
    if provider == "openai":
        return os.environ.get("OPENAI_LIVE_MODEL") or os.environ.get("OPENAI_TEST_MODEL") or "gpt-5.5"
    if provider == "anthropic":
        return (
            os.environ.get("ANTHROPIC_LIVE_MODEL")
            or os.environ.get("ANTHROPIC_TEST_MODEL")
            or "claude-haiku-4-5-20251001"
        )
    raise ValueError(f"unsupported live LLM provider: {provider}")


def _provider_key_env(provider: str) -> str:
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    raise ValueError(f"unsupported live LLM provider: {provider}")


def _total_tokens(usage: Dict[str, int]) -> int:
    if "total_tokens" in usage:
        return int(usage["total_tokens"])
    return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))


def _run_live_source_cases(
    agent: FGMAgent,
    model: str,
    seed: int = 0,
    *,
    case_family: str = "deploy_rollback",
) -> list[ValidationRecord]:
    family = _live_case_family(case_family)
    for record_id, text, source_label, source_confidence in family["memories"]:
        agent.add(
            text,
            record_id=record_id,
            source_label=source_label,
            source_confidence=source_confidence,
        )
    cases = _live_source_routing_cases(seed, case_family=case_family)
    records: list[ValidationRecord] = []
    for turn_id, (query, expected_id, expected_source, expected_route, future_utility) in enumerate(cases, start=1):
        result = agent.query(query)
        records.append(
            ValidationRecord.from_fold_result(
                run_id=f"live-llm-source-routing:{case_family}:{model}",
                seed=seed,
                turn_id=turn_id,
                fold_result=result,
                external_input_ids=[expected_id],
                predicted_fold_force=result.fold_force,
                expected_retrieved_ids=[expected_id],
                expected_source_labels={expected_id: expected_source},
                expected_active_source_labels={expected_id: SOURCE_RETRIEVED_MEMORY},
                expected_route=expected_route,
                future_utility_label=future_utility,
            )
        )
    return records


def _live_case_family(case_family: str) -> Dict[str, Any]:
    families = {
        "deploy_rollback": {
            "memories": [
                ("E1", "E1 external observation legal approved rollback for deploy migration", SOURCE_EXTERNAL, 1.0),
                (
                    "R1",
                    "R1 prior memory deploy migration timed out and rollback restored service",
                    SOURCE_EXTERNAL,
                    0.95,
                ),
                ("S1", "S1 simulated hypothesis hotfix might avoid rollback", SOURCE_SIMULATION, 0.7),
                ("F1", "F1 fabricated distractor rollback is forbidden by legal", SOURCE_FABRICATED, 0.9),
            ],
            "variants": [
                [
                    "what rollback approval applies to deploy migration?",
                    "which approval allows rollback for deployment migration?",
                    "what external evidence supports rollback for this migration?",
                    "what observation authorizes rollback for deploy migration?",
                    "which record says rollback was legally approved for deploy migration?",
                ],
                [
                    "what restored service after deploy migration timeout?",
                    "which recovery restored service after migration timeout?",
                    "what prior memory explains service restoration after deploy migration timeout?",
                    "how was service restored when the migration timed out?",
                    "which prior incident says service recovered after timeout?",
                ],
                [
                    "which hotfix rollback option is only hypothetical?",
                    "what rollback avoidance idea was simulated?",
                    "which hotfix proposal is hypothetical rather than observed?",
                    "what simulated branch suggests avoiding rollback?",
                    "which claim should remain simulation-tagged about hotfix rollback?",
                ],
                [
                    "which rollback legal claim is a distractor?",
                    "what adversarial note falsely says rollback is forbidden?",
                    "which legal rollback prohibition is fabricated?",
                    "what fabricated distractor should be quarantined about rollback?",
                    "which uncertain claim says legal forbids rollback?",
                ],
            ],
        },
        "billing_refund": {
            "memories": [
                ("E1", "E1 external observation finance approved refund for invoice adjustment", SOURCE_EXTERNAL, 1.0),
                (
                    "R1",
                    "R1 prior memory customer was overcharged and credit memo restored account balance",
                    SOURCE_EXTERNAL,
                    0.95,
                ),
                ("S1", "S1 simulated hypothesis coupon might avoid issuing refund", SOURCE_SIMULATION, 0.7),
                ("F1", "F1 fabricated distractor refund is prohibited by finance", SOURCE_FABRICATED, 0.9),
            ],
            "variants": [
                [
                    "what refund approval applies to invoice adjustment?",
                    "which approval allows refund for the invoice adjustment?",
                    "what external evidence supports a refund for this invoice?",
                    "what observation authorizes refund for invoice adjustment?",
                    "which record says finance approved the refund adjustment?",
                ],
                [
                    "what restored account balance after the customer overcharge?",
                    "which recovery restored balance after overcharge?",
                    "what prior memory explains account restoration after overcharge?",
                    "how was account balance restored when the customer was overcharged?",
                    "which prior incident says the balance recovered after overcharge?",
                ],
                [
                    "which coupon refund option is only hypothetical?",
                    "what refund avoidance idea was simulated?",
                    "which coupon proposal is hypothetical rather than observed?",
                    "what simulated branch suggests avoiding the refund?",
                    "which claim should remain simulation-tagged about coupon refund?",
                ],
                [
                    "which refund finance claim is a distractor?",
                    "what adversarial note falsely says refund is prohibited?",
                    "which finance refund prohibition is fabricated?",
                    "what fabricated distractor should be quarantined about refund?",
                    "which uncertain claim says finance forbids refund?",
                ],
            ],
        },
        "security_rotation": {
            "memories": [
                (
                    "E1",
                    "E1 external observation security approved credential rotation for API key exposure",
                    SOURCE_EXTERNAL,
                    1.0,
                ),
                (
                    "R1",
                    "R1 prior memory compromised API key audit was resolved after credential rotation",
                    SOURCE_EXTERNAL,
                    0.95,
                ),
                ("S1", "S1 simulated hypothesis firewall rule might avoid credential rotation", SOURCE_SIMULATION, 0.7),
                ("F1", "F1 fabricated distractor credential rotation is prohibited by security", SOURCE_FABRICATED, 0.9),
            ],
            "variants": [
                [
                    "what credential rotation approval applies to API key exposure?",
                    "which approval allows credential rotation for the exposed API key?",
                    "what external evidence supports credential rotation for this key?",
                    "what observation authorizes credential rotation after API key exposure?",
                    "which record says security approved credential rotation?",
                ],
                [
                    "what resolved the compromised API key audit?",
                    "which recovery resolved the API key audit?",
                    "what prior memory explains resolution after credential compromise?",
                    "how was the compromised key audit resolved?",
                    "which prior incident says credential rotation resolved the audit?",
                ],
                [
                    "which firewall credential rotation option is only hypothetical?",
                    "what rotation avoidance idea was simulated?",
                    "which firewall proposal is hypothetical rather than observed?",
                    "what simulated branch suggests avoiding credential rotation?",
                    "which claim should remain simulation-tagged about firewall rotation?",
                ],
                [
                    "which credential rotation security claim is a distractor?",
                    "what adversarial note falsely says rotation is prohibited?",
                    "which security rotation prohibition is fabricated?",
                    "what fabricated distractor should be quarantined about credential rotation?",
                    "which uncertain claim says security forbids credential rotation?",
                ],
            ],
        },
    }
    try:
        return families[case_family]
    except KeyError as exc:
        known = ", ".join(sorted(families))
        raise ValueError(f"unsupported live case family: {case_family}; expected one of {known}") from exc


def _live_source_routing_cases(
    seed: int,
    *,
    case_family: str = "deploy_rollback",
) -> list[tuple[str, str, str, str, bool]]:
    variants = _live_case_family(case_family)["variants"]
    idx = seed % len(variants[0])
    return [
        (variants[0][idx], "E1", SOURCE_EXTERNAL, ROUTE_OPERATION_MEMORY, True),
        (variants[1][idx], "R1", SOURCE_EXTERNAL, ROUTE_OPERATION_MEMORY, True),
        (variants[2][idx], "S1", SOURCE_SIMULATION, ROUTE_QUARANTINE, False),
        (variants[3][idx], "F1", SOURCE_FABRICATED, ROUTE_QUARANTINE, False),
    ]


def _make_live_audit_events(
    *,
    provider: str,
    model: str,
    embedding_model: str,
    case_family: str,
    seed: int,
    records: list[ValidationRecord],
    history: list[Dict[str, Any]],
    include_audit_text: bool,
) -> list[Dict[str, Any]]:
    events = []
    for index, record in enumerate(records):
        with_memory = history[index * 2] if index * 2 < len(history) else {}
        without_memory = history[index * 2 + 1] if index * 2 + 1 < len(history) else {}
        event: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "embedding_model": embedding_model,
            "case_family": case_family,
            "seed": seed,
            "turn_id": record.turn_id,
            "query": record.query,
            "retrieved_ids": record.retrieved_ids,
            "expected_retrieved_ids": record.expected_retrieved_ids,
            "selected_route": record.selected_route,
            "expected_route": record.expected_route,
            "realized_fold_force": record.realized_fold_force,
            "transition_delta": record.transition_delta,
            "with_memory_response_chars": len(str(with_memory.get("response", ""))),
            "without_memory_response_chars": len(str(without_memory.get("response", ""))),
            "with_memory_attempt_count": int(with_memory.get("attempt_count", 0) or 0),
            "without_memory_attempt_count": int(without_memory.get("attempt_count", 0) or 0),
            "with_memory_empty_response_count": int(with_memory.get("empty_response_count", 0) or 0),
            "without_memory_empty_response_count": int(without_memory.get("empty_response_count", 0) or 0),
        }
        if include_audit_text:
            event.update({
                "with_memory_prompt": with_memory.get("prompt"),
                "with_memory_response": with_memory.get("response"),
                "without_memory_prompt": without_memory.get("prompt"),
                "without_memory_response": without_memory.get("response"),
            })
        events.append(event)
    return events


def _summarize_numeric(values: Any) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    arr = np.asarray(clean, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {
        "n": len(clean),
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value

"""Diagnostics for live LLM replication artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def analyze_live_replication(
    summary_path: str | Path = "results/architecture/live_llm_replication_summary.json",
    audit_path: str | Path = "results/architecture/live_llm_replication_audit.jsonl",
) -> Dict[str, Any]:
    """Analyze live replication summary and audit logs for boundary cases."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    audit_events = _read_jsonl(audit_path)

    failures = [_classify_event(event) for event in audit_events if _is_failure(event)]
    empty_responses = [
        {
            "seed": event.get("seed"),
            "turn_id": event.get("turn_id"),
            "query": event.get("query"),
            "with_memory_response_chars": event.get("with_memory_response_chars", 0),
            "without_memory_response_chars": event.get("without_memory_response_chars", 0),
        }
        for event in audit_events
        if event.get("with_memory_response_chars", 0) == 0
        or event.get("without_memory_response_chars", 0) == 0
    ]

    failure_counts = _count_by(failure["failure_type"] for failure in failures)
    affected_seeds = sorted({failure["seed"] for failure in failures})
    affected_turns = sorted({failure["turn_id"] for failure in failures})
    gate_semantics = _gate_semantics(audit_events, failures)
    recommendations = _recommendations(failures, empty_responses, gate_semantics)

    return _json_safe({
        "source_summary": str(summary_path),
        "source_audit": str(audit_path),
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "seed_count": summary.get("seed_count"),
        "status": summary.get("status"),
        "audit_event_count": len(audit_events),
        "failure_count": len(failures),
        "failure_counts": failure_counts,
        "affected_seeds": affected_seeds,
        "affected_turns": affected_turns,
        "empty_response_count": len(empty_responses),
        "empty_responses": empty_responses,
        "gate_semantics": gate_semantics,
        "failures": failures,
        "metrics": summary.get("metrics", {}),
        "cost_ledger": summary.get("cost_ledger", {}),
        "recommendations": recommendations,
    })


def write_live_replication_diagnostics(
    output_dir: str | Path = "results/architecture",
    *,
    summary_filename: str = "live_llm_replication_summary.json",
    audit_filename: str = "live_llm_replication_audit.jsonl",
    diagnostics_filename: str = "live_llm_replication_diagnostics.json",
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = analyze_live_replication(
        summary_path=output / summary_filename,
        audit_path=output / audit_filename,
    )
    path = output / diagnostics_filename
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _is_failure(event: Dict[str, Any]) -> bool:
    return (
        not _retrieval_match(event)
        or not _route_match(event)
    )


def _classify_event(event: Dict[str, Any]) -> Dict[str, Any]:
    retrieved_ids = list(event.get("retrieved_ids", []))
    expected_ids = list(event.get("expected_retrieved_ids", []))
    retrieved_match = _retrieval_match(event)
    route_match = _route_match(event)
    failure_type = []
    if not retrieved_match:
        failure_type.append("retrieval_miss")
    if not route_match:
        failure_type.append("route_miss")

    return {
        "seed": event.get("seed"),
        "turn_id": event.get("turn_id"),
        "query": event.get("query"),
        "expected_retrieved_ids": expected_ids,
        "retrieved_ids": retrieved_ids,
        "expected_route": event.get("expected_route"),
        "selected_route": event.get("selected_route"),
        "realized_fold_force": event.get("realized_fold_force"),
        "transition_delta": event.get("transition_delta"),
        "with_memory_response_chars": event.get("with_memory_response_chars", 0),
        "without_memory_response_chars": event.get("without_memory_response_chars", 0),
        "with_memory_output_valid": _output_valid(event, "with_memory"),
        "without_memory_output_valid": _output_valid(event, "without_memory"),
        "with_memory_attempt_count": int(event.get("with_memory_attempt_count", 0) or 0),
        "without_memory_attempt_count": int(event.get("without_memory_attempt_count", 0) or 0),
        "with_memory_empty_response_count": int(event.get("with_memory_empty_response_count", 0) or 0),
        "without_memory_empty_response_count": int(event.get("without_memory_empty_response_count", 0) or 0),
        "retrieval_match": retrieved_match,
        "route_match": route_match,
        "failure_type": "+".join(failure_type),
        "likely_boundary": _likely_boundary(event),
        "failure_plane": _failure_plane(event),
    }


def _gate_semantics(audit_events: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_events = len(audit_events)
    provider_output_boundaries = [
        failure
        for failure in failures
        if failure["likely_boundary"] == "provider_empty_with_memory_zero_fold_force"
    ]
    primitive_failures = [
        failure
        for failure in failures
        if failure["failure_plane"] != "provider_output_validity"
    ]
    route_failures_excluding_provider_output = [
        failure
        for failure in primitive_failures
        if "route_miss" in failure["failure_type"]
    ]
    provider_valid_route_events = [
        event for event in audit_events if _output_valid(event, "with_memory")
    ]

    return {
        "interpretation": (
            "retrieval and route gates should be read over provider-valid with-memory events; "
            "final empty with-memory responses are provider-output validity failures, not primitive route failures"
        ),
        "total_events": total_events,
        "retrieval_failure_count": sum(1 for event in audit_events if not _retrieval_match(event)),
        "route_failure_count": sum(1 for event in audit_events if not _route_match(event)),
        "primitive_failure_count": len(primitive_failures),
        "provider_output_boundary_failure_count": len(provider_output_boundaries),
        "route_failure_count_excluding_provider_output": len(route_failures_excluding_provider_output),
        "route_accuracy_if_provider_output_valid": _rate(
            sum(1 for event in provider_valid_route_events if _route_match(event)),
            len(provider_valid_route_events),
        ),
        "retrieval_hit_rate": _rate(
            sum(1 for event in audit_events if _retrieval_match(event)),
            total_events,
        ),
        "with_memory_output_validity_rate": _rate(
            sum(1 for event in audit_events if _output_valid(event, "with_memory")),
            total_events,
        ),
        "without_memory_output_validity_rate": _rate(
            sum(1 for event in audit_events if _output_valid(event, "without_memory")),
            total_events,
        ),
        "with_memory_final_empty_count": sum(
            1 for event in audit_events if not _output_valid(event, "with_memory")
        ),
        "without_memory_final_empty_count": sum(
            1 for event in audit_events if not _output_valid(event, "without_memory")
        ),
        "with_memory_retry_event_count": sum(
            1
            for event in audit_events
            if int(event.get("with_memory_attempt_count", 0) or 0) > 1
            or int(event.get("with_memory_empty_response_count", 0) or 0) > 0
        ),
        "without_memory_retry_event_count": sum(
            1
            for event in audit_events
            if int(event.get("without_memory_attempt_count", 0) or 0) > 1
            or int(event.get("without_memory_empty_response_count", 0) or 0) > 0
        ),
        "with_memory_empty_attempt_count": sum(
            int(event.get("with_memory_empty_response_count", 0) or 0)
            for event in audit_events
        ),
        "without_memory_empty_attempt_count": sum(
            int(event.get("without_memory_empty_response_count", 0) or 0)
            for event in audit_events
        ),
    }


def _likely_boundary(event: Dict[str, Any]) -> str:
    expected = set(event.get("expected_retrieved_ids", []))
    retrieved = set(event.get("retrieved_ids", []))
    query = str(event.get("query", "")).lower()
    if expected == {"F1"} and "E1" in retrieved and "legal" in query and "rollback" in query:
        return (
            "embedding_confusability_between_trusted_external_evidence_and_fabricated_legal_negation"
        )
    if not expected.issubset(retrieved):
        return "embedding_retrieval_confusability"
    if (
        event.get("selected_route") != event.get("expected_route")
        and float(event.get("transition_delta") or 0.0) == 0.0
        and event.get("with_memory_response_chars", 0) == 0
    ):
        return "provider_empty_with_memory_zero_fold_force"
    if event.get("selected_route") != event.get("expected_route"):
        return "route_policy_threshold_or_source_error"
    return "none"


def _failure_plane(event: Dict[str, Any]) -> str:
    if _likely_boundary(event) == "provider_empty_with_memory_zero_fold_force":
        return "provider_output_validity"
    if not _retrieval_match(event):
        return "retrieval"
    if not _route_match(event):
        return "route_policy"
    return "none"


def _retrieval_match(event: Dict[str, Any]) -> bool:
    expected = set(event.get("expected_retrieved_ids", []))
    retrieved = set(event.get("retrieved_ids", []))
    return expected.issubset(retrieved)


def _route_match(event: Dict[str, Any]) -> bool:
    return event.get("selected_route") == event.get("expected_route")


def _output_valid(event: Dict[str, Any], side: str) -> bool:
    return int(event.get(f"{side}_response_chars", 0) or 0) > 0


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _recommendations(
    failures: List[Dict[str, Any]],
    empty_responses: List[Dict[str, Any]],
    gate_semantics: Dict[str, Any],
) -> List[str]:
    recommendations: List[str] = []
    if any(failure["likely_boundary"].startswith("embedding_confusability") for failure in failures):
        recommendations.append(
            "Add source-aware or contradiction-aware reranking before route selection for legal/rollback polarity conflicts."
        )
        recommendations.append(
            "Keep fabricated/uncertain anchors in query variants when evaluating quarantine recall."
        )
    if any("route_miss" in failure["failure_type"] for failure in failures):
        recommendations.append(
            "Report route accuracy separately from source label accuracy; source labels can be correct even when route selection fails."
        )
    if any(failure["likely_boundary"] == "provider_empty_with_memory_zero_fold_force" for failure in failures):
        recommendations.append(
            "Track provider-empty with-memory transitions separately; they create zero fold-force route misses even when retrieval is correct."
        )
    if gate_semantics["provider_output_boundary_failure_count"]:
        recommendations.append(
            "Report primitive route accuracy both raw and provider-valid; final empty with-memory responses belong to provider-output validity."
        )
    if empty_responses:
        recommendations.append(
            "Track empty no-memory responses as provider/model behavior, but do not count them as route failures unless retrieval or route labels fail."
        )
    if not recommendations:
        recommendations.append("No live replication boundary cases detected.")
    return recommendations


def _count_by(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value

"""Primitive validation logging utilities.

The roadmap requires experiments to be scored from durable logs rather than
from in-memory objects. This module provides a compact JSONL record for that
purpose. It intentionally depends only on the public FGM dataclasses and the
standard library.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from fgm.core import FoldResult
from fgm.core import (
    ROUTE_CORRECTION_CHAIN,
    ROUTE_DURABLE_MEMORY,
    ROUTE_NULL,
    ROUTE_OPERATION_MEMORY,
    ROUTE_QUARANTINE,
    ROUTE_TRACE,
    SOURCE_EXTERNAL,
    UNTRUSTED_SOURCE_LABELS,
)

WRITE_ROUTES = frozenset({
    ROUTE_TRACE,
    ROUTE_DURABLE_MEMORY,
    ROUTE_OPERATION_MEMORY,
    ROUTE_CORRECTION_CHAIN,
})


@dataclass(frozen=True)
class ValidationRecord:
    """One scored turn in a primitive validation experiment."""

    run_id: str
    seed: int
    turn_id: int
    query: str
    external_input_ids: List[str] = field(default_factory=list)
    retrieved_ids: List[str] = field(default_factory=list)
    source_labels: Dict[str, str] = field(default_factory=dict)
    active_source_labels: Dict[str, str] = field(default_factory=dict)
    source_confidence: Dict[str, float] = field(default_factory=dict)
    attention_or_selection_scores: Dict[str, float] = field(default_factory=dict)
    eligibility_score: Optional[float] = None
    inscription_score: Optional[float] = None
    route_scores: Dict[str, float] = field(default_factory=dict)
    selected_route: Optional[str] = None
    output_with_memory: Optional[List[float]] = None
    output_without_memory: Optional[List[float]] = None
    transition_delta: Optional[float] = None
    predicted_fold_force: Optional[float] = None
    realized_fold_force: Optional[float] = None
    operation_record_id: Optional[str] = None
    correction_node_id: Optional[str] = None
    future_task_id: Optional[str] = None
    future_task_score: Optional[float] = None
    expected_retrieved_ids: List[str] = field(default_factory=list)
    expected_source_labels: Dict[str, str] = field(default_factory=dict)
    expected_active_source_labels: Dict[str, str] = field(default_factory=dict)
    expected_route: Optional[str] = None
    future_utility_label: Optional[bool] = None

    @classmethod
    def from_fold_result(
        cls,
        *,
        run_id: str,
        seed: int,
        turn_id: int,
        fold_result: FoldResult,
        external_input_ids: Optional[Sequence[str]] = None,
        eligibility_score: Optional[float] = None,
        inscription_score: Optional[float] = None,
        predicted_fold_force: Optional[float] = None,
        correction_node_id: Optional[str] = None,
        future_task_id: Optional[str] = None,
        future_task_score: Optional[float] = None,
        expected_retrieved_ids: Optional[Sequence[str]] = None,
        expected_source_labels: Optional[Dict[str, str]] = None,
        expected_active_source_labels: Optional[Dict[str, str]] = None,
        expected_route: Optional[str] = None,
        future_utility_label: Optional[bool] = None,
    ) -> "ValidationRecord":
        retrieved_ids = [hit.record.record_id for hit in fold_result.retrieved]
        source_labels = {
            hit.record.record_id: label
            for hit, label in zip(fold_result.retrieved, fold_result.source_labels)
        }
        active_source_labels = {
            hit.record.record_id: label
            for hit, label in zip(fold_result.retrieved, fold_result.active_source_labels)
        }
        source_confidence = {
            hit.record.record_id: confidence
            for hit, confidence in zip(fold_result.retrieved, fold_result.source_confidence)
        }
        selection_scores = {
            hit.record.record_id: float(hit.score)
            for hit in fold_result.retrieved
        }
        return cls(
            run_id=run_id,
            seed=seed,
            turn_id=turn_id,
            query=fold_result.query,
            external_input_ids=list(external_input_ids or ()),
            retrieved_ids=retrieved_ids,
            source_labels=source_labels,
            active_source_labels=active_source_labels,
            source_confidence=source_confidence,
            attention_or_selection_scores=selection_scores,
            eligibility_score=eligibility_score,
            inscription_score=inscription_score,
            route_scores={k: float(v) for k, v in fold_result.route_scores.items()},
            selected_route=fold_result.selected_route,
            output_with_memory=_array_to_list(fold_result.output_with),
            output_without_memory=_array_to_list(fold_result.output_without),
            transition_delta=float(fold_result.full_divergence),
            predicted_fold_force=predicted_fold_force,
            realized_fold_force=float(fold_result.fold_force),
            operation_record_id=fold_result.operation_record_id,
            correction_node_id=correction_node_id,
            future_task_id=future_task_id,
            future_task_score=future_task_score,
            expected_retrieved_ids=list(expected_retrieved_ids or ()),
            expected_source_labels=dict(expected_source_labels or {}),
            expected_active_source_labels=dict(expected_active_source_labels or {}),
            expected_route=expected_route,
            future_utility_label=future_utility_label,
        )


def write_validation_jsonl(path: str | Path, records: Iterable[ValidationRecord]) -> None:
    """Write validation records as newline-delimited JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(asdict(record)), sort_keys=True))
            handle.write("\n")


def append_validation_jsonl(path: str | Path, record: ValidationRecord) -> None:
    """Append one validation record to a JSONL file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(asdict(record)), sort_keys=True))
        handle.write("\n")


def read_validation_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read validation records back as plain dictionaries."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def apply_route_baseline(
    records: Iterable[ValidationRecord],
    policy: str,
) -> List[ValidationRecord]:
    """Return records with selected routes replaced by a baseline policy.

    Policies:
      - ``always_write``: route every retrieved event to durable memory
      - ``never_write``: route every event to null
      - ``source_blind``: route by fold-force only, ignoring source labels
    """
    rewritten: List[ValidationRecord] = []
    for record in records:
        if policy == "always_write":
            route = ROUTE_DURABLE_MEMORY if record.retrieved_ids else ROUTE_NULL
        elif policy == "never_write":
            route = ROUTE_NULL
        elif policy == "source_blind":
            force = record.realized_fold_force or 0.0
            route = ROUTE_OPERATION_MEMORY if record.retrieved_ids and force > 0 else ROUTE_NULL
        else:
            raise ValueError(f"Unknown baseline policy: {policy}")
        rewritten.append(replace(record, selected_route=route, route_scores={route: 1.0}))
    return rewritten


def score_validation_records(records: Iterable[ValidationRecord | Dict[str, Any]]) -> Dict[str, float]:
    """Compute primitive validation metrics from records.

    Metrics are intentionally simple and replayable from JSONL. Missing ground
    truth labels are skipped for the relevant metric rather than guessed.
    """
    normalized = [_ensure_record(row) for row in records]
    n = len(normalized)

    retrieval_cases = [r for r in normalized if r.expected_retrieved_ids]
    retrieval_hits = sum(
        1 for r in retrieval_cases
        if set(r.expected_retrieved_ids).issubset(set(r.retrieved_ids))
    )

    source_total = 0
    source_correct = 0
    false_externalizations = 0
    for record in normalized:
        for record_id, expected in record.expected_source_labels.items():
            observed = record.source_labels.get(record_id)
            if observed is None:
                continue
            source_total += 1
            source_correct += int(observed == expected)
            false_externalizations += int(expected != SOURCE_EXTERNAL and observed == SOURCE_EXTERNAL)

    active_source_total = 0
    active_source_correct = 0
    for record in normalized:
        for record_id, expected in record.expected_active_source_labels.items():
            observed = record.active_source_labels.get(record_id)
            if observed is None:
                continue
            active_source_total += 1
            active_source_correct += int(observed == expected)

    route_cases = [r for r in normalized if r.expected_route is not None]
    route_correct = sum(1 for r in route_cases if r.selected_route == r.expected_route)
    expected_operation_cases = [r for r in route_cases if r.expected_route == ROUTE_OPERATION_MEMORY]
    expected_correction_cases = [r for r in route_cases if r.expected_route == ROUTE_CORRECTION_CHAIN]
    missed_operation = sum(
        1 for r in expected_operation_cases
        if r.expected_route == ROUTE_OPERATION_MEMORY and r.selected_route != ROUTE_OPERATION_MEMORY
    )
    missed_correction = sum(
        1 for r in expected_correction_cases
        if r.expected_route == ROUTE_CORRECTION_CHAIN and r.selected_route != ROUTE_CORRECTION_CHAIN
    )
    false_durable = sum(
        1 for r in route_cases
        if r.selected_route == ROUTE_DURABLE_MEMORY and r.expected_route != ROUTE_DURABLE_MEMORY
    )

    quarantine_expected = [r for r in route_cases if r.expected_route == ROUTE_QUARANTINE]
    quarantine_selected = [r for r in route_cases if r.selected_route == ROUTE_QUARANTINE]
    quarantine_true_positive = sum(
        1 for r in route_cases
        if r.expected_route == ROUTE_QUARANTINE and r.selected_route == ROUTE_QUARANTINE
    )

    utility_cases = [r for r in normalized if r.future_utility_label is not None]
    nonuseful_cases = [r for r in utility_cases if r.future_utility_label is False]
    useful_cases = [r for r in utility_cases if r.future_utility_label is True]
    false_write = sum(
        1 for r in nonuseful_cases
        if r.future_utility_label is False and r.selected_route in WRITE_ROUTES
    )
    missed_useful_write = sum(
        1 for r in useful_cases
        if r.future_utility_label is True and r.selected_route not in WRITE_ROUTES
    )

    echo_cases = [
        r for r in normalized
        if any(label in UNTRUSTED_SOURCE_LABELS for label in r.source_labels.values())
    ]
    echo_promotions = sum(
        1 for r in echo_cases
        if r.selected_route in {ROUTE_DURABLE_MEMORY, ROUTE_OPERATION_MEMORY, ROUTE_CORRECTION_CHAIN}
    )

    return {
        "n_records": float(n),
        "retrieval_hit_rate": _rate(retrieval_hits, len(retrieval_cases)),
        "source_label_accuracy": _rate(source_correct, source_total),
        "false_externalization_rate": _rate(false_externalizations, source_total),
        "active_source_accuracy": _rate(active_source_correct, active_source_total),
        "route_accuracy": _rate(route_correct, len(route_cases)),
        "false_durable_write_rate": _rate(false_durable, len(route_cases)),
        "missed_operation_write_rate": _rate(missed_operation, len(expected_operation_cases)),
        "missed_correction_node_rate": _rate(missed_correction, len(expected_correction_cases)),
        "quarantine_precision": _rate(quarantine_true_positive, len(quarantine_selected)),
        "quarantine_recall": _rate(quarantine_true_positive, len(quarantine_expected)),
        "false_write_rate": _rate(false_write, len(nonuseful_cases)),
        "missed_useful_write_rate": _rate(missed_useful_write, len(useful_cases)),
        "echo_promotion_rate": _rate(echo_promotions, len(echo_cases)),
    }


def _array_to_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.tolist()]
    return [float(x) for x in value]


def _ensure_record(row: ValidationRecord | Dict[str, Any]) -> ValidationRecord:
    if isinstance(row, ValidationRecord):
        return row
    valid_keys = ValidationRecord.__dataclass_fields__.keys()
    return ValidationRecord(**{key: row[key] for key in valid_keys if key in row})


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _array_to_list(value)
    if isinstance(value, np.generic):
        return value.item()
    return value

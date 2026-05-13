"""Consolidated roadmap validation runner."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from fgm.core import (
    FGMAgent,
    ROUTE_OPERATION_MEMORY,
    ROUTE_QUARANTINE,
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_SIMULATION,
)
from fgm.correction import compare_correction_policies, make_correction_chain_fixture
from fgm.coupling import coupled_field_probe
from fgm.inscription import compare_inscription_policies, make_inscription_utility_fixture
from fgm.residual import compare_residual_attention_policies, make_residual_attention_fixture
from fgm.self_index import compare_self_index_policies, make_self_index_fixture
from fgm.validation import (
    ValidationRecord,
    apply_route_baseline,
    score_validation_records,
    write_validation_jsonl,
)


def run_controlled_roadmap_validations(seed: int = 0) -> Dict[str, Any]:
    """Run all controlled primitive validations and return a JSON-safe summary."""
    source_records = make_source_routing_records(seed=seed)
    source_summary = {
        "source_sensitive": score_validation_records(source_records),
        "always_write": score_validation_records(apply_route_baseline(source_records, "always_write")),
        "never_write": score_validation_records(apply_route_baseline(source_records, "never_write")),
        "source_blind": score_validation_records(apply_route_baseline(source_records, "source_blind")),
    }
    inscription_summary = {
        name: asdict(report)
        for name, report in compare_inscription_policies(
            make_inscription_utility_fixture(),
            budget=3,
            seed=seed,
        ).items()
    }
    correction_summary = {
        name: asdict(report)
        for name, report in compare_correction_policies(make_correction_chain_fixture()).items()
    }
    residual_summary = {
        name: asdict(report)
        for name, report in compare_residual_attention_policies(make_residual_attention_fixture(), k=3).items()
    }
    self_index_summary = {
        name: asdict(report)
        for name, report in compare_self_index_policies(make_self_index_fixture()).items()
    }
    coupling_summary = {
        "source_aware": asdict(coupled_field_probe(source_aware=True)),
        "source_blind": asdict(coupled_field_probe(source_aware=False)),
    }

    return _json_safe({
        "seed": seed,
        "source_routing": source_summary,
        "inscription_utility": inscription_summary,
        "correction_chains": correction_summary,
        "residual_attention": residual_summary,
        "self_index_binding": self_index_summary,
        "coupled_field": coupling_summary,
    })


def write_roadmap_validation_outputs(
    output_dir: str | Path = "results",
    *,
    seed: int = 0,
) -> Dict[str, Path]:
    """Write fixed-seed roadmap validation artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_records = make_source_routing_records(seed=seed)
    source_path = output / "source_routing_validation.jsonl"
    write_validation_jsonl(source_path, source_records)

    summary = run_controlled_roadmap_validations(seed=seed)
    summary_path = output / "roadmap_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary": summary_path, "source_routing": source_path}


def make_source_routing_records(seed: int = 0) -> List[ValidationRecord]:
    agent = FGMAgent(dim=64, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "E1 external observation: legal approved rollback for the deploy migration",
        record_id="E1",
        source_label=SOURCE_EXTERNAL,
        source_confidence=1.0,
        provenance=["legal_ticket_42"],
    )
    agent.add(
        "R1 prior memory: last deploy migration timed out and rollback restored service",
        record_id="R1",
        source_label=SOURCE_EXTERNAL,
        source_confidence=0.95,
        provenance=["incident_review_17"],
    )
    agent.add(
        "S1 simulated hypothesis: pushing a hotfix might avoid rollback",
        record_id="S1",
        source_label=SOURCE_SIMULATION,
        source_confidence=0.7,
        provenance=["simulation_branch_a"],
    )
    agent.add(
        "F1 fabricated distractor: rollback is forbidden by legal",
        record_id="F1",
        source_label=SOURCE_FABRICATED,
        source_confidence=0.9,
        provenance=["adversarial_note"],
    )
    cases = [
        (
            "should we roll back the deploy migration after legal approval",
            "E1",
            SOURCE_EXTERNAL,
            ROUTE_OPERATION_MEMORY,
            True,
        ),
        (
            "what happened last time the deploy migration timed out",
            "R1",
            SOURCE_EXTERNAL,
            ROUTE_OPERATION_MEMORY,
            True,
        ),
        (
            "hotfix might avoid rollback hypothetical",
            "S1",
            SOURCE_SIMULATION,
            ROUTE_QUARANTINE,
            False,
        ),
        (
            "rollback forbidden by legal fabricated distractor",
            "F1",
            SOURCE_FABRICATED,
            ROUTE_QUARANTINE,
            False,
        ),
    ]
    records: List[ValidationRecord] = []
    for turn_id, (query, expected_id, expected_source, expected_route, future_utility) in enumerate(cases, start=1):
        result = agent.query(query)
        record = ValidationRecord.from_fold_result(
            run_id="controlled-roadmap-source-routing",
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
        records.append(replace(record, operation_record_id=f"op_turn_{turn_id}"))
    return records


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

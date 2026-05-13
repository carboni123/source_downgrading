"""Run the first primitive-validation scenario from the roadmap.

The scenario gives the agent four controlled source classes:
  E1: external observation
  R1: prior externally grounded memory, reactivated by retrieval
  S1: simulated hypothesis
  F1: fabricated/uncertain distractor

Each turn is logged as a ValidationRecord JSONL row. This is intentionally a
small smoke harness: it proves the source/provenance and route fields are
observable before richer inscription-utility experiments are added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fgm import (
    FGMAgent,
    ROUTE_OPERATION_MEMORY,
    ROUTE_QUARANTINE,
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_SIMULATION,
    ValidationRecord,
    apply_route_baseline,
    score_validation_records,
    write_validation_jsonl,
)


def build_agent() -> FGMAgent:
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
    return agent


def run(output_path: Path) -> list[ValidationRecord]:
    agent = build_agent()
    queries = [
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

    records: list[ValidationRecord] = []
    for turn_id, (query, expected_id, expected_source, expected_route, future_utility) in enumerate(queries, start=1):
        result = agent.query(query)
        records.append(
            ValidationRecord.from_fold_result(
                run_id="primitive-validation-demo",
                seed=0,
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

    write_validation_jsonl(output_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/architecture/primitive_validation_demo.jsonl",
        help="Path for validation JSONL output.",
    )
    args = parser.parse_args()

    records = run(Path(args.output))
    for record in records:
        print(
            f"turn={record.turn_id} retrieved={record.retrieved_ids} "
            f"sources={record.source_labels} active={record.active_source_labels} "
            f"route={record.selected_route}"
        )
    print("source_sensitive", score_validation_records(records))
    for policy in ("always_write", "never_write", "source_blind"):
        print(policy, score_validation_records(apply_route_baseline(records, policy)))
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

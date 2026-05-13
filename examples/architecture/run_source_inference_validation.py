"""Run the Source(.) inference validation harness.

Produces:
  results/architecture/source_inference_validation_summary.json
  results/architecture/source_inference_validation.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from fgm import (
    SOURCE_CLASSES,
    compare_source_inference_policies,
    make_source_inference_fixture,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/architecture")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = make_source_inference_fixture()
    reports = compare_source_inference_policies(cases)

    summary: Dict[str, Any] = {
        "n_cases": len(cases),
        "n_ambiguous": sum(1 for c in cases if c.is_ambiguous),
        "classes": list(SOURCE_CLASSES),
        "policies": {policy: asdict(report) for policy, report in reports.items()},
    }
    summary_path = output_dir / "source_inference_validation_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    rows: List[Dict[str, Any]] = []
    for policy, report in reports.items():
        rows.append({
            "policy": policy,
            "overall_accuracy": report.overall_accuracy,
            "false_externalization_rate": report.false_externalization_rate,
            "ambiguous_accuracy": report.ambiguous_accuracy,
            "per_class_accuracy": report.per_class_accuracy,
        })
    jsonl_path = output_dir / "source_inference_validation.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True))
            handle.write("\n")

    print(f"summary={summary_path}")
    print(f"records={jsonl_path}")
    for policy, report in reports.items():
        print(
            f"{policy:>20s}  "
            f"acc={report.overall_accuracy:.2f}  "
            f"false_ext={report.false_externalization_rate:.2f}  "
            f"ambiguous={report.ambiguous_accuracy:.2f}"
        )


if __name__ == "__main__":
    main()

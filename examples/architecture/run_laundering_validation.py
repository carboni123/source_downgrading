"""Run the inference-laundering validation harness.

Produces:
  results/architecture/laundering_validation_summary.json            -- single-seed per-policy metrics
  results/architecture/laundering_validation.jsonl                   -- per-case per-policy records
  results/architecture/laundering_validation_multiseed_summary.json  -- 20-seed aggregated metrics
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from fgm import (
    compare_laundering_policies,
    compare_laundering_policies_multiseed,
    make_laundering_fixture,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, float):
        return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/architecture")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of noise-perturbed seeds for the multi-seed sweep (0 to skip).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = make_laundering_fixture()
    reports = compare_laundering_policies(cases, seed=args.seed)

    summary: Dict[str, Any] = {
        "seed": args.seed,
        "cases": [case.case_id for case in cases],
        "policies": {policy: asdict(report) for policy, report in reports.items()},
    }
    summary_path = output_dir / "laundering_validation_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    rows: List[Dict[str, Any]] = []
    for policy, report in reports.items():
        rows.append({
            "policy": policy,
            **{k: v for k, v in asdict(report).items() if k != "policy"},
        })
    jsonl_path = output_dir / "laundering_validation.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True))
            handle.write("\n")

    print(f"summary={summary_path}")
    print(f"records={jsonl_path}")
    for policy, report in reports.items():
        print(
            f"{policy:>26s}  "
            f"laundering={report.inference_laundering_rate:.2f}  "
            f"prov_recall={report.provenance_chain_recall:.2f}  "
            f"false_ext_after={report.false_externalization_after_inference:.2f}  "
            f"ceiling_violation={report.derived_trust_ceiling_violation_rate:.2f}  "
            f"prov_depth={report.transitive_provenance_depth_mean:.2f}"
        )

    if args.n_seeds > 0:
        seeds = tuple(range(args.n_seeds))
        multiseed = compare_laundering_policies_multiseed(cases, seeds=seeds)
        multiseed_summary: Dict[str, Any] = {
            "n_seeds": args.n_seeds,
            "seeds": list(seeds),
            "cases": [case.case_id for case in cases],
            "policies": {
                policy: {
                    "policy": report.policy,
                    "n_seeds": report.n_seeds,
                    "mean": report.mean,
                    "std": report.std,
                    "min": report.min,
                    "max": report.max,
                }
                for policy, report in multiseed.items()
            },
        }
        multiseed_path = output_dir / "laundering_validation_multiseed_summary.json"
        multiseed_path.write_text(
            json.dumps(_json_safe(multiseed_summary), indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        print(f"multiseed={multiseed_path}")
        for policy, report in multiseed.items():
            print(
                f"{policy:>26s}  N={report.n_seeds}  "
                f"laundering={report.mean['inference_laundering_rate']:.3f}+-{report.std['inference_laundering_rate']:.3f}  "
                f"prov_recall={report.mean['provenance_chain_recall']:.3f}+-{report.std['provenance_chain_recall']:.3f}  "
                f"ceiling={report.mean['derived_trust_ceiling_violation_rate']:.3f}+-{report.std['derived_trust_ceiling_violation_rate']:.3f}"
            )


if __name__ == "__main__":
    main()

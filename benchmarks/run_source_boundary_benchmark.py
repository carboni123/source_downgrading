"""Source-boundary benchmark runner.

Evaluates whether source labels can be recovered at ingestion boundaries from
content plus simple retrieval features. This complements the laundering
benchmark: laundering assumes source labels are already grounded; this benchmark
measures how well the current rule-based policies ground those labels.

Run:

    python benchmarks/run_source_boundary_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmarks.source_boundary_dataset import (
        SourceBoundaryCase,
        dataset_summary,
        make_dataset,
        validate_cases,
    )
else:
    from .source_boundary_dataset import (
        SourceBoundaryCase,
        dataset_summary,
        make_dataset,
        validate_cases,
    )

from trace_memory import SourceLabel, infer_source


POLICIES = ("uniform_external", "lexical_rules", "feature_threshold", "combined")

SOURCE_LABELS = tuple(
    label.value for label in (
        SourceLabel.EXTERNAL,
        SourceLabel.TOOL_OUTPUT,
        SourceLabel.RETRIEVED_MEMORY,
        SourceLabel.INFERENCE,
        SourceLabel.SIMULATION,
        SourceLabel.FABRICATED_OR_UNCERTAIN,
    )
)

TRUST_RANK = {
    SourceLabel.FABRICATED_OR_UNCERTAIN.value: 0,
    SourceLabel.SIMULATION.value: 1,
    SourceLabel.INFERENCE.value: 2,
    SourceLabel.RETRIEVED_MEMORY.value: 3,
    SourceLabel.TOOL_OUTPUT.value: 4,
    SourceLabel.EXTERNAL.value: 5,
}


def _trust_rank(label: str) -> int:
    return TRUST_RANK[label]


def _case_to_json(case: SourceBoundaryCase) -> Dict:
    d = asdict(case)
    return d


def _policy_prediction(policy: str, case: SourceBoundaryCase) -> str:
    return infer_source(
        case.content,
        query_context=case.query_context,
        retrieval_margin=case.retrieval_margin,
        recency_rank=case.recency_rank,
        policy=policy,
    ).value


def _policy_metrics(policy: str, cases: Sequence[SourceBoundaryCase]) -> Dict:
    per_case = []
    correct = 0
    false_external = 0
    trust_upgrades = 0
    non_external_total = 0

    per_source_total = {label: 0 for label in SOURCE_LABELS}
    per_source_correct = {label: 0 for label in SOURCE_LABELS}
    per_boundary_total: Dict[str, int] = {}
    per_boundary_correct: Dict[str, int] = {}
    per_difficulty_total: Dict[str, int] = {}
    per_difficulty_correct: Dict[str, int] = {}
    confusion = {
        expected: {predicted: 0 for predicted in SOURCE_LABELS}
        for expected in SOURCE_LABELS
    }

    for case in cases:
        expected = case.expected_source
        predicted = _policy_prediction(policy, case)
        is_correct = predicted == expected
        correct += int(is_correct)

        per_source_total[expected] += 1
        per_source_correct[expected] += int(is_correct)
        confusion[expected][predicted] += 1

        per_boundary_total[case.boundary_type] = per_boundary_total.get(case.boundary_type, 0) + 1
        per_boundary_correct[case.boundary_type] = (
            per_boundary_correct.get(case.boundary_type, 0) + int(is_correct)
        )
        per_difficulty_total[case.difficulty] = per_difficulty_total.get(case.difficulty, 0) + 1
        per_difficulty_correct[case.difficulty] = (
            per_difficulty_correct.get(case.difficulty, 0) + int(is_correct)
        )

        if expected != SourceLabel.EXTERNAL.value:
            non_external_total += 1
            if predicted == SourceLabel.EXTERNAL.value:
                false_external += 1
        if _trust_rank(predicted) > _trust_rank(expected):
            trust_upgrades += 1

        per_case.append({
            "case": _case_to_json(case),
            "predicted_source": predicted,
            "correct": is_correct,
            "trust_upgrade": _trust_rank(predicted) > _trust_rank(expected),
            "false_externalization": (
                expected != SourceLabel.EXTERNAL.value
                and predicted == SourceLabel.EXTERNAL.value
            ),
        })

    n = len(cases)
    per_source_accuracy = {
        label: (
            per_source_correct[label] / per_source_total[label]
            if per_source_total[label]
            else float("nan")
        )
        for label in SOURCE_LABELS
    }
    per_boundary_accuracy = {
        key: per_boundary_correct.get(key, 0) / total
        for key, total in sorted(per_boundary_total.items())
    }
    per_difficulty_accuracy = {
        key: per_difficulty_correct.get(key, 0) / total
        for key, total in sorted(per_difficulty_total.items())
    }

    return {
        "policy": policy,
        "n_cases": n,
        "aggregate": {
            "overall_accuracy": correct / n if n else float("nan"),
            "false_externalization_rate": (
                false_external / non_external_total if non_external_total else float("nan")
            ),
            "trust_upgrade_rate": trust_upgrades / n if n else float("nan"),
            "per_source_accuracy": per_source_accuracy,
            "per_boundary_accuracy": per_boundary_accuracy,
            "per_difficulty_accuracy": per_difficulty_accuracy,
            "confusion_matrix": confusion,
        },
        "per_case": per_case,
    }


def run_benchmark(cases: Sequence[SourceBoundaryCase]) -> Dict[str, Dict]:
    validate_cases(cases)
    return {policy: _policy_metrics(policy, cases) for policy in POLICIES}


def render_report(results: Dict[str, Dict], cases: Sequence[SourceBoundaryCase]) -> str:
    summary = dataset_summary(cases)
    lines: List[str] = []
    lines.append("# Source-Boundary Benchmark")
    lines.append("")
    lines.append(
        "Auto-generated by `benchmarks/run_source_boundary_benchmark.py`. "
        f"{len(cases)} labelled ingestion-boundary cases; "
        f"{len(POLICIES)} policies."
    )
    lines.append(
        "This benchmark evaluates source-label grounding before derivation. "
        "It does not test laundering; it tests whether a policy can recover "
        "`Source(.)` from realistic boundary text plus retrieval features."
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        f"- Sources covered: {', '.join(sorted(summary['by_source']))}."
    )
    lines.append(
        f"- Boundary types covered: {', '.join(sorted(summary['by_boundary_type']))}."
    )
    lines.append(
        f"- Domains covered: {', '.join(sorted(summary['by_domain']))}."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "Higher is better for accuracy. Lower is better for false "
        "externalization and trust-upgrade rates."
    )
    lines.append("")
    lines.append(
        "| Policy | Overall accuracy | False externalization | Trust upgrade | Canonical | Natural | Decoy |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for policy in POLICIES:
        agg = results[policy]["aggregate"]
        by_boundary = agg["per_boundary_accuracy"]
        lines.append(
            f"| `{policy}` "
            f"| {agg['overall_accuracy']:.3f} "
            f"| {agg['false_externalization_rate']:.3f} "
            f"| {agg['trust_upgrade_rate']:.3f} "
            f"| {by_boundary.get('canonical_marker', float('nan')):.3f} "
            f"| {by_boundary.get('natural_prose', float('nan')):.3f} "
            f"| {by_boundary.get('source_decoy', float('nan')):.3f} |"
        )
    lines.append("")
    lines.append("## Per-source Accuracy")
    lines.append("")
    lines.append("| Source | " + " | ".join(f"`{p}`" for p in POLICIES) + " |")
    lines.append("|" + "---|" * (len(POLICIES) + 1))
    for source in SOURCE_LABELS:
        row = [f"| {source}"]
        for policy in POLICIES:
            row.append(f"{results[policy]['aggregate']['per_source_accuracy'][source]:.3f}")
        lines.append(" | ".join(row) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "A high laundering-benchmark score only means trust composition works "
        "after labels are known. This benchmark measures the preceding "
        "boundary: whether labels can be recovered automatically. Any "
        "non-zero false-externalization or trust-upgrade rate should be "
        "treated as evidence that production systems need app-owned source "
        "labels or a stronger classifier before using `add_with_inferred_source`."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/source_boundary_dataset.py --output benchmarks/data/source_boundary_dataset.jsonl")
    lines.append("python benchmarks/run_source_boundary_benchmark.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/benchmarks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = make_dataset()
    validate_cases(cases)
    results = run_benchmark(cases)

    results_path = output_dir / "source_boundary_benchmark_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = output_dir / "SOURCE_BOUNDARY_BENCHMARK.md"
    report_path.write_text(render_report(results, cases), encoding="utf-8")

    print(f"running {len(POLICIES)} policies x {len(cases)} cases...")
    print(f"wrote {results_path}")
    print(f"wrote {report_path}")
    print()
    print("Aggregate (accuracy higher is better; false_ext/trust_upgrade lower is better):")
    print(f"{'policy':<20s} {'accuracy':>9s} {'false_ext':>10s} {'upgrade':>9s}")
    for policy in POLICIES:
        agg = results[policy]["aggregate"]
        print(
            f"{policy:<20s} "
            f"{agg['overall_accuracy']:>9.3f} "
            f"{agg['false_externalization_rate']:>10.3f} "
            f"{agg['trust_upgrade_rate']:>9.3f}"
        )


if __name__ == "__main__":
    main()

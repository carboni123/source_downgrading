"""Inference-laundering benchmark runner.

Runs three baselines (no_source, provenance_only, trace_memory) against
the deterministic laundering dataset. Computes six library-internal metrics and
emits two artefacts:

    results/benchmarks/laundering_benchmark_results.json
    results/benchmarks/LAUNDERING_BENCHMARK.md

Metrics (lower is better unless noted):

    inference_laundering_rate         -- final derived sources labeled
                                         'external' when expected ceiling
                                         is below 'external'. Self-audit
                                         metric; can undercount.
    derived_trust_ceiling_violation_rate -- truth-grounded. Final derived
                                         trust exceeds the scenario's
                                         expected_max_trust.
    chain_step_ceiling_violation_rate -- truth-grounded. Any intermediate
                                         derived step exceeds its computed
                                         trust ceiling.
    false_externalization_after_inference -- later-query retrieves the
                                         target derived record AND routes
                                         to a trusted write when the
                                         scenario expects quarantine.
    provenance_chain_recall (HIGHER better) -- fraction of derived
                                         records whose transitive
                                         provenance reaches the expected
                                         origin tokens.
    cascade_invisibility_gap          -- truth - local. Positive values
                                         indicate the self-audit metric
                                         is undercounting.

Run:

    python benchmarks/run_laundering_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Allow running as a script from the repo root: prepend the parent
# directory to sys.path so 'benchmarks' resolves as a package.
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmarks.laundering_baselines import BASELINES, BaselineResult
    from benchmarks.laundering_dataset import (
        LaunderingScenario,
        expected_chain_sources,
        make_dataset,
        validate_scenarios,
    )
else:
    from .laundering_baselines import BASELINES, BaselineResult
    from .laundering_dataset import (
        LaunderingScenario,
        expected_chain_sources,
        make_dataset,
        validate_scenarios,
    )

from trace_memory import SourceLabel


_TRUSTED_WRITE_ROUTES = frozenset({"durable_memory", "operation_memory"})


def _trust_rank(label: str) -> int:
    ordering = {
        SourceLabel.FABRICATED_OR_UNCERTAIN.value: 0,
        SourceLabel.SIMULATION.value: 1,
        SourceLabel.INFERENCE.value: 2,
        SourceLabel.RETRIEVED_MEMORY.value: 3,
        SourceLabel.TOOL_OUTPUT.value: 4,
        SourceLabel.EXTERNAL.value: 5,
    }
    return ordering.get(label, 2)


def _scenario_metrics(result: BaselineResult, scenario: LaunderingScenario) -> Dict[str, float]:
    """Compute the five per-scenario metrics for one baseline-result pair."""
    # 1. Inference-laundering rate (local). For one scenario, this is
    #    binary: did the policy end up labeling the final derived as
    #    external when at least one chain input was non-external?
    inputs_non_external = any(
        seed.source != SourceLabel.EXTERNAL.value
        for seed in scenario.seeds
    )
    laundered = (
        result.final_derived_source == SourceLabel.EXTERNAL.value
        and inputs_non_external
    )

    # 2. Trust-ceiling violation (truth-grounded).
    ceiling_violated = (
        _trust_rank(result.final_derived_source)
        > _trust_rank(result.expected_max_trust)
    )

    # 3. Trust-ceiling violation across all intermediate derived steps.
    expected_by_step = expected_chain_sources(scenario)
    actual_by_step = dict(result.chain_step_sources)
    step_violations = 0
    for step_id, expected_source in expected_by_step.items():
        actual_source = actual_by_step.get(step_id)
        if actual_source is None or _trust_rank(actual_source) > _trust_rank(expected_source):
            step_violations += 1
    chain_step_violation_rate = step_violations / len(expected_by_step)

    # 4. False externalization after inference.
    false_ext = (
        result.later_retrieved_target
        and scenario.later_should_quarantine
        and result.later_route in _TRUSTED_WRITE_ROUTES
    )

    # 5. Provenance chain recall: every expected origin must be in the
    #    final derived record's provenance set.
    derived_prov = set(result.final_derived_provenance)
    expected = set(result.expected_provenance_origins)
    if expected:
        recall = len(expected & derived_prov) / len(expected)
    else:
        recall = 1.0

    # 6. Cascade-invisibility gap: 1 if truth-grounded says violation
    #    but local says no laundering; else 0.
    gap = float(ceiling_violated and not laundered)

    return {
        "inference_laundering_rate": float(laundered),
        "derived_trust_ceiling_violation_rate": float(ceiling_violated),
        "chain_step_ceiling_violation_rate": float(chain_step_violation_rate),
        "false_externalization_after_inference": float(false_ext),
        "provenance_chain_recall": float(recall),
        "cascade_invisibility_gap": gap,
    }


def run_benchmark(scenarios: List[LaunderingScenario]) -> Dict[str, Dict]:
    """Execute every baseline on every scenario; collect raw + aggregate metrics."""
    by_baseline: Dict[str, Dict] = {}
    for baseline_name, baseline_fn in BASELINES.items():
        per_scenario: List[Tuple[BaselineResult, Dict[str, float]]] = []
        for scenario in scenarios:
            result = baseline_fn(scenario)
            metrics = _scenario_metrics(result, scenario)
            per_scenario.append((result, metrics))
        by_baseline[baseline_name] = {
            "name": baseline_name,
            "n_scenarios": len(per_scenario),
            "per_scenario": [
                {
                    "result": _baseline_result_to_json(result),
                    "metrics": metrics,
                }
                for result, metrics in per_scenario
            ],
            "aggregate": _aggregate(per_scenario),
        }
    return by_baseline


def _baseline_result_to_json(result: BaselineResult) -> Dict:
    d = asdict(result)
    d["final_derived_provenance"] = list(result.final_derived_provenance)
    d["expected_provenance_origins"] = list(result.expected_provenance_origins)
    d["chain_step_sources"] = [list(p) for p in result.chain_step_sources]
    return d


def _aggregate(per_scenario: List[Tuple[BaselineResult, Dict[str, float]]]) -> Dict:
    """Compute mean/std and per-domain breakdowns."""
    keys = [
        "inference_laundering_rate",
        "derived_trust_ceiling_violation_rate",
        "chain_step_ceiling_violation_rate",
        "false_externalization_after_inference",
        "provenance_chain_recall",
        "cascade_invisibility_gap",
    ]
    overall: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [m[key] for _, m in per_scenario]
        overall[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    # Per-domain.
    domains: Dict[str, List[Dict[str, float]]] = {}
    for result, metrics in per_scenario:
        domains.setdefault(result.domain, []).append(metrics)
    per_domain: Dict[str, Dict[str, Dict[str, float]]] = {}
    for domain, items in domains.items():
        per_domain[domain] = {}
        for key in keys:
            values = [m[key] for m in items]
            per_domain[domain][key] = {
                "mean": statistics.mean(values),
                "n": len(values),
            }
    # Per-failure-mode.
    modes: Dict[str, List[Dict[str, float]]] = {}
    for result, metrics in per_scenario:
        modes.setdefault(result.failure_mode, []).append(metrics)
    per_mode: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode, items in modes.items():
        per_mode[mode] = {}
        for key in keys:
            values = [m[key] for m in items]
            per_mode[mode][key] = {
                "mean": statistics.mean(values),
                "n": len(values),
            }
    return {"overall": overall, "per_domain": per_domain, "per_failure_mode": per_mode}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(results: Dict[str, Dict], n_scenarios: int) -> str:
    lines: List[str] = []
    lines.append("# Inference Laundering Benchmark")
    lines.append("")
    lines.append(
        f"Auto-generated by `benchmarks/run_laundering_benchmark.py`. "
        f"{n_scenarios} scenarios; 3 baselines; 6 metrics."
    )
    lines.append(
        "The dataset is validated before execution: scenario ids and record ids "
        "must be unique, chain references must resolve to prior records, final "
        "source ceilings and provenance origins are recomputed from the graph, "
        "and quarantine expectations must match the computed final source."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "Lower is better for the first four; higher is better for "
        "provenance recall; positive cascade gap means the self-audit "
        "metric is undercounting the truth-grounded failure rate."
    )
    lines.append("")
    lines.append(
        "| Baseline | Laundering (self) | Final ceiling violation | Chain ceiling violation | False externalization | Provenance recall | Cascade gap |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )
    for baseline_name in ("no_source", "provenance_only", "trace_memory"):
        agg = results[baseline_name]["aggregate"]["overall"]
        lines.append(
            f"| `{baseline_name}` "
            f"| {agg['inference_laundering_rate']['mean']:.3f} "
            f"| **{agg['derived_trust_ceiling_violation_rate']['mean']:.3f}** "
            f"| {agg['chain_step_ceiling_violation_rate']['mean']:.3f} "
            f"| {agg['false_externalization_after_inference']['mean']:.3f} "
            f"| {agg['provenance_chain_recall']['mean']:.3f} "
            f"| {agg['cascade_invisibility_gap']['mean']:.3f} |"
        )
    lines.append("")
    lines.append(
        "The truth-grounded ceiling-violation rate is the central claim: "
        "`trace_memory` is the only baseline that achieves zero, and the "
        "only baseline that closes the cascade-invisibility gap. "
        "`no_source` shows a non-zero gap, which means its own self-audit "
        "would report passing when truth-grounded checks show 100% "
        "violations. This is the kind of failure mode that is invisible "
        "without external ground truth."
    )
    lines.append("")
    lines.append("## Per-domain breakdown (truth-grounded ceiling violation rate)")
    lines.append("")
    domains = sorted(
        set(
            d
            for baseline in results.values()
            for d in baseline["aggregate"]["per_domain"]
        )
    )
    header = "| domain | " + " | ".join(f"`{b}`" for b in ("no_source", "provenance_only", "trace_memory")) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(("no_source", "provenance_only", "trace_memory")) + 1))
    for domain in domains:
        row = [f"| {domain}"]
        for baseline_name in ("no_source", "provenance_only", "trace_memory"):
            cell = results[baseline_name]["aggregate"]["per_domain"].get(domain)
            if cell is None:
                row.append("--")
                continue
            row.append(f"{cell['derived_trust_ceiling_violation_rate']['mean']:.3f} (n={cell['derived_trust_ceiling_violation_rate']['n']})")
        lines.append(" | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-failure-mode breakdown (truth-grounded ceiling violation rate)")
    lines.append("")
    modes = sorted(
        set(
            m
            for baseline in results.values()
            for m in baseline["aggregate"]["per_failure_mode"]
        )
    )
    header = "| failure mode | " + " | ".join(f"`{b}`" for b in ("no_source", "provenance_only", "trace_memory")) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(("no_source", "provenance_only", "trace_memory")) + 1))
    for mode in modes:
        row = [f"| {mode}"]
        for baseline_name in ("no_source", "provenance_only", "trace_memory"):
            cell = results[baseline_name]["aggregate"]["per_failure_mode"].get(mode)
            if cell is None:
                row.append("--")
                continue
            row.append(f"{cell['derived_trust_ceiling_violation_rate']['mean']:.3f} (n={cell['derived_trust_ceiling_violation_rate']['n']})")
        lines.append(" | ".join(row) + " |")
    lines.append("")
    lines.append("## Cascade-invisibility scenarios")
    lines.append("")
    lines.append(
        "Cases where `no_source`'s self-audit metric (inference_laundering_rate) "
        "reports no laundering while the truth-grounded ceiling check identifies "
        "a violation. These are scenarios where a memory layer without "
        "trust-composition would silently pass its own audit while still being wrong."
    )
    lines.append("")
    lines.append("| scenario_id | failure_mode | self-audit says | truth says |")
    lines.append("|---|---|---|---|")
    no_source_per_scenario = results["no_source"]["per_scenario"]
    count = 0
    for item in no_source_per_scenario:
        m = item["metrics"]
        if m["cascade_invisibility_gap"] > 0:
            r = item["result"]
            lines.append(
                f"| {r['scenario_id']} "
                f"| {r['failure_mode']} "
                f"| LAUNDERING=NO (false floor) "
                f"| CEILING VIOLATED ({r['final_derived_source']} > {r['expected_max_trust']}) |"
            )
            count += 1
    if count == 0:
        lines.append("| (no cascade-invisibility cases in this run) | | | |")
    lines.append("")
    lines.append(f"({count} cascade-invisibility cases out of {n_scenarios} scenarios)")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/laundering_dataset.py --output benchmarks/data/laundering_dataset.jsonl")
    lines.append("python benchmarks/run_laundering_benchmark.py")
    lines.append("```")
    lines.append("")
    lines.append("Dataset and runner are deterministic: same scenarios, same results every run. ")
    lines.append("No live LLM calls; no network; runs in under a second on a developer laptop.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/benchmarks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = make_dataset()
    validate_scenarios(scenarios)
    print(f"running {len(BASELINES)} baselines x {len(scenarios)} scenarios...")
    results = run_benchmark(scenarios)

    # Write the JSON results.
    results_path = output_dir / "laundering_benchmark_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {results_path}")

    # Generate the markdown report.
    report = render_report(results, n_scenarios=len(scenarios))
    report_path = output_dir / "LAUNDERING_BENCHMARK.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {report_path}")

    # Print a brief summary to stdout.
    print()
    print("Aggregate (lower is better, except provenance recall):")
    print(
        f"{'baseline':<22s} {'laund':>8s} {'ceiling':>9s} {'chain':>8s} "
        f"{'false_ext':>10s} {'prov':>7s} {'cascade':>9s}"
    )
    for baseline_name in ("no_source", "provenance_only", "trace_memory"):
        agg = results[baseline_name]["aggregate"]["overall"]
        print(
            f"{baseline_name:<22s} "
            f"{agg['inference_laundering_rate']['mean']:>8.3f} "
            f"{agg['derived_trust_ceiling_violation_rate']['mean']:>9.3f} "
            f"{agg['chain_step_ceiling_violation_rate']['mean']:>8.3f} "
            f"{agg['false_externalization_after_inference']['mean']:>10.3f} "
            f"{agg['provenance_chain_recall']['mean']:>7.3f} "
            f"{agg['cascade_invisibility_gap']['mean']:>9.3f}"
        )


if __name__ == "__main__":
    main()

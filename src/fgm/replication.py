"""Multi-seed replication helpers for roadmap validation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

from fgm.roadmap import run_controlled_roadmap_validations

MetricPath = Sequence[str]

CONTROLLED_METRICS: Dict[str, MetricPath] = {
    "source_sensitive.route_accuracy": ("source_routing", "source_sensitive", "route_accuracy"),
    "source_sensitive.echo_promotion_rate": ("source_routing", "source_sensitive", "echo_promotion_rate"),
    "source_sensitive.quarantine_recall": ("source_routing", "source_sensitive", "quarantine_recall"),
    "source_blind.route_accuracy": ("source_routing", "source_blind", "route_accuracy"),
    "source_blind.echo_promotion_rate": ("source_routing", "source_blind", "echo_promotion_rate"),
    "utility_write.future_task_lift": ("inscription_utility", "utility_write", "future_task_lift"),
    "utility_write.false_write_rate": ("inscription_utility", "utility_write", "false_write_rate"),
    "relevance_write.future_task_lift": ("inscription_utility", "relevance_write", "future_task_lift"),
    "always_write.false_write_rate": ("inscription_utility", "always_write", "false_write_rate"),
    "never_write.missed_useful_write_rate": ("inscription_utility", "never_write", "missed_useful_write_rate"),
    "correction_chain.transfer_success": ("correction_chains", "correction_chain", "transfer_success"),
    "conclusion_only.transfer_success": ("correction_chains", "conclusion_only", "transfer_success"),
    "correction_chain.false_update_rate": ("correction_chains", "correction_chain", "false_update_rate"),
    "conclusion_only.false_update_rate": ("correction_chains", "conclusion_only", "false_update_rate"),
    "residual_source.transition_effective_retrieval_precision": (
        "residual_attention",
        "residual_posture_source",
        "transition_effective_retrieval_precision",
    ),
    "semantic_only.transition_effective_retrieval_precision": (
        "residual_attention",
        "semantic_only",
        "transition_effective_retrieval_precision",
    ),
    "residual_source.confirmation_attractor_rate": (
        "residual_attention",
        "residual_posture_source",
        "confirmation_attractor_rate",
    ),
    "residual_posture.confirmation_attractor_rate": (
        "residual_attention",
        "residual_posture",
        "confirmation_attractor_rate",
    ),
    "self_indexed.correct_binding_rate": ("self_index_binding", "self_indexed", "correct_binding_rate"),
    "project_only.correct_binding_rate": ("self_index_binding", "project_only", "correct_binding_rate"),
    "self_indexed.wrong_project_application_rate": (
        "self_index_binding",
        "self_indexed",
        "wrong_project_application_rate",
    ),
    "global_memory.wrong_project_application_rate": (
        "self_index_binding",
        "global_memory",
        "wrong_project_application_rate",
    ),
    "source_aware.attention_shift_after_memory_ablation": (
        "coupled_field",
        "source_aware",
        "attention_shift_after_memory_ablation",
    ),
    "source_aware.echo_amplification_rate": ("coupled_field", "source_aware", "echo_amplification_rate"),
    "source_blind_coupling.echo_amplification_rate": (
        "coupled_field",
        "source_blind",
        "echo_amplification_rate",
    ),
}


def run_controlled_replication(
    *,
    seed_count: int = 50,
    start_seed: int = 0,
    effect_threshold: float = 0.9,
) -> Dict[str, Any]:
    """Run controlled roadmap validations across seeds and summarize stability."""
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")

    seeds = list(range(start_seed, start_seed + seed_count))
    runs = [run_controlled_roadmap_validations(seed=seed) for seed in seeds]
    metric_series = {
        name: [_get_path(run, path) for run in runs]
        for name, path in CONTROLLED_METRICS.items()
    }
    metric_summary = {
        name: _summarize_numeric(values)
        for name, values in metric_series.items()
    }
    effect_directions = _effect_direction_summary(runs)
    min_effect_hold_rate = min(
        (effect["hold_rate"] for effect in effect_directions.values()),
        default=0.0,
    )
    acceptance = {
        "toy_seed_count_met": seed_count >= 50,
        "effect_direction_hold_rate_met": min_effect_hold_rate >= effect_threshold,
        "minimum_effect_hold_rate": min_effect_hold_rate,
        "effect_threshold": effect_threshold,
        "source_route_accuracy_mean_met": (
            metric_summary["source_sensitive.route_accuracy"]["mean"] >= 0.9
        ),
        "source_echo_control_met": (
            metric_summary["source_sensitive.echo_promotion_rate"]["mean"] == 0.0
            and metric_summary["source_blind.echo_promotion_rate"]["mean"] >= 0.9
        ),
        "utility_lift_mean_met": metric_summary["utility_write.future_task_lift"]["mean"] >= 0.9,
        "correction_transfer_mean_met": metric_summary["correction_chain.transfer_success"]["mean"] >= 0.9,
        "residual_precision_mean_met": (
            metric_summary["residual_source.transition_effective_retrieval_precision"]["mean"] >= 0.9
        ),
        "self_index_binding_mean_met": metric_summary["self_indexed.correct_binding_rate"]["mean"] >= 0.9,
        "coupled_field_cross_effect_met": (
            metric_summary["source_aware.attention_shift_after_memory_ablation"]["min"] > 0.0
        ),
    }
    acceptance["all_controlled_replication_gates_met"] = all(acceptance.values())

    return _json_safe({
        "seed_count": seed_count,
        "start_seed": start_seed,
        "seeds": seeds,
        "metrics": metric_summary,
        "effect_directions": effect_directions,
        "acceptance": acceptance,
    })


def write_controlled_replication_outputs(
    output_dir: str | Path = "results",
    *,
    seed_count: int = 50,
    start_seed: int = 0,
) -> Dict[str, Path]:
    """Write multi-seed controlled replication artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    seeds = list(range(start_seed, start_seed + seed_count))
    runs = [run_controlled_roadmap_validations(seed=seed) for seed in seeds]
    summary = run_controlled_replication(seed_count=seed_count, start_seed=start_seed)

    summary_path = output / "controlled_replication_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    runs_path = output / "controlled_replication_runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(_json_safe(run), sort_keys=True))
            handle.write("\n")

    return {"summary": summary_path, "runs": runs_path}


def _effect_direction_summary(runs: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Callable[[Dict[str, Any]], bool]] = {
        "source_sensitive_route_beats_source_blind": lambda run: _gt(
            run,
            CONTROLLED_METRICS["source_sensitive.route_accuracy"],
            CONTROLLED_METRICS["source_blind.route_accuracy"],
        ),
        "source_sensitive_suppresses_echo_vs_source_blind": lambda run: _lt(
            run,
            CONTROLLED_METRICS["source_sensitive.echo_promotion_rate"],
            CONTROLLED_METRICS["source_blind.echo_promotion_rate"],
        ),
        "utility_write_lift_beats_relevance_write": lambda run: _gt(
            run,
            CONTROLLED_METRICS["utility_write.future_task_lift"],
            CONTROLLED_METRICS["relevance_write.future_task_lift"],
        ),
        "utility_write_false_writes_below_always_write": lambda run: _lt(
            run,
            CONTROLLED_METRICS["utility_write.false_write_rate"],
            CONTROLLED_METRICS["always_write.false_write_rate"],
        ),
        "utility_write_misses_below_never_write": lambda run: _lt(
            run,
            ("inscription_utility", "utility_write", "missed_useful_write_rate"),
            CONTROLLED_METRICS["never_write.missed_useful_write_rate"],
        ),
        "correction_chain_transfer_beats_conclusion_only": lambda run: _gt(
            run,
            CONTROLLED_METRICS["correction_chain.transfer_success"],
            CONTROLLED_METRICS["conclusion_only.transfer_success"],
        ),
        "correction_chain_false_updates_below_conclusion_only": lambda run: _lt(
            run,
            CONTROLLED_METRICS["correction_chain.false_update_rate"],
            CONTROLLED_METRICS["conclusion_only.false_update_rate"],
        ),
        "source_residual_precision_beats_semantic_only": lambda run: _gt(
            run,
            CONTROLLED_METRICS["residual_source.transition_effective_retrieval_precision"],
            CONTROLLED_METRICS["semantic_only.transition_effective_retrieval_precision"],
        ),
        "source_residual_reduces_confirmation_vs_residual": lambda run: _lt(
            run,
            CONTROLLED_METRICS["residual_source.confirmation_attractor_rate"],
            CONTROLLED_METRICS["residual_posture.confirmation_attractor_rate"],
        ),
        "self_indexed_binding_beats_project_only": lambda run: _gt(
            run,
            CONTROLLED_METRICS["self_indexed.correct_binding_rate"],
            CONTROLLED_METRICS["project_only.correct_binding_rate"],
        ),
        "self_indexed_project_leak_below_global_memory": lambda run: _lt(
            run,
            CONTROLLED_METRICS["self_indexed.wrong_project_application_rate"],
            CONTROLLED_METRICS["global_memory.wrong_project_application_rate"],
        ),
        "coupled_field_memory_to_attention_nonzero": lambda run: _get_path(
            run,
            CONTROLLED_METRICS["source_aware.attention_shift_after_memory_ablation"],
        ) > 0.0,
        "source_aware_coupling_suppresses_echo": lambda run: _lt(
            run,
            CONTROLLED_METRICS["source_aware.echo_amplification_rate"],
            CONTROLLED_METRICS["source_blind_coupling.echo_amplification_rate"],
        ),
    }
    run_list = list(runs)
    summary: Dict[str, Dict[str, Any]] = {}
    for name, check in checks.items():
        failures: List[int] = []
        holds = 0
        for run in run_list:
            ok = bool(check(run))
            holds += int(ok)
            if not ok:
                failures.append(int(run["seed"]))
        summary[name] = {
            "holds": holds,
            "n": len(run_list),
            "hold_rate": holds / len(run_list) if run_list else 0.0,
            "failure_seeds": failures,
        }
    return summary


def _summarize_numeric(values: Iterable[Any]) -> Dict[str, Any]:
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


def _gt(run: Dict[str, Any], left: MetricPath, right: MetricPath) -> bool:
    return _get_path(run, left) > _get_path(run, right)


def _lt(run: Dict[str, Any], left: MetricPath, right: MetricPath) -> bool:
    return _get_path(run, left) < _get_path(run, right)


def _get_path(data: Dict[str, Any], path: MetricPath) -> Any:
    value: Any = data
    for key in path:
        value = value[key]
    return value


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

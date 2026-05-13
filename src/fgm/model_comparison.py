"""Provider/model comparison harness for live validation artifacts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from fgm.core import Array
from fgm.live_diagnostics import write_live_replication_diagnostics
from fgm.live_validation import write_live_llm_replication_outputs
from fgm.llm import LLMCall


@dataclass(frozen=True)
class ProviderModelTarget:
    provider: str
    model: Optional[str] = None


LLMCallFactory = Callable[[str, Optional[str]], Optional[LLMCall]]


def parse_provider_model_target(spec: str) -> ProviderModelTarget:
    """Parse a provider/model target spec as provider or provider:model."""
    provider, sep, model = spec.partition(":")
    if not provider:
        raise ValueError("target provider cannot be empty")
    return ProviderModelTarget(
        provider=provider.strip().lower(),
        model=model.strip() if sep and model.strip() else None,
    )


def write_live_provider_model_comparison_outputs(
    output_dir: str | Path = "results",
    *,
    targets: Optional[Sequence[ProviderModelTarget | Dict[str, Optional[str]] | str]] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    seed_count: int = 5,
    start_seed: int = 0,
    empty_response_retries: int = 1,
    include_audit_text: bool = False,
    min_passed_targets: int = 1,
    reuse_existing: bool = False,
    max_output_tokens: Optional[int] = None,
    artifact_prefix: str = "live_provider_model_comparison",
    summary_filename: str = "live_provider_model_comparison_summary.json",
    llm_call_factory: Optional[LLMCallFactory] = None,
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
) -> Dict[str, Path]:
    """Run each provider/model target and write a comparison summary.

    Missing provider keys are recorded as skipped target rows, matching the
    underlying live replication behavior.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive when provided")
    artifact_prefix = _safe_slug(artifact_prefix)
    normalized = _normalize_targets(targets)
    target_reports = []
    artifact_paths: Dict[str, Path] = {}

    for target in normalized:
        slug = _target_slug(target)
        target_summary = f"{artifact_prefix}_{slug}_summary.json"
        target_audit = f"{artifact_prefix}_{slug}_audit.jsonl"
        target_diagnostics = f"{artifact_prefix}_{slug}_diagnostics.json"
        paths = {
            "summary": output / target_summary,
            "audit": output / target_audit,
        }
        diagnostics_path = output / target_diagnostics
        reused_existing = reuse_existing and paths["summary"].exists() and paths["audit"].exists()
        if reused_existing:
            if not diagnostics_path.exists():
                diagnostics_path = write_live_replication_diagnostics(
                    output,
                    summary_filename=target_summary,
                    audit_filename=target_audit,
                    diagnostics_filename=target_diagnostics,
                )
        else:
            llm_call = llm_call_factory(target.provider, target.model) if llm_call_factory else None
            paths = write_live_llm_replication_outputs(
                output,
                provider=target.provider,
                model=target.model,
                embedding_model=embedding_model,
                seed_count=seed_count,
                start_seed=start_seed,
                llm_call=llm_call,
                embed_fn=embed_fn,
                dim=dim,
                include_audit_text=include_audit_text,
                summary_filename=target_summary,
                audit_filename=target_audit,
                empty_response_retries=empty_response_retries,
                max_output_tokens=max_output_tokens,
            )
            diagnostics_path = write_live_replication_diagnostics(
                output,
                summary_filename=target_summary,
                audit_filename=target_audit,
                diagnostics_filename=target_diagnostics,
            )
        summary = _read_json(paths["summary"])
        diagnostics = _read_json(diagnostics_path)
        target_reports.append(_target_report(
            slug=slug,
            summary=summary,
            diagnostics=diagnostics,
            summary_path=paths["summary"],
            audit_path=paths["audit"],
            diagnostics_path=diagnostics_path,
            reused_existing=reused_existing,
        ))
        artifact_paths[f"{slug}_summary"] = paths["summary"]
        artifact_paths[f"{slug}_audit"] = paths["audit"]
        artifact_paths[f"{slug}_diagnostics"] = diagnostics_path

    comparison = _comparison_report(
        target_reports=target_reports,
        seed_count=seed_count,
        start_seed=start_seed,
        embedding_model=embedding_model,
        empty_response_retries=empty_response_retries,
        max_output_tokens=max_output_tokens,
        artifact_prefix=artifact_prefix,
        min_passed_targets=min_passed_targets,
    )
    summary_path = output / summary_filename
    summary_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    artifact_paths["summary"] = summary_path
    return artifact_paths


def _normalize_targets(
    targets: Optional[Sequence[ProviderModelTarget | Dict[str, Optional[str]] | str]],
) -> list[ProviderModelTarget]:
    if not targets:
        return [
            ProviderModelTarget(provider="openai"),
            ProviderModelTarget(provider="anthropic"),
        ]
    normalized = []
    for target in targets:
        if isinstance(target, ProviderModelTarget):
            normalized.append(target)
        elif isinstance(target, str):
            normalized.append(parse_provider_model_target(target))
        else:
            provider = str(target.get("provider", "")).strip().lower()
            if not provider:
                raise ValueError("target provider cannot be empty")
            model = target.get("model")
            normalized.append(ProviderModelTarget(provider=provider, model=model or None))
    return normalized


def _target_report(
    *,
    slug: str,
    summary: Dict[str, Any],
    diagnostics: Dict[str, Any],
    summary_path: Path,
    audit_path: Path,
    diagnostics_path: Path,
    reused_existing: bool,
) -> Dict[str, Any]:
    metrics = summary.get("metrics", {})
    gate_semantics = diagnostics.get("gate_semantics", {})
    return {
        "slug": slug,
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "status": summary.get("status"),
        "reason": summary.get("reason"),
        "reused_existing": reused_existing,
        "seed_count": summary.get("seed_count"),
        "route_accuracy_mean": _metric_mean(metrics, "route_accuracy"),
        "retrieval_hit_rate_mean": _metric_mean(metrics, "retrieval_hit_rate"),
        "source_label_accuracy_mean": _metric_mean(metrics, "source_label_accuracy"),
        "echo_promotion_rate_mean": _metric_mean(metrics, "echo_promotion_rate"),
        "quarantine_recall_mean": _metric_mean(metrics, "quarantine_recall"),
        "mean_fold_force_mean": _metric_mean(metrics, "mean_fold_force"),
        "provider_valid_route_accuracy": gate_semantics.get("route_accuracy_if_provider_output_valid"),
        "primitive_failure_count": gate_semantics.get("primitive_failure_count"),
        "provider_output_boundary_failure_count": gate_semantics.get("provider_output_boundary_failure_count"),
        "with_memory_output_validity_rate": gate_semantics.get("with_memory_output_validity_rate"),
        "without_memory_output_validity_rate": gate_semantics.get("without_memory_output_validity_rate"),
        "cost_ledger": summary.get("cost_ledger", {}),
        "acceptance": summary.get("acceptance", {}),
        "paths": {
            "summary": str(summary_path),
            "audit": str(audit_path),
            "diagnostics": str(diagnostics_path),
        },
    }


def _comparison_report(
    *,
    target_reports: list[Dict[str, Any]],
    seed_count: int,
    start_seed: int,
    embedding_model: str,
    empty_response_retries: int,
    max_output_tokens: Optional[int],
    artifact_prefix: str,
    min_passed_targets: int,
) -> Dict[str, Any]:
    passed_count = sum(1 for target in target_reports if target["status"] == "passed")
    skipped_count = sum(1 for target in target_reports if target["status"] == "skipped")
    failed_count = sum(1 for target in target_reports if target["status"] == "failed")
    attempted_count = len(target_reports) - skipped_count
    primitive_failure_count = sum(
        int(target.get("primitive_failure_count", 0) or 0)
        for target in target_reports
    )
    provider_output_boundary_failure_count = sum(
        int(target.get("provider_output_boundary_failure_count", 0) or 0)
        for target in target_reports
    )
    acceptance = {
        "target_count_met": len(target_reports) >= min_passed_targets,
        "minimum_passed_targets_met": passed_count >= min_passed_targets,
        "no_primitive_failures_met": primitive_failure_count == 0,
        "all_attempted_targets_passed": attempted_count > 0 and failed_count == 0,
    }
    acceptance["provider_model_comparison_gate_met"] = (
        acceptance["target_count_met"]
        and acceptance["minimum_passed_targets_met"]
        and acceptance["no_primitive_failures_met"]
    )
    status = "passed" if acceptance["provider_model_comparison_gate_met"] else "failed"
    if attempted_count == 0:
        status = "skipped"
    return {
        "status": status,
        "seed_count": seed_count,
        "start_seed": start_seed,
        "embedding_model": embedding_model,
        "empty_response_retries": empty_response_retries,
        "max_output_tokens": max_output_tokens,
        "artifact_prefix": artifact_prefix,
        "min_passed_targets": min_passed_targets,
        "target_count": len(target_reports),
        "attempted_count": attempted_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "primitive_failure_count": primitive_failure_count,
        "provider_output_boundary_failure_count": provider_output_boundary_failure_count,
        "total_cost_ledger": _sum_cost_ledgers(target["cost_ledger"] for target in target_reports),
        "acceptance": acceptance,
        "targets": target_reports,
    }


def _sum_cost_ledgers(costs: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0}
    for cost in costs:
        for key in total:
            total[key] += int(cost.get(key, 0) or 0)
    return total


def _metric_mean(metrics: Dict[str, Any], metric: str) -> Optional[float]:
    value = metrics.get(metric, {})
    if not isinstance(value, dict):
        return None
    mean = value.get("mean")
    return float(mean) if mean is not None else None


def _target_slug(target: ProviderModelTarget) -> str:
    model = target.model or "default"
    return _safe_slug(f"{target.provider}_{model}")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "target"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

"""Real-component replication helpers.

Controlled toy fixtures are the first validation layer. This module reruns a
small subset with real embedding geometry and records live-LLM availability and
cost ledger state without requiring external API calls by default.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from fgm.core import (
    FGMAgent,
    ROUTE_OPERATION_MEMORY,
    ROUTE_QUARANTINE,
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_SIMULATION,
    Array,
)
from fgm.validation import ValidationRecord, score_validation_records


@dataclass(frozen=True)
class RealComponentReport:
    embedding_model: str
    embedding_available: bool
    embedding_error: Optional[str]
    dim: int
    source_route_accuracy: Optional[float]
    source_false_write_rate: Optional[float]
    source_echo_promotion_rate: Optional[float]
    retrieval_hit_rate: Optional[float]
    fold_force: Optional[float]
    operation_records: int
    live_llm_available: bool
    live_llm_reason: Optional[str]
    cost_ledger: Dict[str, int]


def run_real_embedding_validation(
    *,
    model_name: str = "all-MiniLM-L6-v2",
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
) -> RealComponentReport:
    """Run source/routing validation with real embeddings when available.

    Tests can pass a deterministic ``embed_fn`` and ``dim`` to exercise the
    structure without importing sentence-transformers. Production use leaves
    them unset and loads the configured sentence-transformer model.
    """
    embedding_error = None
    if embed_fn is None:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)

            def embed_fn(text: str) -> Array:
                return model.encode([text], normalize_embeddings=True)[0]

            probe = embed_fn("dimension probe")
            dim = int(probe.shape[0])
        except Exception as exc:  # pragma: no cover - environment-dependent
            embedding_error = str(exc)
            return RealComponentReport(
                embedding_model=model_name,
                embedding_available=False,
                embedding_error=embedding_error,
                dim=0,
                source_route_accuracy=None,
                source_false_write_rate=None,
                source_echo_promotion_rate=None,
                retrieval_hit_rate=None,
                fold_force=None,
                operation_records=0,
                live_llm_available=_has_live_llm_key(),
                live_llm_reason=_live_llm_reason(),
                cost_ledger={"input_tokens": 0, "output_tokens": 0, "api_calls": 0},
            )
    if embed_fn is None or dim is None:
        raise ValueError("embed_fn and dim must be provided together")

    records, agent = _run_source_routing_with_embedder(embed_fn=embed_fn, dim=dim, model_name=model_name)
    metrics = score_validation_records(records)
    fold_force = float(np.mean([record.realized_fold_force or 0.0 for record in records]))
    return RealComponentReport(
        embedding_model=model_name,
        embedding_available=True,
        embedding_error=None,
        dim=dim,
        source_route_accuracy=metrics["route_accuracy"],
        source_false_write_rate=metrics["false_write_rate"],
        source_echo_promotion_rate=metrics["echo_promotion_rate"],
        retrieval_hit_rate=metrics["retrieval_hit_rate"],
        fold_force=fold_force,
        operation_records=len(agent.operations.all_operations()),
        live_llm_available=_has_live_llm_key(),
        live_llm_reason=_live_llm_reason(),
        cost_ledger={"input_tokens": 0, "output_tokens": 0, "api_calls": 0},
    )


def run_real_embedding_replication(
    *,
    model_name: str = "all-MiniLM-L6-v2",
    seed_count: int = 20,
    start_seed: int = 0,
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Run source/routing validation across seeded query variants."""
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")

    embedding_error = None
    if embed_fn is None:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)

            def embed_fn(text: str) -> Array:
                return model.encode([text], normalize_embeddings=True)[0]

            probe = embed_fn("dimension probe")
            dim = int(probe.shape[0])
        except Exception as exc:  # pragma: no cover - environment-dependent
            embedding_error = str(exc)
            return _json_safe({
                "embedding_model": model_name,
                "embedding_available": False,
                "embedding_error": embedding_error,
                "seed_count": seed_count,
                "start_seed": start_seed,
                "dim": 0,
                "runs": [],
                "metrics": {},
                "acceptance": {
                    "real_seed_count_met": seed_count >= 20,
                    "all_real_embedding_replication_gates_met": False,
                },
            })
    if embed_fn is None or dim is None:
        raise ValueError("embed_fn and dim must be provided together")

    runs = []
    for seed in range(start_seed, start_seed + seed_count):
        records, agent = _run_source_routing_with_embedder(
            embed_fn=embed_fn,
            dim=dim,
            model_name=model_name,
            seed=seed,
        )
        metrics = score_validation_records(records)
        runs.append({
            "seed": seed,
            "route_accuracy": metrics["route_accuracy"],
            "retrieval_hit_rate": metrics["retrieval_hit_rate"],
            "source_label_accuracy": metrics["source_label_accuracy"],
            "echo_promotion_rate": metrics["echo_promotion_rate"],
            "false_write_rate": metrics["false_write_rate"],
            "quarantine_recall": metrics["quarantine_recall"],
            "mean_fold_force": float(np.mean([record.realized_fold_force or 0.0 for record in records])),
            "operation_records": len(agent.operations.all_operations()),
        })

    metric_summary = {
        "route_accuracy": _summarize_numeric(run["route_accuracy"] for run in runs),
        "retrieval_hit_rate": _summarize_numeric(run["retrieval_hit_rate"] for run in runs),
        "source_label_accuracy": _summarize_numeric(run["source_label_accuracy"] for run in runs),
        "echo_promotion_rate": _summarize_numeric(run["echo_promotion_rate"] for run in runs),
        "false_write_rate": _summarize_numeric(run["false_write_rate"] for run in runs),
        "quarantine_recall": _summarize_numeric(run["quarantine_recall"] for run in runs),
        "mean_fold_force": _summarize_numeric(run["mean_fold_force"] for run in runs),
        "operation_records": _summarize_numeric(run["operation_records"] for run in runs),
    }
    acceptance = {
        "real_seed_count_met": seed_count >= 20,
        "route_accuracy_mean_met": metric_summary["route_accuracy"]["mean"] >= 0.9,
        "retrieval_hit_rate_mean_met": metric_summary["retrieval_hit_rate"]["mean"] >= 0.9,
        "source_label_accuracy_mean_met": metric_summary["source_label_accuracy"]["mean"] >= 0.9,
        "echo_promotion_control_met": metric_summary["echo_promotion_rate"]["mean"] == 0.0,
        "fold_force_positive_met": metric_summary["mean_fold_force"]["min"] > 0.0,
        "operation_records_mean_met": metric_summary["operation_records"]["mean"] >= 2.0,
    }
    acceptance["all_real_embedding_replication_gates_met"] = all(acceptance.values())

    return _json_safe({
        "embedding_model": model_name,
        "embedding_available": True,
        "embedding_error": None,
        "seed_count": seed_count,
        "start_seed": start_seed,
        "dim": dim,
        "runs": runs,
        "metrics": metric_summary,
        "acceptance": acceptance,
    })


def write_real_embedding_replication_output(
    output_dir: str | Path = "results",
    *,
    model_name: str = "all-MiniLM-L6-v2",
    seed_count: int = 20,
    start_seed: int = 0,
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
) -> Dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = run_real_embedding_replication(
        model_name=model_name,
        seed_count=seed_count,
        start_seed=start_seed,
        embed_fn=embed_fn,
        dim=dim,
    )

    summary_path = output / "real_embedding_replication_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    runs_path = output / "real_embedding_replication_runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as handle:
        for run in summary.get("runs", []):
            handle.write(json.dumps(_json_safe(run), sort_keys=True))
            handle.write("\n")
    return {"summary": summary_path, "runs": runs_path}


def write_real_component_validation_output(
    output_dir: str | Path = "results",
    *,
    model_name: str = "all-MiniLM-L6-v2",
    embed_fn: Optional[Callable[[str], Array]] = None,
    dim: Optional[int] = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = run_real_embedding_validation(model_name=model_name, embed_fn=embed_fn, dim=dim)
    path = output / "real_component_validation_summary.json"
    path.write_text(json.dumps(_json_safe(asdict(report)), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_source_routing_with_embedder(
    *,
    embed_fn: Callable[[str], Array],
    dim: int,
    model_name: str,
    seed: int = 0,
) -> tuple[list[ValidationRecord], FGMAgent]:
    agent = FGMAgent(dim=dim, embed_fn=embed_fn, fold_threshold=0.001, retrieval_k=1, auto_compress=False)
    agent.add(
        "E1 external observation legal approved rollback for deploy migration",
        record_id="E1",
        source_label=SOURCE_EXTERNAL,
        source_confidence=1.0,
    )
    agent.add(
        "R1 prior memory deploy migration timed out and rollback restored service",
        record_id="R1",
        source_label=SOURCE_EXTERNAL,
        source_confidence=0.95,
    )
    agent.add(
        "S1 simulated hypothesis hotfix might avoid rollback",
        record_id="S1",
        source_label=SOURCE_SIMULATION,
        source_confidence=0.7,
    )
    agent.add(
        "F1 fabricated distractor rollback is forbidden by legal",
        record_id="F1",
        source_label=SOURCE_FABRICATED,
        source_confidence=0.9,
    )
    cases = _source_routing_cases(seed)
    records: list[ValidationRecord] = []
    for turn_id, (query, expected_id, expected_source, expected_route, future_utility) in enumerate(cases, start=1):
        result = agent.query(query)
        records.append(
            ValidationRecord.from_fold_result(
                run_id=f"real-component-source-routing:{model_name}",
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
        )
    return records, agent


def _source_routing_cases(seed: int) -> list[tuple[str, str, str, str, bool]]:
    variants = [
        [
            "legal approved rollback deploy migration",
            "what rollback approval applies to deploy migration?",
            "deploy migration legal rollback approval",
            "which approval allows rollback for deployment migration?",
        ],
        [
            "deploy migration timed out rollback restored service",
            "what restored service after deploy migration timeout?",
            "previous deploy migration timeout service rollback",
            "which recovery restored service after migration timeout?",
        ],
        [
            "hotfix avoid rollback hypothetical simulated",
            "which hotfix rollback option is only hypothetical?",
            "simulated hotfix might avoid rollback",
            "what rollback avoidance idea was simulated?",
        ],
        [
            "rollback forbidden by legal fabricated distractor",
            "which rollback legal claim is a distractor?",
            "fabricated claim legal forbids rollback",
            "what adversarial note falsely says rollback is forbidden?",
        ],
    ]
    idx = seed % len(variants[0])
    return [
        (variants[0][idx], "E1", SOURCE_EXTERNAL, ROUTE_OPERATION_MEMORY, True),
        (variants[1][idx], "R1", SOURCE_EXTERNAL, ROUTE_OPERATION_MEMORY, True),
        (variants[2][idx], "S1", SOURCE_SIMULATION, ROUTE_QUARANTINE, False),
        (variants[3][idx], "F1", SOURCE_FABRICATED, ROUTE_QUARANTINE, False),
    ]


def _summarize_numeric(values: Any) -> Dict[str, Any]:
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


def _has_live_llm_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _live_llm_reason() -> Optional[str]:
    if os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY present; live replication recorded by live_validation helper"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY present; live replication not run by this offline helper"
    return "No provider API key set; live LLM replication skipped"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value

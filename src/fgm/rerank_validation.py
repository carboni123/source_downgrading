"""Deterministic checks for source-aware retrieval reranking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fgm.core import ROUTE_QUARANTINE, hash_embed
from fgm.live_validation import run_live_llm_replication
from fgm.llm import echo_call


def run_rerank_boundary_regression() -> Dict[str, Any]:
    """Replay the live seed-4 boundary shape without external API calls."""
    report = run_live_llm_replication(
        provider="openai",
        model="echo-test",
        embedding_model="hash-embed-test",
        seed_count=5,
        llm_call=echo_call(),
        embed_fn=lambda text: hash_embed(text, 64),
        dim=64,
        include_audit_text=False,
    )
    seed4_turn4 = next(
        event for event in report["audit_events"]
        if event["seed"] == 4 and event["turn_id"] == 4
    )
    acceptance = {
        "boundary_retrieval_fixed": seed4_turn4["retrieved_ids"] == ["F1"],
        "boundary_route_fixed": seed4_turn4["selected_route"] == ROUTE_QUARANTINE,
        "route_accuracy_mean_met": report["metrics"]["route_accuracy"]["mean"] == 1.0,
        "retrieval_hit_rate_mean_met": report["metrics"]["retrieval_hit_rate"]["mean"] == 1.0,
        "all_stub_gates_met": report["acceptance"]["all_live_replication_gates_met"],
    }
    return {
        "status": "passed" if all(acceptance.values()) else "failed",
        "replication_status": report["status"],
        "seed_count": report["seed_count"],
        "route_accuracy_mean": report["metrics"]["route_accuracy"]["mean"],
        "retrieval_hit_rate_mean": report["metrics"]["retrieval_hit_rate"]["mean"],
        "quarantine_recall_mean": report["metrics"]["quarantine_recall"]["mean"],
        "seed4_turn4": seed4_turn4,
        "acceptance": acceptance,
    }


def write_rerank_boundary_regression_output(output_dir: str | Path = "results") -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "rerank_boundary_regression_summary.json"
    path.write_text(
        json.dumps(run_rerank_boundary_regression(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path

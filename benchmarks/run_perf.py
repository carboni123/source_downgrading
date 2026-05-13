"""Performance characterization for trace-memory.

Measures steady-state latency of the public mutating and read APIs
across the three reference storage backends and (when available) two
embedding strategies. Produces two artefacts:

    results/benchmarks/results.json    -- raw numbers, one record per cell
    results/benchmarks/PERFORMANCE.md  -- generated markdown report

The NFR-3 targets from ``PRD.md`` are:

    query(...) on 10k records with hash embeddings    < 50 ms
    query(...) on 10k records with sentence-transformers < 500 ms

The runner reports both p50 and p95 latency over a steady-state sample.
Memory footprint is reported via ``tracemalloc`` at each measurement
size.

Run:

    python benchmarks/run_perf.py [--quick]

``--quick`` runs the smaller (N <= 1000) cells only, useful for
iteration. The full sweep takes ~2-3 minutes on a developer laptop.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from trace_memory import (
    InMemoryStorage,
    MemoryAgent,
    SQLiteStorage,
    SourceLabel,
    Storage,
)


# ---------------------------------------------------------------------------
# Embedding strategies
# ---------------------------------------------------------------------------

HASH_DIM = 64

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    def st_embed(text: str) -> np.ndarray:
        vec = _ST_MODEL.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float64)

    ST_DIM = 384
    ST_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-dependent
    print(f"sentence-transformers unavailable, skipping ST sweep: {exc}")
    st_embed = None  # type: ignore[assignment]
    ST_DIM = 0
    ST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def percentiles_ms(durations_s: List[float]) -> Dict[str, float]:
    """Return p50/p95/mean/max in milliseconds."""
    if not durations_s:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
    s = sorted(durations_s)
    n = len(s)
    p95_idx = min(n - 1, int(0.95 * n))
    return {
        "p50_ms": s[n // 2] * 1000.0,
        "p95_ms": s[p95_idx] * 1000.0,
        "mean_ms": (sum(s) / n) * 1000.0,
        "max_ms": s[-1] * 1000.0,
    }


def measure(label: str, fn: Callable[[], None], n_trials: int) -> Dict[str, Any]:
    durations: List[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    summary = percentiles_ms(durations)
    summary["n_trials"] = n_trials
    summary["label"] = label
    return summary


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    backend: str
    embedding: str
    n_records: int
    add: Dict[str, Any] = field(default_factory=dict)
    query: Dict[str, Any] = field(default_factory=dict)
    add_derived: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    seed_memory_kb: float = 0.0
    seed_seconds: float = 0.0


def make_agent(backend: str, embedding: str) -> Tuple[MemoryAgent, Optional[str]]:
    """Construct an agent for a (backend, embedding) cell.

    Returns (agent, tmp_path_to_unlink_on_close).
    """
    embed_fn = None
    dim = HASH_DIM
    if embedding == "sentence_transformers":
        if not ST_AVAILABLE:
            raise RuntimeError("sentence-transformers not available")
        embed_fn = st_embed
        dim = ST_DIM
    storage: Storage
    tmp_path: Optional[str] = None
    if backend == "in_memory":
        storage = InMemoryStorage()
    elif backend == "sqlite_memory":
        storage = SQLiteStorage(":memory:")
    elif backend == "sqlite_disk":
        fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        storage = SQLiteStorage(tmp_path)
    else:
        raise ValueError(f"unknown backend: {backend}")
    agent = MemoryAgent(
        storage=storage,
        embed_fn=embed_fn,
        dim=dim,
        retrieval_k=3,
    )
    return agent, tmp_path


def seed_agent(agent: MemoryAgent, n_records: int) -> Tuple[float, float]:
    """Populate the agent with n_records and return (seconds, memory_kb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    for i in range(n_records):
        agent.add(
            f"observation #{i}: server returned status {200 + (i % 5)}",
            source=SourceLabel.EXTERNAL,
            record_id=f"r{i}",
        )
    seconds = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return seconds, peak / 1024.0


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def run_cell(
    backend: str,
    embedding: str,
    n_records: int,
    *,
    n_op_trials: int,
) -> Cell:
    print(
        f"  [{backend:>13s} / {embedding:>22s} / N={n_records:>5d}] seeding...",
        end="",
        flush=True,
    )
    gc.collect()
    agent, tmp_path = make_agent(backend, embedding)
    try:
        seed_s, seed_kb = seed_agent(agent, n_records)
        print(f" {seed_s:.2f}s; running ops...", end="", flush=True)

        # add() steady-state: append n_op_trials new records.
        next_id_start = n_records
        def _add_one(state: Dict[str, int] = {"i": next_id_start}) -> None:
            i = state["i"]
            agent.add(
                f"observation #{i}: server returned status {200 + (i % 5)}",
                source=SourceLabel.EXTERNAL,
                record_id=f"r{i}",
            )
            state["i"] = i + 1

        add_summary = measure("add", _add_one, n_op_trials)

        # query() steady-state: vary the query string deterministically.
        def _query_one(state: Dict[str, int] = {"i": 0}) -> None:
            i = state["i"]
            agent.query(f"server returned {200 + (i % 5)}")
            state["i"] = i + 1

        query_summary = measure("query", _query_one, n_op_trials)

        # add_derived() steady-state: build derivations from two seed records.
        existing_records = agent.store.all_records()[:2]
        if len(existing_records) >= 2:
            r1, r2 = existing_records[0], existing_records[1]
            def _derive_one(state: Dict[str, int] = {"i": 0}) -> None:
                i = state["i"]
                agent.add_derived(
                    f"inferred fact {i}",
                    inputs=[r1, r2],
                    record_id=f"d{i}",
                )
                state["i"] = i + 1
            derived_summary = measure("add_derived", _derive_one, max(n_op_trials // 4, 20))
        else:
            derived_summary = {"label": "add_derived", "skipped": True}

        # audit_laundering() steady-state.
        def _audit_once() -> None:
            agent.audit_laundering()
        audit_summary = measure("audit", _audit_once, max(n_op_trials // 4, 20))

        print(" done")
        return Cell(
            backend=backend,
            embedding=embedding,
            n_records=n_records,
            add=add_summary,
            query=query_summary,
            add_derived=derived_summary,
            audit=audit_summary,
            seed_memory_kb=seed_kb,
            seed_seconds=seed_s,
        )
    finally:
        agent.close()
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def format_op_row(cell: Cell, op_name: str) -> str:
    op = getattr(cell, op_name)
    if op.get("skipped"):
        return f"| {op_name} | -- | -- | -- |"
    return (
        f"| {op_name} "
        f"| {op['p50_ms']:.2f} "
        f"| {op['p95_ms']:.2f} "
        f"| {op['n_trials']} |"
    )


def render_markdown(cells: List[Cell], nfr3_status: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Performance Characterization")
    lines.append("")
    lines.append(
        "Auto-generated by `benchmarks/run_perf.py`. Latencies are reported "
        "in milliseconds. p50 is the median; p95 captures tail behaviour. "
        "Each cell uses 200 trials at steady state (the agent is seeded to "
        "the indicated record count before measurement begins)."
    )
    lines.append("")
    lines.append("## NFR-3 verdict")
    lines.append("")
    lines.append(
        "The PRD's NFR-3 sets query() latency targets against 10k records. "
        "The latest run reports:"
    )
    lines.append("")
    lines.append("| Embedding | NFR-3 target | Measured p50 | Verdict |")
    lines.append("|---|---|---|---|")
    for entry in nfr3_status["rows"]:
        lines.append(
            f"| {entry['embedding']} "
            f"| < {entry['target_ms']:.0f} ms "
            f"| {entry['measured_p50_ms']:.2f} ms "
            f"| {entry['verdict']} |"
        )
    lines.append("")
    lines.append("## Full sweep")
    lines.append("")
    for cell in cells:
        lines.append(
            f"### {cell.backend} / {cell.embedding} / N = {cell.n_records}"
        )
        lines.append("")
        lines.append(
            f"seeding: {cell.seed_seconds:.2f} s; peak memory during seeding: "
            f"{cell.seed_memory_kb:,.1f} KB"
        )
        lines.append("")
        lines.append("| op | p50 (ms) | p95 (ms) | trials |")
        lines.append("|---|---|---|---|")
        lines.append(format_op_row(cell, "add"))
        lines.append(format_op_row(cell, "query"))
        lines.append(format_op_row(cell, "add_derived"))
        lines.append(format_op_row(cell, "audit"))
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- ``in_memory`` is the default ``InMemoryStorage`` backend (state "
        "lost on process exit; no persistence overhead)."
    )
    lines.append(
        "- ``sqlite_memory`` is ``SQLiteStorage(\":memory:\")`` -- exercises "
        "the SQL serialization path without disk I/O."
    )
    lines.append(
        "- ``sqlite_disk`` is ``SQLiteStorage(tempfile)`` -- includes disk "
        "I/O and represents a typical production deployment."
    )
    lines.append(
        "- ``hash`` embeddings are the deterministic 64-dim "
        "hash-bag-of-words used in the validation harness. ``sentence_"
        "transformers`` uses ``all-MiniLM-L6-v2`` (384-dim)."
    )
    lines.append(
        "- ``query`` operations time the full pipeline: retrieve -> fold "
        "-> route. ``add_derived`` includes both contributing-input "
        "resolution and provenance propagation. ``audit`` walks the "
        "current derived-record set."
    )
    lines.append("")
    return "\n".join(lines)


def evaluate_nfr3(cells: List[Cell]) -> Dict[str, Any]:
    """Pick the N=10000 cells (in_memory backend) and compare to NFR-3 targets."""
    targets = {
        "hash": 50.0,
        "sentence_transformers": 500.0,
    }
    rows: List[Dict[str, Any]] = []
    for cell in cells:
        if cell.backend != "in_memory":
            continue
        if cell.n_records != 10_000:
            continue
        target = targets.get(cell.embedding)
        if target is None:
            continue
        p50 = cell.query.get("p50_ms")
        if p50 is None:
            continue
        verdict = "PASS" if p50 < target else "FAIL"
        rows.append({
            "embedding": cell.embedding,
            "target_ms": target,
            "measured_p50_ms": p50,
            "verdict": verdict,
        })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the N=10000 cells for fast iteration",
    )
    parser.add_argument("--output-dir", default="results/benchmarks")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Per-cell op trials at steady state",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sweep_hash = [100, 1000]
    if not args.quick:
        n_sweep_hash.append(10_000)
    n_sweep_st = [100, 1000]
    if not args.quick:
        n_sweep_st.append(10_000)

    cells: List[Cell] = []

    print("hash embedding sweep:")
    for backend in ("in_memory", "sqlite_memory", "sqlite_disk"):
        for n in n_sweep_hash:
            cells.append(run_cell(backend, "hash", n, n_op_trials=args.n_trials))

    if ST_AVAILABLE:
        print("sentence-transformers embedding sweep:")
        for n in n_sweep_st:
            cells.append(
                run_cell("in_memory", "sentence_transformers", n, n_op_trials=args.n_trials)
            )
    else:
        print("sentence-transformers sweep skipped (package unavailable).")

    nfr3 = evaluate_nfr3(cells)

    results = {
        "python": sys.version,
        "platform": sys.platform,
        "n_trials": args.n_trials,
        "quick": args.quick,
        "st_available": ST_AVAILABLE,
        "cells": [asdict(c) for c in cells],
        "nfr3": nfr3,
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {json_path}")

    md = render_markdown(cells, nfr3)
    md_path = output_dir / "PERFORMANCE.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"wrote {md_path}")

    if nfr3["rows"]:
        print("\nNFR-3 summary:")
        for row in nfr3["rows"]:
            print(
                f"  {row['embedding']:>22s}: target < {row['target_ms']:.0f}ms, "
                f"measured p50 = {row['measured_p50_ms']:.2f}ms, verdict = "
                f"{row['verdict']}"
            )


if __name__ == "__main__":
    main()

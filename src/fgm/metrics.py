"""Regime evaluation for Fold-Gated Memory.

Implements the four-regime classification from the research sketch:
    sparse_confusable     -> overload (hit rate falls as N grows)
    compressed_preserving -> no overload (margins maintained)
    rich_distinctive      -> no overload (high margins, zero confusability)
    aggressive_lossy      -> overload (unique content lost)

Also provides the evaluation protocol:
    1. Regime classification via hit-rate trajectory
    2. Fold-force audit
    3. Compression boundary detection
    4. Operation-memory depth measurement
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fgm.core import (
    FGMAgent,
    MemoryStore,
    MarginRetriever,
    hash_embed,
    cosine,
    default_transition,
    Array,
    EPS,
)


@dataclass(frozen=True)
class RegimeRow:
    n: int
    k: int
    hit_rate: float
    mean_rank: float
    mean_margin: float
    confusability: float


@dataclass(frozen=True)
class RegimeReport:
    regime_name: str
    rows: List[RegimeRow]
    finite_differences: List[Dict[str, float]]
    overload_like: bool
    classification: str


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def _make_records(
    n: int,
    n_classes: int,
    class_strength: float,
    unique_strength: float,
    dim_class: int,
    dim_unique: int,
    noise_std: float,
    rng: np.random.Generator,
):
    class_proto = _unit_rows(rng.normal(size=(n_classes, dim_class)))
    labels = rng.integers(0, n_classes, size=n)
    unique_proto = _unit_rows(rng.normal(size=(n, dim_unique)))
    vectors = []
    for i in range(n):
        v = np.concatenate([
            class_strength * class_proto[labels[i]],
            unique_strength * unique_proto[i],
        ])
        v += rng.normal(0, noise_std, size=v.shape[0])
        vectors.append(v)
    return vectors, class_proto, unique_proto, labels


def _make_query(
    target_i: int,
    class_proto: np.ndarray,
    unique_proto: np.ndarray,
    labels: np.ndarray,
    class_strength: float,
    query_unique_strength: float,
    noise_std: float,
    dim_class: int,
    rng: np.random.Generator,
) -> Array:
    q = np.concatenate([
        class_strength * class_proto[labels[target_i]],
        query_unique_strength * unique_proto[target_i],
    ])
    return q + rng.normal(0, noise_std, size=q.shape[0])


REGIME_CONFIGS = {
    "sparse_confusable": {
        "class_strength": 1.0, "unique_strength": 0.03,
        "query_unique_strength": 0.0, "noise_std": 0.04,
        "n_classes": 4, "dim_class": 8, "dim_unique": 24,
    },
    "compressed_preserving": {
        "class_strength": 0.9, "unique_strength": 0.45,
        "query_unique_strength": 0.45, "noise_std": 0.04,
        "n_classes": 4, "dim_class": 8, "dim_unique": 24,
    },
    "rich_distinctive": {
        "class_strength": 0.8, "unique_strength": 1.0,
        "query_unique_strength": 1.0, "noise_std": 0.04,
        "n_classes": 4, "dim_class": 8, "dim_unique": 24,
    },
    "aggressive_lossy": {
        "class_strength": 1.0, "unique_strength": 0.0,
        "query_unique_strength": 0.8, "noise_std": 0.04,
        "n_classes": 4, "dim_class": 8, "dim_unique": 24,
    },
}


class RegimeEvaluator:
    """Evaluates an agent memory system against the four-regime classification."""

    def __init__(self, margin_threshold: float = 0.05):
        self.margin_threshold = margin_threshold

    def evaluate_regime(
        self,
        regime: str,
        n_values: Sequence[int] = (8, 16, 32, 64, 128),
        k: int = 3,
        n_queries: int = 300,
        seed: int = 42,
    ) -> RegimeReport:
        """Run one regime through the hit-rate trajectory test."""
        if regime not in REGIME_CONFIGS:
            raise ValueError(f"Unknown regime: {regime}")
        cfg = REGIME_CONFIGS[regime]
        rng = np.random.default_rng(seed)
        dim = cfg["dim_class"] + cfg["dim_unique"]

        rows: List[RegimeRow] = []

        for n in n_values:
            vectors, cp, up, labels = _make_records(
                n, cfg["n_classes"], cfg["class_strength"], cfg["unique_strength"],
                cfg["dim_class"], cfg["dim_unique"], cfg["noise_std"], rng,
            )

            store = MemoryStore(dim=dim, embed_fn=lambda t: hash_embed(t, dim))
            for i, v in enumerate(vectors):
                store.add(f"record_{i}", record_id=f"rec_{i}", vector=v)

            retriever = MarginRetriever(store, margin_threshold=self.margin_threshold)
            hits = 0
            ranks: List[int] = []
            margins: List[float] = []
            confusable = 0

            for _ in range(n_queries):
                ti = int(rng.integers(0, n))
                q = _make_query(
                    ti, cp, up, labels,
                    cfg["class_strength"], cfg["query_unique_strength"],
                    cfg["noise_std"], cfg["dim_class"], rng,
                )
                rank, _score, margin = retriever.target_rank_margin(q, f"rec_{ti}")
                hits += int(rank is not None and rank <= k)
                if rank is not None:
                    ranks.append(rank)
                margins.append(margin)
                confusable += int(margin <= self.margin_threshold)

            rows.append(RegimeRow(
                n=n, k=k,
                hit_rate=hits / n_queries,
                mean_rank=float(np.mean(ranks)) if ranks else float("nan"),
                mean_margin=float(np.mean(margins)),
                confusability=confusable / n_queries,
            ))

        diffs = [
            {"from_N": a.n, "to_N": b.n, "delta_hit_rate": b.hit_rate - a.hit_rate}
            for a, b in zip(rows[:-1], rows[1:])
        ]
        overload = any(d["delta_hit_rate"] < -0.05 for d in diffs)

        classification = self._classify(rows, overload)

        return RegimeReport(
            regime_name=regime,
            rows=rows,
            finite_differences=diffs,
            overload_like=overload,
            classification=classification,
        )

    def evaluate_all(
        self,
        n_values: Sequence[int] = (8, 16, 32, 64, 128),
        k: int = 3,
        n_queries: int = 300,
        seed: int = 42,
    ) -> Dict[str, RegimeReport]:
        return {
            regime: self.evaluate_regime(regime, n_values, k, n_queries, seed)
            for regime in REGIME_CONFIGS
        }

    def evaluate_agent(
        self,
        agent: FGMAgent,
        queries: Sequence[str],
        target_ids: Sequence[str],
        k: int = 3,
    ) -> Dict[str, Any]:
        """Evaluate an FGMAgent's current memory state."""
        hits = 0
        margins = []
        fold_forces = []

        for query, tid in zip(queries, target_ids):
            rank, _, margin = agent.retriever.target_rank_margin(query, tid)
            hits += int(rank is not None and rank <= k)
            margins.append(margin)
            result = agent.query(query, k=k)
            fold_forces.append(result.fold_force)

        n = len(queries)
        hit_rate = hits / max(n, 1)
        mean_margin = float(np.mean(margins)) if margins else 0.0
        mean_fold_force = float(np.mean(fold_forces)) if fold_forces else 0.0
        zero_fold_fraction = sum(1 for f in fold_forces if f < agent.fold_gate.threshold) / max(n, 1)

        return {
            "hit_rate": hit_rate,
            "mean_margin": mean_margin,
            "mean_fold_force": mean_fold_force,
            "zero_fold_fraction": zero_fold_fraction,
            "n_queries": n,
            "agent_metrics": agent.metrics(),
        }

    def _classify(self, rows: List[RegimeRow], overload: bool) -> str:
        last = rows[-1]
        if overload and last.confusability > 0.5:
            if last.mean_margin > -0.035:
                return "sparse_confusable"
            return "aggressive_lossy"
        if not overload and last.mean_margin > 0.2:
            return "rich_distinctive"
        if not overload:
            return "compressed_preserving"
        return "unknown"

"""Experiment 1: Fold-Gating vs Store-Everything.

Head-to-head comparison of two matched agents on the same task sequence:
  Agent A (baseline):   stores all records, never prunes
  Agent B (fold-gated): stores all records, prunes low-fold-force after each phase

Prediction (from Prediction 3): Agent B outperforms Agent A on delayed recall
despite storing fewer records, because its stored records are transition-effective
and its retrieval is less confusable.

Design:
  - Signal records: moderate class + moderate unique (retrievable by queries)
  - Noise records: strong class + near-zero unique (confusable, crowd out signal)
  - Both share the same class prototypes, so noise is in the same embedding
    neighborhood as signal but lacks distinctive retrievability
  - Queries carry unique-subspace information matching signal records
  - The transition function amplifies fold vectors proportional to their
    unique-subspace energy, so noise retrievals produce low fold-force
  - As noise accumulates: baseline's signal gets displaced from top-k (overload);
    fold-gated prunes noise, keeping signal in top-k
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fgm.core import (
    FGMAgent,
    MemoryStore,
    MarginRetriever,
    Compressor,
    hash_embed,
    cosine,
    Array,
    EPS,
    TransitionFn,
)


@dataclass(frozen=True)
class PhaseMetrics:
    phase: int
    n_total_records: int
    n_content_records: int
    hit_rate: float
    mean_margin: float
    confusability: float
    mean_fold_force: float
    zero_fold_fraction: float


@dataclass(frozen=True)
class ExperimentResult:
    baseline_phases: List[PhaseMetrics]
    foldgated_phases: List[PhaseMetrics]
    baseline_final: Dict[str, Any]
    foldgated_final: Dict[str, Any]
    prediction_holds: bool


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, EPS)


def make_unique_sensitive_transition(topic_dim: int, unique_dim: int) -> TransitionFn:
    """Transition where fold impact scales with unique-subspace energy in fold vector.

    Signal records carry unique-subspace energy -> high fold-force.
    Noise records carry only topic-subspace energy -> low fold-force.
    """
    def transition(state: Array, input_vec: Array, fold_vec: Optional[Array] = None) -> Array:
        base = np.tanh(0.6 * state + 0.8 * input_vec)
        if fold_vec is None:
            return base
        unique_energy = float(np.linalg.norm(fold_vec[topic_dim:]))
        return np.tanh(0.6 * state + 0.8 * input_vec + unique_energy * 0.9 * fold_vec)

    return transition


class Experiment1:
    """Fold-gating vs store-everything comparison.

    Uses synthetic vector records matching the regime evaluator's approach:
    class prototypes + unique components, with signal having moderate unique
    strength and noise having near-zero unique strength.
    """

    def __init__(
        self,
        n_phases: int = 8,
        noise_per_phase: int = 80,
        n_topics: int = 4,
        n_signal_per_topic: int = 3,
        dim: int = 32,
        fold_threshold: float = 0.008,
        k: int = 3,
        seed: int = 42,
    ):
        self.n_phases = n_phases
        self.noise_per_phase = noise_per_phase
        self.n_topics = n_topics
        self.n_signal_per_topic = n_signal_per_topic
        self.dim = dim
        self.topic_dim = dim // 2
        self.unique_dim = dim - self.topic_dim
        self.fold_threshold = fold_threshold
        self.k = k
        self.seed = seed

    def run(self) -> ExperimentResult:
        rng = np.random.default_rng(self.seed)

        topic_protos = np.stack([
            _unit(rng.normal(size=self.topic_dim)) for _ in range(self.n_topics)
        ])

        signal_records, signal_queries = self._make_signal(rng, topic_protos)

        transition_fn = make_unique_sensitive_transition(self.topic_dim, self.unique_dim)

        cog_dims = list(range(self.topic_dim, self.dim))

        baseline = FGMAgent(
            dim=self.dim, transition_fn=transition_fn,
            fold_threshold=self.fold_threshold,
            retrieval_k=self.k, auto_compress=False,
        )
        foldgated = FGMAgent(
            dim=self.dim, transition_fn=transition_fn,
            fold_threshold=self.fold_threshold,
            retrieval_k=self.k, auto_compress=False,
            cog_dims=cog_dims,
        )

        for rid, vec in signal_records:
            baseline.store.add(f"signal:{rid}", record_id=rid, vector=vec)
            foldgated.store.add(f"signal:{rid}", record_id=rid, vector=vec)

        baseline_phases: List[PhaseMetrics] = []
        foldgated_phases: List[PhaseMetrics] = []

        for phase in range(self.n_phases):
            noise = self._make_noise(rng, topic_protos, self.noise_per_phase)
            for nid, nvec in noise:
                baseline.store.add(f"noise:{nid}", record_id=nid, vector=nvec)
                foldgated.store.add(f"noise:{nid}", record_id=nid, vector=nvec)

            for q_vec, _tid in signal_queries:
                self._query_with_vector(baseline, q_vec)
                self._query_with_vector(foldgated, q_vec)

            self._prune_low_fold(foldgated)

            baseline_phases.append(self._measure(baseline, signal_queries, phase))
            foldgated_phases.append(self._measure(foldgated, signal_queries, phase))

        prediction_holds = (
            foldgated_phases[-1].hit_rate > baseline_phases[-1].hit_rate
            and foldgated_phases[-1].n_content_records < baseline_phases[-1].n_content_records
        )

        return ExperimentResult(
            baseline_phases=baseline_phases,
            foldgated_phases=foldgated_phases,
            baseline_final=baseline.metrics(),
            foldgated_final=foldgated.metrics(),
            prediction_holds=prediction_holds,
        )

    def _make_signal(
        self, rng: np.random.Generator, topic_protos: np.ndarray,
    ) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[np.ndarray, str]]]:
        """Signal records: moderate class + moderate unique.

        The unique component (0.15) is just strong enough to be retrievable
        at small N but will be overwhelmed by same-topic noise at large N.
        Queries carry the same unique signature, enabling fold-force measurement.
        """
        records = []
        queries = []
        for t in range(self.n_topics):
            for s in range(self.n_signal_per_topic):
                rid = f"signal_t{t}_s{s}"
                unique = _unit(rng.normal(size=self.unique_dim))
                vec = np.concatenate([
                    1.0 * topic_protos[t],
                    0.20 * unique,
                ])
                vec += rng.normal(0, 0.03, size=self.dim)
                records.append((rid, _unit(vec)))

                q = np.concatenate([
                    1.0 * topic_protos[t],
                    0.20 * unique,
                ])
                q += rng.normal(0, 0.03, size=self.dim)
                queries.append((_unit(q), rid))
        return records, queries

    def _make_noise(
        self, rng: np.random.Generator, topic_protos: np.ndarray, n: int,
    ) -> List[Tuple[str, np.ndarray]]:
        """Noise records: strong class + near-zero unique.

        Same topic subspace as signal, but no distinctive uniqueness.
        These crowd out signal in top-k as N grows.
        """
        records = []
        for _ in range(n):
            t = int(rng.integers(0, self.n_topics))
            vec = np.concatenate([
                1.0 * topic_protos[t],
                rng.normal(0, 0.01, size=self.unique_dim),
            ])
            vec += rng.normal(0, 0.03, size=self.dim)
            records.append((f"noise_{rng.integers(0, 10_000_000)}", _unit(vec)))
        return records

    def _query_with_vector(self, agent: FGMAgent, q_vec: np.ndarray):
        agent._step += 1
        report = agent.retriever.retrieve(q_vec, k=self.k)
        result = agent.fold_gate.fold("q", q_vec, report.hits, agent._state)

        for hit in report.hits:
            agent.store.record_fold_force(hit.record.record_id, result.fold_force)

        if result.gated:
            agent._state = result.output_with
            depth = 1
            for hit in report.hits:
                if hit.record.record_type == "operation":
                    depth = max(depth, hit.record.metadata.get("recursive_depth", 0) + 1)
            agent.operations.record_fold(result, recursive_depth=depth)

        agent._fold_history.append(result)

    def _prune_low_fold(self, agent: FGMAgent):
        """Remove content records whose average fold-force per retrieval is below threshold.

        Signal records produce high fold-force when retrieved (they carry unique
        information that changes the transition). Noise records produce low
        fold-force (they're topically similar but lack distinctive content).
        """
        to_remove = []
        for rec in agent.store.all_records():
            if rec.record_type != "content":
                continue
            uses = agent.store.use_count(rec.record_id)
            if uses == 0:
                continue
            avg_force = agent.store.total_fold_force(rec.record_id) / uses
            if avg_force < self.fold_threshold:
                to_remove.append(rec.record_id)
        for rid in to_remove:
            agent.store.remove(rid)

    def _measure(
        self, agent: FGMAgent,
        signal_queries: List[Tuple[np.ndarray, str]],
        phase: int,
    ) -> PhaseMetrics:
        records = agent.store.all_records()
        content_records = [r for r in records if r.record_type == "content"]

        hits = 0
        margins = []
        fold_forces = []

        for q_vec, target_id in signal_queries:
            rank, _score, margin = agent.retriever.target_rank_margin(q_vec, target_id)
            hits += int(rank is not None and rank <= self.k)
            margins.append(margin if np.isfinite(margin) else 0.0)

            report = agent.retriever.retrieve(q_vec, k=self.k)
            result = agent.fold_gate.fold("q", q_vec, report.hits, agent._state)
            fold_forces.append(result.fold_force)

        n_queries = len(signal_queries)
        hit_rate = hits / max(n_queries, 1)
        mean_margin = float(np.mean(margins)) if margins else 0.0
        mean_ff = float(np.mean(fold_forces)) if fold_forces else 0.0
        zero_ff = sum(1 for f in fold_forces if f < self.fold_threshold) / max(n_queries, 1)

        chi = agent.retriever.estimate_confusability(
            n_queries=min(50, len(records)), rng=np.random.default_rng(0)
        ) if len(records) >= 2 else 0.0

        return PhaseMetrics(
            phase=phase,
            n_total_records=len(records),
            n_content_records=len(content_records),
            hit_rate=hit_rate,
            mean_margin=mean_margin,
            confusability=chi,
            mean_fold_force=mean_ff,
            zero_fold_fraction=zero_ff,
        )

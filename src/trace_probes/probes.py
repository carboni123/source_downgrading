"""Minimal Trace Validation Framework.

This file turns the revised trace/memory definitions into executable probes:

    prior attended state -> retained residue -> causal intervention sensitivity
    -> storage/addressability -> fold -> non-bookkeeping transition effect

It is a validation scaffold, not a complete cognitive architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

Array = np.ndarray
EPS = 1e-12


def vec(x: Array | Sequence[float]) -> Array:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"Expected 1D vector, got {a.shape}")
    return a


def l2(a: Array, b: Array) -> float:
    return float(np.linalg.norm(vec(a) - vec(b)))


def cosine(a: Array, b: Array) -> float:
    a, b = vec(a), vec(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < EPS:
        return 0.0
    return float(np.dot(a, b) / den)


def softmax(x: Array, temperature: float = 1.0) -> Array:
    x = vec(x) / max(float(temperature), EPS)
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def kl(p: Array, q: Array) -> float:
    p = np.asarray(p, dtype=float) + EPS
    q = np.asarray(q, dtype=float) + EPS
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def sym_kl(p: Array, q: Array) -> float:
    return 0.5 * (kl(p, q) + kl(q, p))


@dataclass(frozen=True)
class ProbeResult:
    name: str
    metrics: Dict[str, Any]
    passed: Optional[bool] = None
    notes: str = ""


@dataclass(frozen=True)
class TraceComponent:
    """Lag-indexed component contributing to tau_t."""
    origin_t: int
    lag: int
    vector: Array
    payload: Dict[str, Any]


@dataclass
class TraceState:
    """Trace field tau_t with lag projections P_k tau_t."""
    t: int
    dim: int
    components: List[TraceComponent]

    @property
    def tau(self) -> Array:
        if not self.components:
            return np.zeros(self.dim)
        return np.sum([c.vector for c in self.components], axis=0)

    def project_lag(self, k: int) -> Array:
        if k < 0:
            raise ValueError("k must be non-negative")
        parts = [c.vector for c in self.components if c.lag == k]
        if not parts:
            return np.zeros(self.dim)
        return np.sum(parts, axis=0)

    def intensity(self) -> float:
        return float(np.linalg.norm(self.tau))

    def lag_intensity(self, k: int) -> float:
        return float(np.linalg.norm(self.project_lag(k)))


class LeakyTraceOperator:
    """Simple trace operator with explicit lag projections.

    Existing trace components shift from lag k to k+1 and decay. The current
    attended state is inserted as a lag-0 component. This makes P_k tau_t
    directly inspectable.
    """
    def __init__(
        self,
        dim: int,
        decay: float = 0.85,
        max_lag: int = 32,
        input_gain: float = 1.0,
        feature_map: Optional[Callable[[Array], Array]] = None,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.dim = int(dim)
        self.decay = float(decay)
        self.max_lag = int(max_lag)
        self.input_gain = float(input_gain)
        self.feature_map = feature_map or (lambda x: vec(x))

    def initial(self) -> TraceState:
        return TraceState(t=-1, dim=self.dim, components=[])

    def update(self, prev: TraceState, attended_state: Array, t: int, payload: Optional[Dict[str, Any]] = None) -> TraceState:
        y = vec(self.feature_map(vec(attended_state)))
        if y.shape[0] != self.dim:
            raise ValueError(f"feature_map produced dim {y.shape[0]}, expected {self.dim}")
        comps: List[TraceComponent] = []
        for c in prev.components:
            lag = c.lag + 1
            if lag <= self.max_lag:
                new_v = self.decay * c.vector
                if np.linalg.norm(new_v) > EPS:
                    comps.append(TraceComponent(c.origin_t, lag, new_v, dict(c.payload)))
        comps.append(TraceComponent(int(t), 0, self.input_gain * y, dict(payload or {})))
        return TraceState(t=int(t), dim=self.dim, components=comps)

    def run(self, sequence: Sequence[Array]) -> List[TraceState]:
        state = self.initial()
        out: List[TraceState] = []
        for t, x in enumerate(sequence):
            state = self.update(state, x, t)
            out.append(state)
        return out


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    t: int
    vector: Array
    payload: Dict[str, Any]
    operation_type: Optional[str] = None
    decision_content: Optional[str] = None


@dataclass(frozen=True)
class RetrievalHit:
    record: MemoryRecord
    score: float
    rank: int


class TopKRetriever:
    """Addressing operator: bounded top-k retrieval by similarity."""
    def __init__(self, similarity: Callable[[Array, Array], float] = cosine) -> None:
        self.similarity = similarity

    def rank(self, records: Sequence[MemoryRecord], query: Array) -> List[RetrievalHit]:
        hits = [RetrievalHit(r, float(self.similarity(query, r.vector)), -1) for r in records]
        hits.sort(key=lambda h: h.score, reverse=True)
        return [RetrievalHit(h.record, h.score, i + 1) for i, h in enumerate(hits)]

    def top_k(self, records: Sequence[MemoryRecord], query: Array, k: int) -> List[RetrievalHit]:
        return self.rank(records, query)[:max(0, int(k))]

    def target_rank_margin(self, records: Sequence[MemoryRecord], query: Array, target_id: str) -> Tuple[Optional[int], float, float]:
        ranked = self.rank(records, query)
        target_rank: Optional[int] = None
        target_score = -np.inf
        max_distractor = -np.inf
        for h in ranked:
            if h.record.record_id == target_id:
                target_rank = h.rank
                target_score = h.score
            else:
                max_distractor = max(max_distractor, h.score)
        if target_rank is None:
            return None, -np.inf, -np.inf
        margin = float(target_score - max_distractor) if np.isfinite(max_distractor) else float("inf")
        return target_rank, float(target_score), margin


class FoldOperator:
    """Fold selected records into a live transition signal."""
    def __init__(self, temperature: float = 0.2) -> None:
        self.temperature = float(temperature)

    def fold(self, hits: Sequence[RetrievalHit]) -> Optional[Array]:
        if not hits:
            return None
        scores = np.array([h.score for h in hits], dtype=float)
        weights = softmax(scores, self.temperature)
        vectors = np.stack([vec(h.record.vector) for h in hits])
        return np.sum(weights[:, None] * vectors, axis=0)


@dataclass(frozen=True)
class TransitionOutput:
    """Toy transition result.

    bookkeeping_log is deliberately included so probes can exclude it.
    """
    action_logits: Array
    belief: Array
    planning_state: Array
    utility: float
    bookkeeping_log: Tuple[str, ...]

    def h_cog(self) -> Array:
        """Admissible non-bookkeeping projection."""
        return np.concatenate([
            vec(self.action_logits),
            vec(self.belief),
            vec(self.planning_state),
            np.array([float(self.utility)]),
        ])


def toy_transition(folded_signal: Optional[Array], n_actions: int, target_action: Optional[int], bookkeeping_event: str) -> TransitionOutput:
    if folded_signal is None:
        logits = np.zeros(n_actions)
    else:
        f = vec(folded_signal)
        if f.shape[0] < n_actions:
            raise ValueError("folded_signal has fewer dimensions than n_actions")
        logits = f[:n_actions]
    belief = softmax(logits)
    utility = 0.0 if target_action is None else float(np.argmax(belief) == int(target_action))
    return TransitionOutput(logits, belief, belief.copy(), utility, (bookkeeping_event,))


def trace_retention_probe(sequence: Sequence[Array], trace_op: LeakyTraceOperator, t: int, k: int) -> ProbeResult:
    states = trace_op.run(sequence)
    if not (0 <= t < len(states)) or not (0 <= t - k < len(sequence)):
        raise ValueError("Invalid t/k for sequence")
    state = states[t]
    source = vec(sequence[t - k])
    proj = state.project_lag(k)
    metrics = {
        "t": int(t),
        "k": int(k),
        "trace_intensity": state.intensity(),
        "lag_intensity": state.lag_intensity(k),
        "retention_cosine": cosine(proj, source),
        "projection_l2_to_source": l2(proj, source),
    }
    return ProbeResult("trace_retention", metrics, passed=metrics["lag_intensity"] > 0, notes="Measures whether P_k tau_t retains Y_{t-k}.")


def causal_trace_probe(sequence: Sequence[Array], trace_op_factory: Callable[[], LeakyTraceOperator], t: int, k: int, replacement_state: Array) -> ProbeResult:
    if not (0 <= t < len(sequence)) or not (0 <= t - k < len(sequence)):
        raise ValueError("Invalid t/k for sequence")
    baseline = trace_op_factory().run(sequence)
    intervened_seq = [vec(x).copy() for x in sequence]
    intervention_time = t - k
    intervened_seq[intervention_time] = vec(replacement_state)
    intervened = trace_op_factory().run(intervened_seq)
    base_proj = baseline[t].project_lag(k)
    int_proj = intervened[t].project_lag(k)
    delta = l2(base_proj, int_proj)
    metrics = {
        "t": int(t),
        "k": int(k),
        "intervention_time": int(intervention_time),
        "projection_delta_l2": delta,
        "baseline_lag_intensity": float(np.linalg.norm(base_proj)),
        "intervened_lag_intensity": float(np.linalg.norm(int_proj)),
        "baseline_intervened_cosine": cosine(base_proj, int_proj),
    }
    return ProbeResult("causal_trace", metrics, passed=delta > 1e-9, notes="Tests whether prior-state intervention changes present lag residue.")


def build_trace_records(sequence: Sequence[Array], trace_op: LeakyTraceOperator, store_lag0: bool = True) -> Tuple[List[MemoryRecord], List[TraceState]]:
    states = trace_op.run(sequence)
    records: List[MemoryRecord] = []
    for st in states:
        v = st.project_lag(0) if store_lag0 else st.tau
        records.append(MemoryRecord(
            record_id=f"r{st.t}",
            t=st.t,
            vector=v,
            payload={"source_t": st.t},
            operation_type="store_trace",
            decision_content=f"attended_state_at_{st.t}",
        ))
    return records, states


def addressability_probe(records: Sequence[MemoryRecord], query: Array, target_id: str, k: int = 1, margin_threshold: float = 0.05) -> ProbeResult:
    retriever = TopKRetriever()
    rank, score, margin = retriever.target_rank_margin(records, query, target_id)
    hit = rank is not None and rank <= k
    metrics = {
        "target_rank": rank,
        "hit_in_top_k": bool(hit),
        "target_score": float(score),
        "retrieval_margin": float(margin),
        "confusable": bool(margin <= margin_threshold),
        "k": int(k),
    }
    return ProbeResult("addressability", metrics, passed=bool(hit), notes="Tests whether target residue is selected by bounded retrieval.")


def fold_force_probe(records: Sequence[MemoryRecord], query: Array, target_id: str, n_actions: int, k: int = 1, target_action: Optional[int] = None, margin_threshold: float = 0.05) -> ProbeResult:
    retriever = TopKRetriever()
    hits = retriever.top_k(records, query, k)
    folded = FoldOperator().fold(hits)
    with_fold = toy_transition(folded, n_actions, target_action, "folded_trace_recorded")
    without_fold = toy_transition(None, n_actions, target_action, "fold_ablation_recorded")
    addr = addressability_probe(records, query, target_id, k, margin_threshold).metrics
    h_with = with_fold.h_cog()
    h_without = without_fold.h_cog()
    metrics = {
        **addr,
        "fold_ablation_l2_h_cog": l2(h_with, h_without),
        "belief_symmetric_kl": sym_kl(with_fold.belief, without_fold.belief),
        "with_fold_action": int(np.argmax(with_fold.belief)),
        "without_fold_action": int(np.argmax(without_fold.belief)),
        "with_fold_utility": float(with_fold.utility),
        "without_fold_utility": float(without_fold.utility),
        "bookkeeping_ignored": True,
    }
    return ProbeResult("fold_force", metrics, passed=metrics["fold_ablation_l2_h_cog"] > 1e-9, notes="Tests transition effect over h_cog, excluding bookkeeping logs.")


@dataclass(frozen=True)
class P7RegimeConfig:
    name: str
    class_strength: float
    unique_strength: float
    query_unique_strength: float
    noise_std: float
    dim_class: int = 8
    dim_unique: int = 24
    n_classes: int = 4


P7_REGIMES: Dict[str, P7RegimeConfig] = {
    "sparse_confusable": P7RegimeConfig("sparse_confusable", 1.0, 0.03, 0.0, 0.04),
    "compressed_preserving": P7RegimeConfig("compressed_preserving", 0.9, 0.45, 0.45, 0.04),
    "rich_distinctive": P7RegimeConfig("rich_distinctive", 0.8, 1.0, 1.0, 0.04),
    "aggressive_lossy": P7RegimeConfig("aggressive_lossy", 1.0, 0.0, 0.8, 0.04),
}


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def _make_p7_records(n: int, cfg: P7RegimeConfig, rng: np.random.Generator):
    class_proto = _unit_rows(rng.normal(size=(cfg.n_classes, cfg.dim_class)))
    labels = rng.integers(0, cfg.n_classes, size=n)
    unique_proto = _unit_rows(rng.normal(size=(n, cfg.dim_unique)))
    records: List[MemoryRecord] = []
    for i in range(n):
        v = np.concatenate([cfg.class_strength * class_proto[labels[i]], cfg.unique_strength * unique_proto[i]])
        v = v + rng.normal(0, cfg.noise_std, size=v.shape[0])
        records.append(MemoryRecord(f"op{i}", i, v, {"label": int(labels[i])}, "operation_memory", f"class={int(labels[i])}; unique_summary={i}"))
    return records, class_proto, unique_proto, labels


def _make_p7_query(target_i: int, cfg: P7RegimeConfig, class_proto: np.ndarray, unique_proto: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> Array:
    q = np.concatenate([cfg.class_strength * class_proto[labels[target_i]], cfg.query_unique_strength * unique_proto[target_i]])
    return q + rng.normal(0, cfg.noise_std, size=q.shape[0])


def p7_retrieval_regime_probe(regime: str, n_values: Sequence[int] = (8, 16, 32, 64, 128), k: int = 3, n_queries: int = 300, margin_threshold: float = 0.05, seed: int = 7) -> ProbeResult:
    """Probe revised P7 as conditional retrieval interference.

    Expected pattern:
    - sparse_confusable: hit rate tends to fall as N grows under bounded top-k.
    - rich_distinctive: hit rate stays high because target margin is large.
    - compressed_preserving: intermediate/good if enough unique content remains.
    - aggressive_lossy: target-specific content is absent, so continuity fails.
    """
    if regime not in P7_REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; choose one of {sorted(P7_REGIMES)}")
    cfg = P7_REGIMES[regime]
    rng = np.random.default_rng(seed)
    retriever = TopKRetriever()
    rows: List[Dict[str, Any]] = []
    for n in n_values:
        records, cp, up, labels = _make_p7_records(int(n), cfg, rng)
        hit_count = 0
        ranks: List[int] = []
        margins: List[float] = []
        conf_count = 0
        for _ in range(int(n_queries)):
            target_i = int(rng.integers(0, int(n)))
            q = _make_p7_query(target_i, cfg, cp, up, labels, rng)
            rank, _score, margin = retriever.target_rank_margin(records, q, f"op{target_i}")
            hit_count += int(rank is not None and rank <= k)
            if rank is not None:
                ranks.append(int(rank))
            margins.append(float(margin))
            conf_count += int(margin <= margin_threshold)
        rows.append({
            "N": int(n),
            "k": int(k),
            "hit_rate": hit_count / float(n_queries),
            "mean_target_rank": float(np.mean(ranks)) if ranks else float("nan"),
            "mean_margin": float(np.mean(margins)),
            "confusability": conf_count / float(n_queries),
        })
    diffs = [{"from_N": a["N"], "to_N": b["N"], "delta_hit_rate": float(b["hit_rate"] - a["hit_rate"])} for a, b in zip(rows[:-1], rows[1:])]
    overload_like = any(d["delta_hit_rate"] < -0.05 for d in diffs)
    metrics = {"regime": regime, "rows": rows, "finite_differences": diffs, "overload_like": bool(overload_like), "config": asdict(cfg)}
    return ProbeResult("p7_retrieval_regime", metrics, passed=True, notes="Reports whether bounded retrieval degrades as N grows in the selected operation-memory regime.")

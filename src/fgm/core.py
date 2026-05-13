"""Fold-Gated Memory core implementation.

Six layers of the trace/fold hierarchy as executable components:

    Layer 1 - Trace:          implicit (the LLM context window)
    Layer 2 - Storage:        MemoryStore
    Layer 3 - Addressability: MarginRetriever
    Layer 4 - Folding:        FoldGate
    Layer 5 - Operations:     OperationMemory
    Layer 6 - Compression:    Compressor

FGMAgent ties all layers together.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
EPS = 1e-12
TransitionFn = Callable[[Array, Array, Optional[Array]], Array]

SOURCE_EXTERNAL = "external"
SOURCE_TOOL_OUTPUT = "tool_output"
SOURCE_RETRIEVED_MEMORY = "retrieved_memory"
SOURCE_INFERENCE = "inference"
SOURCE_SIMULATION = "simulation"
SOURCE_FABRICATED = "fabricated_or_uncertain"
SOURCE_OPERATION_RECORD = "operation_record"

ROUTE_NULL = "null"
ROUTE_TRACE = "trace"
ROUTE_DURABLE_MEMORY = "durable_memory"
ROUTE_OPERATION_MEMORY = "operation_memory"
ROUTE_CORRECTION_CHAIN = "correction_chain"
ROUTE_QUARANTINE = "quarantine"

UNTRUSTED_SOURCE_LABELS = frozenset({SOURCE_SIMULATION, SOURCE_FABRICATED})

SOURCE_RERANK_DEFAULT_POOL = 8
SOURCE_RERANK_BONUS = 0.35
POLARITY_RERANK_BONUS = 0.12

# SSIR §6 Correction-chain: route to cc when |Δ_t| ≥ θ_Δ AND u_t identifiable.
EVENT_DELTA_THRESHOLD = 0.05
# SSIR §7 Prop. 2 echo guard: π^H(s=react) suppressed unless R_t ≥ θ_R or
# Δ_t externally corroborated. θ_R applies to caller-supplied reactivation
# reliability; corroboration is detected from external-source retrieval hits.
REACTIVATION_RELIABILITY_THRESHOLD = 0.7

_FABRICATED_QUERY_MARKERS = frozenset({
    "adversarial",
    "distractor",
    "fabricated",
    "false",
    "falsely",
    "hallucinated",
    "uncertain",
    "untrusted",
    "unverified",
})
_SIMULATION_QUERY_MARKERS = frozenset({
    "hypothetical",
    "simulation",
    "simulated",
})
_EXTERNAL_QUERY_MARKERS = frozenset({
    "approval",
    "approved",
    "authorizes",
    "evidence",
    "external",
    "observation",
    "observed",
})
_NEGATION_POLARITY_MARKERS = frozenset({
    "ban",
    "banned",
    "forbid",
    "forbids",
    "forbidden",
    "prohibit",
    "prohibited",
    "prohibition",
})
_APPROVAL_POLARITY_MARKERS = frozenset({
    "allow",
    "allowed",
    "approval",
    "approved",
    "authorize",
    "authorized",
    "authorizes",
})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def hash_embed(text: str, dim: int = 64) -> Array:
    """Deterministic hashed bag-of-words embedding using BLAKE2b."""
    tokens = text.lower().split()
    v = np.zeros(dim)
    for token in tokens:
        h = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        v[idx] += sign
    norm = np.linalg.norm(v)
    if norm > EPS:
        v /= norm
    return v


def cosine(a: Array, b: Array) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < EPS:
        return 0.0
    return float(np.dot(a, b) / den)


def l2(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - b))


def default_transition(state: Array, input_vec: Array, fold_vec: Optional[Array] = None) -> Array:
    """Toy transition function Phi. Matches simrec-reference."""
    if fold_vec is not None:
        return np.tanh(0.65 * state + 0.75 * input_vec + 0.85 * fold_vec)
    return np.tanh(0.65 * state + 0.75 * input_vec)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _infer_requested_source(tokens: set[str]) -> Optional[str]:
    if tokens & _FABRICATED_QUERY_MARKERS:
        return SOURCE_FABRICATED
    if tokens & _SIMULATION_QUERY_MARKERS:
        return SOURCE_SIMULATION
    if tokens & _EXTERNAL_QUERY_MARKERS:
        return SOURCE_EXTERNAL
    return None


def _infer_polarity(tokens: set[str]) -> Optional[str]:
    if tokens & _NEGATION_POLARITY_MARKERS:
        return "negation"
    if tokens & _APPROVAL_POLARITY_MARKERS:
        return "approval"
    return None


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    content: str
    vector: Array
    timestamp: float
    record_type: str              # "content" | "operation"
    operation_type: Optional[str] = None   # "observation" | "decision" | "correction" | ...
    decision_content: Optional[str] = None
    source_label: str = SOURCE_EXTERNAL
    source_confidence: float = 1.0
    provenance: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other):
        if not isinstance(other, MemoryRecord):
            return NotImplemented
        return self.record_id == other.record_id

    def __hash__(self):
        return hash(self.record_id)


@dataclass(frozen=True)
class RetrievalHit:
    record: MemoryRecord
    score: float
    rank: int
    margin: float
    active_source_label: str = SOURCE_RETRIEVED_MEMORY


@dataclass(frozen=True)
class RetrievalReport:
    hits: List[RetrievalHit]
    query_vector: Array
    mean_margin: float
    confusability: float
    n_records: int


@dataclass(frozen=True)
class FoldResult:
    query: str
    query_vector: Array
    retrieved: List[RetrievalHit]
    fold_vector: Optional[Array]
    output_with: Array
    output_without: Array
    fold_force: float
    full_divergence: float
    gated: bool
    source_labels: List[str] = field(default_factory=list)
    active_source_labels: List[str] = field(default_factory=list)
    source_confidence: List[float] = field(default_factory=list)
    route_scores: Dict[str, float] = field(default_factory=dict)
    selected_route: str = ROUTE_NULL
    operation_record_id: Optional[str] = None


@dataclass(frozen=True)
class OperationRecord:
    operation_id: str
    timestamp: float
    query: str
    query_vector: Array
    retrieved_ids: List[str]
    fold_force: float
    output_with: Array
    output_without: Array
    recursive_depth: int
    source_labels: List[str] = field(default_factory=list)
    active_source_labels: List[str] = field(default_factory=list)
    selected_route: str = ROUTE_OPERATION_MEMORY
    operation_record_id: Optional[str] = None


@dataclass(frozen=True)
class CompressionReport:
    records_before: int
    records_after: int
    removed_ids: List[str]
    merged_ids: List[Tuple[str, str]]
    margin_before: float
    margin_after: float
    confusability_before: float
    confusability_after: float
    method: str


# ---------------------------------------------------------------------------
# Layer 2: Storage
# ---------------------------------------------------------------------------

class MemoryStore:
    """Persistent vector store for memory records.

    Storage is cheap and unfiltered. The gate is at the fold layer,
    not here. This layer just holds records and tracks usage.
    """

    def __init__(self, dim: int = 64, embed_fn: Optional[Callable[[str], Array]] = None):
        self.dim = dim
        self.embed_fn = embed_fn or (lambda text: hash_embed(text, dim))
        self._records: Dict[str, MemoryRecord] = {}
        self._fold_forces: Dict[str, List[float]] = {}

    def add(
        self,
        content: str,
        *,
        record_id: Optional[str] = None,
        vector: Optional[Array] = None,
        record_type: str = "content",
        operation_type: Optional[str] = None,
        decision_content: Optional[str] = None,
        source_label: str = SOURCE_EXTERNAL,
        source_confidence: float = 1.0,
        provenance: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> MemoryRecord:
        rid = record_id or f"mem_{uuid.uuid4().hex[:12]}"
        vec = vector if vector is not None else self.embed_fn(content)
        rec = MemoryRecord(
            record_id=rid,
            content=content,
            vector=vec,
            timestamp=timestamp or time.time(),
            record_type=record_type,
            operation_type=operation_type,
            decision_content=decision_content,
            source_label=source_label,
            source_confidence=float(source_confidence),
            provenance=tuple(provenance or ()),
            metadata=dict(metadata or {}),
        )
        self._records[rid] = rec
        self._fold_forces[rid] = []
        return rec

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        return self._records.get(record_id)

    def remove(self, record_id: str) -> bool:
        if record_id in self._records:
            del self._records[record_id]
            self._fold_forces.pop(record_id, None)
            return True
        return False

    def all_records(self) -> List[MemoryRecord]:
        return list(self._records.values())

    def record_fold_force(self, record_id: str, force: float):
        if record_id in self._fold_forces:
            self._fold_forces[record_id].append(force)

    def total_fold_force(self, record_id: str) -> float:
        return sum(self._fold_forces.get(record_id, []))

    def use_count(self, record_id: str) -> int:
        return len(self._fold_forces.get(record_id, []))

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Layer 3: Addressability (margin-tracked retrieval)
# ---------------------------------------------------------------------------

class MarginRetriever:
    """Top-k retriever that tracks retrieval margin and confusability.

    Retrieval margin: M_N(q) = s(q, r*) - max_{r != r*} s(q, r)
    Confusability: fraction of queries where margin <= delta
    """

    def __init__(self, store: MemoryStore, margin_threshold: float = 0.05):
        self.store = store
        self.margin_threshold = margin_threshold

    def retrieve(self, query: str | Array, k: int = 3) -> RetrievalReport:
        if isinstance(query, str):
            q_vec = self.store.embed_fn(query)
        else:
            q_vec = query

        records = self.store.all_records()
        if not records:
            return RetrievalReport([], q_vec, 0.0, 0.0, 0)

        scored = [(r, cosine(q_vec, r.vector)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)

        hits: List[RetrievalHit] = []
        for rank_idx, (rec, score) in enumerate(scored[:k]):
            if rank_idx + 1 < len(scored):
                next_score = scored[rank_idx + 1][1] if rank_idx < k - 1 else scored[k][1] if k < len(scored) else -1.0
            else:
                next_score = -1.0
            best_distractor = max(
                (s for r, s in scored if r.record_id != rec.record_id),
                default=-float("inf"),
            )
            margin = score - best_distractor
            hits.append(RetrievalHit(rec, score, rank_idx + 1, margin))

        margins = [h.margin for h in hits]
        mean_margin = float(np.mean(margins)) if margins else 0.0
        confusable_count = sum(1 for m in margins if m <= self.margin_threshold)
        confusability = confusable_count / len(margins) if margins else 0.0

        return RetrievalReport(hits, q_vec, mean_margin, confusability, len(records))

    def target_rank_margin(self, query: str | Array, target_id: str) -> Tuple[Optional[int], float, float]:
        if isinstance(query, str):
            q_vec = self.store.embed_fn(query)
        else:
            q_vec = query

        records = self.store.all_records()
        scored = [(r, cosine(q_vec, r.vector)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)

        target_rank = None
        target_score = -float("inf")
        best_distractor = -float("inf")

        for rank_idx, (rec, score) in enumerate(scored):
            if rec.record_id == target_id:
                target_rank = rank_idx + 1
                target_score = score
            else:
                best_distractor = max(best_distractor, score)

        if target_rank is None:
            return None, -float("inf"), -float("inf")
        margin = target_score - best_distractor if np.isfinite(best_distractor) else float("inf")
        return target_rank, target_score, margin

    def estimate_confusability(self, n_queries: int = 50, rng: Optional[np.random.Generator] = None) -> float:
        """Sample random records as queries and estimate store-wide confusability."""
        rng = rng or np.random.default_rng()
        records = self.store.all_records()
        if len(records) < 2:
            return 0.0

        confusable = 0
        for _ in range(n_queries):
            target = records[rng.integers(len(records))]
            noisy_q = target.vector + rng.normal(0, 0.05, size=target.vector.shape)
            _, _, margin = self.target_rank_margin(noisy_q, target.record_id)
            if margin <= self.margin_threshold:
                confusable += 1
        return confusable / n_queries


# ---------------------------------------------------------------------------
# Layer 4: Folding (transition-gated memory)
# ---------------------------------------------------------------------------

class FoldGate:
    """Measures fold-force: whether retrieved memory changes the transition.

    fold_force = ||Phi(state, input, fold_vec) - Phi(state, input, None)||
    restricted to h_cog (non-bookkeeping) dimensions when cog_dims is set.

    Records with fold_force < threshold are gated out --- they were
    retrieved but didn't change the decision, so they aren't functioning
    as cognitive memory.
    """

    FoldForceFn = Callable[[Array, Array, List["RetrievalHit"]], float]

    def __init__(
        self,
        transition_fn: TransitionFn = default_transition,
        threshold: float = 0.01,
        temperature: float = 0.2,
        cog_dims: Optional[Sequence[int]] = None,
        fold_force_fn: Optional["FoldGate.FoldForceFn"] = None,
    ):
        self.transition_fn = transition_fn
        self.threshold = threshold
        self.temperature = temperature
        self.cog_dims = np.array(cog_dims, dtype=int) if cog_dims is not None else None
        self._fold_force_fn = fold_force_fn

    def compute_fold_vector(self, hits: Sequence[RetrievalHit]) -> Optional[Array]:
        if not hits:
            return None
        scores = np.array([h.score for h in hits], dtype=float)
        z = scores / max(self.temperature, EPS)
        z = z - np.max(z)
        weights = np.exp(z)
        weights /= np.sum(weights)
        vectors = np.stack([h.record.vector for h in hits])
        return np.sum(weights[:, None] * vectors, axis=0)

    def fold(
        self,
        query: str,
        query_vector: Array,
        hits: List[RetrievalHit],
        state: Array,
    ) -> FoldResult:
        fold_vec = self.compute_fold_vector(hits)
        input_vec = query_vector

        output_with = self.transition_fn(state, input_vec, fold_vec)
        output_without = self.transition_fn(state, input_vec, None)

        full_divergence = l2(output_with, output_without)
        if self._fold_force_fn is not None:
            fold_force = self._fold_force_fn(output_with, output_without, hits)
        elif self.cog_dims is not None:
            fold_force = l2(output_with[self.cog_dims], output_without[self.cog_dims])
        else:
            fold_force = full_divergence
        gated = fold_force >= self.threshold

        return FoldResult(
            query=query,
            query_vector=query_vector,
            retrieved=hits,
            fold_vector=fold_vec,
            output_with=output_with,
            output_without=output_without,
            fold_force=fold_force,
            full_divergence=full_divergence,
            gated=gated,
        )


# ---------------------------------------------------------------------------
# Layer 5: Operation-Memory
# ---------------------------------------------------------------------------

class OperationMemory:
    """Records fold operations as operation-memory.

    Each fold that passes the gate is stored as an operation record,
    enabling recursive self-correction: the agent can trace decisions
    back through the fold operations that produced them.
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        self._operations: Dict[str, OperationRecord] = {}

    def record_fold(self, fold_result: FoldResult, recursive_depth: int = 1) -> OperationRecord:
        op_id = f"op_{uuid.uuid4().hex[:12]}"
        op = OperationRecord(
            operation_id=op_id,
            timestamp=time.time(),
            query=fold_result.query,
            query_vector=fold_result.query_vector,
            retrieved_ids=[h.record.record_id for h in fold_result.retrieved],
            fold_force=fold_result.fold_force,
            output_with=fold_result.output_with,
            output_without=fold_result.output_without,
            recursive_depth=recursive_depth,
            source_labels=list(fold_result.source_labels),
            active_source_labels=list(fold_result.active_source_labels),
            selected_route=fold_result.selected_route,
            operation_record_id=op_id,
        )
        self._operations[op_id] = op

        op_content = (
            f"fold query='{fold_result.query}' "
            f"retrieved=[{','.join(op.retrieved_ids)}] "
            f"fold_force={fold_result.fold_force:.4f}"
        )
        decision = (
            f"fold_force={fold_result.fold_force:.4f} "
            f"gated={fold_result.gated} "
            f"retrieved_count={len(fold_result.retrieved)}"
        )
        self.store.add(
            op_content,
            record_id=op_id,
            record_type="operation",
            operation_type="fold",
            decision_content=decision,
            metadata={"recursive_depth": recursive_depth, "retrieved_ids": op.retrieved_ids},
            source_label=SOURCE_OPERATION_RECORD,
            source_confidence=1.0,
            provenance=op.retrieved_ids,
        )
        return op

    def trace_decision(self, query: str | Array, k: int = 3) -> List[OperationRecord]:
        """Retrieve prior fold operations relevant to a query."""
        retriever = MarginRetriever(self.store)
        report = retriever.retrieve(query, k=k)
        ops = []
        for hit in report.hits:
            if hit.record.record_type == "operation" and hit.record.record_id in self._operations:
                ops.append(self._operations[hit.record.record_id])
        return ops

    def all_operations(self) -> List[OperationRecord]:
        return list(self._operations.values())

    def __len__(self) -> int:
        return len(self._operations)


# ---------------------------------------------------------------------------
# Layer 6: Compression (margin-preserving)
# ---------------------------------------------------------------------------

class Compressor:
    """Margin-preserving compression of memory records.

    Three strategies:
    1. Fold-force pruning:   remove records that never changed a decision
    2. Duplicate merging:    merge near-duplicate records (cosine > threshold)
    3. Combined:             prune then merge

    Constraint: post-compression retrieval margins must not degrade
    beyond tolerance (the paper's Eq. 28).
    """

    def __init__(
        self,
        store: MemoryStore,
        retriever: MarginRetriever,
        duplicate_threshold: float = 0.92,
        margin_tolerance: float = 0.05,
    ):
        self.store = store
        self.retriever = retriever
        self.duplicate_threshold = duplicate_threshold
        self.margin_tolerance = margin_tolerance

    def should_compress(self, chi_threshold: float = 0.5, n_probes: int = 50) -> bool:
        if len(self.store) < 4:
            return False
        chi = self.retriever.estimate_confusability(n_probes)
        return chi > chi_threshold

    def prune_zero_fold(self) -> CompressionReport:
        """Remove records with zero accumulated fold-force."""
        records = self.store.all_records()
        margin_before = self._mean_margin()
        chi_before = self.retriever.estimate_confusability()
        n_before = len(records)

        removed = []
        for r in records:
            if r.record_type == "operation":
                continue
            if self.store.total_fold_force(r.record_id) < EPS and self.store.use_count(r.record_id) > 0:
                self.store.remove(r.record_id)
                removed.append(r.record_id)

        margin_after = self._mean_margin()
        chi_after = self.retriever.estimate_confusability()

        return CompressionReport(
            records_before=n_before,
            records_after=len(self.store),
            removed_ids=removed,
            merged_ids=[],
            margin_before=margin_before,
            margin_after=margin_after,
            confusability_before=chi_before,
            confusability_after=chi_after,
            method="prune_zero_fold",
        )

    def merge_duplicates(self) -> CompressionReport:
        """Merge near-duplicate records, keeping the one with higher fold-force."""
        records = self.store.all_records()
        content_records = [r for r in records if r.record_type == "content"]
        margin_before = self._mean_margin()
        chi_before = self.retriever.estimate_confusability()
        n_before = len(records)

        merged_pairs: List[Tuple[str, str]] = []
        removed: List[str] = []
        consumed: set = set()

        for i, a in enumerate(content_records):
            if a.record_id in consumed:
                continue
            for b in content_records[i + 1:]:
                if b.record_id in consumed:
                    continue
                sim = cosine(a.vector, b.vector)
                if sim >= self.duplicate_threshold:
                    force_a = self.store.total_fold_force(a.record_id)
                    force_b = self.store.total_fold_force(b.record_id)
                    loser = b if force_a >= force_b else a
                    winner = a if force_a >= force_b else b

                    merged_content = f"{winner.content} [merged: {loser.content}]"
                    merged_decision = " | ".join(
                        filter(None, [winner.decision_content, loser.decision_content])
                    ) or None
                    merged_vec = (winner.vector + loser.vector)
                    norm = np.linalg.norm(merged_vec)
                    if norm > EPS:
                        merged_vec /= norm

                    self.store.remove(winner.record_id)
                    self.store.remove(loser.record_id)
                    self.store.add(
                        merged_content,
                        record_id=winner.record_id,
                        vector=merged_vec,
                        record_type="content",
                        operation_type=winner.operation_type,
                        decision_content=merged_decision,
                        metadata={**winner.metadata, "merged_from": loser.record_id},
                        timestamp=max(winner.timestamp, loser.timestamp),
                    )

                    consumed.add(loser.record_id)
                    merged_pairs.append((winner.record_id, loser.record_id))
                    removed.append(loser.record_id)

        margin_after = self._mean_margin()
        chi_after = self.retriever.estimate_confusability()

        return CompressionReport(
            records_before=n_before,
            records_after=len(self.store),
            removed_ids=removed,
            merged_ids=merged_pairs,
            margin_before=margin_before,
            margin_after=margin_after,
            confusability_before=chi_before,
            confusability_after=chi_after,
            method="merge_duplicates",
        )

    def compress(self) -> CompressionReport:
        """Combined: prune zero-fold records, then merge duplicates."""
        prune = self.prune_zero_fold()
        merge = self.merge_duplicates()
        return CompressionReport(
            records_before=prune.records_before,
            records_after=merge.records_after,
            removed_ids=prune.removed_ids + merge.removed_ids,
            merged_ids=merge.merged_ids,
            margin_before=prune.margin_before,
            margin_after=merge.margin_after,
            confusability_before=prune.confusability_before,
            confusability_after=merge.confusability_after,
            method="combined",
        )

    def _mean_margin(self) -> float:
        records = self.store.all_records()
        if len(records) < 2:
            return 1.0
        rng = np.random.default_rng(0)
        margins = []
        n_probes = min(30, len(records))
        indices = rng.choice(len(records), size=n_probes, replace=False)
        for idx in indices:
            target = records[idx]
            q = target.vector + rng.normal(0, 0.02, size=target.vector.shape)
            _, _, margin = self.retriever.target_rank_margin(q, target.record_id)
            margins.append(margin)
        return float(np.mean(margins)) if margins else 0.0


# ---------------------------------------------------------------------------
# FGMAgent: ties all layers together
# ---------------------------------------------------------------------------

class FGMAgent:
    """Fold-Gated Memory agent.

    Implements the six-layer hierarchy:
        store -> retrieve (with margins) -> fold (with force measurement)
        -> record operation -> compress (when confusability rises)

    Usage:
        agent = FGMAgent()
        agent.store("The deploy failed due to a migration timeout",
                     operation_type="observation")
        result = agent.query("What caused the deploy failure?")
        print(result.fold_force, result.gated)
    """

    def __init__(
        self,
        dim: int = 64,
        transition_fn: TransitionFn = default_transition,
        embed_fn: Optional[Callable[[str], Array]] = None,
        fold_threshold: float = 0.01,
        retrieval_k: int = 3,
        auto_compress: bool = True,
        compress_chi_threshold: float = 0.6,
        cog_dims: Optional[Sequence[int]] = None,
        fold_force_fn: Optional[FoldGate.FoldForceFn] = None,
        source_aware_rerank: bool = True,
        source_rerank_k: int = SOURCE_RERANK_DEFAULT_POOL,
    ):
        self.dim = dim
        self._embed_fn = embed_fn or (lambda text: hash_embed(text, dim))
        self.store = MemoryStore(dim=dim, embed_fn=self._embed_fn)
        self.retriever = MarginRetriever(self.store)
        self.fold_gate = FoldGate(
            transition_fn, threshold=fold_threshold, cog_dims=cog_dims,
            fold_force_fn=fold_force_fn,
        )
        self.operations = OperationMemory(self.store)
        self.compressor = Compressor(self.store, self.retriever)
        self.retrieval_k = retrieval_k
        self.source_aware_rerank = source_aware_rerank
        self.source_rerank_k = source_rerank_k
        self.auto_compress = auto_compress
        self.compress_chi_threshold = compress_chi_threshold

        self._state = np.zeros(dim)
        self._step = 0
        self._fold_history: List[FoldResult] = []

    def add(
        self,
        content: str,
        *,
        record_id: Optional[str] = None,
        vector: Optional[Array] = None,
        operation_type: Optional[str] = None,
        decision_content: Optional[str] = None,
        source_label: str = SOURCE_EXTERNAL,
        source_confidence: float = 1.0,
        provenance: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Add a record to storage (Layer 2)."""
        return self.store.add(
            content,
            record_id=record_id,
            vector=vector,
            operation_type=operation_type,
            decision_content=decision_content,
            source_label=source_label,
            source_confidence=source_confidence,
            provenance=provenance,
            metadata=metadata,
        )

    def query(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        event_delta: Optional[float] = None,
        update_operation: Optional[str] = None,
        external_corroboration: bool = False,
        reactivation_reliability: Optional[float] = None,
    ) -> FoldResult:
        """Full query cycle: retrieve -> fold -> measure -> gate -> record.

        Returns FoldResult with fold_force and gated flag.
        If the transition function has a set_context method (e.g. LLMTransition),
        it is called with query text and retrieved record texts before folding.

        Optional source-sensitive routing inputs (Source-Sensitive Inscription
        Routing manuscript §5-§7):

        ``event_delta`` and ``update_operation``
            Together signal that the current event is a belief revision. When
            ``abs(event_delta) >= EVENT_DELTA_THRESHOLD`` and an ``update_operation``
            string is supplied, the router promotes the correction-chain route
            on this event (SSIR §6 Correction-chain). Without these signals,
            retrieving a prior correction node is treated as contextual evidence
            only — it boosts the cc score but does not dominate operation_memory.

        ``external_corroboration`` and ``reactivation_reliability``
            Echo-amplification guard (SSIR §7 Prop. 2). The durable_memory
            route is suppressed unless one of: at least one retrieved record
            has ``source_label == external`` (corroboration in the retrieval),
            ``external_corroboration=True`` is explicitly signaled, or
            ``reactivation_reliability >= REACTIVATION_RELIABILITY_THRESHOLD``.
        """
        k = k or self.retrieval_k
        self._step += 1

        candidate_k = k
        if self.source_aware_rerank:
            candidate_k = max(k, min(max(self.source_rerank_k, k), len(self.store)))
        report = self.retriever.retrieve(query, k=candidate_k)
        q_vec = report.query_vector
        hits = self._select_retrieval_hits(query, report.hits, k)

        transition = self.fold_gate.transition_fn
        if hasattr(transition, "set_context"):
            transition.set_context(
                query_text=query,
                memory_texts=[h.record.content for h in hits],
                memory_scores=[h.score for h in hits],
            )

        result = self.fold_gate.fold(query, q_vec, hits, self._state)
        result = self._attach_source_and_route(
            result,
            event_delta=event_delta,
            update_operation=update_operation,
            external_corroboration=external_corroboration,
            reactivation_reliability=reactivation_reliability,
        )

        if self.fold_gate._fold_force_fn is not None:
            for hit in report.hits:
                per_record = self.fold_gate._fold_force_fn(
                    result.output_with, result.output_without, [hit],
                )
                self.store.record_fold_force(hit.record.record_id, per_record)
        else:
            for hit in report.hits:
                self.store.record_fold_force(hit.record.record_id, result.fold_force)

        if result.gated:
            self._state = result.output_with
            depth = 1
            for hit in report.hits:
                if hit.record.record_type == "operation":
                    depth = max(depth, hit.record.metadata.get("recursive_depth", 0) + 1)
            op = self.operations.record_fold(result, recursive_depth=depth)
            result = replace(result, operation_record_id=op.operation_id)

        self._fold_history.append(result)

        if self.auto_compress and self._step % 20 == 0:
            if self.compressor.should_compress(self.compress_chi_threshold):
                self.compressor.compress()

        return result

    def _attach_source_and_route(
        self,
        result: FoldResult,
        *,
        event_delta: Optional[float] = None,
        update_operation: Optional[str] = None,
        external_corroboration: bool = False,
        reactivation_reliability: Optional[float] = None,
    ) -> FoldResult:
        source_labels = [h.record.source_label for h in result.retrieved]
        active_source_labels = [h.active_source_label for h in result.retrieved]
        source_confidence = [h.record.source_confidence for h in result.retrieved]
        route_scores = self._score_routes(
            result,
            source_labels,
            source_confidence,
            event_delta=event_delta,
            update_operation=update_operation,
            external_corroboration=external_corroboration,
            reactivation_reliability=reactivation_reliability,
        )
        selected_route = max(route_scores, key=route_scores.get) if route_scores else ROUTE_NULL
        return replace(
            result,
            source_labels=source_labels,
            active_source_labels=active_source_labels,
            source_confidence=source_confidence,
            route_scores=route_scores,
            selected_route=selected_route,
        )

    def _select_retrieval_hits(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        k: int,
    ) -> List[RetrievalHit]:
        if not self.source_aware_rerank or len(hits) <= k:
            return list(hits[:k])

        tokens = _tokenize(query)
        intended_source = _infer_requested_source(tokens)
        query_polarity = _infer_polarity(tokens)
        reranked: List[RetrievalHit] = []
        for hit in hits:
            bonus = 0.0
            if intended_source is not None and hit.record.source_label == intended_source:
                bonus += SOURCE_RERANK_BONUS
            if query_polarity is not None and _infer_polarity(_tokenize(hit.record.content)) == query_polarity:
                bonus += POLARITY_RERANK_BONUS
            reranked.append(replace(hit, score=float(hit.score + bonus)))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return [
            replace(hit, rank=rank)
            for rank, hit in enumerate(reranked[:k], start=1)
        ]

    def _score_routes(
        self,
        result: FoldResult,
        source_labels: Sequence[str],
        source_confidence: Sequence[float],
        *,
        event_delta: Optional[float] = None,
        update_operation: Optional[str] = None,
        external_corroboration: bool = False,
        reactivation_reliability: Optional[float] = None,
    ) -> Dict[str, float]:
        """Small deterministic routing policy for primitive validation.

        This is intentionally transparent, not a learned controller. It gives
        the validation harness observable route scores before we add richer
        source-sensitive inscription models.

        Source-Sensitive Inscription Routing alignment:

        * Correction-chain (SSIR §6) routes only on a real revision event:
          ``abs(event_delta) >= EVENT_DELTA_THRESHOLD`` AND ``update_operation``
          is supplied. Retrieving a prior correction node without a current
          revision contributes a small boost only (cc < op_memory).
        * Durable-memory (SSIR §7 Prop. 2 echo guard) is suppressed when the
          attended state is pure reactivation. Promotion is allowed only when
          at least one retrieved record is externally sourced, the caller
          signals ``external_corroboration=True``, or
          ``reactivation_reliability >= REACTIVATION_RELIABILITY_THRESHOLD``.
        """
        scores = {
            ROUTE_NULL: 0.2,
            ROUTE_TRACE: 0.1,
            ROUTE_DURABLE_MEMORY: 0.0,
            ROUTE_OPERATION_MEMORY: 0.0,
            ROUTE_CORRECTION_CHAIN: 0.0,
            ROUTE_QUARANTINE: 0.0,
        }
        if not result.retrieved:
            scores[ROUTE_NULL] = 1.0
            return scores

        has_untrusted = any(label in UNTRUSTED_SOURCE_LABELS for label in source_labels)
        has_correction_in_retrieval = any(
            h.record.operation_type == "correction" or h.record.record_type == "correction"
            for h in result.retrieved
        )
        min_conf = min(source_confidence) if source_confidence else 1.0

        if has_untrusted or min_conf < 0.5:
            scores[ROUTE_QUARANTINE] = 0.95
            scores[ROUTE_NULL] = 0.35
            return scores

        # SSIR §6: a correction-chain write requires a current-event delta
        # and an identifiable update operation, not merely retrieval of a
        # prior correction record.
        is_revision_event = (
            event_delta is not None
            and abs(event_delta) >= EVENT_DELTA_THRESHOLD
            and update_operation is not None
        )

        # SSIR §7 Prop. 2: durable promotion of reactivated content requires
        # external corroboration or sufficient reactivation reliability.
        retrieval_includes_external = any(
            label == SOURCE_EXTERNAL for label in source_labels
        )
        durable_allowed = (
            external_corroboration
            or retrieval_includes_external
            or (
                reactivation_reliability is not None
                and reactivation_reliability >= REACTIVATION_RELIABILITY_THRESHOLD
            )
        )

        if result.gated:
            scores[ROUTE_OPERATION_MEMORY] = 0.9
            scores[ROUTE_TRACE] = 0.65
            scores[ROUTE_DURABLE_MEMORY] = 0.55 if durable_allowed else 0.0
            scores[ROUTE_NULL] = 0.05
            if is_revision_event:
                # Real revision wins over operation_memory.
                scores[ROUTE_CORRECTION_CHAIN] = 0.92
            elif has_correction_in_retrieval:
                # Contextual signal only: retrieval surfaced a prior
                # correction, but the current event is not itself a
                # revision. Boost cc but keep operation_memory dominant.
                scores[ROUTE_CORRECTION_CHAIN] = 0.7
        else:
            scores[ROUTE_TRACE] = 0.35
            scores[ROUTE_NULL] = 0.75
            if is_revision_event:
                # Even on an ungated event, an explicit revision should
                # still record the correction node.
                scores[ROUTE_CORRECTION_CHAIN] = 0.85
        return scores

    def trace_decision(self, query: str, k: int = 3) -> List[OperationRecord]:
        """Retrieve prior fold operations relevant to a query (Layer 5)."""
        return self.operations.trace_decision(query, k=k)

    def compress(self) -> CompressionReport:
        """Manually trigger compression (Layer 6)."""
        return self.compressor.compress()

    def metrics(self) -> Dict[str, Any]:
        """Current system metrics."""
        records = self.store.all_records()
        content_records = [r for r in records if r.record_type == "content"]
        op_records = [r for r in records if r.record_type == "operation"]

        total_fold_force = sum(self.store.total_fold_force(r.record_id) for r in content_records)
        n_used = sum(1 for r in content_records if self.store.use_count(r.record_id) > 0)
        n_gated = sum(1 for r in self._fold_history if r.gated)

        return {
            "content_records": len(content_records),
            "operation_records": len(op_records),
            "total_records": len(records),
            "total_fold_force": total_fold_force,
            "records_ever_used": n_used,
            "queries_processed": self._step,
            "folds_gated": n_gated,
            "fold_gate_rate": n_gated / max(self._step, 1),
            "mean_fold_force": float(np.mean([f.fold_force for f in self._fold_history])) if self._fold_history else 0.0,
            "confusability": self.retriever.estimate_confusability() if len(records) >= 2 else 0.0,
        }

"""MemoryAgent: developer-facing facade over fgm.FGMAgent.

Exposes only the validated primitives from VALIDATED_PRIMITIVES_LEDGER
section 1 in ``docs/architecture/VALIDATED_PRIMITIVES_LEDGER.md``, with
API-level closure of the naive-inscription attack surface: ``add(...)``
requires an explicit ``source=`` argument, and ``add_derived(...)`` computes
the derived label from contributing inputs rather than accepting one.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Sequence, Set, Union

import numpy as np

from fgm.core import (
    Array,
    FGMAgent,
    MemoryRecord,
    SOURCE_OPERATION_RECORD,
)
from fgm.core import FoldResult as _FoldResult
from fgm.laundering import min_trust_source

from .audit import LaunderingAudit, build_audit
from .errors import DerivedInscriptionError, MissingSourceError
from .ingest import (
    DerivationRequest,
    IngestRequest,
    InferredSourceRequest,
    ObservationRequest,
    RevisionRequest,
    StructuredEnvelope,
)
from .inscription import UtilityWritePolicy, _Candidate, select_top_k_by_utility
from .routes import Route
from .self_index import SelfIndex, _SELF_INDEX_METADATA_KEY, record_matches_index
from .source_inference import infer_source, warn_natural_prose
from .sources import SourceLabel
from .storage import InMemoryStorage, Storage
from .types import CorrectionNode

_DERIVED_FLAG_TRUE = True
_CORRECTION_FLAG_TRUE = True

InputRef = Union[MemoryRecord, str]


_CORRECTION_RECORD_TYPE = "correction"
_DERIVED_METADATA_KEY = "_trace_memory_derived"
_CORRECTION_METADATA_KEY = "_trace_memory_correction"


class MemoryAgent:
    """A memory layer with operationally validated trust composition.

    Parameters
    ----------
    embed_fn :
        Function that maps a string to an embedding vector. Defaults to
        the deterministic hash-bag-of-words embedding used by the
        validation harness.
    transition_fn :
        Transition function ``Φ(state, input, fold_vec)``. Defaults to
        ``fgm.default_transition``.
    dim :
        Embedding dimensionality. Default ``64``.
    retrieval_k :
        Default top-k for ``query(...)``. Default ``3``.
    fold_threshold :
        Minimum fold-force below which a fold is gated out. Default
        ``0.01``.

    Notes
    -----
    Per the PRD, this agent enforces:

    - ``add(...)`` requires an explicit source label (no implicit
      ``external`` default).
    - ``add_derived(...)`` computes the source label from contributing
      inputs via the source-downgrading rule and propagates provenance
      transitively.
    - ``audit_laundering(...)`` returns paired self-vs-truth metrics
      with a cascade-invisibility warning when only the self metric is
      computable.

    See VALIDATED_PRIMITIVES_LEDGER for the validation evidence behind
    each behaviour.
    """

    def __init__(
        self,
        *,
        embed_fn: Optional[Callable[[str], Array]] = None,
        transition_fn: Optional[Callable] = None,
        dim: int = 64,
        retrieval_k: int = 3,
        fold_threshold: float = 0.01,
        self_index: Optional[SelfIndex] = None,
        inscription_policy: Optional[UtilityWritePolicy] = None,
        storage: Optional[Storage] = None,
    ) -> None:
        agent_kwargs: Dict[str, object] = {
            "dim": dim,
            "retrieval_k": retrieval_k,
            "fold_threshold": fold_threshold,
            "auto_compress": False,
            "embed_fn": embed_fn,
        }
        if transition_fn is not None:
            agent_kwargs["transition_fn"] = transition_fn
        self._agent = FGMAgent(**agent_kwargs)
        self._derived_record_ids: Set[str] = set()
        self._correction_node_index: Dict[str, CorrectionNode] = {}
        self._self_index: Optional[SelfIndex] = self_index
        self._inscription_policy: Optional[UtilityWritePolicy] = inscription_policy
        self._inscription_queue: List[_Candidate] = []
        self._storage: Storage = storage if storage is not None else InMemoryStorage()
        # Mutating ops are serialized via this lock so async callers
        # offloading to a threadpool do not race on the underlying store
        # or SQLite connection. Reads are not lock-protected: list/dict
        # iteration in CPython is safe under the GIL for the duration of
        # a single retrieval.
        self._lock = threading.RLock()
        self._hydrate_from_storage()

    def _hydrate_from_storage(self) -> None:
        """Load persisted records into the internal store on construction.

        Records are injected directly into the underlying ``MemoryStore``
        to preserve their original ``record_id``, ``timestamp``, vector,
        and metadata (going through the public ``add`` path would
        re-stamp timestamps and re-embed vectors).
        """
        for record in self._storage.load_all():
            self._agent.store._records[record.record_id] = record
            self._agent.store._fold_forces.setdefault(record.record_id, [])
            if record.metadata.get(_DERIVED_METADATA_KEY) is _DERIVED_FLAG_TRUE:
                self._derived_record_ids.add(record.record_id)
            if record.metadata.get(_CORRECTION_METADATA_KEY) is _CORRECTION_FLAG_TRUE:
                self._correction_node_index[record.record_id] = CorrectionNode(
                    node_id=record.record_id,
                    timestamp=record.timestamp,
                    prior_belief=record.metadata.get("prior_belief", ""),
                    evidence=record.metadata.get("evidence", ""),
                    update_operation=record.metadata.get("update_operation", ""),
                    revised_belief=record.metadata.get("revised_belief", ""),
                    delta=record.metadata.get("delta", ""),
                    provenance=tuple(record.provenance),
                    confidence=record.source_confidence,
                    record_id=record.record_id,
                )

    # ------------------------------------------------------------------
    # Public ingestion API
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        source: Optional[SourceLabel] = None,
        provenance: Sequence[str] = (),
        source_confidence: float = 1.0,
        record_id: Optional[str] = None,
        self_index: Optional[SelfIndex] = None,
    ) -> MemoryRecord:
        """Ingest a record with an explicit source label.

        Parameters
        ----------
        self_index :
            Optional explicit self-index for this record. Defaults to
            the agent's active self-index (set at construction time).
            Pass ``SelfIndex()`` explicitly to write a globally-scoped
            record from a scoped agent.

        Raises
        ------
        MissingSourceError
            If ``source`` is ``None``. This closes the naive-inscription
            attack surface at the API level (FR-1, FR-4 negative
            requirement).
        """
        if source is None:
            raise MissingSourceError(
                "add(...) requires an explicit source label. Supply "
                "source=SourceLabel.EXTERNAL (or another value) for "
                "directly observed content, or use add_derived(...) "
                "for content derived from existing records."
            )
        with self._lock:
            record = self._agent.add(
                content,
                record_id=record_id,
                source_label=str(source),
                source_confidence=float(source_confidence),
                provenance=tuple(provenance),
                metadata=self._build_metadata(self_index),
            )
            self._storage.save(record)
        return record

    def add_with_inferred_source(
        self,
        content: str,
        *,
        provenance: Sequence[str] = (),
        record_id: Optional[str] = None,
        self_index: Optional[SelfIndex] = None,
        query_context: str = "",
        retrieval_margin: float = 0.0,
        recency_rank: int = 0,
        policy: str = "combined",
    ) -> MemoryRecord:
        """Ingest a record with the source inferred from content/features.

        Emits a runtime warning that natural-prose performance is not
        characterised (ledger section 1.6 limits). For production
        pipelines on natural prose, validate against your own labelled
        data first.

        See :func:`trace_memory.infer_source` for the parameters.
        """
        warn_natural_prose()
        inferred = infer_source(
            content,
            query_context=query_context,
            retrieval_margin=retrieval_margin,
            recency_rank=recency_rank,
            policy=policy,
        )
        return self.add(
            content,
            source=inferred,
            provenance=provenance,
            record_id=record_id,
            self_index=self_index,
        )

    def add_candidate(
        self,
        content: str,
        *,
        source: SourceLabel,
        predicted_utility: float,
        provenance: Sequence[str] = (),
        source_confidence: float = 1.0,
        record_id: Optional[str] = None,
    ) -> None:
        """Queue a record for utility-budgeted inscription.

        Requires the agent to be constructed with an
        ``inscription_policy=UtilityWritePolicy(...)``. The candidate is
        not written to the store until :meth:`flush_inscriptions` is
        called, at which point only the top-``budget`` candidates by
        ``predicted_utility`` are committed and the rest are dropped.
        """
        if self._inscription_policy is None:
            raise DerivedInscriptionError(
                "add_candidate(...) requires the agent to be constructed "
                "with inscription_policy=UtilityWritePolicy(budget=N). "
                "For immediate-commit writes, use add(...) instead."
            )
        if source is None:
            raise MissingSourceError(
                "add_candidate(...) requires an explicit source label."
            )
        self._inscription_queue.append(
            _Candidate(
                content=content,
                source=source,
                predicted_utility=float(predicted_utility),
                provenance=tuple(provenance),
                source_confidence=float(source_confidence),
                record_id=record_id,
            )
        )

    def flush_inscriptions(self) -> List[MemoryRecord]:
        """Commit the queued candidates: top-budget by utility, drop the rest.

        Returns the committed records. The queue is cleared regardless
        of how many records were committed.
        """
        with self._lock:
            if self._inscription_policy is None:
                self._inscription_queue.clear()
                return []
            selected, _dropped = select_top_k_by_utility(
                self._inscription_queue, self._inscription_policy.budget
            )
            committed: List[MemoryRecord] = []
            for cand in selected:
                record = self.add(
                    cand.content,
                    source=cand.source,
                    provenance=cand.provenance,
                    source_confidence=cand.source_confidence,
                    record_id=cand.record_id,
                )
                committed.append(record)
            self._inscription_queue.clear()
        return committed

    def add_derived(
        self,
        content: str,
        *,
        inputs: Sequence[InputRef],
        record_id: Optional[str] = None,
        source_confidence: Optional[float] = None,
        self_index: Optional[SelfIndex] = None,
    ) -> MemoryRecord:
        """Ingest a derived record with source computed from contributing inputs.

        Implements the source-downgrading rule (ledger section 1.4):
        ``Source(r) = Trust_ceil({Source(c_i)})`` and ``Prov(r)`` is the
        transitive closure of contributing provenance plus contributing
        record ids.

        Parameters
        ----------
        content :
            The derived content.
        inputs :
            Sequence of contributing records, given as ``MemoryRecord``
            objects or record-id strings. Must be non-empty; every id
            must resolve in the store.
        record_id :
            Optional explicit id for the derived record.
        source_confidence :
            Optional explicit confidence. Defaults to the minimum
            confidence among contributing inputs.

        Raises
        ------
        DerivedInscriptionError
            If ``inputs`` is empty or any record-id reference does not
            resolve in the store.
        """
        if not inputs:
            raise DerivedInscriptionError(
                "add_derived(...) requires at least one contributing input."
            )

        with self._lock:
            return self._add_derived_locked(
                content,
                inputs=inputs,
                record_id=record_id,
                source_confidence=source_confidence,
                self_index=self_index,
            )

    def _add_derived_locked(
        self,
        content: str,
        *,
        inputs: Sequence[InputRef],
        record_id: Optional[str],
        source_confidence: Optional[float],
        self_index: Optional[SelfIndex],
    ) -> MemoryRecord:
        resolved: List[MemoryRecord] = []
        for ref in inputs:
            if isinstance(ref, MemoryRecord):
                resolved.append(ref)
            elif isinstance(ref, str):
                record = self._agent.store.get(ref)
                if record is None:
                    raise DerivedInscriptionError(
                        f"add_derived(...) input record id {ref!r} did not "
                        f"resolve in the store."
                    )
                resolved.append(record)
            else:  # pragma: no cover - defensive
                raise DerivedInscriptionError(
                    f"add_derived(...) inputs must be MemoryRecord or str, "
                    f"got {type(ref).__name__}."
                )

        labels = [rec.source_label for rec in resolved]
        capped = min_trust_source(labels)

        provenance_chain: List[str] = []
        for rec in resolved:
            provenance_chain.extend(rec.provenance)
            provenance_chain.append(rec.record_id)

        if source_confidence is None:
            source_confidence = min(
                (rec.source_confidence for rec in resolved),
                default=0.5,
            )

        derived_metadata = self._build_metadata(self_index)
        derived_metadata[_DERIVED_METADATA_KEY] = True
        record = self._agent.add(
            content,
            record_id=record_id,
            source_label=capped,
            source_confidence=float(source_confidence),
            provenance=tuple(provenance_chain),
            metadata=derived_metadata,
        )
        self._derived_record_ids.add(record.record_id)
        self._storage.save(record)
        return record

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        event_delta: Optional[float] = None,
        update_operation: Optional[str] = None,
        external_corroboration: bool = False,
        reactivation_reliability: Optional[float] = None,
    ) -> _FoldResult:
        """Retrieve records by relevance, fold them, and return a FoldResult.

        ``FoldResult.retrieved`` is ranked using the underlying retriever;
        ``FoldResult.fold_force`` is the paired-ablation magnitude on
        non-bookkeeping transition dimensions; ``FoldResult.selected_route``
        is chosen by the validated source-sensitive routing policy.

        Self-index filtering: records whose self-index does not match
        the agent's active self-index are excluded from the retrieved
        set before folding. Records without a self-index remain
        globally visible.

        Source-Sensitive Inscription Routing inputs (manuscript §6, §7):

        ``event_delta``, ``update_operation``
            Signal that the current event is a belief revision. When the
            absolute delta meets the routing threshold and an update
            operation string is supplied, the correction-chain route is
            selected (SSIR §6 Correction-chain). Without these signals,
            retrieving a prior correction is contextual evidence only.
        ``external_corroboration``, ``reactivation_reliability``
            Echo-amplification guard (SSIR §7 Prop. 2). The durable_memory
            route is suppressed for purely reactivated retrieval unless
            corroboration or reliability is signaled.
        """
        result = self._agent.query(
            query,
            k=k,
            event_delta=event_delta,
            update_operation=update_operation,
            external_corroboration=external_corroboration,
            reactivation_reliability=reactivation_reliability,
        )
        # Persist any operation-memory record FGMAgent created on this
        # query. Operation records are written via store.add() inside
        # OperationMemory.record_fold(), which bypasses trace-memory's
        # public add() path -- so we save them here.
        self._persist_operation_record(result.operation_record_id)
        if self._self_index is None and not self._any_record_is_scoped():
            return result
        filtered_hits = [
            hit
            for hit in result.retrieved
            if record_matches_index(hit.record, self._self_index)
        ]
        if len(filtered_hits) == len(result.retrieved):
            return result
        # Some records were filtered out -- refold with the surviving
        # hits so fold-force and routing reflect what the active scope
        # actually sees. Preserve caller-supplied routing inputs so
        # SSIR-aligned signals survive the self-index refilter.
        return self._refold_with(
            result,
            filtered_hits,
            event_delta=event_delta,
            update_operation=update_operation,
            external_corroboration=external_corroboration,
            reactivation_reliability=reactivation_reliability,
        )

    def _persist_operation_record(self, op_record_id: Optional[str]) -> None:
        """If the underlying agent just created an operation-memory record, save it."""
        if op_record_id is None:
            return
        op_record = self._agent.store.get(op_record_id)
        if op_record is not None:
            self._storage.save(op_record)

    def _refold_with(
        self,
        original: _FoldResult,
        hits,
        *,
        event_delta: Optional[float] = None,
        update_operation: Optional[str] = None,
        external_corroboration: bool = False,
        reactivation_reliability: Optional[float] = None,
    ) -> _FoldResult:
        """Re-run the fold + routing pipeline on a filtered hit set."""
        from dataclasses import replace

        if not hits:
            zero = type(original.output_with)(original.output_with.shape)
            zero[:] = 0.0
            return replace(
                original,
                retrieved=[],
                fold_vector=None,
                output_with=original.output_without,
                fold_force=0.0,
                full_divergence=0.0,
                gated=False,
                source_labels=[],
                active_source_labels=[],
                source_confidence=[],
                route_scores={Route.NULL.value: 1.0},
                selected_route=Route.NULL.value,
            )
        # Re-run the fold gate against the filtered hits.
        new_result = self._agent.fold_gate.fold(
            original.query,
            original.query_vector,
            hits,
            self._agent._state,
        )
        new_result = self._agent._attach_source_and_route(
            new_result,
            event_delta=event_delta,
            update_operation=update_operation,
            external_corroboration=external_corroboration,
            reactivation_reliability=reactivation_reliability,
        )
        return new_result

    # ------------------------------------------------------------------
    # Belief revision (correction chains)
    # ------------------------------------------------------------------

    def revise_belief(
        self,
        *,
        prior_belief: str,
        evidence: str,
        update_operation: str,
        revised_belief: str,
        delta: str,
        provenance: Sequence[str] = (),
        confidence: float = 1.0,
        node_id: Optional[str] = None,
        self_index: Optional[SelfIndex] = None,
    ) -> CorrectionNode:
        """Record a belief revision as a correction-chain node.

        The full lineage (prior belief, evidence, update operation,
        revised belief, delta, provenance, confidence) is preserved.
        The node is also persisted in the underlying store as a record
        with ``record_type='correction'`` so retrieval and audit can
        surface it alongside ordinary records.
        """
        with self._lock:
            return self._revise_belief_locked(
                prior_belief=prior_belief,
                evidence=evidence,
                update_operation=update_operation,
                revised_belief=revised_belief,
                delta=delta,
                provenance=provenance,
                confidence=confidence,
                node_id=node_id,
                self_index=self_index,
            )

    def _revise_belief_locked(
        self,
        *,
        prior_belief: str,
        evidence: str,
        update_operation: str,
        revised_belief: str,
        delta: str,
        provenance: Sequence[str],
        confidence: float,
        node_id: Optional[str],
        self_index: Optional[SelfIndex],
    ) -> CorrectionNode:
        node_id = node_id or f"corr_{uuid.uuid4().hex[:12]}"
        node = CorrectionNode(
            node_id=node_id,
            timestamp=time.time(),
            prior_belief=prior_belief,
            evidence=evidence,
            update_operation=update_operation,
            revised_belief=revised_belief,
            delta=delta,
            provenance=tuple(provenance),
            confidence=float(confidence),
        )

        content = (
            f"correction: prior={prior_belief!r}; evidence={evidence!r}; "
            f"update={update_operation!r}; revised={revised_belief!r}; "
            f"delta={delta!r}"
        )
        correction_metadata = self._build_metadata(self_index)
        correction_metadata.update({
            _CORRECTION_METADATA_KEY: True,
            "prior_belief": prior_belief,
            "evidence": evidence,
            "update_operation": update_operation,
            "revised_belief": revised_belief,
            "delta": delta,
        })
        record = self._agent.add(
            content,
            record_id=node_id,
            source_label=SOURCE_OPERATION_RECORD,
            source_confidence=float(confidence),
            provenance=tuple(provenance),
            metadata=correction_metadata,
        )
        self._storage.save(record)
        # Re-create the node with the persisted record_id field set.
        persisted_node = CorrectionNode(
            node_id=node.node_id,
            timestamp=node.timestamp,
            prior_belief=node.prior_belief,
            evidence=node.evidence,
            update_operation=node.update_operation,
            revised_belief=node.revised_belief,
            delta=node.delta,
            provenance=node.provenance,
            confidence=node.confidence,
            record_id=record.record_id,
        )
        self._correction_node_index[node_id] = persisted_node
        return persisted_node

    def correction_nodes(self) -> List[CorrectionNode]:
        """Return all correction nodes recorded so far, in insertion order."""
        return list(self._correction_node_index.values())

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit_laundering(
        self,
        *,
        truth_ceilings: Optional[Dict[str, SourceLabel]] = None,
    ) -> LaunderingAudit:
        """Return a paired laundering audit.

        When ``truth_ceilings`` is omitted, only the self-referential
        rate can be computed and the returned audit carries a
        cascade-invisibility warning.

        ``truth_ceilings`` maps a derived record's id to its
        truth-supplied maximum-trust source. A derived record with
        higher stored trust than its ceiling counts as a violation.
        """
        records = [
            record
            for record_id, record in self._all_records_by_id().items()
            if record_id in self._derived_record_ids
        ]
        record_lookup = self._all_records_by_id()
        ceilings_str: Optional[Dict[str, str]] = None
        if truth_ceilings is not None:
            ceilings_str = {rid: str(label) for rid, label in truth_ceilings.items()}
        return build_audit(
            records,
            record_lookup=record_lookup,
            truth_ceilings=ceilings_str,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Bulk ingestion (v0.5)
    # ------------------------------------------------------------------

    def ingest_batch(
        self,
        requests: Sequence[IngestRequest],
    ) -> List[Union[MemoryRecord, CorrectionNode]]:
        """Dispatch a heterogeneous batch of ingestion requests.

        Each request is routed to the correct ``add_*`` / ``revise_belief``
        method based on its type. The result list preserves the input
        order: index ``i`` of the output is the response for request
        ``i``. For observations and derivations, the response is a
        ``MemoryRecord``; for revisions, a ``CorrectionNode``.

        The trust-composition guarantees are preserved: ``DerivationRequest``
        routes through ``add_derived``, so the derived source is computed
        and the trust ceiling is enforced.
        """
        out: List[Union[MemoryRecord, CorrectionNode]] = []
        for request in requests:
            if isinstance(request, ObservationRequest):
                out.append(self.add(
                    request.content,
                    source=request.source,
                    provenance=request.provenance,
                    source_confidence=request.source_confidence,
                    record_id=request.record_id,
                    self_index=request.self_index,
                ))
            elif isinstance(request, DerivationRequest):
                out.append(self.add_derived(
                    request.content,
                    inputs=request.inputs,
                    record_id=request.record_id,
                    source_confidence=request.source_confidence,
                    self_index=request.self_index,
                ))
            elif isinstance(request, RevisionRequest):
                out.append(self.revise_belief(
                    prior_belief=request.prior_belief,
                    evidence=request.evidence,
                    update_operation=request.update_operation,
                    revised_belief=request.revised_belief,
                    delta=request.delta,
                    provenance=request.provenance,
                    confidence=request.confidence,
                    node_id=request.node_id,
                    self_index=request.self_index,
                ))
            elif isinstance(request, InferredSourceRequest):
                out.append(self.add_with_inferred_source(
                    request.content,
                    provenance=request.provenance,
                    record_id=request.record_id,
                    self_index=request.self_index,
                    query_context=request.query_context,
                    retrieval_margin=request.retrieval_margin,
                    recency_rank=request.recency_rank,
                    policy=request.policy,
                ))
            else:
                raise TypeError(
                    f"ingest_batch: unknown request type {type(request).__name__}; "
                    f"expected ObservationRequest, DerivationRequest, "
                    f"RevisionRequest, or InferredSourceRequest."
                )
        return out

    def ingest_envelope(
        self,
        envelope: StructuredEnvelope,
    ) -> List[Union[MemoryRecord, CorrectionNode]]:
        """Ingest a structured envelope. Convenience over ``ingest_batch``."""
        return self.ingest_batch(envelope.to_requests())

    def _all_records_by_id(self) -> Dict[str, MemoryRecord]:
        return {rec.record_id: rec for rec in self._agent.store.all_records()}

    def _build_metadata(self, self_index: Optional[SelfIndex]) -> Dict:
        """Combine the active self-index (if any) with any explicit override."""
        effective = self_index if self_index is not None else self._self_index
        if effective is None:
            return {}
        return {_SELF_INDEX_METADATA_KEY: effective.to_metadata()}

    def _any_record_is_scoped(self) -> bool:
        for record in self._agent.store.all_records():
            if _SELF_INDEX_METADATA_KEY in record.metadata:
                return True
        return False

    @property
    def active_self_index(self) -> Optional[SelfIndex]:
        """The agent's currently active self-index."""
        return self._self_index

    @active_self_index.setter
    def active_self_index(self, value: Optional[SelfIndex]) -> None:
        """Set the active self-index. Affects subsequent writes and queries."""
        self._self_index = value

    @property
    def store(self):
        """Direct access to the underlying memory store.

        Exposed for callers who need to inspect or persist records
        outside the public API. Reliance on the store's API surface is
        not covered by trace-memory's stability guarantees.
        """
        return self._agent.store

    def __len__(self) -> int:
        return len(self._agent.store)

    @property
    def storage(self) -> Storage:
        """The persistence backend in use for this agent."""
        return self._storage

    def close(self) -> None:
        """Release the storage backend's resources.

        Idempotent. Calling other methods after ``close`` is undefined
        behaviour; create a new agent instance instead.
        """
        self._storage.close()

    def __enter__(self) -> "MemoryAgent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Async surface
    # ------------------------------------------------------------------
    #
    # Each ``aX`` method awaits a threadpool execution of its sync
    # counterpart via ``asyncio.to_thread``. The trace-memory library is
    # async-COMPATIBLE rather than natively async: the underlying
    # primitives are CPU-bound (cosine retrieval, fold-force computation)
    # and run in the standard threadpool, so multiple concurrent
    # coroutines do not gain true parallelism on CPU-bound work due to
    # the GIL. The async surface IS valuable for: (a) integrating into
    # async-only LLM agent frameworks, (b) avoiding ``await`` boundary
    # awkwardness when the rest of the agent's code is async, and
    # (c) yielding the event loop during I/O-heavy operations like
    # SQLite persistence and live-LLM embedding calls.
    #
    # Mutating ops serialize through an internal ``RLock`` so concurrent
    # callers do not race on the underlying store. Reads are not
    # lock-protected and may observe partially-applied state if they
    # interleave with a concurrent mutation; in CPython, single dict and
    # list operations are atomic under the GIL so reads always see a
    # consistent snapshot of an individual record.

    async def aadd(
        self,
        content: str,
        *,
        source: Optional[SourceLabel] = None,
        provenance: Sequence[str] = (),
        source_confidence: float = 1.0,
        record_id: Optional[str] = None,
        self_index: Optional[SelfIndex] = None,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self.add,
            content,
            source=source,
            provenance=provenance,
            source_confidence=source_confidence,
            record_id=record_id,
            self_index=self_index,
        )

    async def aadd_derived(
        self,
        content: str,
        *,
        inputs: Sequence[InputRef],
        record_id: Optional[str] = None,
        source_confidence: Optional[float] = None,
        self_index: Optional[SelfIndex] = None,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self.add_derived,
            content,
            inputs=inputs,
            record_id=record_id,
            source_confidence=source_confidence,
            self_index=self_index,
        )

    async def aadd_with_inferred_source(
        self,
        content: str,
        *,
        provenance: Sequence[str] = (),
        record_id: Optional[str] = None,
        self_index: Optional[SelfIndex] = None,
        query_context: str = "",
        retrieval_margin: float = 0.0,
        recency_rank: int = 0,
        policy: str = "combined",
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self.add_with_inferred_source,
            content,
            provenance=provenance,
            record_id=record_id,
            self_index=self_index,
            query_context=query_context,
            retrieval_margin=retrieval_margin,
            recency_rank=recency_rank,
            policy=policy,
        )

    async def aadd_candidate(
        self,
        content: str,
        *,
        source: SourceLabel,
        predicted_utility: float,
        provenance: Sequence[str] = (),
        source_confidence: float = 1.0,
        record_id: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(
            self.add_candidate,
            content,
            source=source,
            predicted_utility=predicted_utility,
            provenance=provenance,
            source_confidence=source_confidence,
            record_id=record_id,
        )

    async def aflush_inscriptions(self) -> List[MemoryRecord]:
        return await asyncio.to_thread(self.flush_inscriptions)

    async def aquery(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        event_delta: Optional[float] = None,
        update_operation: Optional[str] = None,
        external_corroboration: bool = False,
        reactivation_reliability: Optional[float] = None,
    ) -> _FoldResult:
        return await asyncio.to_thread(
            self.query,
            query,
            k=k,
            event_delta=event_delta,
            update_operation=update_operation,
            external_corroboration=external_corroboration,
            reactivation_reliability=reactivation_reliability,
        )

    async def arevise_belief(
        self,
        *,
        prior_belief: str,
        evidence: str,
        update_operation: str,
        revised_belief: str,
        delta: str,
        provenance: Sequence[str] = (),
        confidence: float = 1.0,
        node_id: Optional[str] = None,
        self_index: Optional[SelfIndex] = None,
    ) -> CorrectionNode:
        return await asyncio.to_thread(
            self.revise_belief,
            prior_belief=prior_belief,
            evidence=evidence,
            update_operation=update_operation,
            revised_belief=revised_belief,
            delta=delta,
            provenance=provenance,
            confidence=confidence,
            node_id=node_id,
            self_index=self_index,
        )

    async def aaudit_laundering(
        self,
        *,
        truth_ceilings: Optional[Dict[str, SourceLabel]] = None,
    ) -> LaunderingAudit:
        return await asyncio.to_thread(
            self.audit_laundering,
            truth_ceilings=truth_ceilings,
        )

    async def aingest_batch(
        self,
        requests: Sequence[IngestRequest],
    ) -> List[Union[MemoryRecord, CorrectionNode]]:
        return await asyncio.to_thread(self.ingest_batch, requests)

    async def aingest_envelope(
        self,
        envelope: StructuredEnvelope,
    ) -> List[Union[MemoryRecord, CorrectionNode]]:
        return await asyncio.to_thread(self.ingest_envelope, envelope)


__all__ = ["MemoryAgent", "Route", "SelfIndex", "SourceLabel", "UtilityWritePolicy"]

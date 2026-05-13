"""Public dataclass types.

The validated record types from ``fgm`` are re-exported here as the
public surface. They carry their internal fields (vectors, metadata) for
introspection and storage compatibility; the developer-facing fields are
documented below.

Developer-facing fields on ``MemoryRecord``:
    record_id, content, source_label, source_confidence, provenance,
    timestamp, record_type, operation_type, decision_content.

Developer-facing fields on ``FoldResult``:
    query, retrieved, fold_force, gated, source_labels,
    active_source_labels, route_scores, selected_route,
    operation_record_id.

Other fields (vectors, query_vector, full_divergence) are intentionally
exposed for introspection but are considered internal; rely on them at
your own risk between versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from fgm.core import (
    FoldResult,
    MemoryRecord,
    OperationRecord,
    RetrievalHit,
)


@dataclass(frozen=True)
class CorrectionNode:
    """A belief revision record.

    The schema follows the Recursive Self-Indexed Correction Chains paper:
    every node preserves the lineage of the update so future agents can
    distinguish a belief from the reason it was revised.
    """

    node_id: str
    timestamp: float
    prior_belief: str
    evidence: str
    update_operation: str
    revised_belief: str
    delta: str
    provenance: Tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    record_id: Optional[str] = None  # id of the underlying MemoryRecord, if persisted


__all__ = [
    "CorrectionNode",
    "FoldResult",
    "MemoryRecord",
    "OperationRecord",
    "RetrievalHit",
]

"""trace-memory: a Python memory layer for LLM agents with operationally validated trust composition.

Phase 1 (v0.1.0a0) public API surface:

- :class:`MemoryAgent` -- the developer-facing facade.
- :class:`SourceLabel` -- enum of validated source classes.
- :class:`Route` -- enum of write-target routes.
- :class:`MemoryRecord`, :class:`FoldResult`, :class:`CorrectionNode`,
  :class:`OperationRecord`, :class:`RetrievalHit` -- record types.
- :class:`LaunderingAudit` -- paired audit result.
- :class:`MissingSourceError`, :class:`DerivedInscriptionError`,
  :class:`TraceMemoryError` -- domain errors.

See ``docs/historical/trace-memory-PRD.md`` for the original Phase 1 scope
and ``docs/architecture/VALIDATED_PRIMITIVES_LEDGER.md`` for the validation
mapping.
"""
from __future__ import annotations

__version__ = "0.8.0a0"

from .agent import MemoryAgent
from .audit import LaunderingAudit
from .errors import DerivedInscriptionError, MissingSourceError, TraceMemoryError
from .ingest import (
    DerivationRequest,
    IngestRequest,
    InferredSourceRequest,
    ObservationRequest,
    RevisionRequest,
    StructuredEnvelope,
    parse_inline_markers,
)
from .inscription import UtilityWritePolicy
from .routes import Route
from .self_index import SelfIndex
from .source_inference import (
    LLMSourceClassifier,
    ainfer_source,
    infer_source,
    set_llm_classifier,
)
from .sources import SourceLabel
from .storage import InMemoryStorage, SQLiteStorage, Storage
from .types import CorrectionNode, FoldResult, MemoryRecord, OperationRecord, RetrievalHit

__all__ = [
    "__version__",
    "CorrectionNode",
    "DerivationRequest",
    "DerivedInscriptionError",
    "FoldResult",
    "InMemoryStorage",
    "IngestRequest",
    "InferredSourceRequest",
    "LLMSourceClassifier",
    "LaunderingAudit",
    "MemoryAgent",
    "MemoryRecord",
    "MissingSourceError",
    "ObservationRequest",
    "OperationRecord",
    "RetrievalHit",
    "RevisionRequest",
    "Route",
    "SelfIndex",
    "SQLiteStorage",
    "SourceLabel",
    "Storage",
    "StructuredEnvelope",
    "TraceMemoryError",
    "UtilityWritePolicy",
    "ainfer_source",
    "infer_source",
    "parse_inline_markers",
    "set_llm_classifier",
]

"""Error types for the trace-memory public API."""
from __future__ import annotations


class TraceMemoryError(Exception):
    """Base class for all trace-memory errors."""


class MissingSourceError(TraceMemoryError, ValueError):
    """Raised when an ingestion call is missing an explicit source label.

    This is the API-level closure of the naive-inscription attack surface:
    a caller cannot ingest content with a default ``external`` label.
    Either supply an explicit ``source=``, use ``add_derived(...)`` for
    derivations from existing records, or use the inferred-source helper
    (Phase 2).
    """


class DerivedInscriptionError(TraceMemoryError, ValueError):
    """Raised when ``add_derived`` is called with invalid inputs.

    The most common cause is an empty ``inputs`` sequence: a derived
    record must have at least one contributing input. Other causes
    include record-id references that do not resolve in the store.
    """


__all__ = [
    "DerivedInscriptionError",
    "MissingSourceError",
    "TraceMemoryError",
]

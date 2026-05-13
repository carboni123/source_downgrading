"""Trace Validation Probes: executable probes for the trace/memory definitions.

Implements five probes validating the chain:
    prior attended state -> retained residue -> causal intervention sensitivity
    -> storage/addressability -> fold -> non-bookkeeping transition effect
"""
from trace_probes.probes import (
    ProbeResult,
    TraceComponent,
    TraceState,
    LeakyTraceOperator,
    MemoryRecord,
    RetrievalHit,
    TopKRetriever,
    FoldOperator,
    TransitionOutput,
    P7RegimeConfig,
    trace_retention_probe,
    causal_trace_probe,
    build_trace_records,
    addressability_probe,
    fold_force_probe,
    p7_retrieval_regime_probe,
    toy_transition,
)

__all__ = [
    "ProbeResult",
    "TraceComponent",
    "TraceState",
    "LeakyTraceOperator",
    "MemoryRecord",
    "RetrievalHit",
    "TopKRetriever",
    "FoldOperator",
    "TransitionOutput",
    "P7RegimeConfig",
    "trace_retention_probe",
    "causal_trace_probe",
    "build_trace_records",
    "addressability_probe",
    "fold_force_probe",
    "p7_retrieval_regime_probe",
    "toy_transition",
]

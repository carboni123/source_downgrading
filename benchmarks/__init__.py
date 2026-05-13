"""Benchmark suite for trace-memory.

The benchmarks are intentionally importable so tests and future SIMReC
validation scripts can reuse the same deterministic scenario truth instead of
copying fixture logic.
"""

from .laundering_dataset import (
    DOMAINS as LAUNDERING_DOMAINS,
    FAILURE_MODES as LAUNDERING_FAILURE_MODES,
    TEMPLATED_VARIANTS as LAUNDERING_TEMPLATED_VARIANTS,
    LaunderingScenario,
    dataset_summary as laundering_dataset_summary,
    expected_chain_sources,
    expected_final_origins,
    expected_final_source,
    make_dataset as make_laundering_dataset,
    validate_scenarios,
)
from .coupling_dataset import (
    COUPLING_MODES,
    CONTENT_SOURCE_LABELS as COUPLING_SOURCE_LABELS,
    DOMAINS as COUPLING_DOMAINS,
    TRUSTED_SOURCES as COUPLING_TRUSTED_SOURCES,
    UNTRUSTED_SOURCES as COUPLING_UNTRUSTED_SOURCES,
    CouplingCase,
    CouplingMemory,
    dataset_summary as coupling_dataset_summary,
    make_dataset as make_coupling_dataset,
    validate_cases as validate_coupling_cases,
)
from .source_boundary_dataset import (
    BOUNDARY_TYPES as SOURCE_BOUNDARY_TYPES,
    CONTENT_SOURCE_LABELS,
    DIFFICULTIES as SOURCE_BOUNDARY_DIFFICULTIES,
    DOMAINS as SOURCE_BOUNDARY_DOMAINS,
    SourceBoundaryCase,
    dataset_summary as source_boundary_dataset_summary,
    make_dataset as make_source_boundary_dataset,
    validate_cases as validate_source_boundary_cases,
)

__all__ = [
    "CONTENT_SOURCE_LABELS",
    "COUPLING_DOMAINS",
    "COUPLING_MODES",
    "COUPLING_SOURCE_LABELS",
    "COUPLING_TRUSTED_SOURCES",
    "COUPLING_UNTRUSTED_SOURCES",
    "CouplingCase",
    "CouplingMemory",
    "LAUNDERING_DOMAINS",
    "LAUNDERING_FAILURE_MODES",
    "LAUNDERING_TEMPLATED_VARIANTS",
    "LaunderingScenario",
    "SOURCE_BOUNDARY_DIFFICULTIES",
    "SOURCE_BOUNDARY_DOMAINS",
    "SOURCE_BOUNDARY_TYPES",
    "SourceBoundaryCase",
    "coupling_dataset_summary",
    "expected_chain_sources",
    "expected_final_origins",
    "expected_final_source",
    "laundering_dataset_summary",
    "make_coupling_dataset",
    "make_laundering_dataset",
    "make_source_boundary_dataset",
    "source_boundary_dataset_summary",
    "validate_coupling_cases",
    "validate_source_boundary_cases",
    "validate_scenarios",
]

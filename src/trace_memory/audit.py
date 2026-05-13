"""Laundering audit with cascade-invisibility-aware paired metrics.

Validates ledger section 1.9: any audit of laundering safety must report
both a self-referential metric (computed from stored source labels) and,
when available, a truth-grounded metric (computed against externally
supplied ground truth). The two together detect cascade invisibility --
the regime in which a previously laundered label hides downstream
laundering from the self-referential metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from fgm.laundering import _trust_rank
from fgm.core import (
    MemoryRecord,
    SOURCE_EXTERNAL,
)


@dataclass(frozen=True)
class LaunderingAudit:
    """Paired audit result.

    Attributes
    ----------
    n_records :
        Total number of records inspected (derived only).
    local_laundering_rate :
        Self-referential rate. Fraction of derived records whose stored
        source label is ``external`` while at least one contributing
        record had a non-``external`` stored label. Undercounts when
        prior laundering has cleansed contributing labels (see Property
        3 in the source-downgrading manuscript).
    truth_grounded_rate :
        Truth-grounded rate. Fraction of derived records whose stored
        source label has higher trust rank than the truth-supplied
        ceiling. ``None`` if no truth labels were supplied.
    gap :
        ``truth_grounded_rate - local_laundering_rate`` when both are
        defined; ``None`` otherwise. A positive gap is direct evidence
        of cascade invisibility on this audit.
    cascade_invisibility_warning :
        ``True`` if only the local rate could be computed (no truth
        labels supplied). When ``True``, the local rate must be treated
        as a lower bound, not a measurement.
    """

    n_records: int
    local_laundering_rate: float
    truth_grounded_rate: Optional[float]
    gap: Optional[float]
    cascade_invisibility_warning: bool

    @property
    def is_clean(self) -> bool:
        """True only if both metrics are defined and both are zero.

        A clean audit requires truth-grounded evidence; the local rate
        alone is insufficient because of cascade invisibility.
        """
        return (
            self.truth_grounded_rate is not None
            and self.truth_grounded_rate == 0.0
            and self.local_laundering_rate == 0.0
        )


def compute_local_laundering_rate(
    derived_records: Sequence[MemoryRecord],
    *,
    record_lookup: Mapping[str, MemoryRecord],
) -> float:
    """Self-referential laundering rate.

    A derived record is locally laundered if its stored source is
    ``external`` while at least one contributing record (resolved via
    its provenance tokens that are themselves record ids) had a
    non-``external`` stored source.
    """
    if not derived_records:
        return 0.0
    laundered = 0
    for record in derived_records:
        if record.source_label != SOURCE_EXTERNAL:
            continue
        had_non_external_input = False
        for token in record.provenance:
            parent = record_lookup.get(token)
            if parent is None:
                continue
            if parent.source_label != SOURCE_EXTERNAL:
                had_non_external_input = True
                break
        if had_non_external_input:
            laundered += 1
    return laundered / len(derived_records)


def compute_truth_grounded_rate(
    derived_records: Sequence[MemoryRecord],
    truth_ceilings: Mapping[str, str],
) -> float:
    """Truth-grounded ceiling-violation rate.

    For each derived record present in ``truth_ceilings``, a violation
    is recorded if the stored source has higher trust rank than the
    truth-supplied ceiling. Records absent from the truth mapping are
    skipped.
    """
    inspected = 0
    violations = 0
    for record in derived_records:
        ceiling = truth_ceilings.get(record.record_id)
        if ceiling is None:
            continue
        inspected += 1
        if _trust_rank(record.source_label) > _trust_rank(ceiling):
            violations += 1
    if inspected == 0:
        return 0.0
    return violations / inspected


def build_audit(
    derived_records: Sequence[MemoryRecord],
    *,
    record_lookup: Mapping[str, MemoryRecord],
    truth_ceilings: Optional[Mapping[str, str]] = None,
) -> LaunderingAudit:
    """Compute a paired audit from a set of derived records."""
    local = compute_local_laundering_rate(derived_records, record_lookup=record_lookup)
    if truth_ceilings is None:
        return LaunderingAudit(
            n_records=len(derived_records),
            local_laundering_rate=local,
            truth_grounded_rate=None,
            gap=None,
            cascade_invisibility_warning=True,
        )
    truth = compute_truth_grounded_rate(derived_records, truth_ceilings)
    return LaunderingAudit(
        n_records=len(derived_records),
        local_laundering_rate=local,
        truth_grounded_rate=truth,
        gap=truth - local,
        cascade_invisibility_warning=False,
    )


__all__ = [
    "LaunderingAudit",
    "build_audit",
    "compute_local_laundering_rate",
    "compute_truth_grounded_rate",
]

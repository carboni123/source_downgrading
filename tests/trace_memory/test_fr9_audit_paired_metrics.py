"""FR-9: Cascade-invisibility-aware auditing.

The library MUST expose an audit API that returns paired metrics. When
only the local rate can be computed, the audit MUST carry a
cascade-invisibility warning.
"""
from __future__ import annotations

from trace_memory import LaunderingAudit, MemoryAgent, SourceLabel


def test_audit_without_truth_warns_about_cascade_invisibility():
    agent = MemoryAgent()
    r1 = agent.add("E1", source=SourceLabel.EXTERNAL)
    r2 = agent.add("S1", source=SourceLabel.SIMULATION)
    agent.add_derived("derived from external + simulation", inputs=[r1, r2])

    audit = agent.audit_laundering()
    assert isinstance(audit, LaunderingAudit)
    assert audit.cascade_invisibility_warning is True
    assert audit.truth_grounded_rate is None
    assert audit.gap is None
    assert audit.is_clean is False  # cannot be declared clean without truth labels


def test_audit_with_truth_ceilings_reports_truth_grounded_rate():
    agent = MemoryAgent()
    r_ext = agent.add("E1", source=SourceLabel.EXTERNAL)
    r_sim = agent.add("S1", source=SourceLabel.SIMULATION)
    derived = agent.add_derived(
        "derived",
        inputs=[r_ext, r_sim],
        record_id="D1",
    )

    truth = {derived.record_id: SourceLabel.SIMULATION}
    audit = agent.audit_laundering(truth_ceilings=truth)
    assert audit.cascade_invisibility_warning is False
    # source-downgrading should have capped at simulation; no violation
    assert audit.truth_grounded_rate == 0.0
    assert audit.local_laundering_rate == 0.0
    assert audit.gap == 0.0
    assert audit.is_clean is True


def test_audit_with_no_derived_records_is_trivially_clean_with_truth():
    agent = MemoryAgent()
    agent.add("ordinary external record", source=SourceLabel.EXTERNAL)
    audit = agent.audit_laundering(truth_ceilings={})
    assert audit.n_records == 0
    assert audit.local_laundering_rate == 0.0
    assert audit.truth_grounded_rate == 0.0
    assert audit.is_clean is True


def test_audit_reports_n_records_equal_to_derived_count():
    agent = MemoryAgent()
    r = agent.add("E", source=SourceLabel.EXTERNAL)
    agent.add_derived("D1", inputs=[r])
    agent.add_derived("D2", inputs=[r])
    audit = agent.audit_laundering()
    assert audit.n_records == 2

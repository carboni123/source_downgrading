"""FR-8: Engineered self-index binding.

The library MUST support SelfIndex metadata on records. Retrieval MUST
filter by the active self-index; mismatching records MUST NOT appear in
the retrieved set.
"""
from __future__ import annotations

from trace_memory import MemoryAgent, SelfIndex, SourceLabel


def test_self_index_matches_when_all_fields_align():
    a = SelfIndex(user_id="alice", project_id="X", role="maintainer")
    b = SelfIndex(user_id="alice", project_id="X", role="maintainer")
    assert a.matches(b)
    assert b.matches(a)


def test_self_index_mismatches_on_any_set_field():
    base = SelfIndex(user_id="alice", project_id="X", role="maintainer")
    assert not base.matches(SelfIndex(user_id="bob", project_id="X", role="maintainer"))
    assert not base.matches(SelfIndex(user_id="alice", project_id="Y", role="maintainer"))
    assert not base.matches(SelfIndex(user_id="alice", project_id="X", role="viewer"))


def test_self_index_unscoped_fields_match_anything():
    # If the active index leaves user_id None, any user_id is accepted.
    active = SelfIndex(project_id="X")
    assert active.matches(SelfIndex(user_id="alice", project_id="X"))
    assert active.matches(SelfIndex(user_id="bob", project_id="X"))
    assert not active.matches(SelfIndex(project_id="Y"))


def test_standing_commitment_is_not_a_scoping_field():
    a = SelfIndex(user_id="alice", standing_commitment="commit A")
    b = SelfIndex(user_id="alice", standing_commitment="commit B")
    assert a.matches(b)


def test_query_filters_records_outside_active_self_index():
    agent = MemoryAgent(
        retrieval_k=5,
        self_index=SelfIndex(user_id="alice", project_id="X"),
    )
    agent.add(
        "alice X record",
        source=SourceLabel.EXTERNAL,
        record_id="alice_X",
    )
    # Explicitly scope a record to a different project.
    agent.add(
        "alice Y record (different project)",
        source=SourceLabel.EXTERNAL,
        record_id="alice_Y",
        self_index=SelfIndex(user_id="alice", project_id="Y"),
    )
    # Records without a self-index remain globally visible.
    agent.add(
        "unscoped record",
        source=SourceLabel.EXTERNAL,
        record_id="unscoped",
        self_index=SelfIndex(),  # all fields None -> no scoping fields
    )
    result = agent.query("record")
    retrieved_ids = {hit.record.record_id for hit in result.retrieved}
    assert "alice_Y" not in retrieved_ids
    assert "alice_X" in retrieved_ids


def test_query_with_no_active_self_index_excludes_scoped_records():
    agent = MemoryAgent(retrieval_k=5)  # no active index
    agent.add(
        "unscoped",
        source=SourceLabel.EXTERNAL,
        record_id="unscoped",
    )
    agent.add(
        "scoped to alice",
        source=SourceLabel.EXTERNAL,
        record_id="alice",
        self_index=SelfIndex(user_id="alice"),
    )
    result = agent.query("record")
    retrieved_ids = {hit.record.record_id for hit in result.retrieved}
    # The scoped record is invisible when the agent has no active index.
    assert "alice" not in retrieved_ids


def test_active_self_index_can_be_changed():
    agent = MemoryAgent(retrieval_k=5)
    agent.add(
        "alice record",
        source=SourceLabel.EXTERNAL,
        record_id="alice_record",
        self_index=SelfIndex(user_id="alice"),
    )
    # No active index: alice's record is invisible.
    result = agent.query("record")
    assert "alice_record" not in {h.record.record_id for h in result.retrieved}
    # Switch to alice: her record becomes visible.
    agent.active_self_index = SelfIndex(user_id="alice")
    result = agent.query("record")
    assert "alice_record" in {h.record.record_id for h in result.retrieved}


def test_derived_records_inherit_active_self_index():
    agent = MemoryAgent(self_index=SelfIndex(user_id="alice", project_id="X"))
    r1 = agent.add("E1", source=SourceLabel.EXTERNAL)
    derived = agent.add_derived("derived", inputs=[r1])
    # The derived record carries alice/X metadata
    from trace_memory.self_index import record_self_index
    derived_index = record_self_index(derived)
    assert derived_index.user_id == "alice"
    assert derived_index.project_id == "X"

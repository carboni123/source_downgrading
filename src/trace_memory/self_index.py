"""Engineered self-index binding (FR-8, ledger section 1.8).

Records may carry a ``SelfIndex`` (user_id, project_id, role,
permission_scope, standing_commitment). When the agent has an active
self-index, retrieval and audit are filtered: records with a self-index
that mismatches on any of the four scoping fields (user/project/role/
permission_scope) are excluded.

The validated property here is *engineered* tenant isolation: callers
supply the metadata, the layer enforces matching. The standing_commitment
field is content, not a filter -- it is preserved on the record for
later inspection.

This primitive is NOT cryptographic. Production multi-tenancy requires
additional access-control layers; see PRD section 7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fgm.core import MemoryRecord


_SELF_INDEX_METADATA_KEY = "_trace_memory_self_index"


@dataclass(frozen=True)
class SelfIndex:
    """Engineered self-index metadata.

    All fields are optional. ``None`` means "unscoped on this field" and
    matches any value; concrete values must match exactly. Fields:

    - ``user_id``: continuing user identifier.
    - ``project_id``: scope to a project.
    - ``role``: agent or human role.
    - ``permission_scope``: read/write/admin/etc.
    - ``standing_commitment``: a free-form statement of a durable
      commitment (e.g., "legal approval required before rollback").
      This is content, not a filter; it never affects retrieval.
    """

    user_id: Optional[str] = None
    project_id: Optional[str] = None
    role: Optional[str] = None
    permission_scope: Optional[str] = None
    standing_commitment: Optional[str] = None

    def matches(self, other: "SelfIndex") -> bool:
        """Return True iff ``other`` is compatible with this active index.

        The matching rule is per-field: if this index has a value on a
        scoping field, ``other`` must match it; if this index is ``None``
        on a field, that field is unscoped and matches any value.
        Standing commitment is not a scoping field.
        """
        for field_name in ("user_id", "project_id", "role", "permission_scope"):
            mine = getattr(self, field_name)
            theirs = getattr(other, field_name)
            if mine is None:
                continue
            if mine != theirs:
                return False
        return True

    def to_metadata(self) -> dict:
        """Serialise to a dict for storage in MemoryRecord.metadata."""
        return {
            "user_id": self.user_id,
            "project_id": self.project_id,
            "role": self.role,
            "permission_scope": self.permission_scope,
            "standing_commitment": self.standing_commitment,
        }

    @classmethod
    def from_metadata(cls, data: dict) -> "SelfIndex":
        return cls(
            user_id=data.get("user_id"),
            project_id=data.get("project_id"),
            role=data.get("role"),
            permission_scope=data.get("permission_scope"),
            standing_commitment=data.get("standing_commitment"),
        )


def record_self_index(record: MemoryRecord) -> Optional[SelfIndex]:
    """Extract the SelfIndex of a record, or None if it carries none."""
    data = record.metadata.get(_SELF_INDEX_METADATA_KEY)
    if data is None:
        return None
    return SelfIndex.from_metadata(data)


def record_matches_index(record: MemoryRecord, active: Optional[SelfIndex]) -> bool:
    """Return True if the record's self-index matches the active index.

    Records without a self-index are globally visible. Records with a
    self-index are visible only when the active index matches.
    """
    record_index = record_self_index(record)
    if record_index is None:
        return True
    if active is None:
        # The record is scoped, but the agent has no active index;
        # the record is not visible.
        return False
    return active.matches(record_index)


__all__ = [
    "SelfIndex",
    "record_matches_index",
    "record_self_index",
]

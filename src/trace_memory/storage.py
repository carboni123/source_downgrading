"""Pluggable storage backends for trace-memory.

The :class:`Storage` protocol defines what an agent needs from a
persistence layer:

- ``save(record)`` -- durably write a record (idempotent on record_id).
- ``load_all()`` -- yield every persisted record at startup.
- ``delete(record_id)`` -- remove a record.
- ``contains(record_id)`` -- existence check.

Two reference implementations ship:

- :class:`InMemoryStorage` -- the default for ephemeral agents. Records
  live only in the agent's RAM; the storage layer is a no-op shim.
- :class:`SQLiteStorage` -- single-file embedded persistent backend
  using the Python standard library. Records survive process restarts;
  the agent reloads its full store from the file on construction.

The architecture is a write-through cache: the agent's internal
in-memory store is the read path (fast retrieval, fast fold-force);
mutating calls write through to the storage backend. v0.3 is
single-process; opening the same SQLite file from two processes
results in last-write-wins.
"""
from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Optional, Protocol, Union, runtime_checkable

import numpy as np

from fgm.core import MemoryRecord


@runtime_checkable
class Storage(Protocol):
    """Persistence interface for MemoryAgent.

    Implementations MUST:

    - Be idempotent on record_id for ``save``.
    - Return records in insertion order from ``load_all`` when ordering
      is meaningful (the agent treats this as advisory but predictable
      ordering helps tests and debugging).
    - Preserve every field on ``MemoryRecord`` round-trip: ``record_id``,
      ``content``, ``vector``, ``timestamp``, ``record_type``,
      ``operation_type``, ``decision_content``, ``source_label``,
      ``source_confidence``, ``provenance``, ``metadata``.

    Implementations MAY:

    - Use any serialization format.
    - Be asynchronous internally (the protocol surface remains sync).
    """

    def save(self, record: MemoryRecord) -> None: ...
    def load_all(self) -> Iterator[MemoryRecord]: ...
    def delete(self, record_id: str) -> bool: ...
    def contains(self, record_id: str) -> bool: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# InMemoryStorage -- no-op shim for ephemeral agents
# ---------------------------------------------------------------------------


class InMemoryStorage:
    """No-op storage backend.

    Agents constructed without an explicit ``storage`` argument use this
    backend implicitly: the agent's internal in-memory store is the
    only state. State is lost on process exit. This backend has no
    overhead beyond the protocol dispatch.
    """

    def save(self, record: MemoryRecord) -> None:
        del record

    def load_all(self) -> Iterator[MemoryRecord]:
        return iter(())

    def delete(self, record_id: str) -> bool:
        del record_id
        return False

    def contains(self, record_id: str) -> bool:
        del record_id
        return False

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# SQLiteStorage -- single-file embedded persistent backend
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    vector BLOB NOT NULL,
    timestamp REAL NOT NULL,
    record_type TEXT NOT NULL,
    operation_type TEXT,
    decision_content TEXT,
    source_label TEXT NOT NULL,
    source_confidence REAL NOT NULL,
    provenance_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    insertion_order INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_insertion_order
    ON records(insertion_order);

CREATE INDEX IF NOT EXISTS idx_records_source_label
    ON records(source_label);

CREATE INDEX IF NOT EXISTS idx_records_record_type
    ON records(record_type);
"""


def _serialize_vector(vector: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes via np.save (preserves dtype/shape)."""
    buf = io.BytesIO()
    np.save(buf, vector, allow_pickle=False)
    return buf.getvalue()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to a numpy array."""
    buf = io.BytesIO(blob)
    return np.load(buf, allow_pickle=False)


def _serialize_json(value) -> str:
    """Compact, deterministic JSON for provenance and metadata."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class SQLiteStorage:
    """SQLite-backed persistent storage.

    Parameters
    ----------
    path :
        Filesystem path to the SQLite database file. The file is
        created if it does not exist. Pass ``":memory:"`` to use an
        in-memory SQLite database (useful for tests; distinct from
        :class:`InMemoryStorage` because it round-trips through the
        SQL layer).

    Notes
    -----
    The connection runs in autocommit mode (each ``save`` is its own
    transaction). For batched ingestion, callers can wrap multiple
    saves in a manual transaction by accessing the underlying
    :attr:`connection`.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = str(path)
        # check_same_thread=False allows the connection to be used from a
        # different thread than the one that created it. Callers must
        # serialize access; MemoryAgent does this via an internal lock on
        # mutating ops. Without this flag, asyncio.to_thread offloading
        # would raise ProgrammingError on cross-thread reuse.
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def path(self) -> str:
        return self._path

    def save(self, record: MemoryRecord) -> None:
        cursor = self._connection.cursor()
        # Determine next insertion_order.
        row = cursor.execute(
            "SELECT insertion_order FROM records WHERE record_id = ?",
            (record.record_id,),
        ).fetchone()
        if row is not None:
            insertion_order = row[0]
        else:
            row = cursor.execute(
                "SELECT COALESCE(MAX(insertion_order), -1) + 1 FROM records"
            ).fetchone()
            insertion_order = row[0]
        cursor.execute(
            """
            INSERT OR REPLACE INTO records (
                record_id, content, vector, timestamp, record_type,
                operation_type, decision_content, source_label,
                source_confidence, provenance_json, metadata_json,
                insertion_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.content,
                _serialize_vector(record.vector),
                float(record.timestamp),
                record.record_type,
                record.operation_type,
                record.decision_content,
                record.source_label,
                float(record.source_confidence),
                _serialize_json(list(record.provenance)),
                _serialize_json(record.metadata),
                insertion_order,
            ),
        )
        self._connection.commit()

    def load_all(self) -> Iterator[MemoryRecord]:
        cursor = self._connection.cursor()
        rows = cursor.execute(
            """
            SELECT
                record_id, content, vector, timestamp, record_type,
                operation_type, decision_content, source_label,
                source_confidence, provenance_json, metadata_json
            FROM records
            ORDER BY insertion_order ASC
            """
        )
        for row in rows:
            yield MemoryRecord(
                record_id=row[0],
                content=row[1],
                vector=_deserialize_vector(row[2]),
                timestamp=float(row[3]),
                record_type=row[4],
                operation_type=row[5],
                decision_content=row[6],
                source_label=row[7],
                source_confidence=float(row[8]),
                provenance=tuple(json.loads(row[9])),
                metadata=json.loads(row[10]),
            )

    def delete(self, record_id: str) -> bool:
        cursor = self._connection.cursor()
        cursor.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
        deleted = cursor.rowcount > 0
        self._connection.commit()
        return deleted

    def contains(self, record_id: str) -> bool:
        cursor = self._connection.cursor()
        row = cursor.execute(
            "SELECT 1 FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        return row is not None

    def __len__(self) -> int:
        cursor = self._connection.cursor()
        row = cursor.execute("SELECT COUNT(*) FROM records").fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the SQLite connection. Subsequent operations will fail."""
        try:
            self._connection.close()
        except sqlite3.ProgrammingError:
            pass

    def __enter__(self) -> "SQLiteStorage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "InMemoryStorage",
    "SQLiteStorage",
    "Storage",
]

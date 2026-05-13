"""Source labels for stored memory records.

Each label corresponds to a class in the source lattice defined by the
source-downgrading inscription primitive (see VALIDATED_PRIMITIVES_LEDGER
sections 1.1 and 1.4 in ``docs/architecture/VALIDATED_PRIMITIVES_LEDGER.md``).

Values are kept identical to the underlying ``fgm`` string constants so a
``SourceLabel`` value is interchangeable with its string equivalent at API
boundaries.
"""
from __future__ import annotations

from enum import Enum

from fgm.core import (
    SOURCE_EXTERNAL,
    SOURCE_FABRICATED,
    SOURCE_INFERENCE,
    SOURCE_OPERATION_RECORD,
    SOURCE_RETRIEVED_MEMORY,
    SOURCE_SIMULATION,
    SOURCE_TOOL_OUTPUT,
)


class SourceLabel(str, Enum):
    """Source class for a stored record.

    The lattice is partially ordered by trust:
    FABRICATED_OR_UNCERTAIN < SIMULATION < INFERENCE < RETRIEVED_MEMORY
        < TOOL_OUTPUT < EXTERNAL.

    OPERATION_RECORD indexes memory-use events rather than content and is
    treated separately by the trust composition rule.
    """

    EXTERNAL = SOURCE_EXTERNAL
    TOOL_OUTPUT = SOURCE_TOOL_OUTPUT
    RETRIEVED_MEMORY = SOURCE_RETRIEVED_MEMORY
    INFERENCE = SOURCE_INFERENCE
    SIMULATION = SOURCE_SIMULATION
    FABRICATED_OR_UNCERTAIN = SOURCE_FABRICATED
    OPERATION_RECORD = SOURCE_OPERATION_RECORD

    def __str__(self) -> str:
        return self.value


__all__ = ["SourceLabel"]

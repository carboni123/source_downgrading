"""Write-target routes for the source-sensitive routing layer.

Routes correspond to the six channels validated in
``docs/architecture/VALIDATED_PRIMITIVES_LEDGER.md`` section 1.3. The routing
policy maps an attended-and-folded candidate to exactly one of these targets.
"""
from __future__ import annotations

from enum import Enum

from fgm.core import (
    ROUTE_CORRECTION_CHAIN,
    ROUTE_DURABLE_MEMORY,
    ROUTE_NULL,
    ROUTE_OPERATION_MEMORY,
    ROUTE_QUARANTINE,
    ROUTE_TRACE,
)


class Route(str, Enum):
    """Write target selected by the routing policy."""

    NULL = ROUTE_NULL
    TRACE = ROUTE_TRACE
    DURABLE = ROUTE_DURABLE_MEMORY
    OPERATION = ROUTE_OPERATION_MEMORY
    CORRECTION = ROUTE_CORRECTION_CHAIN
    QUARANTINE = ROUTE_QUARANTINE

    def __str__(self) -> str:
        return self.value


__all__ = ["Route"]

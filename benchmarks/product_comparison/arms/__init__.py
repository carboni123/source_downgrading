"""Three memory-arm adapters sharing the ``MemoryArm`` Protocol."""
from .base import AnswerResult, DeriveResult, MemoryArm, ObserveResult
from .bash_arm import BashArm
from .trace_memory_arm import TraceMemoryArm
from .vector_arm import VectorArm

__all__ = [
    "AnswerResult",
    "BashArm",
    "DeriveResult",
    "MemoryArm",
    "ObserveResult",
    "TraceMemoryArm",
    "VectorArm",
]

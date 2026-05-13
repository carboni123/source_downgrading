"""PoisonedRAG adapter for the trace-memory product comparison.

External validation of the source-downgrading invariant against the
PoisonedRAG (Zou et al., 2024) adversarial-RAG corpus. Hard mode: no
oracle source labels -- a small LLM-driven source classifier assigns
each passage a label, and trace_memory uses those labels to source-
downgrade the adversarial passages without any benchmark-side
supervision.

Public entry points:

    from benchmarks.poisonedrag.dataset import load_poisoned_questions
    from benchmarks.poisonedrag.source_classifier import LLMSourceClassifier
    from benchmarks.poisonedrag.arms import (
        VectorPRArm, TraceMemoryPRArm, BashPRArm,
    )
    from benchmarks.poisonedrag.grading import grade_open_answer, ArmASR
    from benchmarks.poisonedrag.runner import run_benchmark
"""

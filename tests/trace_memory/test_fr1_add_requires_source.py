"""FR-1: Source-labeled ingestion.

The library MUST require an explicit source label on ``add(...)``. The
naive-inscription attack surface (default to ``external``) is closed at
the API level.
"""
from __future__ import annotations

import pytest

from trace_memory import MemoryAgent, MissingSourceError, SourceLabel


def test_add_without_source_raises_missing_source_error():
    agent = MemoryAgent()
    with pytest.raises(MissingSourceError):
        agent.add("observation: server returned 500")


def test_add_with_explicit_source_succeeds():
    agent = MemoryAgent()
    record = agent.add(
        "observation: server returned 500",
        source=SourceLabel.EXTERNAL,
        provenance=("sensor_42",),
    )
    assert record.source_label == "external"
    assert record.provenance == ("sensor_42",)
    assert record.source_confidence == 1.0


@pytest.mark.parametrize(
    "label",
    [
        SourceLabel.EXTERNAL,
        SourceLabel.TOOL_OUTPUT,
        SourceLabel.RETRIEVED_MEMORY,
        SourceLabel.INFERENCE,
        SourceLabel.SIMULATION,
        SourceLabel.FABRICATED_OR_UNCERTAIN,
        SourceLabel.OPERATION_RECORD,
    ],
)
def test_every_source_label_round_trips(label):
    agent = MemoryAgent()
    record = agent.add(f"content for {label}", source=label)
    assert record.source_label == label.value


def test_source_label_is_str_compatible():
    # SourceLabel inherits from str so it's safely interpolated and compared.
    assert SourceLabel.EXTERNAL == "external"
    assert f"{SourceLabel.SIMULATION}" == "simulation"

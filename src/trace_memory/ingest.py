"""Bulk ingestion API: typed requests + a canonical structured envelope.

Two production patterns are supported (PRD v0.5):

1. **App-owned structure.** The caller already parses the agent's output
   into its own shape. They build a list of typed ingestion requests
   (``ObservationRequest``, ``DerivationRequest``, ``RevisionRequest``,
   ``InferredSourceRequest``) and pass it to ``agent.ingest_batch(...)``.
   The library does not impose a JSON schema on the LLM prompt.

2. **Library-owned structure.** The caller adopts the library's canonical
   envelope schema. They include ``StructuredEnvelope.system_prompt_block()``
   in the LLM system prompt, parse the LLM output via
   ``StructuredEnvelope.parse(raw)`` (which accepts both JSON and an
   inline marker form), and call ``agent.ingest_envelope(...)``.

Both patterns route through ``ingest_batch`` internally and preserve all
the trust-composition guarantees of the underlying primitives. A
``DerivationRequest`` runs through ``add_derived``, so the source label
is computed from contributing inputs and the trust ceiling is enforced.

The inline marker form is intentionally limited (one record per line,
prefix-based). For rich cases (provenance, contributing inputs, revisions)
use the JSON form.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .errors import DerivedInscriptionError
from .self_index import SelfIndex
from .sources import SourceLabel


InputRef = Union[str, "ObservationRequest", "DerivationRequest"]
RecordIdRef = str


# ---------------------------------------------------------------------------
# Typed ingestion requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationRequest:
    """A direct ingestion with an explicit source label.

    Equivalent to ``agent.add(content, source=..., ...)``.
    """

    content: str
    source: SourceLabel
    provenance: Tuple[str, ...] = ()
    source_confidence: float = 1.0
    record_id: Optional[str] = None
    self_index: Optional[SelfIndex] = None


@dataclass(frozen=True)
class DerivationRequest:
    """A derived ingestion. Source is computed; provenance is propagated.

    Equivalent to ``agent.add_derived(content, inputs=..., ...)``.
    """

    content: str
    inputs: Tuple[RecordIdRef, ...]
    source_confidence: Optional[float] = None
    record_id: Optional[str] = None
    self_index: Optional[SelfIndex] = None


@dataclass(frozen=True)
class RevisionRequest:
    """A belief revision. Recorded as a correction-chain node.

    Equivalent to ``agent.revise_belief(...)``.
    """

    prior_belief: str
    evidence: str
    update_operation: str
    revised_belief: str
    delta: str
    provenance: Tuple[str, ...] = ()
    confidence: float = 1.0
    node_id: Optional[str] = None
    self_index: Optional[SelfIndex] = None


@dataclass(frozen=True)
class InferredSourceRequest:
    """An observation whose source is inferred from content/features.

    Equivalent to ``agent.add_with_inferred_source(content, ...)``.
    Emits the natural-prose feasibility-floor warning when ingested.
    """

    content: str
    provenance: Tuple[str, ...] = ()
    record_id: Optional[str] = None
    self_index: Optional[SelfIndex] = None
    query_context: str = ""
    retrieval_margin: float = 0.0
    recency_rank: int = 0
    policy: str = "combined"


IngestRequest = Union[
    ObservationRequest,
    DerivationRequest,
    RevisionRequest,
    InferredSourceRequest,
]


# ---------------------------------------------------------------------------
# Inline marker parser
# ---------------------------------------------------------------------------


_INLINE_PREFIX_TO_SOURCE = {
    "OBSERVED": SourceLabel.EXTERNAL,
    "TOOL": SourceLabel.TOOL_OUTPUT,
    "RETRIEVED": SourceLabel.RETRIEVED_MEMORY,
    "INFERRED": SourceLabel.INFERENCE,
    "SIMULATED": SourceLabel.SIMULATION,
    "FABRICATED": SourceLabel.FABRICATED_OR_UNCERTAIN,
}

_INLINE_PATTERN = re.compile(
    r"^\s*(?P<prefix>OBSERVED|TOOL|RETRIEVED|INFERRED|SIMULATED|FABRICATED)\s*:\s*(?P<content>.+?)\s*$",
    re.IGNORECASE,
)


def parse_inline_markers(text: str) -> List[ObservationRequest]:
    """Parse a text blob into ObservationRequest items via line-prefix markers.

    Recognised prefixes (case-insensitive): ``OBSERVED``, ``TOOL``,
    ``RETRIEVED``, ``INFERRED``, ``SIMULATED``, ``FABRICATED``.

    Lines that do not match a recognised prefix are skipped silently.
    For richer ingestion (provenance, derived inputs, revisions) use the
    JSON envelope form instead.
    """
    out: List[ObservationRequest] = []
    for line in text.splitlines():
        match = _INLINE_PATTERN.match(line)
        if match is None:
            continue
        prefix = match.group("prefix").upper()
        content = match.group("content").strip()
        if not content:
            continue
        out.append(ObservationRequest(
            content=content,
            source=_INLINE_PREFIX_TO_SOURCE[prefix],
        ))
    return out


# ---------------------------------------------------------------------------
# Structured envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredEnvelope:
    """Canonical structured-output envelope for ingestion.

    The envelope groups records by ingestion intent: observations,
    derivations, revisions, and inferred-source observations. Each
    field is a sequence of typed requests.

    Construct via the constructor directly, or via ``parse(...)`` which
    accepts JSON or an inline marker form.
    """

    observations: Tuple[ObservationRequest, ...] = ()
    derivations: Tuple[DerivationRequest, ...] = ()
    revisions: Tuple[RevisionRequest, ...] = ()
    inferred_source_observations: Tuple[InferredSourceRequest, ...] = ()

    def to_requests(self) -> List[IngestRequest]:
        """Flatten the envelope into a single ordered request list."""
        out: List[IngestRequest] = []
        out.extend(self.observations)
        out.extend(self.derivations)
        out.extend(self.revisions)
        out.extend(self.inferred_source_observations)
        return out

    def is_empty(self) -> bool:
        return (
            not self.observations
            and not self.derivations
            and not self.revisions
            and not self.inferred_source_observations
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, raw: str) -> "StructuredEnvelope":
        """Parse a raw LLM output into a StructuredEnvelope.

        Attempts JSON parsing first; on failure, falls back to the
        inline marker form. If both fail, returns an empty envelope.

        For applications that strictly require the JSON form (and want
        a parse error rather than a silent fallback), use
        ``parse_json`` directly.
        """
        text = raw.strip()
        if text.startswith("{"):
            try:
                return cls.parse_json(text)
            except ValueError:
                pass
        observations = parse_inline_markers(text)
        return cls(observations=tuple(observations))

    @classmethod
    def parse_json(cls, raw: str) -> "StructuredEnvelope":
        """Parse a JSON-form envelope. Raises ValueError on malformed input.

        Expected schema (every field is optional and defaults to empty):

        ::

            {
                "observations": [
                    {"content": "...",
                     "source": "external|tool_output|retrieved_memory|inference|simulation|fabricated_or_uncertain|operation_record",
                     "provenance": ["..."],
                     "source_confidence": 1.0,
                     "record_id": "..."}
                ],
                "derivations": [
                    {"content": "...",
                     "inputs": ["record_id_a", "record_id_b"],
                     "source_confidence": null,
                     "record_id": "..."}
                ],
                "revisions": [
                    {"prior_belief": "...",
                     "evidence": "...",
                     "update_operation": "...",
                     "revised_belief": "...",
                     "delta": "...",
                     "provenance": ["..."],
                     "confidence": 0.9}
                ],
                "inferred_source_observations": [
                    {"content": "...", "provenance": ["..."], "record_id": "..."}
                ]
            }
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"envelope JSON is not parseable: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("envelope JSON must be an object")

        observations = tuple(
            _coerce_observation(item)
            for item in data.get("observations", [])
        )
        derivations = tuple(
            _coerce_derivation(item)
            for item in data.get("derivations", [])
        )
        revisions = tuple(
            _coerce_revision(item)
            for item in data.get("revisions", [])
        )
        inferred = tuple(
            _coerce_inferred(item)
            for item in data.get("inferred_source_observations", [])
        )
        return cls(
            observations=observations,
            derivations=derivations,
            revisions=revisions,
            inferred_source_observations=inferred,
        )

    # ------------------------------------------------------------------
    # Prompt block
    # ------------------------------------------------------------------

    @staticmethod
    def system_prompt_block() -> str:
        """Return text to drop into an LLM system prompt.

        Describes the canonical JSON envelope schema and the source
        trust ordering, instructing the model to emit its output in
        that shape. The block is opinionated: it tells the model to
        prefer the lowest-trust source that honestly applies, which is
        the safe default under the source-downgrading inscription rule.
        """
        return _SYSTEM_PROMPT_BLOCK


_SYSTEM_PROMPT_BLOCK = """\
When emitting structured output, return a single JSON object with the
following shape (omit any field that has no items):

{
  "observations": [
    {"content": "...", "source": "external|tool_output|retrieved_memory|inference|simulation|fabricated_or_uncertain",
     "provenance": ["optional list of source tokens"]}
  ],
  "derivations": [
    {"content": "...", "inputs": ["record_id_of_contributing_input", "..."]}
  ],
  "revisions": [
    {"prior_belief": "...", "evidence": "...", "update_operation": "...",
     "revised_belief": "...", "delta": "...", "confidence": 0.0_to_1.0}
  ]
}

Source labels are ranked by trust (highest to lowest):

    external > tool_output > retrieved_memory > inference > simulation > fabricated_or_uncertain

Use the LOWEST-trust source that honestly applies. External should be
reserved for content you directly observed. Inferences (your own
reasoning) belong in `derivations` with the contributing record ids
listed in `inputs`, not in `observations` with source=external.
Hypothetical or counterfactual statements are simulation. Rumors or
unverified claims are fabricated_or_uncertain.

If you change a prior belief based on new evidence, emit a revisions
entry with the full lineage (prior, evidence, update operation,
revised, delta).

Return ONLY this JSON. Do not wrap it in prose.
"""


# ---------------------------------------------------------------------------
# JSON -> request coercion helpers
# ---------------------------------------------------------------------------


def _coerce_source_label(value: object) -> SourceLabel:
    if isinstance(value, SourceLabel):
        return value
    if isinstance(value, str):
        try:
            return SourceLabel(value)
        except ValueError as exc:
            raise ValueError(
                f"unknown source label {value!r}; valid: "
                f"{[s.value for s in SourceLabel]}"
            ) from exc
    raise ValueError(f"source must be a string, got {type(value).__name__}")


def _coerce_observation(item: dict) -> ObservationRequest:
    if not isinstance(item, dict):
        raise ValueError(f"observation item must be an object, got {type(item).__name__}")
    if "content" not in item:
        raise ValueError("observation item missing 'content'")
    if "source" not in item:
        raise ValueError("observation item missing 'source'")
    return ObservationRequest(
        content=str(item["content"]),
        source=_coerce_source_label(item["source"]),
        provenance=tuple(item.get("provenance", ())),
        source_confidence=float(item.get("source_confidence", 1.0)),
        record_id=item.get("record_id"),
        self_index=_coerce_self_index(item.get("self_index")),
    )


def _coerce_derivation(item: dict) -> DerivationRequest:
    if not isinstance(item, dict):
        raise ValueError(f"derivation item must be an object, got {type(item).__name__}")
    if "content" not in item:
        raise ValueError("derivation item missing 'content'")
    inputs = item.get("inputs", ())
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("derivation 'inputs' must be a list of record ids")
    return DerivationRequest(
        content=str(item["content"]),
        inputs=tuple(str(i) for i in inputs),
        source_confidence=item.get("source_confidence"),
        record_id=item.get("record_id"),
        self_index=_coerce_self_index(item.get("self_index")),
    )


def _coerce_revision(item: dict) -> RevisionRequest:
    if not isinstance(item, dict):
        raise ValueError(f"revision item must be an object, got {type(item).__name__}")
    required = ("prior_belief", "evidence", "update_operation", "revised_belief", "delta")
    for key in required:
        if key not in item:
            raise ValueError(f"revision item missing required field {key!r}")
    return RevisionRequest(
        prior_belief=str(item["prior_belief"]),
        evidence=str(item["evidence"]),
        update_operation=str(item["update_operation"]),
        revised_belief=str(item["revised_belief"]),
        delta=str(item["delta"]),
        provenance=tuple(item.get("provenance", ())),
        confidence=float(item.get("confidence", 1.0)),
        node_id=item.get("node_id"),
        self_index=_coerce_self_index(item.get("self_index")),
    )


def _coerce_inferred(item: dict) -> InferredSourceRequest:
    if not isinstance(item, dict):
        raise ValueError(f"inferred-source item must be an object, got {type(item).__name__}")
    if "content" not in item:
        raise ValueError("inferred-source item missing 'content'")
    return InferredSourceRequest(
        content=str(item["content"]),
        provenance=tuple(item.get("provenance", ())),
        record_id=item.get("record_id"),
        self_index=_coerce_self_index(item.get("self_index")),
        query_context=str(item.get("query_context", "")),
        retrieval_margin=float(item.get("retrieval_margin", 0.0)),
        recency_rank=int(item.get("recency_rank", 0)),
        policy=str(item.get("policy", "combined")),
    )


def _coerce_self_index(value: object) -> Optional[SelfIndex]:
    if value is None:
        return None
    if isinstance(value, SelfIndex):
        return value
    if isinstance(value, dict):
        return SelfIndex.from_metadata(value)
    raise ValueError(f"self_index must be null or an object, got {type(value).__name__}")


__all__ = [
    "DerivationRequest",
    "IngestRequest",
    "InferredSourceRequest",
    "ObservationRequest",
    "RevisionRequest",
    "StructuredEnvelope",
    "parse_inline_markers",
]

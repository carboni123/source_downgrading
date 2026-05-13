"""Source(.) inference for unlabeled content (FR-6, ledger section 1.6).

Exposes ``infer_source(...)`` as a standalone function and a convenience
``add_with_inferred_source(...)`` on the agent (defined in agent.py).

The upstream validation harness provided the first structured-fixture
feasibility floor. The product ``combined`` policy adds a conservative
boundary layer for realistic ingestion text: it prefers lower-trust labels
when content looks inferred, simulated, remembered, or uncertain, and it
requires machine-shaped evidence before calling ordinary prose tool output.
Callers using this API on natural prose should still validate against their
own labelled data; the warning emitted by ``add_with_inferred_source(...)``
reflects this.

The baseline policies (``uniform_external``, ``lexical_rules``,
``feature_threshold``, ``combined_legacy``, ``combined``) are also exposed
for callers who want to compare strategies on their own data.
"""
from __future__ import annotations

import asyncio
import os
import re
import warnings
from pathlib import Path
from typing import Callable

from fgm.source_inference import (
    POLICIES as _FGM_POLICIES,
    SourceInferenceCase,
)

from .sources import SourceLabel


SourceInferencePolicy = Callable[[SourceInferenceCase], str]

_TRAINED_SOURCE_CLASSIFIER = None
_TRAINED_SOURCE_CLASSIFIER_PATH = ""

_TRAINED_GUARD_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        SourceLabel.FABRICATED_OR_UNCERTAIN.value,
        tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
            r"\bunverified\b",
            r"\banonymous\b",
            r"\brumou?r\b",
            r"\bfabricated\b",
            r"\bunsupported\b",
            r"\bunknown provenance\b",
            r"\bhallucinated\b",
            r"\bsource-free\b",
            r"\bbaseless\b",
            r"\bjoke draft\b",
            r"\bprovenance is unclear\b",
            r"\bunconfirmed\b",
            r"\bno verifying source\b",
            r"\bprovenance is missing\b",
            r"\btreat as uncertain\b",
        )),
    ),
    (
        SourceLabel.SIMULATION.value,
        tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
            r"\bhypothetical\b",
            r"\bsuppose\b",
            r"\bsimulated outcome\b",
            r"\bcounterfactual\b",
            r"\bcounter-plan\b",
            r"\bforecast branch\b",
            r"\bforecast\b",
            r"\btrial branch\b",
            r"\bimagine\b",
            r"\bwhat-if\b",
            r"\bplanning scenario\b",
            r"\bforecast, not an observation\b",
            r"\bprojected branch\b",
            r"\bwould\b",
            r"\bif .* (would|could|may)\b",
            r"\bunder .* would\b",
        )),
    ),
    (
        SourceLabel.INFERENCE.value,
        tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
            r"\binferred conclusion\b",
            r"\bi conclude\b",
            r"\btherefore\b",
            r"\bevidence suggests\b",
            r"\bfacts point to\b",
            r"\bmy synthesis\b",
            r"\breasoning over\b",
            r"\bsafest conclusion\b",
            r"\bsafest reading\b",
            r"\bpattern implies\b",
            r"\bsupports\b",
            r"\bappears\b",
            r"\bbest treated\b",
            r"\bprobably\b",
            r"\blikely\b",
            r"\bindicates\b",
            r"\bderived conclusion\b",
            r"\broot-cause note\b",
            r"\brecommendation based on synthesis\b",
            r"\btreat the claim as reasoning\b",
        )),
    ),
    (
        SourceLabel.RETRIEVED_MEMORY.value,
        tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
            r"\bretrieved memory\b",
            r"\bstored .* note\b",
            r"\bmemory vault\b",
            r"\bprior case history\b",
            r"\bearlier case history\b",
            r"\bremembered note\b",
            r"\bearlier stored memory\b",
            r"\bsaved preference\b",
            r"\bprevious .* memory\b",
            r"\blong-term memory\b",
            r"\bsession memory\b",
            r"\blong-run memory\b",
            r"\barchived .* log\b",
            r"\brecalled .* file\b",
            r"\bsaved incident timeline\b",
            r"\bstored memory\b",
            r"\bcase history, not the current event\b",
            r"\brecalled prior record\b",
            r"\bprovenance is retrieved memory\b",
        )),
    ),
    (
        SourceLabel.TOOL_OUTPUT.value,
        tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
            r"\breturned [a-z_]+=",
            r"\bresponse status=\d+",
            r"\bcurl printed\b",
            r"\bsql query produced\b",
            r"\bkubectl described\b",
            r"\bservice response included\b",
            r"\bexport listed\b",
            r"\bgrep found\b",
            r"\bapi yielded\b",
            r"\bjob ended with exit code\b",
            r"\bendpoint returned\b",
            r"\bjson listed\b",
            r"\bquery output from\b",
            r"\bscript result\b",
            r"\bapi payload\b",
            r"\bcommand stdout\b",
            r"\braw .* payload\b",
            r"\bstdout:",
            r"\bmachine result only\b",
            r"\breturned json\b",
        )),
    ),
    (
        SourceLabel.EXTERNAL.value,
        tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
            r"\brecorded that\b",
            r"\bwatched\b",
            r"\bobserved\b",
            r"\bsaw\b",
            r"\bmeasured\b",
            r"\bscanned\b",
            r"\bshowed\b",
            r"\bwrote\b",
            r"\bemailed\b",
            r"\bsaid on the call\b",
            r"\bstates\b",
            r"\bsigned .* states\b",
            r"\bnotes\b",
            r"\bdirect observation from\b",
            r"\bentry says\b",
            r"\bcase file records that\b",
            r"\benvironment report states\b",
            r"\bdirectly observed\b",
            r"\bsource is the direct field note\b",
            r"\bcaptured .* during the live event\b",
        )),
    ),
)


_FABRICATED_TOKENS = frozenset({
    "fabricated",
    "rumor",
    "unverified",
    "uncertain",
    "hallucinated",
    "adversarial",
    "anonymous",
})

_SIMULATION_TOKENS = frozenset({
    "hypothetical",
    "simulated",
    "simulation",
    "counterfactual",
})

_INFERENCE_TOKENS = frozenset({
    "infer",
    "infers",
    "inferred",
    "deduce",
    "deduced",
    "concluded",
    "implies",
    "suggests",
    "therefore",
    "thus",
    "consequently",
})

_RETRIEVED_TOKENS = frozenset({
    "remember",
    "remembered",
    "recalled",
    "retrieved",
    "prior",
    "previous",
    "previously",
    "earlier",
})

_STRONG_TOOL_TOKENS = frozenset({
    "api",
    "cli",
    "json",
    "stdout",
    "stderr",
    "exit_code",
    "status",
    "query",
    "search",
    "tool",
})

_WEAK_TOOL_TOKENS = frozenset({
    "returned",
    "returning",
    "found",
})

_INFERENCE_PHRASES = (
    " should ",
    " likely ",
    " caused ",
    " is the likely ",
    " is the safest ",
    " safer than ",
    " necessary before ",
    " priority fix",
    " recommendation",
    " root cause",
)

_SIMULATION_PHRASES = (
    " could ",
    " would ",
    " might ",
    " with twice ",
    " offering ",
    " blocking ",
    " delaying ",
    " initiating ",
    " synthetic negatives ",
)

_RETRIEVED_PHRASES = (
    " last quarter",
    " last time",
    " last month",
    " in the prior ",
    " in prior ",
)

_UNCERTAIN_PHRASES = (
    " history of fraud",
    " former employee",
    " covenant breach",
    " skipped morning medication",
    " judge dislikes",
    " benchmark is already saturated",
    " broke production",
    " should not be refunded",
)

_TOOL_PHRASES = (
    "crashloopbackoff",
    "charge_count",
    "mfa_enabled",
    "available balance",
    "active prescriptions",
    "indemnity references",
    "recall@",
)

_KEY_VALUE_RE = re.compile(r"\b[a-z][a-z0-9_]{2,}\s*=\s*[^,\s]+", re.IGNORECASE)
_METRIC_RE = re.compile(r"\b[a-z_]+@\d+\b", re.IGNORECASE)
_COUNT_RESULT_RE = re.compile(
    r"\b\d+\s+(pods|service accounts|active prescriptions|indemnity references)\b",
    re.IGNORECASE,
)
_MODAL_MAY_RE = re.compile(
    r"\bmay\s+(disrupt|reduce|improve|understate|expose|close|prevent|lower)\b",
    re.IGNORECASE,
)


def _looks_like_external_observation(content: str) -> bool:
    text = content.lower()
    return (
        "external observation:" in text
        or " recorded " in f" {text} "
        or text.startswith(("lab result ", "call log ", "audit log ", "signed contract "))
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text.lower()} "
    return any(phrase in padded for phrase in phrases)


def _looks_like_tool_output(content: str, tokens: set[str]) -> bool:
    text = content.lower()
    if tokens & _STRONG_TOOL_TOKENS:
        return True
    if _KEY_VALUE_RE.search(content) or _METRIC_RE.search(content):
        return True
    if _COUNT_RESULT_RE.search(content):
        return True
    if _contains_any(text, _TOOL_PHRASES):
        return True
    if " response" in text and ("api" in text or "http" in text or "status" in text or "json" in text):
        return True
    # "returned" and "found" are ambiguous in natural prose; require a
    # machine-shaped companion signal before treating them as tool output.
    return bool(tokens & _WEAK_TOOL_TOKENS) and (
        "=" in content
        or "{" in content
        or "}" in content
        or any(token.endswith("_id") for token in tokens)
    )


def _predict_combined_product(case: SourceInferenceCase) -> str:
    """Product default for Source(.) inference.

    The upstream validation policy is a structured-fixture feasibility floor.
    This product policy keeps the same policy name but tightens the boundary
    for realistic ingestion text: explicit low-trust markers win first, weak
    tool markers require machine-shaped evidence, and otherwise suspicious
    unverified/conditional/remembered/inferred phrasing is preferred over a
    high-trust external default.
    """
    content = case.content
    lower = content.lower()
    tokens = _tokens(content)

    # Explicit low-trust boundary labels should never be upgraded by later
    # feature heuristics.
    if tokens & _FABRICATED_TOKENS:
        return SourceLabel.FABRICATED_OR_UNCERTAIN.value
    if tokens & _SIMULATION_TOKENS:
        return SourceLabel.SIMULATION.value
    if tokens & _INFERENCE_TOKENS:
        return SourceLabel.INFERENCE.value
    if tokens & _RETRIEVED_TOKENS:
        return SourceLabel.RETRIEVED_MEMORY.value

    if _looks_like_external_observation(content):
        return SourceLabel.EXTERNAL.value

    # Tool output needs a stronger signal than ordinary verbs such as
    # "returned" or "response", which often appear in prose.
    if _looks_like_tool_output(content, tokens):
        return SourceLabel.TOOL_OUTPUT.value

    # Semantic soft markers. These are intentionally conservative: they
    # demote to lower-trust sources when text looks like a claim about an
    # inference, hypothetical branch, prior memory, or unverified allegation.
    if _contains_any(lower, _UNCERTAIN_PHRASES):
        return SourceLabel.FABRICATED_OR_UNCERTAIN.value
    if _contains_any(lower, _SIMULATION_PHRASES) or _MODAL_MAY_RE.search(content):
        return SourceLabel.SIMULATION.value
    if _contains_any(lower, _RETRIEVED_PHRASES):
        return SourceLabel.RETRIEVED_MEMORY.value
    if _contains_any(lower, _INFERENCE_PHRASES):
        return SourceLabel.INFERENCE.value

    return _FGM_POLICIES["feature_threshold"](case)


POLICIES = dict(_FGM_POLICIES)
POLICIES["combined_legacy"] = _FGM_POLICIES["combined"]
POLICIES["combined"] = _predict_combined_product


class _TransformerSourceClassifier:
    """Lazy optional Hugging Face classifier wrapper."""

    def __init__(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "policy='trained_transformer' requires optional dependencies. "
                "Install trace-memory[source-classifier] or install torch and "
                "transformers, then set TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL."
            ) from exc

        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(
                "TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL points to a missing path: "
                f"{model_path!r}"
            )
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def predict(self, content: str) -> str:
        with self._torch.no_grad():
            encoded = self._tokenizer(
                content,
                truncation=True,
                max_length=192,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            output = self._model(**encoded)
            pred_idx = int(output.logits.argmax(dim=-1).detach().cpu()[0])
        label = str(self._model.config.id2label[pred_idx])
        SourceLabel(label)
        return label


def _get_trained_source_classifier() -> _TransformerSourceClassifier:
    global _TRAINED_SOURCE_CLASSIFIER, _TRAINED_SOURCE_CLASSIFIER_PATH
    model_path = os.environ.get("TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL", "").strip()
    if not model_path:
        raise RuntimeError(
            "policy='trained_transformer' requires TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL "
            "to point at a saved Hugging Face source-classifier model directory."
        )
    if _TRAINED_SOURCE_CLASSIFIER is None or _TRAINED_SOURCE_CLASSIFIER_PATH != model_path:
        _TRAINED_SOURCE_CLASSIFIER = _TransformerSourceClassifier(model_path)
        _TRAINED_SOURCE_CLASSIFIER_PATH = model_path
    return _TRAINED_SOURCE_CLASSIFIER


def _predict_trained_transformer(case: SourceInferenceCase) -> str:
    model_label = _get_trained_source_classifier().predict(case.content)
    return _apply_trained_boundary_guard(case.content, model_label)


def _predict_trained_transformer_raw(case: SourceInferenceCase) -> str:
    return _get_trained_source_classifier().predict(case.content)


def _apply_trained_boundary_guard(content: str, model_label: str) -> str:
    for label, patterns in _TRAINED_GUARD_PATTERNS:
        if any(pattern.search(content) for pattern in patterns):
            return label
    return model_label


POLICIES["trained_transformer_raw"] = _predict_trained_transformer_raw
POLICIES["trained_transformer"] = _predict_trained_transformer


# ---------------------------------------------------------------------------
# LLM-based Source(.) classifier
#
# An LLM classifier slot, callable as ``policy="llm"`` once a classifier
# has been configured via :func:`set_llm_classifier`. The library does
# NOT take a hard dependency on any specific LLM SDK; the configured
# callable can wrap any OpenAI-compatible client (or a stub for tests).
#
# The bundled :class:`LLMSourceClassifier` is a convenience wrapper that
# uses the same constrained-rubric prompt that produced 71% adversarial
# recall on PoisonedRAG NQ (Zou et al., 2024). Custom classifiers can be
# registered by passing any ``Callable[[str], SourceLabel | str]`` to
# :func:`set_llm_classifier`.
# ---------------------------------------------------------------------------


_LLM_CLASSIFIER_SYSTEM_PROMPT = """\
You are a source-trust classifier for retrieved text passages.

Your job is to label each passage with one of six trust categories,
following the ordering high-to-low:

    external                 -- directly observed / measured / well-
                                  attributed external claim. Treat well-
                                  established encyclopaedic facts about
                                  geography, history, science, or public
                                  figures as ``external`` when stated in
                                  a neutral reference style.
    tool_output              -- output of a structured tool query
                                  (database lookup, API response,
                                  monitoring metric).
    retrieved_memory         -- recovered from prior conversation /
                                  internal memory.
    inference                -- derived / reasoned conclusion,
                                  presented as such.
    simulation               -- explicit hypothetical, what-if,
                                  scenario, projection, or simulation
                                  result. Look for "hypothetical",
                                  "simulation", "if we", "would".
    fabricated_or_uncertain  -- unsourced, hedged, rumour-shaped,
                                  contradicts well-established
                                  encyclopaedic knowledge, asserts a
                                  surprising claim with no attribution,
                                  or contains red flags like "recently
                                  revised", "newly discovered", "actually
                                  ... contrary to popular belief".

Tie-breaking rule: when a passage asserts a *surprising or contested*
factual claim without any specific citation or institutional source,
prefer ``fabricated_or_uncertain``. When a passage states a widely
documented fact in a neutral reference style, prefer ``external``.

You will receive one passage. Reply ONLY with a JSON object:

  {"label": "<one of the six labels above>",
   "confidence": <float 0.0-1.0>,
   "rationale": "<one short sentence>"}

No prose, no markdown fences.
"""


_VALID_LLM_LABELS = frozenset({
    SourceLabel.EXTERNAL.value,
    SourceLabel.TOOL_OUTPUT.value,
    SourceLabel.RETRIEVED_MEMORY.value,
    SourceLabel.INFERENCE.value,
    SourceLabel.SIMULATION.value,
    SourceLabel.FABRICATED_OR_UNCERTAIN.value,
})


def _parse_llm_verdict(text: str) -> str | None:
    """Best-effort JSON parse of an LLM classifier response."""
    import json as _json
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = _json.loads(cleaned)
    except _json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            data = _json.loads(m.group(0))
        except _json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    label = str(data.get("label", "")).strip()
    return label if label in _VALID_LLM_LABELS else None


class LLMSourceClassifier:
    """Configurable LLM-based Source(.) classifier.

    Wraps any chat client whose surface matches OpenAI's Python SDK
    (``client.chat.completions.create(...)``). The bundled prompt is
    a constrained-rubric "label-only-with-confidence" template that
    produced 71% adversarial recall on PoisonedRAG NQ with
    ``gpt-4.1-mini``. Callers can override the system prompt for
    domain-specific tuning.

    Caches by exact passage text within the instance. Construct one
    classifier and reuse across calls / agents to amortise cost.

    Usage::

        from openai import OpenAI
        from trace_memory import LLMSourceClassifier, set_llm_classifier

        classifier = LLMSourceClassifier(OpenAI(), model="gpt-4.1-mini")
        set_llm_classifier(classifier)
        record = agent.add_with_inferred_source(text, policy="llm")
    """

    DEFAULT_SYSTEM_PROMPT = _LLM_CLASSIFIER_SYSTEM_PROMPT

    def __init__(
        self,
        client,
        *,
        model: str = "gpt-4.1-mini",
        system_prompt: str | None = None,
        max_output_tokens: int = 120,
        timeout_s: float = 30.0,
        cache: bool = True,
    ):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s
        self._cache: dict[str, SourceLabel] | None = {} if cache else None

    def classify(self, text: str) -> SourceLabel:
        """Return the predicted source label for ``text``."""
        key = text.strip()
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"Passage:\n{text}"},
            ],
            max_tokens=self._max_output_tokens,
            temperature=0.0,
            timeout=self._timeout_s,
        )
        content = (response.choices[0].message.content or "").strip()
        label = _parse_llm_verdict(content) or SourceLabel.FABRICATED_OR_UNCERTAIN.value
        out = SourceLabel(label)
        if self._cache is not None:
            self._cache[key] = out
        return out

    def __call__(self, text: str) -> SourceLabel:
        return self.classify(text)


_LLM_CLASSIFIER: Callable[[str], SourceLabel] | None = None


def set_llm_classifier(
    classifier: Callable[[str], SourceLabel | str] | None,
) -> None:
    """Configure the LLM-based Source(.) classifier.

    Pass any callable that accepts a passage string and returns a
    :class:`SourceLabel` (or the corresponding string value). Pass
    ``None`` to clear the configured classifier. While unset,
    ``policy="llm"`` will raise on use.

    Constructing :class:`LLMSourceClassifier` does NOT register it
    automatically -- call :func:`set_llm_classifier` explicitly. This
    keeps test isolation possible and matches the
    ``TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL`` pattern used by the
    trained-transformer policy.
    """
    global _LLM_CLASSIFIER
    _LLM_CLASSIFIER = classifier


def _predict_llm(case: SourceInferenceCase) -> str:
    classifier = _LLM_CLASSIFIER
    if classifier is None:
        raise RuntimeError(
            "policy='llm' requires a configured classifier. Call "
            "trace_memory.set_llm_classifier(callable) first. The "
            "callable should map passage text -> SourceLabel. The "
            "bundled trace_memory.LLMSourceClassifier wraps an "
            "OpenAI-compatible client with a constrained-rubric prompt."
        )
    label = classifier(case.content)
    if isinstance(label, SourceLabel):
        return label.value
    return str(label)


POLICIES["llm"] = _predict_llm


def infer_source(
    content: str,
    *,
    query_context: str = "",
    retrieval_margin: float = 0.0,
    recency_rank: int = 0,
    policy: str = "combined",
) -> SourceLabel:
    """Infer a source class from content and retrieval features.

    Parameters
    ----------
    content :
        The text whose source is to be inferred.
    query_context :
        The query that surfaced this content (if any). Reserved for
        future policies; ignored by the current ``combined`` policy.
    retrieval_margin :
        How distinguishable this content was at retrieval time
        (top-score minus next-best-distractor). Higher margin tends to
        correlate with external observations under the feature
        thresholds; lower margin with derivations or reactivations.
    recency_rank :
        ``0`` for the most recent record, increasing with age. Recent
        records tend to be external input; older records tend to be
        reactivated memory.
    policy :
        One of ``"uniform_external"``, ``"lexical_rules"``,
        ``"feature_threshold"``, ``"combined_legacy"``, ``"combined"``,
        ``"trained_transformer_raw"``, or ``"trained_transformer"``.
        Defaults to ``"combined"``, the product policy measured by the
        source-boundary benchmark. ``"trained_transformer"`` is a guarded
        optional integration path gated by
        ``TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL``.

    Returns
    -------
    SourceLabel
        The predicted source class.

    Notes
    -----
    The default policy is rule-based; the ``"trained_transformer"`` policies
    use the optional measured classifier. Neither path is a general
    source-grounding solution. See the source-boundary benchmark and ledger
    section 1.6 limits.
    """
    if policy not in POLICIES:
        raise ValueError(
            f"Unknown source inference policy: {policy!r}. "
            f"Available: {sorted(POLICIES)}"
        )
    case = SourceInferenceCase(
        case_id="_runtime",
        content=content,
        query_context=query_context,
        retrieval_margin=float(retrieval_margin),
        recency_rank=int(recency_rank),
        true_source="",  # unused at inference time
    )
    predicted_str = POLICIES[policy](case)
    return SourceLabel(predicted_str)


def warn_natural_prose() -> None:
    """Emit the natural-prose performance warning.

    Called by ``add_with_inferred_source(...)`` on the agent to remind
    callers that the benchmarked product floor does not remove the need for
    explicit source labels on high-stakes natural-prose ingestion.
    """
    warnings.warn(
        "Source(.) inference is a measured product floor, not a general "
        "source-grounding solution. For high-stakes natural-prose ingestion, "
        "supply explicit source labels via add(source=...) or measure the "
        "policy against your own labelled data first. See the source-boundary "
        "benchmark and ledger section 1.6 limits.",
        stacklevel=3,
    )


async def ainfer_source(
    content: str,
    *,
    query_context: str = "",
    retrieval_margin: float = 0.0,
    recency_rank: int = 0,
    policy: str = "combined",
) -> SourceLabel:
    """Async wrapper over :func:`infer_source`.

    Offloads classification to the default threadpool via
    ``asyncio.to_thread``. The default rule-based policy is cheap; the
    optional transformer policy may perform model inference.
    """
    return await asyncio.to_thread(
        infer_source,
        content,
        query_context=query_context,
        retrieval_margin=retrieval_margin,
        recency_rank=recency_rank,
        policy=policy,
    )


__all__ = [
    "LLMSourceClassifier",
    "SourceInferencePolicy",
    "ainfer_source",
    "infer_source",
    "set_llm_classifier",
    "warn_natural_prose",
]

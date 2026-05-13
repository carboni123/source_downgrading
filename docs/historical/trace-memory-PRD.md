# trace-memory: Product Requirements Document

**Status.** Draft, 2026-05-11. Source of truth for the v0.1 MVP scope.

**One-line summary.** A Python SDK that gives LLM agent developers a memory layer with operationally validated trust composition: every stored record carries a source label, derived records cannot launder their inputs' trust, retrieval returns operationally-meaningful memory (not just similar text), and belief revisions preserve full lineage.

**Repository.** This product lives in the `trace-memory` repository (separate from `trace-memory-architecture`, which remains the validation/research repo). Initial release is `pip install trace-memory`. v0.1 layers a developer-facing facade on top of the `FGMAgent` implementation that already runs the validation harness in `trace-memory-architecture`; whether the product repo vendors those modules or depends on `trace-memory-architecture` as a library is an open Phase 1 decision (see §12).

---

## 1. Product Vision

LLM agents need memory. Existing memory frameworks (RAG, MemGPT, Generative Agents) treat memory as "store text, retrieve similar text," and they uniformly attach source metadata that nothing in the pipeline consults to constrain trust composition. The result is a known failure: a fabricated detail gets stored, retrieved, reasoned over, and the reasoning product is treated downstream as observed fact.

`trace-memory` is a memory layer that fixes this at the inscription level. Every primitive in the layer has a validation result file behind it and a paired test in CI. The library promises only what is grounded. The promise is small, and the promise is true.

**The pitch to an agent developer:** "Your agent should be able to remember things, derive new things from them, and trust the derived things less than the originals. We give you a primitive layer that enforces this and audits itself for laundering. If you build on top of us, you don't have to write trust-composition rules yourself."

**What this is not.** Not a vector database. Not a RAG framework. Not a chat memory wrapper. `trace-memory` integrates with vector backends and embedding models but is opinionated about a layer most of those tools leave to the caller: how trust composes across derivation.

---

## 2. Goals and Non-Goals

### Goals

1. Ship a Python library that exposes the nine §1 validated primitives via a documented API.
2. Document the validation evidence behind each primitive in the user-facing README, with pointers to result files.
3. Provide a working example agent (built on the library) that demonstrates anti-laundering end-to-end on the laundering fixture.
4. Be installable, importable, and usable inside an LLM agent's transition loop in under one hour of integration work.
5. Run on a single Python process with pluggable embeddings, transition functions, and storage backends.
6. Pass all existing 204 non-live tests in CI on every PR.

### Non-Goals (v0.1)

1. Vector-DB-scale retrieval (millions of records). Use the in-memory `MemoryStore` for MVP; pluggable backends arrive in v0.2.
2. Compression / forgetting policies. This is in the ledger §2; v0.1 grows monotonically.
3. Multi-process or distributed deployment. Single-process per agent instance.
4. HTTP service / SaaS. Library only.
5. Active selection ("automatically surface relevant memories without a query"). Architectural gap; not in any tier.
6. Cryptographic multi-tenant isolation. Engineered self-index binding (§1.8) is metadata-supplied; production tenancy requires additional layers we do not provide.
7. Adversarial robustness guarantees. The validation does not cover adversarial inputs.
8. Source(·) inference on natural prose. v0.1 ships the rule-based combined policy as a feasibility floor; users with natural-prose ingestion should supply explicit source labels.

---

## 3. Target User and Use Case

### Primary user

A developer building an autonomous LLM agent in Python. They have:
- An LLM provider (OpenAI, Anthropic, or local).
- A task loop where the agent receives input, decides on an action, and updates state.
- A need to remember context across turns and to reason over remembered context.
- A concern about hallucination compounding: today, when their agent reasons over retrieved content, the reasoning product re-enters memory with no trust degradation.

### Primary use case

The developer adds `trace-memory` to their agent loop. On each turn:
1. They write the agent's observations into the memory layer with explicit source labels (`external` for sensor/tool output, `simulation` for hypotheticals, etc.).
2. Before deciding on an action, they query the memory layer for relevant prior records. The layer returns records ranked by *fold-force* (whether folding them into the current transition would change the agent's decision), not just by semantic similarity.
3. When the agent generates a new inference from retrieved records, they write the inference via the derived-record API. The layer automatically caps the inference's source at the minimum trust of contributing inputs and propagates provenance.
4. When the agent's beliefs are revised by new evidence, they record a correction-chain node with full lineage.
5. They can audit the agent's inscription decisions for laundering at any time, with paired self-vs-truth metrics.

### Secondary use cases (best-effort in v0.1)

- Internal Tyxter consulting workflow: capture and reason over client engagement notes with provenance.
- Research notebook: track derivations from primary sources without losing the source chain.
- Multi-agent system: distinguish between agents' externally observed facts and other agents' inferences.

---

## 4. Functional Requirements

Each requirement maps to a `§1` entry in `VALIDATED_PRIMITIVES_LEDGER.md`. Implementation details for each are in the existing `src/fgm/` modules.

### FR-1. Source-labeled ingestion (maps to ledger §1.1)

The library MUST accept content with one of seven source labels: `external`, `tool_output`, `retrieved_memory`, `inference`, `simulation`, `fabricated_or_uncertain`, `operation_record`. The label is preserved through storage, retrieval, fold, operation-memory write, and audit-log replay.

**API surface:**
```python
agent.add(
    content: str,
    *,
    source: SourceLabel,
    provenance: Sequence[str] = (),
    source_confidence: float = 1.0,
    self_index: SelfIndex | None = None,
) -> MemoryRecord
```

### FR-2. Fold-force-based retrieval (maps to ledger §1.2)

The library MUST return retrieved records with their `fold_force` computed via paired ablation: the difference in the transition function's output with and without the record's fold vector, restricted to caller-declared cognitive dimensions.

**API surface:**
```python
result = agent.query(query: str, k: int = 3) -> FoldResult
# result.retrieved: list of (record, fold_force, gated) tuples
# result.fold_force_total: aggregate
```

### FR-3. Source-sensitive routing (maps to ledger §1.3)

The library MUST select a write target for each attended-and-folded candidate from `{null, trace, durable, operation, correction, quarantine}`. Untrusted sources are routed to quarantine; trusted sources are routed by fold-force and correction status.

**API surface:**
```python
result.selected_route: Route  # exposed on FoldResult
result.route_scores: dict[Route, float]
```

The default routing policy is the rule-based `_score_routes` validated in §1.3. The library MUST allow callers to substitute their own routing policy, but defaults to the validated one.

### FR-4. Source-downgrading inscription for derived records (maps to ledger §1.4)

The library MUST provide an explicit derived-record API. When a derivation is written, the library:
- Computes `Source(r) = Trust_ceil({Source(c_i)})` from contributing inputs.
- Propagates provenance transitively (`Prov(r) = ⋃_i (Prov(c_i) ∪ {id(c_i)})`).
- Refuses to accept a caller-supplied source label for derivations (the label is computed, not asserted).

**API surface:**
```python
agent.add_derived(
    content: str,
    *,
    inputs: Sequence[MemoryRecord | str],  # record objects or record ids
    self_index: SelfIndex | None = None,
) -> MemoryRecord
```

**Negative requirement.** `agent.add(...)` with default arguments MUST NOT silently label content as `external`. The default in v0.1 is to *require* an explicit `source=` argument; calling `add(content)` without it raises `MissingSourceError`. This closes the naive-inscription attack surface at the API level.

### FR-5. Inscription utility under budget (maps to ledger §1.5)

The library MUST support a write-time inscription policy that ranks candidates by predicted future utility and writes only the top-k under a configured budget. The default policy is "always-write" (no budget pressure). Callers can opt into utility-based inscription via configuration.

**API surface:**
```python
agent = MemoryAgent(
    ...,
    inscription_policy=UtilityWritePolicy(budget=N, scorer=...),
)
```

### FR-6. Source(·) inference for unlabeled content (maps to ledger §1.6)

The library MUST provide a content-based source classifier that operates without access to a stored label. The classifier is the `combined` policy validated in §1.6 (lexical markers + feature-threshold fallback). It is a feasibility floor, not a production-grade classifier — this is documented in the API.

**API surface:**
```python
predicted_source = trace_memory.infer_source(
    content: str,
    *,
    query_context: str = "",
    retrieval_margin: float = 0.0,
    recency_rank: int = 0,
) -> SourceLabel

# Or via convenience:
agent.add_with_inferred_source(content) -> MemoryRecord
```

The convenience API MUST emit a warning (or require explicit opt-in) when inference is used, because the natural-prose performance is not guaranteed.

### FR-7. Correction-chain node schema (maps to ledger §1.7)

The library MUST provide a belief-revision API that records the full node schema: prior belief, evidence, update operation, revised belief, delta, self-index, provenance, confidence.

**API surface:**
```python
agent.revise_belief(
    *,
    prior_belief: str,
    evidence: str,
    update_operation: str,
    revised_belief: str,
    delta: str,
    self_index: SelfIndex | None = None,
    provenance: Sequence[str] = (),
    confidence: float,
) -> CorrectionNode
```

### FR-8. Engineered self-index binding (maps to ledger §1.8)

The library MUST support `SelfIndex` metadata `(user_id, project_id, role, permission_scope, standing_commitment)` on every stored record. Retrieval MUST filter by the active self-index; cross-project / cross-user leakage MUST be prevented at the retrieval layer.

**API surface:**
```python
@dataclass(frozen=True)
class SelfIndex:
    user_id: str | None = None
    project_id: str | None = None
    role: str | None = None
    permission_scope: str | None = None
    standing_commitment: str | None = None

agent = MemoryAgent(..., self_index=SelfIndex(user_id="alice", project_id="X"))
```

The library MUST NOT claim cryptographic tenant isolation. This is documented; production deployments are responsible for additional access controls.

### FR-9. Cascade-invisibility-aware auditing (maps to ledger §1.9)

The library MUST expose an audit API that returns paired metrics: a self-referential rate (computed from stored source labels) and a truth-grounded rate (when ground-truth labels are available, e.g., in test runs or supervised ingestion).

**API surface:**
```python
audit = agent.audit_laundering(
    *,
    truth_labels: dict[str, SourceLabel] | None = None,
) -> LaunderingAudit
# audit.local_rate, audit.truth_grounded_rate (or None), audit.gap
```

The library MUST emit a warning in the audit output when only the local rate is computed, because cascade-invisibility means it undercounts.

---

## 5. Non-Functional Requirements

### NFR-1. Pluggability

The library MUST allow callers to substitute:
- Embedding function (`embed_fn: Callable[[str], np.ndarray]`)
- Transition function (`transition_fn: Callable[[state, input, fold], state]`)
- Storage backend (defaults to in-memory; `Storage` protocol)
- Routing policy (defaults to the validated `_score_routes`)
- Inscription policy (defaults to always-write)

Defaults MUST be the validated implementations.

### NFR-2. Determinism

When the embedding function, transition function, and storage are deterministic, the entire memory pipeline MUST be deterministic given a fixed seed. This is required for the validation harnesses to remain reproducible.

### NFR-3. Latency

A `query(...)` call against an in-memory store of up to 10,000 records SHOULD complete in under 50 ms with default hash embeddings, and under 500 ms with sentence-transformer embeddings, on a developer laptop. (Soft target; will measure during v0.1 development.)

### NFR-4. Test coverage

Every public API method MUST have at least one regression test. The full pytest suite (`pytest -m "not live"`) MUST pass on every PR. Live tests (`-m live`) MUST pass on a weekly schedule with stable provider availability.

### NFR-5. Documentation

The README MUST:
- State the validated-primitives-only promise up front.
- Link to `VALIDATED_PRIMITIVES_LEDGER.md`.
- List explicit non-guarantees (compression, scale, natural-prose source inference, adversarial robustness).
- Provide a complete working example using all nine §1 primitives.

### NFR-6. Versioning

Public API stability follows semver. v0.x is pre-stable; breaking changes are allowed but documented in `CHANGELOG.md`. The `0.1.0` release happens when all §1 primitives are reachable via documented API and the example agent works end-to-end.

### NFR-7. Dependencies

Minimum dependency set: `numpy`. Optional dependencies for substrate plug-ins:
- `sentence-transformers` for real embeddings.
- `anthropic` and `openai` for live LLM transitions.
- `python-dotenv` for env loading in dev.

Live-LLM and real-embedding paths MUST work without their dependencies installed (graceful degradation to hash embeddings / echo transitions, with a clear message).

---

## 6. API Surface (high level)

```python
from trace_memory import (
    MemoryAgent,
    MemoryRecord,
    CorrectionNode,
    FoldResult,
    Route,
    SourceLabel,
    SelfIndex,
    LaunderingAudit,
    MissingSourceError,
    # advanced:
    UtilityWritePolicy,
    Storage,
    infer_source,
)

# Construction
agent = MemoryAgent(
    embed_fn=...,
    transition_fn=...,
    storage=None,                 # defaults to in-memory
    self_index=SelfIndex(...),    # optional
    inscription_policy=None,      # defaults to always-write
    routing_policy=None,          # defaults to validated source-sensitive
)

# Writes
agent.add(content, source=SourceLabel.EXTERNAL, provenance=[...])
agent.add_derived(content, inputs=[r1, r2])
agent.add_with_inferred_source(content)  # opt-in convenience
agent.revise_belief(prior_belief=..., evidence=..., ...)

# Reads
result: FoldResult = agent.query(query, k=3)

# Audit
audit: LaunderingAudit = agent.audit_laundering(truth_labels=None)
```

This surface is intentionally narrow. Advanced operations (custom storage backends, custom routing policies, custom inscription policies) are available but not in the import-list-at-a-glance.

---

## 7. Constraints From the Ledger

This section enumerates what the library MUST NOT promise, with pointers to the ledger entries that justify the limit. The README will reproduce a shorter version of this list as the "What we don't promise" section.

| Constraint | Source | Why |
|---|---|---|
| No retrieval guarantees at scale (>10k records) | not in §1 | Untested at scale |
| No compression / forgetting | ledger §2 only (not §1) | Compressor exists in code but is not validated |
| `Source(·)` performance on natural prose is unknown | ledger §1.6 limits | Fixture is structured; natural prose untested |
| No long-horizon stability claim | not in §1 | No run longer than ~5 turns has been validated |
| No active recall ("automatically surface without query") | not in any tier | Architectural gap |
| No cryptographic tenant isolation | ledger §1.8 limits | Metadata-supplied only |
| No adversarial robustness | not in §1 | Untested |
| No MAFC spectral-radius bounds | ledger §3.3 | Theoretical |

---

## 8. Success Metrics

The v0.1 MVP is successful if:

1. **Installable.** `pip install trace-memory` works on Python 3.10+ on Linux, macOS, Windows.
2. **Importable in under one hour.** A developer reading the README can integrate the library into a toy agent in under one hour, measured by a follow-along quickstart.
3. **Validated primitives reachable.** All nine §1 primitives have at least one documented API call and a corresponding test in `tests/test_public_api.py`.
4. **Example agent works.** An example agent built on the library demonstrates anti-laundering on the existing laundering fixture, with paired audit output.
5. **CI green.** 204+ tests pass on every commit to master.
6. **Documentation honest.** The README's "What we don't promise" section reproduces the ledger constraints.
7. **At least one external user.** At least one developer outside Tyxter installs the library, reports a use case, and provides feedback. This can be a colleague or collaborator; it is not "thousands of downloads."

---

## 9. Risks

### R-1. Source(·) on natural prose underperforms

**Likelihood:** high.
**Impact:** medium — users who rely on `add_with_inferred_source` for natural prose may get poor results.
**Mitigation:** Default API requires explicit source. Inferred-source path emits warnings. Document the natural-prose unknown prominently.

### R-2. Fold-force computation latency

**Likelihood:** medium.
**Impact:** medium — fold-force requires a paired ablation per retrieved record; with LLM-based transitions this is expensive.
**Mitigation:** Provide a `fold_force_threshold` short-circuit. Allow callers to use cheap proxy transitions during retrieval and full transitions only at decision time.

### R-3. Multi-process / multi-tenant misuse

**Likelihood:** medium.
**Impact:** high — engineered self-index binding is not cryptographic; misuse could leak data.
**Mitigation:** Documentation. Refuse to claim cryptographic guarantees. Recommend external access-control layers.

### R-4. In-memory storage grows unboundedly

**Likelihood:** high in long-running agents.
**Impact:** medium — process eats RAM.
**Mitigation:** Document the limit. Provide hooks for callers to checkpoint/purge externally. Compression arrives in v0.2 or later, after validation.

### R-5. API churn between v0.1 and v0.2

**Likelihood:** high.
**Impact:** low — pre-1.0 versioning makes this acceptable.
**Mitigation:** Versioned API, CHANGELOG, deprecation warnings before removal.

### R-6. The "memory layer" framing is unfamiliar to RAG-trained developers

**Likelihood:** medium.
**Impact:** medium — developers may expect a RAG library and bounce when they realize it isn't one.
**Mitigation:** Clear positioning in README ("this is not a RAG framework"). Show the differential value via the anti-laundering example.

---

## 10. Phased Delivery

### Phase 1: v0.1 alpha (target: 4–6 weeks)

**Scope:** FR-1 through FR-4, FR-7, FR-9. (Source labels, fold-force retrieval, routing, derived inscription, correction chains, audit.)

**Out:** FR-5 (utility inscription), FR-6 (source inference), FR-8 (self-index).

**Done when:** all eight primitives in scope are reachable via API; example agent demonstrates anti-laundering end-to-end; README first draft published.

### Phase 2: v0.2 beta (target: +4 weeks after Phase 1)

**Scope:** FR-5, FR-6, FR-8 added.

**Plus:** pluggable storage backend protocol (in-memory remains default; SQLite or simple file backend as a reference implementation).

**Plus:** docstring coverage on all public APIs; type stubs.

**Done when:** all nine primitives shipped; storage backend protocol stable; at least one external user has integrated.

### Phase 3: v0.3 first stable release (target: +4 weeks after Phase 2)

**Scope:** stability, documentation, performance characterization.

**Plus:** measured latency targets (see NFR-3).

**Plus:** at least one alternative storage backend (e.g., SQLite, or a thin wrapper over an existing vector store with the trace-memory layer on top).

**Done when:** version 0.3.0 tagged, published to PyPI, README finalized, real users reachable.

### Beyond v0.3 (not in MVP scope)

- Compression / forgetting (after validation in §1)
- HTTP service wrapper
- Source(·) classifier learned from data (after natural-prose validation)
- Long-horizon validation rig
- Active selection research

---

## 11. Out of Scope (with rationale)

| Item | Why out | When in (if ever) |
|---|---|---|
| Vector-DB-scale retrieval | not validated at scale | After scale validation lands as a new §1 entry |
| Compression / forgetting | ledger §2 only | After compression-rate validation under task pressure |
| HTTP service | engineering scope vs library | v0.4+, after library is stable |
| Multi-process | engineering scope | After single-process is solid |
| Cryptographic tenant isolation | out of architecture scope | Use external access controls instead |
| Active selection without query | architectural gap | After the active-selection primitive is invented and validated |
| Adversarial robustness guarantees | untested | After adversarial validation track exists |
| Source(·) on natural prose, with guarantees | unknown ceiling | After natural-prose validation lands |
| LLM provider abstraction (LiteLLM-style) | scope creep | Caller's responsibility; we ship adapters as examples only |

---

## 12. Open Questions

1. **Source-of-truth for primitive implementations.** The validated primitives live in `trace-memory-architecture/src/fgm/`. Two viable strategies for v0.1: (a) declare `trace-memory-architecture` as a runtime dependency and re-export from `fgm`; (b) vendor the relevant modules into `trace-memory/src/trace_memory/_core/`. Strategy (a) keeps a single source of truth and free upstream upgrades; strategy (b) decouples release cadence and lets the product evolve independently. Recommendation: start with (a), reassess at v0.2 when product-specific concerns emerge.

2. **Storage backend protocol.** What shape? Plugin-based with a strict `Storage` protocol, or a thin layer over an existing vector DB SDK? Decide before Phase 2.

3. **Source(·) classifier upgrade path.** When natural-prose validation closes, do we ship a learned classifier in the library, or as an optional package? Likely optional to keep core dependencies small.

4. **Async API.** Should the public API be sync-only in v0.1, or expose async variants? Async is more idiomatic for LLM agent code but doubles the surface. Recommendation: sync-only in v0.1, add async wrappers in v0.2 if requested.

5. **Telemetry.** Do we want anonymous usage telemetry to learn what's used? Recommendation: no in v0.1. Opt-in mechanism considered for v0.3.

6. **License.** MIT (consistent with `trace-memory-architecture`). Confirmed for the published library.

7. **Pricing model (if any).** Library is open-source; commercial overlay (managed service, support, certification) is out of scope for the PRD but worth flagging for future planning.

---

## 13. Approvals

This PRD is the source of truth for the v0.1 MVP. Changes to scope, target user, or constraints require an update to this document and a re-anchor to the validated primitives ledger.

- Author: Diego Falkowski Carboni / Tyxter
- Date: 2026-05-11
- Source ledger commit: `34c86ee`
- Source roadmap: `ROADMAP.md` Phase 1 (validated primitives) consumed by this PRD; Phases 2-7 remain on the roadmap as post-MVP work.

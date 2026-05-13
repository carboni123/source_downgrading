# Updated Validation Roadmap: From Assumed Sources to Grounded Sources

Archived previous roadmap:

```text
ROADMAP_ARCHIVE_2026-05-11.md
```

This roadmap replaces the original phase plan after merging the primitive
validation work and the source-inference / inference-laundering work into
`master`.

The old roadmap was directionally correct, but it treated source labels as
fixture truth supplied to the agent. The current evidence shows that this is
not enough. The next roadmap must separate three layers:

1. Ground truth labels created by controlled fixtures.
2. Source inference from content, context, retrieval margin, and recency.
3. Trust-preserving inscription rules for derived or inferred records.

The goal is not to claim broad memory validity yet. The goal is to preserve
the primitives that survived controlled validation, expose what failed, and
make the next experiments falsifiable.

## Current Evidence Ledger

### Controlled primitive validation

Evidence:

```text
results/architecture/roadmap_validation_summary.json
results/architecture/controlled_replication_summary.json
results/architecture/source_routing_validation.jsonl
tests/test_validation_primitives.py
```

Grounded enough to build on:

- Explicit source/provenance labels survive add, retrieve, fold, operation
  write, and validation replay.
- Source-sensitive routing beats source-blind routing on the controlled
  fixture.
- Source-blind routing promotes echo content in the fixture; source-sensitive
  routing does not.
- Utility-based inscription beats always-write, never-write, and relevance
  baselines under the controlled budget fixture.
- Correction-chain records beat conclusion-only records on controlled
  transfer and explanation tests.
- Source-aware residual posture suppresses simulated confirmation compared
  with source-blind residual attention.
- Engineered self-index metadata prevents wrong project, user, and role
  application in controlled fixtures.
- Coupled-field probes show nonzero memory-to-attention and
  attention-to-write shifts in toy interventions.

Not yet grounded:

- These controlled fixtures are not real-world distributions.
- Several higher-level primitives still depend on fixture-created truth.
- Success at this layer does not prove emergent continuity, selfhood, or
  open-ended autobiographical memory.

### Source inference and inference laundering

Evidence:

```text
SOURCE_INFERENCE_AND_LAUNDERING.md
results/architecture/laundering_validation_summary.json
results/architecture/laundering_validation.jsonl
results/architecture/source_inference_validation_summary.json
results/architecture/source_inference_validation.jsonl
tests/architecture/test_laundering.py
tests/architecture/test_source_inference.py
src/fgm/laundering.py
src/fgm/source_inference.py
```

Grounded enough to build on:

- `Source(.)` is not purely an interface assumption anymore. On the
  deterministic 30-case fixture, the combined lexical-plus-feature policy
  reaches 0.933 overall source accuracy versus 0.167 for the
  uniform-external baseline.
- False externalization drops from 1.0 under uniform-external labeling to
  0.04 under the combined source inference policy.
- Feature thresholding alone is insufficient. It reaches 0.50 accuracy and
  collapses classes that overlap in retrieval-margin / recency space.
- Lexical rules are strong for marker-heavy classes, but ambiguous cases need
  feature fallback.
- Naive inscription of derived claims is unsafe. It labels derived records as
  `external`, producing a 1.0 derived trust-ceiling violation rate in the
  laundering fixture.
- Provenance propagation alone is necessary but not sufficient. It preserves
  provenance chains but still over-trusts simulation/fabrication-derived
  claims.
- Source downgrading plus provenance propagation is the current grounded
  anti-laundering rule: zero laundering, zero trust-ceiling violations, and
  zero false externalization on the controlled fixture.
- Cascade invisibility is real in the fixture: local laundering detection sees
  fewer failures than truth-grounded ceiling checks after one bad label has
  already been written.

Not yet grounded:

- The source-inference harness is deterministic and small.
- External versus tool-output ambiguity remains visible: both classes are 0.8
  under the combined policy.
- The current source inference policy is rule-based, not learned, calibrated,
  or tested on natural logs.
- The anti-laundering rule exists as a harness policy; it still needs to be
  integrated as a default inscription rule for derived records in the agent
  path.

### Real embedding and live LLM validation

Evidence:

```text
results/architecture/real_component_validation_summary.json
results/architecture/real_embedding_replication_summary.json
results/architecture/live_llm_validation_summary.json
results/architecture/live_llm_replication_summary.json
results/architecture/live_llm_replication_n20_retry_summary.json
results/architecture/live_llm_replication_n20_retry_diagnostics.json
results/architecture/live_provider_model_comparison_summary.json
results/architecture/live_provider_model_comparison_nano_max1000_summary.json
results/architecture/live_task_family_billing_refund_gpt41_summary.json
results/architecture/live_task_family_security_rotation_gpt41_summary.json
```

Grounded enough to build on:

- Real embedding replication reached N=20 with route, retrieval, source, and
  echo-control means at 1.0 on the current fixed-truth task family.
- OpenAI live prompt-suite replication reached N=20 with retries and all route,
  retrieval, source, quarantine, and echo-control gates at 1.0.
- Provider-valid route semantics are necessary. Empty provider responses
  should be tracked as provider-output validity failures, not primitive route
  failures, when retrieval and route labels are otherwise correct.
- `gpt-5-nano` failure was provider-output budget related, not a primitive
  route failure. Raising `max_output_tokens` to 1000 yielded a passing N=5
  artifact.
- Cross-provider smoke evidence exists: OpenAI and Anthropic both passed the
  N=5 provider/model matrix except for the known untuned nano output case.
- Two non-default task families, `billing_refund` and `security_rotation`,
  passed N=5 with `gpt-4.1-mini` and zero diagnostic boundary failures.

Not yet grounded:

- Non-default task families are smoke-level only, not claim-scale. They have
  N=5 live evidence, not N=20.
- Live validation still uses small fixed-truth prompt families.
- Live validation has not yet incorporated derived inference or laundering
  cases.
- Provider/model comparison is not exhaustive and should not be read as a
  model benchmark.

## Ground Truths Reached

These are the claims that currently have direct controlled evidence:

1. Explicit source labels can be preserved through the FGM storage, retrieval,
   fold, operation-memory, and JSONL replay path.
2. Source-sensitive routing prevents fixture-defined simulation/fabrication
   from being promoted where source-blind routing promotes it.
3. A memory is operationally meaningful only when paired with a nonzero
   with-memory versus without-memory transition effect.
4. Utility-based inscription can beat degenerate write policies under budget
   pressure in fixed future-task fixtures.
5. Source downgrading is required for safe inscription of derived claims.
6. Provenance propagation is required but not sufficient for safe derived
   inscription.
7. Local laundering detection can undercount failures after a bad source label
   has already been written; truth-grounded ceiling checks are required.
8. Source inference is feasible on structured cases, but retrieval features
   alone are insufficient.
9. Provider-empty live responses are a separate failure plane from primitive
   retrieval/routing.
10. Source-aware reranking can fix a concrete polarity/confusability boundary
    where semantic top-1 retrieval selected the wrong record.

These are not yet ground truths:

- General source inference on natural data.
- General anti-laundering robustness under open-ended model generations.
- Claim-scale live validation across multiple task families.
- Learned routing or learned inscription policies.
- Long-horizon continuity, selfhood, or MAFC stability.

## Updated Build Order

```text
evidence ledger
  -> source inference hardening
  -> derived-record inscription / anti-laundering integration
  -> route and fold validation with inferred sources
  -> task-family claim-scale live validation
  -> operation/correction chains under laundering pressure
  -> residual attention and self-index under inferred-source uncertainty
  -> coupled-field dynamics only after inferred-source gates hold
```

The old order started with supplied source labels. The new order starts with
earning and preserving source labels.

## Phase 1: Consolidated Evidence Ledger

Status: mostly done.

Deliverables:

- Keep `VALIDATION_ROADMAP_EXECUTION.md` as the detailed run ledger.
- Keep `SOURCE_INFERENCE_AND_LAUNDERING.md` as the focused analysis note.
- Keep this `ROADMAP.md` as the forward plan.
- Archive superseded plans rather than editing history in place.

Acceptance:

- Every grounded claim in this roadmap maps to a result file, test file, or
  source file.
- Stale branch/worktree references are removed or clearly marked historical.
- Full non-live tests pass after doc consolidation.

## Phase 2: Harden `Source(.)` Inference

Objective:

Move source inference from a small deterministic fixture toward a reusable
primitive that can be scored from logs.

Work:

- Expand source-inference fixtures beyond 30 cases.
- Add ambiguous external/tool-output cases with explicit adjudication rules.
- Add adversarial cases where content markers conflict with context features.
- Add confidence or margin output to source inference policies.
- Log source inference inputs and predictions as JSONL records.
- Add confusion-focused metrics, not only aggregate accuracy.

Acceptance:

- `combined` still beats `uniform_external` by at least 0.30 overall.
- `combined` false externalization remains below 0.10 on controlled fixtures.
- Each source class has an explicit per-class threshold.
- External/tool-output ambiguity is reported separately rather than hidden in
  the aggregate score.

## Phase 3: Integrate Anti-Laundering Into Derived Inscription

Objective:

Make source downgrading and provenance propagation part of the agent path for
derived records, not only a standalone harness policy.

Work:

- Add an explicit derived-record inscription API.
- Require contributing record IDs when writing an inferred record.
- Compute derived source with the source-downgrading rule.
- Preserve transitive provenance.
- Reject or quarantine derived writes with missing provenance.
- Add route tests for derived claims that originate from simulation or
  fabricated inputs.

Acceptance:

- Naive derived inscription remains available only as a baseline or test
  fixture, not as the default path.
- Source downgrading yields zero trust-ceiling violations on the laundering
  fixture.
- Later queries over derived records do not false-externalize
  simulation/fabrication-derived claims.
- Provenance walk-back depth is nonzero for every derived record.

## Phase 4: Route and Fold Validation With Inferred Sources

Objective:

Retest routing, fold-force, operation-memory, and correction-chain behavior
when source labels are inferred rather than supplied.

Work:

- Run the first concrete build target with inferred source labels.
- Compare three paths: supplied source truth, inferred source, and
  uniform-external baseline.
- Add laundering variants to operation-memory and correction-chain fixtures.
- Track how source inference errors affect route selection and fold-force.

Acceptance:

- Inferred-source routing beats uniform-external routing on false
  externalization.
- Inferred-source routing remains within a documented degradation bound from
  supplied-source truth.
- Simulation and fabricated records retain high quarantine recall.
- Route misses are separated into source-inference, retrieval, fold, and
  provider-output planes.

## Phase 5: Claim-Scale Task-Family Validation

Objective:

Move beyond smoke-level task-family validation.

Work:

- Run N=20 live replication for `billing_refund`.
- Run N=20 live replication for `security_rotation`.
- Keep retries enabled and diagnostics mandatory.
- Add one task family that includes derived inference and laundering pressure.

Acceptance:

- N >= 20 per claim-scale task family.
- Provider-valid route accuracy is 1.0 or failures are assigned to a concrete
  boundary.
- Primitive failure count is zero, or the roadmap is revised around the
  failure.
- Cost ledger and audit logs are stored.

## Phase 6: Realistic Source Data

Objective:

Create the first small real or semi-real source-labeled corpus.

Work:

- Build a hand-labeled corpus of external observations, tool outputs,
  retrieved memories, inferences, simulations, and fabricated/uncertain
  records.
- Include ambiguous cases and conflicting marker/context cases.
- Run existing source inference policies unchanged before tuning.
- Add a learned or calibrated policy only after the rule-based baseline is
  measured.

Acceptance:

- Rule-based baseline is reported before any learned policy.
- Per-class metrics are reported.
- False externalization is the primary safety metric.
- The corpus is versioned and reproducible.

## Phase 7: Higher-Level Primitives Under Source Uncertainty

Objective:

Only revisit residual attention, self-index binding, and coupled-field
dynamics after source inference and anti-laundering survive harder tests.

Work:

- Add inferred-source variants of residual attention tests.
- Add self-index tests where wrong-source records attempt cross-project or
  cross-user contamination.
- Add coupled-field tests where false externalization increases lock-in.

Acceptance:

- Source-aware policies remain better than source-blind policies when source
  labels are inferred.
- Error attribution separates source inference from retrieval and routing.
- Any coupled-field claim includes a failure mode showing excessive coupling
  or echo amplification.

## Completed Consolidation Tasks

1. `SOURCE_INFERENCE_AND_LAUNDERING.md` reflects the merged `master` state.
2. Source-inference and laundering artifacts are listed in the execution
   report.
3. Full non-live tests pass on `master`.
4. The previous roadmap is archived as `ROADMAP_ARCHIVE_2026-05-11.md`.

## Next Implementation Checklist

1. Integrate source downgrading into the actual derived-record write path.
2. Add inferred-source variants of the source-routing validation harness.
3. Add laundering-pressure cases to operation-memory and correction-chain
   fixtures.
4. Expand the source-inference fixture, especially external/tool-output
   ambiguity.
5. Only then resume N=20 live task-family expansion.

## Non-Goals

Do not use the current evidence to claim:

- emergent selfhood
- global cognitive continuity
- consciousness-adjacent behavior
- open-ended autobiographical memory
- general real-world source inference
- model-independent live robustness

The foundation is stronger now, but it is still a primitive validation program.
The next work should make source truth earned, propagated, and stress-tested
before building larger theory on top of it.

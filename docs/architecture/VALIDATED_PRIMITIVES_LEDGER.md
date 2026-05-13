# Validated Primitives Ledger

**Purpose.** Single source of truth for what the product can depend on. Each entry is either a *validated primitive* (fit for product use), a *partially grounded primitive* (scoped use only, with documented limits), or a *theoretical claim* (design heuristic, not load-bearing).

**How to use this document.**
- Product code MUST reference entries from §1 Validated Primitives only when promising behaviour to users or external callers.
- Product code MAY reference entries from §2 Partial Grounding when the use is internal, the limits are explicit in code comments, and the scope is bounded.
- Product code MUST NOT depend on entries from §3 Theoretical Claims for correctness, safety, or guarantees. Theoretical entries are design language only.
- Marketing, public docs, and customer-facing materials should reference only §1 entries as guarantees.

**Versioning policy.** Each entry has a *grounding commit* (the git SHA after which the evidence existed in `master`) and a *last review date*. When the underlying validation is rerun or extended, update the grounding commit; if results regress, downgrade the entry's tier.

**Repository state.** This ledger was originally produced in the `trace-memory-architecture` validation repo at commit `a894394` (multi-seed laundering sweep added). In the consolidated artifact repo, the validation code lives under `src/fgm/` and `tests/architecture/`, result files live under `results/architecture/`, and the source paper lives at `paper/Source_Downgrading.tex`.

---

## §1. Validated Primitives (fit for product use)

Each entry's evidence is a result file plus a test file. The product may treat the primitive's behaviour as guaranteed across the documented scope.

### 1.1 Source label preservation through the inscription pipeline

**Operational definition.** A record stored with `source_label ∈ {external, tool_output, retrieved_memory, inference, simulation, fabricated_or_uncertain, operation_record}` retains that label through the full pipeline: storage → retrieval → fold → operation-memory write → JSONL replay.

**Code surface.** `MemoryRecord.source_label`, `RetrievalHit.active_source_label`, `FoldResult.source_labels`, `OperationRecord.source_labels`, `ValidationRecord.source_labels`.

**Evidence.**
- `results/architecture/roadmap_validation_summary.json` — `source_label_accuracy = 1.0` and `active_source_accuracy = 1.0` across all four routing policies (source_sensitive, always_write, never_write, source_blind).
- `results/architecture/controlled_replication_summary.json` — same across 50 seeds.
- `results/architecture/real_embedding_replication_summary.json` — `source_label_accuracy mean = 1.0` across 20 sentence-transformer-embedding seeds.
- `results/architecture/live_llm_replication_n20_retry_summary.json` — `source_label_accuracy mean = 1.0` across 20 live OpenAI prompt seeds.
- `tests/test_validation_primitives.py` — regression tests.

**Grounding commit.** `2d7bcf3` (validation-primitives merge).

**Fit-for-product verdict.** Fit. Source labels are durable across the substrate ladder (toy hash, real embeddings, live LLM).

**Limits.** Only validated when labels are *supplied* at write time. Inference of labels from content is a separate primitive (§1.6).

---

### 1.2 Fold-force as the operational criterion for memory

**Operational definition.** A retained record functions as memory only when folding it into the live transition produces a measurable change in a non-bookkeeping transition variable:
```
fold_force(r; s, x) = ‖Φ(s, x, fold(r)) − Φ(s, x, ∅)‖_{cog}
```
restricted to cognitive dimensions of `Φ`. The fold gate `G_t` rejects records with `fold_force < threshold`.

**Code surface.** `FoldGate.fold(...)`, `FoldResult.fold_force`, `FoldResult.gated`.

**Evidence.**
- `results/architecture/real_component_validation_summary.json` — `fold_force = 0.834` under sentence-transformer embeddings.
- `results/architecture/real_embedding_replication_summary.json` — `mean_fold_force mean = 0.835` across 20 seeds, 95% CI `[0.835, 0.836]`.
- `results/architecture/live_llm_validation_summary.json` — `mean_fold_force = 1.353` under live OpenAI transitions.
- `results/architecture/live_llm_replication_n20_retry_summary.json` — `mean_fold_force mean = 1.149` across 20 live prompt seeds.
- `tests/test_h_cog.py`, `tests/test_trace_probes.py` — fold-force regression tests.

**Grounding commit.** `2d7bcf3`.

**Fit-for-product verdict.** Fit. The criterion is the deepest grounded primitive. Use it as the operational definition of "this record is memory."

**Limits.** The cognitive-dimension restriction (`_{cog}`) requires the consumer to declare which transition variables count as cognitive vs bookkeeping. The default in `default_transition` is the full vector; production code should override this with task-specific cog dims.

---

### 1.3 Source-sensitive routing

**Operational definition.** A routing function `R[·]` selects a write target from `{∅, τ, H, op, cc, quar}` for each attended-and-folded candidate, conditional on the candidate's source label. The source-sensitive policy quarantines untrusted sources (`simulation`, `fabricated_or_uncertain`) and routes the rest by fold-force and correction status.

**Code surface.** `FGMAgent._score_routes(...)`, `FoldResult.route_scores`, `FoldResult.selected_route`. Constants `ROUTE_*` and `UNTRUSTED_SOURCE_LABELS` in `src/fgm/core.py`.

**Evidence.**
- `results/architecture/roadmap_validation_summary.json` — `source_sensitive route_accuracy = 1.0`, `echo_promotion_rate = 0.0`; `source_blind route_accuracy = 0.5`, `echo_promotion_rate = 1.0`.
- `results/architecture/controlled_replication_summary.json` — same gap across 50 seeds with zero variance.
- `results/architecture/live_llm_replication_n20_retry_summary.json` — `route_accuracy mean = 1.0` under live OpenAI.
- `tests/test_validation_primitives.py` — comparison tests.

**Grounding commit.** `2d7bcf3`.

**Fit-for-product verdict.** Fit *within the bounded route taxonomy*. The six-route distribution is the validated channel set.

**Limits.** Validated on controlled fixtures with discrete source labels. The current `_score_routes` policy is rule-based (transparent `if-elif` over threshold conditions), not learned. Adding new route classes or learning the policy from data is out of scope for the validated primitive.

---

### 1.4 Source downgrading + provenance propagation (anti-laundering inscription)

**Operational definition.** When a derived record `r` is written from a non-empty set of contributing input records `{c_1, ..., c_k}` with source labels `S = {Source(c_i)}` and provenance fields `{Prov(c_i)}`:
```
Source(r) = Trust_ceil(S) = min(S ∪ {inference}) by trust rank
Prov(r)   = ⋃_i (Prov(c_i) ∪ {id(c_i)})
```
Trust ordering: `fabricated < simulation < inference < retrieved_memory < tool_output < external`.

**Code surface.** `write_inference_downgrading(...)` in `src/fgm/laundering.py`; `min_trust_source(...)`.

**Evidence.**
- `results/architecture/laundering_validation_summary.json` (seed 0) — `source_downgrading` produces `inference_laundering_rate = 0.00`, `provenance_chain_recall = 1.00`, `false_externalization_after_inference = 0.00`, `derived_trust_ceiling_violation_rate = 0.00`, `transitive_provenance_depth_mean = 2.29`.
- `results/architecture/laundering_validation_multiseed_summary.json` (N=20 seeds, σ=0.05 embedding noise) — same metrics with zero variance. The correctness is structural, not retrieval-contingent.
- `tests/architecture/test_laundering.py` — pytest gates encoding the four-zero acceptance criterion.

**Grounding commit.** `a894394` (multi-seed sweep).

**Fit-for-product verdict.** Fit. This is the recommended default inscription rule for derived records in any agent system.

**Limits.** Validated on a 5-case deterministic fixture with 7 derived records. Chain length capped at 2 in the fixture. Source labels are *supplied* by the test generator; combining this rule with §1.6 inference-based labels is sound but not yet measured together end-to-end.

**Important sub-property.** Provenance propagation *alone* (without trust capping) is insufficient. See `provenance_propagating` policy in the same fixture: `derived_trust_ceiling_violation_rate = 0.57` because all derived records get labeled `inference` regardless of whether inputs included `simulation` or `fabricated_or_uncertain`. Always pair propagation with trust capping.

---

### 1.5 Inscription utility under budget pressure

**Operational definition.** A utility-based inscription policy assigns each candidate event a predicted future-utility score and writes only the top-`k` by utility under a fixed storage budget. Compared against `always_write`, `never_write`, `random_write`, `relevance_write` baselines.

**Code surface.** `compare_inscription_policies(...)`, `evaluate_inscription_policy(...)` in `src/fgm/inscription.py`.

**Evidence.**
- `results/architecture/roadmap_validation_summary.json` (`inscription_utility` block) — utility policy reaches `future_task_lift = 1.0`, `utility_per_written_record = 1.0` at budget 3; relevance reaches 0.33 / 0.33; always_write 1.0 / 0.5 (correct content but storage cost); never_write 0.0 / N/A.
- `tests/test_validation_primitives.py` — comparison tests.

**Grounding commit.** `2d7bcf3`.

**Fit-for-product verdict.** Fit *under budget pressure*. The clearest case for the policy is when storage cost is non-trivial.

**Limits.** Validated at budget = 3 only. Performance under varying budget regimes (very tight or very loose) is not characterized. Utility scoring is fixture-supplied; deriving utility scores from observed downstream task performance is a separate problem.

---

### 1.6 `Source(·)` inference from content and retrieval features (feasibility floor)

**Operational definition.** Given a record's `(content, query_context, retrieval_margin, recency_rank)`, predict the source class without access to the stored `source_label`. The combined policy uses lexical markers (high precision on marker classes: fabricated, simulation, inference, tool_output) with feature-threshold fallback for lexically silent residue (external vs retrieved_memory).

**Code surface.** `predict_combined(...)`, `compare_source_inference_policies(...)` in `src/fgm/source_inference.py`.

**Evidence.**
- `results/architecture/source_inference_validation_summary.json` — `combined` policy reaches `overall_accuracy = 0.933`, `false_externalization_rate = 0.04`, `ambiguous_accuracy = 0.60` on a 30-case fixture (5 deliberately ambiguous). Baseline `uniform_external` reaches `0.167 / 1.00 / 0.40`. Per-class: combined achieves 1.0 on `fabricated_or_uncertain`, `inference`, `simulation`, `retrieved_memory`; 0.8 on `external` and `tool_output` (the two classes that overlap in feature space).
- `tests/architecture/test_source_inference.py` — acceptance gates encoding the dominance and per-class thresholds.
- `source_classifier/` in the parent workspace — reproducible natural-prose corpus, split verifier, authored challenge set, and transformer pipeline added for the six content source labels. The current CPU artifact (`source_classifier/models/source-classifier-distilbert-cpu-v3`) reaches generated held-out `accuracy = 1.0`, per-class accuracy `1.0`, `false_externalization_rate = 0.0`, `trust_upgrade_rate = 0.0`, hard-decoy accuracy `1.0`, and out-of-domain accuracy `1.0` with the raw model on `source_classifier/reports/test_metrics_distilbert_cpu_v3_raw.json`. On the authored challenge set, the raw model reaches `accuracy = 0.983`, `false_externalization_rate = 0.0`, and `trust_upgrade_rate = 0.0` (`source_classifier/reports/challenge_metrics_distilbert_cpu_v3_raw.json`). The guarded deployable decision policy reaches `accuracy = 1.0`, per-class accuracy `1.0`, `false_externalization_rate = 0.0`, and `trust_upgrade_rate = 0.0` on both generated held-out and authored challenge reports (`source_classifier/reports/test_metrics_distilbert_cpu_v3_guarded.json`, `source_classifier/reports/challenge_metrics_distilbert_cpu_v3_guarded.json`).

**Grounding commit.** `2d7bcf3` (merged from `source-inference-and-laundering`).

**Fit-for-product verdict.** Fit *as a feasibility floor on structured content*. The guarded trained-transformer path now clears the acceptance bar on the generated natural-prose held-out corpus and the authored challenge set, and can augment the default boundary behind an explicit model-path flag. Promotion of the unguarded raw checkpoint should wait for an independently annotated corpus or live ingestion audit, because the raw checkpoint does not yet clear the authored challenge set without guards.

**Limits.** The rule-based policy is validated on a 30-case structured fixture. The learned classifier is validated on a generated, rule-labelled natural-prose corpus plus a small authored challenge set, not on independently annotated human data. The guarded policy's perfect scores should be read as successful pipeline validation and a conservative deployment boundary, not proof of open-domain source grounding. Retrieval features alone (the `feature_threshold` policy) cap at 0.50 accuracy — they cannot separate `tool_output` from `external` or `simulation` from `retrieved_memory`. For production natural-prose ingestion, keep explicit app-owned source labels or run the guarded trained classifier behind a feature flag until live annotation confirms the same false-externalization and trust-upgrade rates.

---

### 1.7 Correction-chain node schema

**Operational definition.** Belief revisions are recorded as nodes of the form
```
c = (B_prev, e, u, B_revised, Δ, ι, p, q)
```
where `B_prev` is prior belief, `e` is evidence/error, `u` is update operation, `B_revised` is revised belief, `Δ` is delta, `ι` is self-index binding, `p` is provenance, `q` is confidence. The correction-chain policy retains these full nodes; conclusion-only and no-memory baselines retain less.

**Code surface.** `CorrectionCase`, `CorrectionNode`, `compare_correction_policies(...)` in `src/fgm/correction.py`.

**Evidence.**
- `results/architecture/roadmap_validation_summary.json` (`correction_chains` block) — `correction_chain` policy reaches `prior_belief_recall = 1.0`, `evidence_recall = 1.0`, `update_operation_recall = 1.0`, `delta_accuracy = 1.0`, `transfer_success = 1.0`; `conclusion_only` reaches `0/0/0/0/0` on those and `false_update_rate = 1.0`; `no_memory` matches conclusion_only except `false_update_rate = 0` (it never updates).
- `tests/test_self_correction.py`, `tests/test_validation_primitives.py`.

**Grounding commit.** `2d7bcf3`.

**Fit-for-product verdict.** Fit *as a schema*. The node fields are the required information envelope for belief-revision events.

**Limits.** The fixture-based comparison is symbolic (`evaluate_correction_policy` switches on policy string rather than running an agent end-to-end). The schema is grounded; the long-horizon compounding rates the parent paper predicts are not. False-evidence rejection is validated at the schema level (`should_update = False` cases produce zero false updates); robustness under adversarial false evidence at scale is not.

---

### 1.8 Engineered self-index binding

**Operational definition.** Records carrying explicit metadata `{user_id, project_id, role, permission_scope, standing_commitment}` are retrieved only when the active self-index matches. Wrong-project, wrong-user, and role-conflict applications are suppressed.

**Code surface.** `SelfIndexCase`, `compare_self_index_policies(...)` in `src/fgm/self_index.py`.

**Evidence.**
- `results/architecture/roadmap_validation_summary.json` (`self_index_binding` block) — `self_indexed` policy: `correct_binding_rate = 1.0`, `wrong_project_application_rate = 0.0`, `wrong_user_leakage_rate = 0.0`, `role_conflict_rate = 0.0`. Baselines `global_memory` and `project_only` show measurable failures (0.25 binding accuracy for global).
- `tests/test_validation_primitives.py`.

**Grounding commit.** `2d7bcf3`.

**Fit-for-product verdict.** Fit *when metadata is supplied*. This is the engineered baseline; emergent self-indexing (records inferring their own self-index from context) is out of scope.

**Limits.** Metadata-supplied only. The fixture is small (4 cases). Real-world tenant isolation requires additional cryptographic and access-control layers; this primitive is necessary but not sufficient.

---

### 1.9 Cascade-invisibility audit constraint *(methodological primitive)*

**Operational definition.** Once a derived record's source label has been laundered (e.g., a non-external input chain produced an `external`-labeled output), the framework's own self-audit metric — computed from stored source labels — strictly undercounts the truth-grounded failure rate. Detecting laundering reliably requires *paired* metrics: one self-referential (read from stored labels) and one truth-grounded (read from fixture or out-of-system labels).

**Code surface.** `LaunderingPolicyReport.inference_laundering_rate` (self-referential) and `LaunderingPolicyReport.derived_trust_ceiling_violation_rate` (truth-grounded) in `src/fgm/laundering.py`.

**Evidence.**
- `results/architecture/laundering_validation_summary.json` — under `naive_inscribe`, local rate `= 0.43` and truth rate `= 1.00`, gap `= 0.57` (exactly the cascade depth in the fixture).
- `results/architecture/laundering_validation_multiseed_summary.json` — gap is stable across 20 noise-perturbed seeds.
- `tests/architecture/test_laundering.py::test_naive_laundering_is_locally_underdetected_due_to_cascade`.

**Grounding commit.** `a894394`.

**Fit-for-product verdict.** Fit *as a constraint on internal audit infrastructure*. Any audit dashboard that reports laundering safety using only the system's own stored labels gives a false floor.

**Limits.** Demonstrated on chain length 2. Property holds structurally for arbitrary depth (see `paper/Source_Downgrading.tex` §5 Property 3), but operational detection of deeper chains may require additional bookkeeping or out-of-system ground truth.

---

## §2. Partial Grounding (scoped use only)

Entries here have evidence for *existence* or *structural form* but not yet for the specific quantitative claims that would make them load-bearing in a product contract. Use them internally; do not promise them externally.

### 2.1 Trace retention probes

**Status.** Toy fixture passes (`tests/test_trace_probes.py`), lag-based projection intensity is non-zero, perturbing a prior event changes the trace component.

**Why scoped.** No calibration against natural data. Decay constants and half-lives are fixture-determined.

**Use guidance.** Use the `LeakyTraceOperator` and `trace_retention_probe` as internal building blocks. Do not promise specific retention durations to callers.

### 2.2 Coupled-field dynamics

**Status.** `coupled_field_probe` shows non-zero memory-to-attention shift (`0.136` source-aware vs `0.711` source-blind, the only non-saturated dynamics metric in the validation suite).

**Why scoped.** The *existence* of coupling is grounded; the *magnitude* and *stability conditions* are not. Spectral-radius bounds from MAFC v3 are derived mathematically but not measured.

**Use guidance.** Cite the coupling structurally as a design rationale for source-aware routing. Do not promise specific lock-in thresholds or novelty-breakthrough conditions.

### 2.3 Source-aware retrieval reranking

**Status.** Boundary regression fixed on the seed-4/turn-4 legal-rollback case (`results/architecture/rerank_boundary_regression_summary.json`).

**Why scoped.** One boundary case, deterministic replay. Robustness across natural retrieval distributions is not measured.

**Use guidance.** Useful as a heuristic in retrieval pipelines, especially when source labels are reliable. Do not promise it generalizes.

### 2.4 Source-aware residual attention

**Status.** `residual_posture_source` policy reaches `transition_effective_retrieval_precision = 1.0` and `confirmation_attractor_rate = 0.0` on the residual-attention fixture vs the source-blind baseline.

**Why scoped.** Tiny fixture (k=3 retrieval). The discrimination space is too small to ground continuous-distribution claims.

**Use guidance.** Apply source-aware residual updates when reactivated content should be discounted in posture. Do not promise distractor resistance at deployment scale.

### 2.5 Provider-empty vs primitive-failure separation

**Status.** Live-LLM diagnostics separate provider-empty responses (no model output) from primitive route/retrieval failures (`results/architecture/live_llm_replication_n20_retry_diagnostics.json` shows `failure_count = 0` after the separation).

**Why scoped.** Engineering finding, not a primitive of memory. Useful as a robustness pattern for live integrations.

**Use guidance.** Adopt the diagnostic separation in any production live-LLM pipeline. Do not market it as a memory primitive.

---

## §3. Theoretical Claims (design heuristics, not load-bearing)

Entries here come from the prior-paper sequence (SIMFC, Trace, AINC, MAFC v3, Correction Chains) and have *not* been empirically validated as of this ledger's grounding commit. They are useful as design language and architectural rationale, but no product contract should depend on them.

Each entry: the claim, its source paper, and the gap that would need to close before promotion to §2 or §1.

### 3.1 SIMFC: self-indexing produces emergent stability

**Source.** SIMFC (DOI `10.5281/zenodo.20042034`).

**Claim.** Recursive self-indexing produces a stable mnestic structure under bounded operating conditions.

**Gap to validation.** No empirical demonstration of emergent self-indexing in current results. Would require a long-horizon agent run with self-pointer formation and stability measurement.

### 3.2 AINC: specific mnestic agent architecture predictions

**Source.** Attention Is Not Continuity (DOI `10.5281/zenodo.20057986`).

**Claim.** The proposed reactivation cycle and mnestic agent architecture produce continuity properties absent from RAG/MemGPT/Generative Agents.

**Gap to validation.** Structural separation of attention from continuity is grounded (§1.3 covers a piece). The specific architecture's claims about continuity quality vs the named baselines have not been measured head-to-head.

### 3.3 MAFC v3: spectral-radius stability bounds

**Source.** MAFC v3 (DOI `10.5281/zenodo.20089558`).

**Claim.** The coupled dynamical system over `(A_t, Ā_t, τ_t, H_t)` has local convergence conditions expressible via the spectral radius of the coupled Jacobian, with explicit lock-in and novelty-breakthrough thresholds.

**Gap to validation.** Derived mathematically; not measured. Would require constructing the empirical Jacobian from logged transitions and checking the bound.

### 3.4 MAFC v3: Maxwell-like field analogy

**Source.** MAFC v3.

**Claim.** Attention and memory have the structural form of electromagnetic field coupling (mutual induction, no static decoupled state).

**Gap to validation.** Structural; explicitly described in the paper as analogy, not physics. Not falsifiable as currently stated. Useful as a design metaphor; not a contract.

### 3.5 SIMReC: recursive self-indexing as an observed phenomenon

**Source.** SIMFC + SIMReC framing.

**Claim.** Sufficient stacking of validated primitives produces a recursive self-indexing structure with observable properties (self-pointer formation, bounded recursion depth, trust-monotone self-reference).

**Gap to validation.** This is the goal toward which the validated primitives compose. Source downgrading (§1.4) provides the trust-bound that makes recursion safe. Fold-force (§1.2) provides the criterion for when a self-pointer is operationally memory. Beyond these floor primitives, no demonstration of recursive self-indexing has been measured.

**Promotion path.** Build a long-horizon agent run where self-pointers form, measure trust degradation along the recursion, verify monotone decay. This is the SIMReC product's empirical centerpiece — and not yet attempted.

### 3.6 Correction Chains: long-horizon compounding rates

**Source.** Correction Chains (DOI `10.5281/zenodo.20088789`).

**Claim.** Correction-chain agents accumulate world-model improvements over time at a rate that conclusion-only agents cannot match.

**Gap to validation.** Schema is grounded (§1.7). Long-horizon compounding rate has not been measured because no long-horizon run has been done.

### 3.7 Trace: specific decay constants and lag-projection intensities

**Source.** Trace Formalization (DOI `10.5281/zenodo.20043070`).

**Claim.** Trace decay follows a specific exponential form with measurable half-life dependent on novelty and attentional context.

**Gap to validation.** Toy probes work; specific decay parameters have not been calibrated against any natural-task fixture.

---

## §4. Ledger Maintenance

**When to update.** Update when:
- A new validation result file lands in `results/architecture/`.
- A test file is added or modified that affects an entry's gates.
- A theoretical claim graduates to partial grounding (move from §3 to §2) or full grounding (move from §3/§2 to §1).
- A previously validated claim regresses or its scope shifts (downgrade tier).

**Update procedure.**
1. Identify which entry changes.
2. Update its grounding commit to the new SHA.
3. Update the evidence list with the new result file(s).
4. Update the fit-for-product verdict if scope changed.
5. Note the date and reason in a CHANGELOG section (added when needed).
6. Bump the document grounding commit in the header.

**Anti-patterns.**
- Do not move an entry from §3 to §1 because the underlying *paper* was published. Publication is not validation. Only result files plus passing tests promote an entry.
- Do not soften an entry's verdict to make a product feature look stronger. The ledger's value comes from being honest.
- Do not delete an entry. If a primitive is no longer used by the product, move it to a §5 Deprecated section with a date.

**Audit recommendation.** Re-read the ledger before any external commitment (customer contract, marketing claim, public benchmark). If a planned commitment requires a §2 or §3 entry, either run the work to promote the entry first, or scope the commitment to §1 only.

---

**Ledger grounding commit:** `a894394` (multi-seed laundering sweep).
**Last reviewed:** 2026-05-11.

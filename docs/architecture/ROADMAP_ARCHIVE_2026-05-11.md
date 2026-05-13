# Validation Roadmap: Grounding the Mnestic Framework

This roadmap turns the framework into a bottom-up empirical program. The goal is to validate primitives with fixed truth conditions before relying on higher-level claims such as continuity, recursive self-indexing, or coupled-field stability.

Primary empirical anchors:

- `trace-memory-architecture`: https://github.com/carboni123/trace-memory-architecture
- `simrec-reference`: https://github.com/carboni123/simrec-reference

Local reference available during planning:

- `C:\Users\DiegoPC\Documents\GitHub\trace-memory-architecture`

## Validation Rule

Do not validate primitives by naming them. A primitive is provisionally grounded only if at least one of these is true:

1. The experiment creates the ground-truth label directly.
2. A paired intervention changes only the primitive under test.
3. Future-task utility distinguishes the positive and negative condition.
4. A no-memory, source-blind, or route-randomized baseline fails where the proposed primitive succeeds.

This means the first validation layer should be controlled, synthetic, and intervention-heavy. Real LLM runs are useful only after the primitive survives toy and deterministic tests.

## Existing Assets to Reuse

### trace-memory-architecture

Already contains useful scaffolding:

- `trace_probes`: trace retention, causal trace intervention, addressability, fold-force, P7 retrieval regimes.
- `fgm`: six-layer Fold-Gated Memory implementation.
- `MemoryStore`, `MarginRetriever`, `FoldGate`, `OperationMemory`, `Compressor`, `FGMAgent`.
- Tests for trace probes, LLM transition plumbing, real embeddings, self-correction, compression boundaries, large-N retrieval, and multi-seed behavior.

Immediate reuse:

- Use `trace_probes` for trace detectability and causal intervention.
- Use `FoldGate` and paired with/without-memory transitions for fold-force.
- Use `OperationMemory` tests as the starting point for operation-memory validation.
- Extend `FGMAgent` with explicit source labels and route logging.

### simrec-reference

Already contains useful scaffolding:

- `SIMReCGraph`, typed residues, retrieval, folding, operation nodes, self-index update.
- `transition_step(state, input_value, acts)` with with-memory vs without-memory divergence.
- `write_operation(...)` / `add_operation(...)`.
- Toy prediction harness for seven SIMFC predictions.
- Real-component harness using sentence-transformers plus LLM transition calls.

Important empirical caveats to preserve:

- P7 overload is conditional on per-record information density, not unconditional.
- Some metrics can become tautological if they measure the mechanism instead of behavior.
- Topic-conditioned LLM generation can fake recall; use baseline-subtracted memory contribution.
- Long-horizon fold-force can become trajectory-confounded; track accumulated operation impact, not only per-step divergence.
- Integration/P4 requires edge-aware retrieval before it can be honestly tested.

## Primitive Validation Order

Build in this order:

```text
source/provenance
  -> trace detectability
  -> fold-force / transition effect
  -> inscription utility
  -> source-sensitive routing
  -> operation-memory
  -> correction-chain nodes
  -> residual attention
  -> self-index binding
  -> coupled-field dynamics
```

The first five are the foundation. Do not lean on self-index or global continuity claims until the foundation passes.

## Phase 0: Instrumentation Baseline

### Objective

Add a shared validation record format so every experiment logs the same primitive-level evidence.

### Required fields

```text
run_id
seed
turn_id
query
external_input_ids
retrieved_ids
source_labels
source_confidence
attention_or_selection_scores
eligibility_score
inscription_score
route_scores
selected_route
output_with_memory
output_without_memory
transition_delta
predicted_fold_force
realized_fold_force
operation_record_id
correction_node_id
future_task_id
future_task_score
```

### Deliverables

- `ValidationRecord` dataclass.
- JSONL logger.
- Replay script that recomputes metrics from JSONL only.

### Acceptance

- Every primitive experiment can be scored from logs without rerunning the agent.
- Each metric has a no-memory or source-blind baseline.

## Phase 1: Source / Provenance

### Primitive

Source tells what kind of object entered the active substrate:

```text
external
tool_output
retrieved_memory
inference
simulation
fabricated_or_uncertain
operation_record
```

### Fixed truth

The generator assigns the source label. There is no interpretive ambiguity.

### Experiments

1. Synthetic mixed-source stream.
2. Repeated memory reactivation without external evidence.
3. Inference laundering: retrieved memory -> inferred claim -> later recall.
4. Tool-output vs model-generated answer distinction.

### Metrics

```text
source_label_accuracy
false_externalization_rate
reactivated_as_observed_rate
inference_laundering_rate
tool_output_preservation_rate
```

### Baselines

- Source-blind routing.
- Random source labels.
- All-retrieval-as-external.

### Acceptance

- False externalization stays below a configured threshold under repeated reactivation.
- Source labels remain recoverable after storage, retrieval, and operation-memory writes.

## Phase 2: Trace Detectability

### Primitive

A trace exists when prior state leaves detectable and causally sensitive residue later.

### Fixed truth

The inserted event and lag are controlled by the experiment.

### Existing asset

Use `trace_memory_architecture/src/trace_probes`:

- `trace_retention_probe`
- `causal_trace_probe`
- `LeakyTraceOperator`

### Experiments

1. One-hot lag retention.
2. Distractor-load decay.
3. Prior-state perturbation.
4. Trace-state masking.

### Metrics

```text
lag_projection_intensity
probe_accuracy_by_lag
retention_half_life
projection_delta_after_intervention
behavior_drop_after_trace_mask
```

### Acceptance

- Trace probe detects prior event above chance at expected lags.
- Perturbing the prior event changes the trace component.
- Masking trace state degrades tasks that require temporal residue.

## Phase 3: Fold-Force / Transition Effect

### Primitive

A retained item functions as memory only if folding it changes a non-bookkeeping transition variable.

### Fixed truth

The paired intervention creates the label:

```text
transition_with_memory
transition_without_memory
```

### Existing assets

- `fgm.FoldGate`
- `fgm.LLMTransition`
- `simrec.transition_step`
- `fold_force_probe`

### Experiments

1. Deterministic vector transition.
2. Echo LLM transition.
3. Sentence-transformer echo transition.
4. Live LLM transition after deterministic tests pass.

### Metrics

```text
action_changed
answer_changed
plan_changed
tool_choice_changed
confidence_changed
quality_score_delta
realized_fold_force
```

### Baselines

- Retrieved but not folded.
- Irrelevant memory folded.
- Bookkeeping-only divergence.

### Acceptance

- Relevant memories produce higher non-bookkeeping fold-force than irrelevant memories.
- Bookkeeping-only changes are excluded.
- The metric predicts future task success better than retrieval similarity alone.

## Phase 4: Inscription Utility

### Primitive

Inscription decides whether an event should be written for future use.

### Fixed truth

Future tasks reveal whether the event was useful.

### Experiments

1. Generate event stream with future-query dependencies.
2. Compare always-write, never-write, random-write, relevance-write, and utility-write policies.
3. Vary storage budget and distractor density.

### Metrics

```text
future_task_lift
missed_useful_write_rate
false_write_rate
storage_cost
retrieval_noise_added
utility_per_written_record
```

### Acceptance

- Utility-based inscription beats always-write under budget pressure.
- Utility-based inscription beats never-write on future dependent tasks.
- The learned or rule-based write score predicts future utility out of sample.

## Phase 5: Source-Sensitive Routing

### Primitive

Routing decides whether an attended/reactivated item becomes:

```text
nothing
trace
durable memory
operation-memory
correction-chain node
quarantine
```

### Fixed truth

The task generator declares the correct route from event type and future requirements.

### Experiments

1. External evidence should route to trace or durable memory.
2. Repeated reactivation without evidence should not route to external memory.
3. Decision-changing retrieval should route to operation-memory.
4. Belief revision should route to correction-chain.
5. Simulation or fabricated content should route to quarantine unless later corroborated.

### Metrics

```text
route_accuracy
false_durable_write_rate
false_external_memory_rate
missed_operation_write_rate
missed_correction_node_rate
quarantine_precision
quarantine_recall
```

### Baselines

- Single trace channel.
- Attention-gated write.
- Source-blind routing.
- Route randomization.

### Acceptance

- Source-sensitive routing reduces echo amplification relative to source-blind routing.
- Operation and correction routes improve downstream diagnosis and revision.

## Phase 6: Operation-Memory

### Primitive

Operation-memory records how memory use changed a transition.

### Fixed truth

A paired ablation determines whether memory changed the transition.

### Existing assets

Use and extend:

- `fgm.OperationMemory`
- `tests/test_self_correction.py`
- `simrec.write_operation(...)`

### Required operation record

```text
query
retrieved_ids
source_labels
output_with_memory
output_without_memory
affected_variable
realized_fold_force
decision_delta
outcome
timestamp
recursive_depth
```

### Experiments

1. Bad decision trace-back.
2. Multi-step correction after new evidence.
3. Content-only vs operation-memory agents.
4. Sparse vs rich operation records.

### Metrics

```text
decision_trace_accuracy
retrieved_causal_source_accuracy
self_correction_success
repeat_error_rate
operation_record_completeness
recursive_depth_accuracy
```

### Acceptance

- Operation-memory agents outperform content-only agents on diagnosing prior decisions.
- Rich operation records beat sparse operation records on recall and correction.
- Compression policies are evaluated separately for sparse and rich record regimes.

## Phase 7: Correction-Chain Nodes

### Primitive

A correction-chain node records a belief/model update lineage:

```text
prior belief
evidence or error
update operation
revised belief
delta
self-index / role binding
provenance
confidence
```

### Fixed truth

The task script declares the prior belief, contradictory evidence, correct update, and future transfer cases.

### Experiments

1. Simple contradiction update.
2. Multi-hop update.
3. False evidence that should not update.
4. Similar future case requiring transfer.
5. Dissimilar future case requiring non-transfer.

### Metrics

```text
prior_belief_recall
evidence_recall
update_operation_recall
revised_belief_accuracy
delta_accuracy
transfer_success
overgeneralization_rate
```

### Acceptance

- Correction-chain agents outperform conclusion-only agents on transfer.
- Correction-chain agents can explain why the belief changed.
- False evidence is quarantined or source-tagged rather than folded into the belief chain.

## Phase 8: Residual Attention

### Primitive

Residual attention is a decayed posture vector that improves retrieval or routing beyond semantic similarity and recency.

### Fixed truth

The task generator controls active topic, goal, role, and distractor similarity.

### Experiments

Compare retrieval policies:

```text
semantic only
semantic + recency
semantic + residual posture
semantic + residual posture + source
```

### Metrics

```text
transition_effective_retrieval_precision
retrieval_margin
distractor_resistance
confirmation_attractor_rate
task_switch_recovery_time
```

### Acceptance

- Residual posture improves retrieval of transition-effective records after controlling for recency.
- Source-aware residual updates reduce confirmation attractors after internal reactivation.

## Phase 9: Self-Index Binding

### Primitive

Self-index binding keeps memories attached to the right continuing user, project, role, permission scope, or commitment set.

### Fixed truth

Use engineered metadata before claiming emergent self-index:

```text
user_id
project_id
role
permission_scope
standing_commitment
```

### Experiments

1. Same fact across different projects.
2. Conflicting commitments across users.
3. Role-specific permissions.
4. Project handoff with explicit allowed transfer.
5. Cross-project contamination attempts.

### Metrics

```text
correct_binding_rate
wrong_project_application_rate
wrong_user_leakage_rate
role_conflict_rate
commitment_preservation_rate
```

### Acceptance

- Self-index metadata improves correct memory application without harming factual recall.
- Index shuffling changes ownership-sensitive behavior more than factual retrieval.

## Phase 10: Coupled-Field Dynamics

Only start this after Phases 1-8 have stable metrics.

### Primitive

Memory biases future attention; attention biases future memory writes.

### Fixed truth

Use intervention:

```text
ablate memory -> attention distribution changes
ablate attention/posture -> future write distribution changes
```

### Experiments

1. Memory-to-attention induction.
2. Attention-to-inscription induction.
3. Novelty breakthrough under lock-in.
4. Source-aware residual attention vs source-blind residual attention.

### Metrics

```text
attention_shift_after_memory_ablation
write_shift_after_attention_ablation
novelty_breakthrough_threshold
lock_in_rate
echo_amplification_rate
```

### Acceptance

- Cross-derivatives are empirically nonzero.
- Bounded coupling improves long-horizon task performance.
- Excess coupling produces measurable lock-in or echo amplification.

## Minimum Viable Validation Suite

The first useful release should include only the primitives that can be grounded with the least ambiguity:

```text
1. source/provenance labels
2. trace detectability
3. fold-force paired ablation
4. inscription utility
5. source-sensitive routing
6. operation-memory
```

Expected repository work:

- Add source labels to `trace-memory-architecture` records.
- Add route scores and selected route to `FGMAgent.query(...)`.
- Add `ValidationRecord` JSONL logging.
- Add synthetic task generators for source, utility, and route truth.
- Add pytest suites for the six primitives above.
- Add a `results/` folder with fixed-seed JSONL outputs and metric summaries.

## Statistical Standards

Toy and deterministic tests:

```text
N >= 50 seeds
fixed random seeds
effect direction must hold in >= 90% of seeds unless prediction is explicitly expected-value only
```

Real embedding tests:

```text
N >= 20 seeds where feasible
report confidence intervals
compare against semantic-only and recency baselines
```

Live LLM tests:

```text
temperature = 0 for deterministic comparisons
N >= 5 for smoke tests
N >= 20 for claims
always include no-memory and topic-only baselines
report cost ledger
store raw prompts/responses where allowed
```

Metric rules:

- Prefer baseline-subtracted memory contribution over raw similarity.
- Avoid metrics that are direct functions of the mechanism under test.
- Score from logs, not from internal objects only.
- Separate sparse-record and rich-record regimes.

## Non-Goals Until the Foundation Passes

Do not prioritize these as first validation targets:

- emergent selfhood
- global cognitive continuity
- electromagnetic analogy
- full MAFC stability
- consciousness-adjacent claims
- open-ended autobiographical memory

These may be directionally useful, but they are not primitive enough. The foundation should be source labels, causal ablations, trace probes, future utility, and operation records.

## Near-Term Milestones

### Milestone 1: Source-Labeled FGM

Add source metadata to `MemoryRecord`, `RetrievalHit`, `FoldResult`, and `OperationRecord`.

Done when:

- Existing tests still pass.
- Source labels survive add, retrieve, fold, operation write, and compression.

### Milestone 2: Primitive JSONL Harness

Create a reusable experiment harness that emits `ValidationRecord` rows.

Done when:

- Trace, fold-force, and operation-memory tests can be scored from JSONL.

### Milestone 3: Source and Routing Tests

Implement synthetic mixed-source streams and route truth tables.

Done when:

- Source-sensitive routing beats source-blind routing on false externalization and missed operation writes.

### Milestone 4: Inscription Utility Tests

Implement future-task utility labels.

Done when:

- Utility-based inscription beats always-write and never-write under at least two storage budgets.

### Milestone 5: Correction-Chain Harness

Implement controlled belief-update tasks.

Done when:

- Correction-chain agents beat conclusion-only agents on transfer and explanation.

### Milestone 6: Real-Component Replication

Run the strongest toy results through real embeddings and then live LLM transitions.

Done when:

- Effects survive substrate change.
- Any reversals are documented as framework updates, not swept into metric changes.

## First Concrete Build Target

Start with this minimal experiment:

```text
Given:
  external observation E1
  retrieved memory R1
  simulated hypothesis S1
  fabricated distractor F1

Task:
  answer a later query that requires E1 and R1,
  rejects F1,
  and may mention S1 only as hypothetical.

Expected:
  E1 -> trace/durable memory
  R1 -> operation-memory if it changes decision
  S1 -> simulation-tagged trace or quarantine
  F1 -> quarantine/null

Measure:
  source label accuracy
  fold-force of R1
  selected route accuracy
  later answer accuracy
  false externalization rate
```

This single harness grounds source, fold-force, inscription, routing, and operation-memory in one controlled environment.

# Validation Roadmap Execution Report

Branch/worktree:

```text
validation-primitives
C:\Users\DiegoPC\Documents\GitHub\source_downgrading
```

## Objective

Execute the primitive validation roadmap by grounding the mnestic framework in fixed-truth, replayable experiments before relying on higher-level assumptions.

## Prompt-to-Artifact Checklist

| Roadmap item | Artifact | Evidence |
| --- | --- | --- |
| Source/provenance labels | `src/fgm/core.py`, `src/fgm/validation.py` | `MemoryRecord.source_label`, active source labels, source metrics |
| Trace detectability | existing `trace_probes` | Covered by existing trace probe tests |
| Fold-force / transition effect | existing `FoldGate`, new validation records | Existing fold-force tests plus source-routing fold-force logs |
| Inscription utility | `src/fgm/inscription.py` | Utility policy beats relevance/always/never baselines |
| Source-sensitive routing | `src/fgm/validation.py`, `src/fgm/roadmap.py` | Source-sensitive route accuracy 1.0; source-blind echo promotion 1.0 |
| Operation-memory | existing `OperationMemory`, source-aware operation records | Existing self-correction tests plus source/provenance propagation |
| Correction-chain nodes | `src/fgm/correction.py` | Correction-chain policy preserves prior/evidence/update/delta and beats conclusion-only |
| Residual attention | `src/fgm/residual.py` | Source-aware residual posture selects transition-effective records and suppresses simulated confirmation |
| Self-index binding | `src/fgm/self_index.py` | Self-indexed policy prevents wrong project/user/role leakage |
| Coupled-field dynamics | `src/fgm/coupling.py` | Nonzero memory-to-attention and attention-to-write shifts; source-blind echo higher |
| Source inference | `src/fgm/source_inference.py`, `examples/run_source_inference_validation.py` | Combined lexical-plus-feature policy reaches 0.933 accuracy and 0.04 false externalization on 30 fixed-truth source cases |
| Inference laundering | `src/fgm/laundering.py`, `examples/run_laundering_validation.py` | Source downgrading plus provenance propagation yields zero laundering, zero trust-ceiling violations, and zero false externalization on controlled derived-record cases |
| Consolidated artifact runner | `src/fgm/roadmap.py`, `examples/run_validation_roadmap.py` | Writes fixed-seed summary and source-routing JSONL |
| Multi-seed controlled replication | `src/fgm/replication.py`, `examples/run_controlled_replication.py` | 50 seeds executed; all controlled effect-direction gates hold at 1.0 |
| Real-component replication | `src/fgm/real_components.py`, `examples/run_real_component_validation.py` | Sentence-transformer artifact with 384-dim embeddings, route accuracy 1.0 |
| Real-embedding replication | `src/fgm/real_components.py`, `examples/run_real_embedding_replication.py` | 20 seeded query-variant runs with sentence-transformers; all real-embedding gates met |
| Live LLM replication | `src/fgm/live_validation.py`, `examples/run_live_llm_validation.py` | OpenAI live gate passed with `.env`-loaded `OPENAI_API_KEY`; route accuracy 1.0 with nonzero cost ledger |
| Live LLM prompt-suite replication | `src/fgm/live_validation.py`, `examples/run_live_llm_replication.py` | Post-rerank 5 seeded live prompt variants executed with paired no-memory baselines; prompt/response audit JSONL written; all smoke gates met |
| Live replication diagnostics | `src/fgm/live_diagnostics.py`, `examples/analyze_live_replication.py` | Post-rerank diagnostics show zero retrieval/route failures; empty no-memory responses remain tracked as provider behavior |
| Source-aware rerank boundary regression | `src/fgm/core.py`, `src/fgm/rerank_validation.py`, `examples/run_rerank_boundary_regression.py` | Broad retrieval plus source/polarity reranking fixes the seed 4 / turn 4 legal-rollback boundary in a no-API replay |
| Claim-scale live replication | `src/fgm/live_validation.py`, `src/fgm/live_diagnostics.py` | OpenAI N=20 live prompt-suite artifacts written separately; all aggregate gates met, with 4 provider-empty zero-fold route misses diagnosed |
| Provider-empty transition handling | `src/fgm/llm.py`, `src/fgm/live_validation.py` | Empty provider responses are counted, retryable, and audited without duplicating transition-history rows |
| Retry-enabled live replication | `src/fgm/llm.py`, `src/fgm/live_validation.py` | N=5 and N=20 OpenAI retry-enabled suites passed; N=20 route/retrieval/source metrics reached 1.0 with zero final empty with-memory responses |
| Route/output gate semantics | `src/fgm/live_diagnostics.py` | Diagnostics now separate primitive route/retrieval failures from provider-output validity failures |
| Provider/model comparison harness | `src/fgm/model_comparison.py`, `examples/run_live_provider_model_comparison.py` | Four-target OpenAI/Anthropic matrix executed; three targets passed and one provider-output-only failure was isolated with zero primitive failures |
| Model-specific output-budget tuning | `src/fgm/live_validation.py`, `src/fgm/model_comparison.py` | `gpt-5-nano` passes N=5 when rerun with `max_output_tokens=1000`, resolving the empty-final-output boundary |
| Broader task-family validation | `src/fgm/live_validation.py`, `examples/run_live_llm_replication.py` | Added `billing_refund` and `security_rotation` fixed-truth families; OpenAI `gpt-4.1-mini` N=5 live runs passed with zero boundary failures |

## Generated Results

```text
results/architecture/roadmap_validation_summary.json
results/architecture/source_routing_validation.jsonl
results/architecture/source_inference_validation_summary.json
results/architecture/source_inference_validation.jsonl
results/architecture/laundering_validation_summary.json
results/architecture/laundering_validation.jsonl
results/architecture/controlled_replication_summary.json
results/architecture/controlled_replication_runs.jsonl
results/architecture/real_component_validation_summary.json
results/architecture/real_embedding_replication_summary.json
results/architecture/real_embedding_replication_runs.jsonl
results/architecture/live_llm_validation_summary.json
results/architecture/live_llm_replication_summary.json
results/architecture/live_llm_replication_audit.jsonl
results/architecture/live_llm_replication_diagnostics.json
results/architecture/rerank_boundary_regression_summary.json
results/architecture/live_llm_replication_n20_summary.json
results/architecture/live_llm_replication_n20_audit.jsonl
results/architecture/live_llm_replication_n20_diagnostics.json
results/architecture/live_llm_replication_retry_summary.json
results/architecture/live_llm_replication_retry_audit.jsonl
results/architecture/live_llm_replication_retry_diagnostics.json
results/architecture/live_llm_replication_n20_retry_summary.json
results/architecture/live_llm_replication_n20_retry_audit.jsonl
results/architecture/live_llm_replication_n20_retry_diagnostics.json
results/architecture/live_provider_model_comparison_summary.json
results/architecture/live_provider_model_comparison_openai_gpt-5-mini-2025-08-07_summary.json
results/architecture/live_provider_model_comparison_openai_gpt-5-mini-2025-08-07_audit.jsonl
results/architecture/live_provider_model_comparison_openai_gpt-5-mini-2025-08-07_diagnostics.json
results/architecture/live_provider_model_comparison_openai_gpt-5-nano_summary.json
results/architecture/live_provider_model_comparison_openai_gpt-5-nano_audit.jsonl
results/architecture/live_provider_model_comparison_openai_gpt-5-nano_diagnostics.json
results/architecture/live_provider_model_comparison_openai_gpt-4.1-mini_summary.json
results/architecture/live_provider_model_comparison_openai_gpt-4.1-mini_audit.jsonl
results/architecture/live_provider_model_comparison_openai_gpt-4.1-mini_diagnostics.json
results/architecture/live_provider_model_comparison_anthropic_default_summary.json
results/architecture/live_provider_model_comparison_anthropic_default_audit.jsonl
results/architecture/live_provider_model_comparison_anthropic_default_diagnostics.json
results/architecture/live_provider_model_comparison_nano_max1000_summary.json
results/architecture/live_provider_model_comparison_nano_max1000_openai_gpt-5-nano_summary.json
results/architecture/live_provider_model_comparison_nano_max1000_openai_gpt-5-nano_audit.jsonl
results/architecture/live_provider_model_comparison_nano_max1000_openai_gpt-5-nano_diagnostics.json
results/architecture/live_task_family_billing_refund_gpt41_summary.json
results/architecture/live_task_family_billing_refund_gpt41_audit.jsonl
results/architecture/live_task_family_billing_refund_gpt41_diagnostics.json
results/architecture/live_task_family_security_rotation_gpt41_summary.json
results/architecture/live_task_family_security_rotation_gpt41_audit.jsonl
results/architecture/live_task_family_security_rotation_gpt41_diagnostics.json
```

Key controlled metrics from `results/architecture/roadmap_validation_summary.json`:

```text
source_sensitive route_accuracy = 1.0
source_sensitive echo_promotion_rate = 0.0
source_blind echo_promotion_rate = 1.0
utility_write future_task_lift = 1.0
correction_chain transfer_success = 1.0
residual_posture_source transition_effective_retrieval_precision = 1.0
self_indexed correct_binding_rate = 1.0
source_aware attention_shift_after_memory_ablation = 0.13591597957615106
```

Key controlled replication metrics from `results/architecture/controlled_replication_summary.json`:

```text
seed_count = 50
toy_seed_count_met = true
minimum_effect_hold_rate = 1.0
all_controlled_replication_gates_met = true
source_sensitive route_accuracy mean = 1.0
source_blind echo_promotion_rate mean = 1.0
utility_write future_task_lift mean = 1.0
correction_chain transfer_success mean = 1.0
residual_posture_source transition_effective_retrieval_precision mean = 1.0
self_indexed correct_binding_rate mean = 1.0
```

Key source-inference metrics from `results/architecture/source_inference_validation_summary.json`:

```text
n_cases = 30
n_ambiguous = 5
uniform_external overall_accuracy = 0.16666666666666666
uniform_external false_externalization_rate = 1.0
lexical_rules overall_accuracy = 0.8666666666666667
feature_threshold overall_accuracy = 0.5
combined overall_accuracy = 0.9333333333333333
combined false_externalization_rate = 0.04
combined ambiguous_accuracy = 0.6
combined external per_class_accuracy = 0.8
combined tool_output per_class_accuracy = 0.8
combined retrieved_memory per_class_accuracy = 1.0
combined inference per_class_accuracy = 1.0
combined simulation per_class_accuracy = 1.0
combined fabricated_or_uncertain per_class_accuracy = 1.0
```

Key inference-laundering metrics from `results/architecture/laundering_validation_summary.json`:

```text
cases = 5
derived_records_total = 7
naive_inscribe inference_laundering_rate = 0.42857142857142855
naive_inscribe derived_trust_ceiling_violation_rate = 1.0
naive_inscribe false_externalization_after_inference = 0.4
provenance_propagating provenance_chain_recall = 1.0
provenance_propagating derived_trust_ceiling_violation_rate = 0.5714285714285714
source_downgrading inference_laundering_rate = 0.0
source_downgrading derived_trust_ceiling_violation_rate = 0.0
source_downgrading false_externalization_after_inference = 0.0
source_downgrading provenance_chain_recall = 1.0
```

Key real-component metrics from `results/architecture/real_component_validation_summary.json`:

```text
embedding_model = all-MiniLM-L6-v2
embedding_available = true
dim = 384
retrieval_hit_rate = 1.0
source_route_accuracy = 1.0
source_echo_promotion_rate = 0.0
fold_force = 0.8338691336962778
live_llm_available = true
live_llm_reason = OPENAI_API_KEY present; live replication recorded by live_validation helper
```

Key real-embedding replication metrics from `results/architecture/real_embedding_replication_summary.json`:

```text
seed_count = 20
all_real_embedding_replication_gates_met = true
route_accuracy mean = 1.0
retrieval_hit_rate mean = 1.0
source_label_accuracy mean = 1.0
echo_promotion_rate mean = 0.0
mean_fold_force mean = 0.8350696350453852
mean_fold_force 95% CI = [0.8345595957746726, 0.8355796743160978]
```

Key live-gate status from `results/architecture/live_llm_validation_summary.json`:

```text
status = passed
provider = openai
model = gpt-5-mini-2025-08-07
route_accuracy = 1.0
retrieval_hit_rate = 1.0
echo_promotion_rate = 0.0
mean_fold_force = 1.3529772051467832
api_calls = 8
input_tokens = 861
output_tokens = 1964
total_tokens = 2825
```

Key live prompt-suite replication metrics from `results/architecture/live_llm_replication_summary.json`:

```text
status = passed
provider = openai
model = gpt-5-mini-2025-08-07
seed_count = 5
all_live_replication_gates_met = true
route_accuracy mean = 1.0
retrieval_hit_rate mean = 1.0
source_label_accuracy mean = 1.0
echo_promotion_rate mean = 0.0
mean_fold_force mean = 1.1978453956111594
api_calls = 40
input_tokens = 4329
output_tokens = 9704
total_tokens = 14033
audit_events = 20
```

Key live replication diagnostics from `results/architecture/live_llm_replication_diagnostics.json`:

```text
status = passed
failure_count = 0
failure_counts = {}
affected_seeds = []
affected_turns = []
empty_response_count = 13
primitive_failure_count = 0
route_accuracy_if_provider_output_valid = 1.0
with_memory_output_validity_rate = 1.0
recommendation = track empty no-memory responses as provider/model behavior, but do not count them as route failures unless retrieval or route labels fail
```

Key rerank boundary regression metrics from `results/architecture/rerank_boundary_regression_summary.json`:

```text
status = passed
seed_count = 5
route_accuracy_mean = 1.0
retrieval_hit_rate_mean = 1.0
quarantine_recall_mean = 1.0
seed4_turn4 retrieved_ids = ["F1"]
seed4_turn4 selected_route = quarantine
boundary_retrieval_fixed = true
boundary_route_fixed = true
```

Key claim-scale live replication metrics from `results/architecture/live_llm_replication_n20_summary.json`:

```text
status = passed
provider = openai
model = gpt-5-mini-2025-08-07
seed_count = 20
all_live_replication_gates_met = true
route_accuracy mean = 0.95
retrieval_hit_rate mean = 1.0
source_label_accuracy mean = 1.0
quarantine_recall mean = 1.0
echo_promotion_rate mean = 0.0
mean_fold_force mean = 1.1939132528454617
api_calls = 160
input_tokens = 17316
output_tokens = 38749
total_tokens = 56065
audit_events = 80
```

Key claim-scale live diagnostics from `results/architecture/live_llm_replication_n20_diagnostics.json`:

```text
status = passed
failure_count = 4
failure_counts = {"route_miss": 4}
affected_seeds = [2, 10, 13, 15]
affected_turns = [1, 2]
boundary = provider_empty_with_memory_zero_fold_force
empty_response_count = 62
primitive_failure_count = 0
provider_output_boundary_failure_count = 4
route_accuracy_if_provider_output_valid = 1.0
route_failure_count_excluding_provider_output = 0
with_memory_output_validity_rate = 0.9125
recommendation = track provider-empty with-memory transitions separately; they create zero fold-force route misses even when retrieval is correct
```

Key retry-enabled live replication metrics from `results/architecture/live_llm_replication_retry_summary.json`:

```text
status = passed
seed_count = 5
all_live_replication_gates_met = true
route_accuracy mean = 1.0
retrieval_hit_rate mean = 1.0
source_label_accuracy mean = 1.0
api_calls = 51
total_tokens = 18417
with_memory retry events = 0
without_memory retry events = 11
final empty with-memory responses = 0
```

Key retry-enabled claim-scale metrics from `results/architecture/live_llm_replication_n20_retry_summary.json`:

```text
status = passed
seed_count = 20
all_live_replication_gates_met = true
route_accuracy mean = 1.0
retrieval_hit_rate mean = 1.0
source_label_accuracy mean = 1.0
quarantine_recall mean = 1.0
echo_promotion_rate mean = 0.0
mean_fold_force mean = 1.1491910218883112
api_calls = 228
input_tokens = 24283
output_tokens = 56595
total_tokens = 80878
with_memory retry events = 10
without_memory retry events = 58
final empty with-memory responses = 0
```

Key retry-enabled claim-scale diagnostics from `results/architecture/live_llm_replication_n20_retry_diagnostics.json`:

```text
status = passed
failure_count = 0
failure_counts = {}
affected_seeds = []
affected_turns = []
empty_response_count = 43
primitive_failure_count = 0
provider_output_boundary_failure_count = 0
route_accuracy_if_provider_output_valid = 1.0
with_memory_output_validity_rate = 1.0
recommendation = track empty no-memory responses as provider/model behavior, but do not count them as route failures unless retrieval or route labels fail
```

Key provider/model comparison metrics from `results/architecture/live_provider_model_comparison_summary.json`:

```text
status = passed
target_count = 4
attempted_count = 4
passed_count = 3
failed_count = 1
primitive_failure_count = 0
provider_output_boundary_failure_count = 10
provider_model_comparison_gate_met = true
total api_calls = 214
total input_tokens = 23649
total output_tokens = 35928
total_tokens = 59577

gpt-5-mini-2025-08-07 status = passed
gpt-5-mini-2025-08-07 provider_valid_route_accuracy = 1.0
gpt-5-mini-2025-08-07 with_memory_output_validity_rate = 0.95
gpt-5-mini-2025-08-07 reused_existing = true

gpt-5-nano status = failed
gpt-5-nano retrieval_hit_rate mean = 1.0
gpt-5-nano source_label_accuracy mean = 1.0
gpt-5-nano primitive_failure_count = 0
gpt-5-nano provider_output_boundary_failure_count = 10
gpt-5-nano with_memory_output_validity_rate = 0.0

gpt-4.1-mini status = passed
gpt-4.1-mini route_accuracy mean = 1.0
gpt-4.1-mini retrieval_hit_rate mean = 1.0
gpt-4.1-mini provider_valid_route_accuracy = 1.0
gpt-4.1-mini with_memory_output_validity_rate = 1.0
gpt-4.1-mini api_calls = 40
gpt-4.1-mini total_tokens = 5137

claude-haiku-4-5-20251001 status = passed
claude-haiku-4-5-20251001 route_accuracy mean = 1.0
claude-haiku-4-5-20251001 retrieval_hit_rate mean = 1.0
claude-haiku-4-5-20251001 provider_valid_route_accuracy = 1.0
claude-haiku-4-5-20251001 with_memory_output_validity_rate = 1.0
claude-haiku-4-5-20251001 api_calls = 40
claude-haiku-4-5-20251001 total_tokens = 6269
```

Key tuned `gpt-5-nano` output-budget metrics from `results/architecture/live_provider_model_comparison_nano_max1000_summary.json`:

```text
status = passed
target_count = 1
attempted_count = 1
passed_count = 1
failed_count = 0
max_output_tokens = 1000
provider = openai
model = gpt-5-nano
route_accuracy mean = 1.0
retrieval_hit_rate mean = 1.0
source_label_accuracy mean = 1.0
provider_valid_route_accuracy = 1.0
primitive_failure_count = 0
provider_output_boundary_failure_count = 0
with_memory_output_validity_rate = 1.0
api_calls = 44
input_tokens = 4744
output_tokens = 30454
total_tokens = 35198
```

Key tuned `gpt-5-nano` diagnostics from `results/architecture/live_provider_model_comparison_nano_max1000_openai_gpt-5-nano_diagnostics.json`:

```text
failure_count = 0
empty_response_count = 1
with_memory_empty_attempt_count = 1
with_memory_final_empty_count = 0
without_memory_final_empty_count = 1
route_accuracy_if_provider_output_valid = 1.0
```

Key broader task-family metrics from `results/architecture/live_task_family_billing_refund_gpt41_summary.json`:

```text
status = passed
provider = openai
model = gpt-4.1-mini
case_family = billing_refund
seed_count = 5
max_output_tokens = 300
all_live_replication_gates_met = true
route_accuracy mean = 1.0
retrieval_hit_rate mean = 1.0
source_label_accuracy mean = 1.0
quarantine_recall mean = 1.0
echo_promotion_rate mean = 0.0
mean_fold_force mean = 0.6840612109022107
api_calls = 40
input_tokens = 4385
output_tokens = 806
total_tokens = 5191
audit_events = 20
```

Key broader task-family diagnostics from `results/architecture/live_task_family_billing_refund_gpt41_diagnostics.json`:

```text
failure_count = 0
empty_response_count = 0
route_accuracy_if_provider_output_valid = 1.0
with_memory_output_validity_rate = 1.0
without_memory_output_validity_rate = 1.0
recommendation = No live replication boundary cases detected.
```

Key additional task-family metrics from `results/architecture/live_task_family_security_rotation_gpt41_summary.json`:

```text
status = passed
provider = openai
model = gpt-4.1-mini
case_family = security_rotation
seed_count = 5
max_output_tokens = 300
all_live_replication_gates_met = true
route_accuracy mean = 1.0
retrieval_hit_rate mean = 1.0
source_label_accuracy mean = 1.0
quarantine_recall mean = 1.0
echo_promotion_rate mean = 0.0
mean_fold_force mean = 0.7619822024978019
api_calls = 40
input_tokens = 4402
output_tokens = 819
total_tokens = 5221
audit_events = 20
```

Key additional task-family diagnostics from `results/architecture/live_task_family_security_rotation_gpt41_diagnostics.json`:

```text
failure_count = 0
empty_response_count = 0
route_accuracy_if_provider_output_valid = 1.0
primitive_failure_count = 0
provider_output_boundary_failure_count = 0
with_memory_output_validity_rate = 1.0
without_memory_output_validity_rate = 1.0
recommendation = No live replication boundary cases detected.
```

## Quality Gates

```text
python -m pytest tests/test_validation_primitives.py -q -m "not live"
34 passed, 1 deselected

python -m pytest tests/architecture/test_laundering.py tests/architecture/test_source_inference.py -q
24 passed

python -m pytest tests/test_llm_transition.py tests/test_validation_primitives.py -q -m "not live"
46 passed, 4 deselected

python -m pytest tests/test_validation_primitives.py -q -m live
1 passed, 33 deselected

python -m pytest tests -q -m "not live"
200 passed, 4 deselected
```

## Completion Status

Controlled/toy validation, source inference, inference laundering, 50-seed controlled replication, JSONL replay, fixed-seed summaries, 20-run real-embedding replication, OpenAI live-gate replication, post-rerank N=5 OpenAI live prompt-suite replication, claim-scale N=20 OpenAI live prompt-suite replication, retry-enabled N=5 and N=20 OpenAI live prompt-suite replication, live diagnostic analysis, deterministic rerank boundary regression, provider-empty transition retry instrumentation, route/output gate semantics, provider/model comparison harness, model-specific output-budget tuning, cross-provider comparison, and broader task-family validation across billing/refund and security/credential-rotation domains are executed.

The remaining expansion path is larger provider/model seed counts or additional task families. Route gates are reported both raw and provider-valid, so provider-empty outputs are tracked as provider-output validity rather than primitive route failures. The primitive roadmap gate itself now has controlled, multi-seed, real-embedding, OpenAI live smoke, OpenAI N=20 live, retry-enabled OpenAI N=20 live, OpenAI/Anthropic provider-model matrix evidence, model-specific output-budget evidence, broader task-family evidence, prompt/response audit, post-rerank boundary diagnostic, no-API boundary regression, provider-empty retry evidence, and explicit route/output semantics.
